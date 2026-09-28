"""
Fitted Q-Iteration (Ernst, Geurts & Wehenkel, 2005) for the tree_agent.
Trees can't be updated per sample, so transitions are collected in a replay
window and every REFIT_INTERVAL rounds a fresh ExtraTreesRegressor is fit on
it, FQI_SWEEPS times, bootstrapping targets off the previous fit.
"""

import json
import os
import pickle
from collections import deque
from typing import List

import events as e
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

import settings as s

from .features import (
    ACTIONS,
    ESCAPE_BUDGET,
    F_CRATES_IF_BOMB,
    F_ESCAPE_IF_BOMB,
    F_ESCAPE_MOVE,
    F_HITS_OPPONENT,
    F_INTO_DANGER,
    F_IS_BOMB,
    F_IS_WAIT,
    F_MOVES_TO_COIN,
    F_MOVES_TO_OPPONENT,
    F_TARGET_DELTA,
    F_VALID,
    GAMMA,
    HUNT_EXEMPT_FROM_AWAY_PENALTY,
    N_FEATURES,
    coin_potential,
    crate_potential,
    escape_potential,
    opponent_potential,
    state_action_features,
    state_features,
)

# Replay window in transitions (about 250 rounds). A shorter window fills
# up with loop data as soon as the greedy policy gets stuck, after which no
# refit can recover; a refit on 100k x 16 still takes only seconds.
REPLAY_CAPACITY = 100_000

# Rounds between refits. 10 looked better on training-time stats but the
# greedy policy behind it was locked into a two-tile oscillation.
REFIT_INTERVAL = 25

# Every this many rounds (on a refit round) the model is also saved to
# checkpoints/model_round_{N}.pt, so a run can be evaluated per checkpoint
# afterwards; greedy quality is not monotone in training length. 0 disables.
CHECKPOINT_INTERVAL = 250

# If set, the replay window is pickled there on every refit round.
REPLAY_DUMP_PATH = os.environ.get("REPLAY_DUMP_PATH")

# FQI sweeps per refit; the first sweep of the first refit fits raw rewards.
FQI_SWEEPS = 3

N_ESTIMATORS = 50
MIN_SAMPLES_LEAF = 5

# steps a self-placed bomb stays dangerous: countdown plus lingering blast
BOMB_TRACK_STEPS = ESCAPE_BUDGET + s.EXPLOSION_TIMER

DEATH_PENALTY = -10.0
KILL_REWARD = 10.0

# Paid on the BOMB transition itself when the blast covers a crate /
# an opponent. The engine's CRATE_DESTROYED / KILLED_OPPONENT events arrive
# several escape steps later and one-step TD never carried that credit back
# to the bomb.
CRATE_HIT_BONUS = 5.0
OPPONENT_HIT_BONUS = 5.0

# Paid for a step that reduces the distance to the navigation target.
# Without it the greedy policy sat on WAIT for almost the whole round; the
# shaping differential alone was too small for the trees to pick up.
MOVE_TOWARD_BONUS = 3.0

# Also charge -MOVE_TOWARD_BONUS for a step away from the target. One-sided
# credit can be farmed by oscillating next to a target.
MOVE_BONUS_ANTISYMMETRIC = True


def setup_training(self):
    """
    Initialise self for training. Called once, after callbacks.setup().
    """
    self.replay = deque(maxlen=REPLAY_CAPACITY)

    # instrumentation only: what happens in the steps after each own bomb
    self.bomb_tracker = None
    self.tracked_bomb_log = []
    self.tracked_bomb_log_flushed = 0
    open("tracked_bomb_log.jsonl", "w").close()


def game_events_occurred(
    self,
    old_game_state: dict,
    self_action: str,
    new_game_state: dict,
    events: List[str],
):
    """
    Called once per non-terminal step. Appends the transition to the replay
    window; learning only happens in end_of_round's periodic refit.
    """
    self.logger.debug(
        f"Encountered game event(s) {', '.join(map(repr, events))} in step {new_game_state['step']}"
    )

    old_state = state_features(
        old_game_state, use_target_aware_escape=False, opponent_has_bombed=self.opponent_has_bombed
    )
    new_state = state_features(
        new_game_state, use_target_aware_escape=False, opponent_has_bombed=self.opponent_has_bombed
    )
    phi = state_action_features(old_state, self_action)
    reward = _transition_reward(self, old_state, phi, self_action, events, new_state)
    self.logger.debug(f"Reward for action {self_action}: {reward}")

    next_phis = np.stack([state_action_features(new_state, a) for a in ACTIONS])
    self.replay.append((phi, reward, next_phis, False))

    _record_escape_step(self, old_state, self_action)
    if self.bomb_tracker is not None and len(self.bomb_tracker["steps"]) >= BOMB_TRACK_STEPS:
        _close_bomb_record(self, survived=True)

    if self.bomb_tracker is None and self_action == "BOMB":
        if phi[F_VALID] == 1.0:
            self.bomb_tracker = {
                "round": int(old_game_state["round"]),
                "drop_step": int(old_game_state["step"]),
                "drop_position": [int(old_state["position"][0]), int(old_state["position"][1])],
                "escape_exists_at_drop": float(phi[F_ESCAPE_IF_BOMB]),
                "crate_count_at_drop": float(phi[F_CRATES_IF_BOMB]),
                "escape_distance_at_drop": new_state["escape_distance"],
                "steps": [],
            }


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    self.logger.debug(f"Encountered event(s) {', '.join(map(repr, events))} in final step")

    old_state = state_features(
        last_game_state, use_target_aware_escape=False, opponent_has_bombed=self.opponent_has_bombed
    )
    phi = state_action_features(old_state, last_action)
    reward = _transition_reward(self, old_state, phi, last_action, events, None)
    self.logger.debug(f"Reward for action {last_action}: {reward}")

    # terminal: the next_phis placeholder is masked out via `done`
    self.replay.append((phi, reward, np.zeros((len(ACTIONS), N_FEATURES)), True))

    if self.bomb_tracker is not None:
        _record_escape_step(self, old_state, last_action)
        _close_bomb_record(self, survived=(e.KILLED_SELF not in events))
    elif e.KILLED_SELF in events:
        self.tracked_bomb_log.append(
            {
                "round": int(last_game_state["round"]),
                "drop_step": None,
                "drop_position": None,
                "escape_exists_at_drop": None,
                "steps": [],
                "survived": False,
                "death_step_index": None,
                "untracked_death": True,
            }
        )

    with open("tracked_bomb_log.jsonl", "a") as f:
        for record in self.tracked_bomb_log[self.tracked_bomb_log_flushed:]:
            f.write(json.dumps(record) + "\n")
    self.tracked_bomb_log_flushed = len(self.tracked_bomb_log)

    round_num = int(last_game_state["round"])
    if round_num % REFIT_INTERVAL == 0 and len(self.replay) > 0:
        self.model = fitted_q_iteration(self.replay, self.model, sweeps=FQI_SWEEPS)
        if REPLAY_DUMP_PATH:
            with open(REPLAY_DUMP_PATH, "wb") as f:
                pickle.dump(list(self.replay), f)

    if self.model is not None:
        with open("tree-model.pt", "wb") as f:
            pickle.dump(self.model, f)
        if CHECKPOINT_INTERVAL and round_num % CHECKPOINT_INTERVAL == 0:
            os.makedirs("checkpoints", exist_ok=True)
            with open(os.path.join("checkpoints", f"model_round_{round_num}.pt"), "wb") as f:
                pickle.dump(self.model, f)


def _transition_reward(self, old_state, phi, action, events, new_state):
    reward = reward_from_events(self, events) + potential_shaping(old_state, new_state)
    if action == "BOMB" and phi[F_VALID] == 1.0 and phi[F_CRATES_IF_BOMB] >= 1.0:
        reward += CRATE_HIT_BONUS
    if action == "BOMB" and phi[F_VALID] == 1.0 and phi[F_HITS_OPPONENT] == 1.0:
        reward += OPPONENT_HIT_BONUS
    reward += move_bonus(old_state, phi)
    return reward


def _record_escape_step(self, old_state, action_taken):
    if self.bomb_tracker is None:
        return

    taken = state_action_features(old_state, action_taken)

    safer_available = False
    for other in ACTIONS:
        if other == action_taken:
            continue
        other_feat = state_action_features(old_state, other)
        if other_feat[F_VALID] == 1.0 and other_feat[F_INTO_DANGER] == 0.0:
            safer_available = True
            break

    # dangerous / safer_available are always 0 inside an escape window;
    # took_correct_escape_move is the informative one there
    in_escape_window = bool(old_state["in_escape_window"])
    escape_direction_exists = old_state["escape_direction"] is not None
    self.bomb_tracker["steps"].append(
        {
            "coinward": bool(taken[F_MOVES_TO_COIN] == 1.0),
            "dangerous": bool(taken[F_INTO_DANGER] == 1.0),
            "safer_available": safer_available,
            "in_escape_window": in_escape_window,
            "escape_direction_exists": escape_direction_exists,
            "took_correct_escape_move": bool(taken[F_ESCAPE_MOVE] == 1.0),
        }
    )


def _close_bomb_record(self, survived):
    self.bomb_tracker["survived"] = survived
    if not survived:
        self.bomb_tracker["death_step_index"] = len(self.bomb_tracker["steps"])
    self.tracked_bomb_log.append(self.bomb_tracker)
    self.bomb_tracker = None


def fitted_q_iteration(replay, prev_model, sweeps: int):
    """target = r + GAMMA * max_a Q_prev(s', a) (just r if done or no model yet), refit, repeat."""
    phis = np.stack([t[0] for t in replay])
    rewards = np.array([t[1] for t in replay])
    next_phis = np.stack([t[2] for t in replay])  # (N, len(ACTIONS), N_FEATURES)
    done = np.array([t[3] for t in replay])

    model = prev_model
    for _ in range(sweeps):
        if model is None:
            targets = rewards.copy()
        else:
            flat_next = next_phis.reshape(-1, N_FEATURES)
            next_q = model.predict(flat_next).reshape(len(replay), len(ACTIONS))
            max_next_q = next_q.max(axis=1)
            max_next_q[done] = 0.0
            targets = rewards + GAMMA * max_next_q

        # n_jobs left at 1: the per-step predict is too small for the
        # multiprocessing overhead to pay off
        model = ExtraTreesRegressor(
            n_estimators=N_ESTIMATORS,
            min_samples_leaf=MIN_SAMPLES_LEAF,
        )
        model.fit(phis, targets)

    return model


def move_bonus(old_state, phi):
    """+bonus toward the target, -bonus away (if antisymmetric); nothing for WAIT/BOMB/invalid/escaping."""
    if phi[F_VALID] != 1.0 or phi[F_IS_WAIT] == 1.0 or phi[F_IS_BOMB] == 1.0:
        return 0.0
    if old_state["in_escape_window"] or old_state["target_kind"] is None:
        return 0.0
    if phi[F_TARGET_DELTA] <= -1.0:
        return MOVE_TOWARD_BONUS
    if not MOVE_BONUS_ANTISYMMETRIC or phi[F_TARGET_DELTA] < 1.0:
        return 0.0
    if HUNT_EXEMPT_FROM_AWAY_PENALTY and phi[F_MOVES_TO_OPPONENT] == 1.0:
        return 0.0
    return -MOVE_TOWARD_BONUS


def reward_from_events(self, events: List[str]) -> float:
    # OPPONENT_ELIMINATED is deliberately absent: it fires for any opponent
    # death, KILLED_OPPONENT is the one that means we did it
    game_rewards = {
        e.COIN_COLLECTED: 1.0,
        e.WAITED: -0.15,
        e.INVALID_ACTION: -0.1,
        e.BOMB_DROPPED: -0.05,
        e.CRATE_DESTROYED: 1.5,
        e.COIN_FOUND: 2.0,
        e.KILLED_SELF: DEATH_PENALTY,
        e.GOT_KILLED: DEATH_PENALTY,
        e.KILLED_OPPONENT: KILL_REWARD,
    }
    reward_sum = 0.0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def _potential(state):
    return (
        coin_potential(state)
        + escape_potential(state)
        + crate_potential(state)
        + opponent_potential(state)
    )


def potential_shaping(old_state, new_state) -> float:
    """GAMMA * phi(s') - phi(s); new_state None is terminal (phi = 0)."""
    old_potential = _potential(old_state)
    new_potential = 0.0 if new_state is None else _potential(new_state)
    return GAMMA * new_potential - old_potential
