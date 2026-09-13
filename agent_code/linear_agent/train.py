"""
Online training loop for the linear_agent: the reward function, the
semi-gradient TD(0) weight update, and read-only bomb-survival
instrumentation. Only active when running with --train (see callbacks.py's
setup(), which calls setup_training() below before the first act() call).

Framework entry points (called automatically by the game engine):
    setup_training(self)                         -- once, before training starts
    game_events_occurred(self, s, a, s', events) -- once per non-terminal step
    end_of_round(self, s, a, events)              -- once per round, on the
                                                      final (possibly terminal) step

Everything else is a private helper.
"""

import json
import os
import pickle
import random
from collections import deque, namedtuple
from typing import List

import events as e
import numpy as np

import settings as s

from .callbacks import _epsilon
from .features import (
    ACTIONS,
    ESCAPE_BUDGET,
    GAMMA,
    N_FEATURES,
    coin_potential,
    crate_potential,
    escape_potential,
    q_values,
    state_action_features,
    state_features,
)

# read-only instrumentation window: covers a self-placed bomb's full danger
# lifetime -- the BOMB_TIMER countdown phase (ESCAPE_BUDGET steps) plus the
# EXPLOSION_TIMER-only lingering-blast phase that follows detonation
TRACK_WINDOW = ESCAPE_BUDGET + s.EXPLOSION_TIMER

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

TRANSITION_HISTORY_SIZE = 3

# Decaying learning rate: ALPHA(round) = max(ALPHA_MIN, ALPHA_START / (1 + ALPHA_DECAY_RATE * round)).
# A fixed step size only ever converges (in the SGD sense) to a noise ball
# around a stationary point, with variance proportional to the step size.
# This schedule satisfies the Robbins-Monro conditions (sum(alpha_t) ->
# infinity, sum(alpha_t^2) < infinity) asymptotically, the classical
# requirement for SGD to converge to a point rather than a noise ball.
# ALPHA_DECAY_RATE gives a half-life at round 5000, keeping the rate large
# enough through the phase where most coarse learning happens (epsilon
# reaches its floor by ~round 1000) while still decaying substantially by
# round 50000 to let close-margin decisions actually settle. ALPHA_MIN is a
# floor, mostly inactive within 50k rounds, present as a safety net for
# longer runs so the rate never fully collapses to zero.
ALPHA_START = 0.01
ALPHA_DECAY_RATE = 0.0002
ALPHA_MIN = 0.0005


def _alpha(round: int) -> float:
    return max(ALPHA_MIN, ALPHA_START / (1 + ALPHA_DECAY_RATE * round))

# separate constant so its magnitude can be swept independently of the other
# event rewards below
KILLED_SELF_PENALTY = -10.0

# Direct, immediate reward for a bomb-drop that hits at least one crate,
# paid on the BOMB transition itself rather than waiting for
# CRATE_DESTROYED (which doesn't fire until ESCAPE_BUDGET steps later, once
# the bomb detonates -- see game_events_occurred). A reward that only
# arrives several steps later, through a chain of escape-phase states with
# no feature distinguishing "escaping after a productive bomb" from
# "escaping any other bomb", gives one-step Q-learning no representational
# bridge to carry that credit back to the BOMB action's own weight. Paying
# it directly removes the need for that bootstrap chain entirely.
#
# The magnitude matters: too small (comparable to ordinary training noise)
# and the model can settle into treating BOMB and WAIT as roughly
# indifferent once already adjacent to a crate, which is a much more
# fragile-looking failure than it sounds -- an near-exact tie between two
# actions gets resolved by argmax's arbitrary tie-break, not by the policy.
# 5.0 was chosen empirically to sit comfortably above that noise floor.
CRATE_HIT_BONUS = 5.0

# analysis-only: every CHECKPOINT_INTERVAL rounds, dump a round-numbered copy
# of the weight vector (see end_of_round) so a long run's trajectory can be
# inspected after the fact, without disturbing linear-model.pt (still
# overwritten every round, unchanged) or anything read during training.
CHECKPOINT_INTERVAL = 2000


def setup_training(self):
    """
    Initialise self for training. Called once, after callbacks.setup().
    """
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

    # pure read-only instrumentation: measures what actually happens in the
    # ESCAPE_BUDGET steps following a real (non-synthetic), escape_exists==1
    # bomb drop -- does the agent complete the escape the BFS found, or does
    # something (e.g. coin-seeking) pull it back into its own blast? Never
    # touches self.weights or action selection.
    self.bomb_tracker = None
    self.tracked_bomb_log = []
    # index into tracked_bomb_log already persisted to disk -- see
    # end_of_round's incremental write below
    self.tracked_bomb_log_flushed = 0
    # truncate any stale file from a previous run; new records are appended
    # to it incrementally from here on, never rewritten wholesale
    open("tracked_bomb_log.jsonl", "w").close()


def game_events_occurred(
    self,
    old_game_state: dict,
    self_action: str,
    new_game_state: dict,
    events: List[str],
):
    """
    Called once per non-terminal step. Computes the reward for the
    transition just taken and performs one SGD weight update on it, plus
    (occasionally) a synthetic counterfactual update and read-only
    bomb-survival tracking.
    """
    self.logger.debug(
        f"Encountered game event(s) {', '.join(map(repr, events))} in step {new_game_state['step']}"
    )

    old_state = state_features(old_game_state, use_target_aware_escape=not self.train)
    new_state = state_features(new_game_state, use_target_aware_escape=not self.train)
    reward = reward_from_events(self, events) + potential_shaping(old_state, new_state)

    bomb_feat = state_action_features(old_state, "BOMB")
    if self_action == "BOMB" and bomb_feat[1] == 1.0 and bomb_feat[6] == 1.0:
        reward += CRATE_HIT_BONUS

    self.logger.debug(f"Reward for action {self_action}: {reward}")

    # (s, a, r, s')
    self.transitions.append(Transition(old_state, self_action, new_state, reward))

    # td target = reward + gamma * max(Q(s',a')) (highest q value in next state)
    next_value = np.max(q_values(self.weights, new_state))
    td_target = reward + GAMMA * next_value
    sgd_update(self, old_state, self_action, td_target, round=old_game_state["round"])

    # Synthetic counterfactual update: escape_exists_if_bomb is a
    # deterministic ground-truth BFS check (features._escape_exists_after_bomb),
    # not a learned estimate -- so when it's 0 we already KNOW with
    # certainty that bombing from this exact state is doomed. The safety
    # filter never lets the agent actually experience that outcome for
    # real, so this weight would otherwise have zero real variance to fit
    # against. This injects that known fact directly as an extra gradient
    # step on (old_state, "BOMB"), independent of whatever action was
    # actually taken this step -- it never touches action selection or
    # game state, purely an additional sgd_update call. Firing it with
    # probability epsilon (the same decaying exploration schedule) rather
    # than every step keeps its influence scaled down as training
    # progresses, instead of injecting a constant-rate signal that would
    # dominate once epsilon has decayed near its floor.
    self.prev_synthetic_update = False
    if bomb_feat[1] == 1.0 and bomb_feat[5] == 0.0:
        epsilon = _epsilon(old_game_state["round"])
        if random.random() < epsilon:
            synthetic_target = (GAMMA ** ESCAPE_BUDGET) * KILLED_SELF_PENALTY
            sgd_update(self, old_state, "BOMB", td_target=synthetic_target, round=old_game_state["round"])
            self.prev_synthetic_update = True

    # --- read-only bomb-danger-window instrumentation (no weights touched) ---
    # append this step's data to whatever window is already in progress,
    # BEFORE checking whether a new one starts -- this guarantees the drop
    # step itself is never counted as a tracked step, only the TRACK_WINDOW
    # steps that follow it (ticking phase + lingering-explosion phase)
    _track_escape_step(self, old_state, self_action)
    if self.bomb_tracker is not None and len(self.bomb_tracker["steps"]) >= TRACK_WINDOW:
        # reaching this call at all means the agent is alive (a death would
        # have routed to end_of_round instead), so the window completed safely
        _finalize_bomb_tracker(self, survived=True)

    if self.bomb_tracker is None and self_action == "BOMB":
        if bomb_feat[1] == 1.0:
            # track every valid bomb drop, not just escape_exists==1 ones,
            # so a death can be attributed to a doomed bomb chosen despite
            # the exploration filter (its fallback path allows an unfiltered
            # choice when the agent is genuinely trapped)
            self.bomb_tracker = {
                "round": int(old_game_state["round"]),
                "drop_step": int(old_game_state["step"]),
                "drop_position": [int(old_state["position"][0]), int(old_state["position"][1])],
                "escape_exists_at_drop": float(bomb_feat[5]),
                "crate_hit_at_drop": float(bomb_feat[6]),
                "escape_distance_at_drop": new_state["escape_distance"],
                "steps": [],
            }


def _track_escape_step(self, old_state, action_taken):
    """
    Read-only: if a bomb escape window is currently being tracked, record
    whether this step's action moved toward the nearest coin, whether it was
    itself a move into danger, and whether a strictly safer alternative
    action existed but wasn't taken. Does not modify self.weights.
    """
    if self.bomb_tracker is None:
        return

    taken = state_action_features(old_state, action_taken)

    safer_available = False
    for other in ACTIONS:
        if other == action_taken:
            continue
        other_feat = state_action_features(old_state, other)
        if other_feat[1] == 1.0 and other_feat[4] == 0.0:
            safer_available = True
            break

    self.bomb_tracker["steps"].append(
        {
            "coinward": bool(taken[0] == 1.0),
            "dangerous": bool(taken[4] == 1.0),
            "safer_available": safer_available,
        }
    )


def _finalize_bomb_tracker(self, survived):
    self.bomb_tracker["survived"] = survived
    if not survived:
        # 1-indexed: the step just appended by _track_escape_step is the one
        # death occurred on, so its count IS the death step index
        self.bomb_tracker["death_step_index"] = len(self.bomb_tracker["steps"])
    self.tracked_bomb_log.append(self.bomb_tracker)
    self.bomb_tracker = None


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called once per round, replacing game_events_occurred for the final
    (possibly terminal) step. Performs the last weight update, finalizes
    any in-progress bomb-survival tracking, persists the model, and
    (every CHECKPOINT_INTERVAL rounds) writes an analysis-only checkpoint.
    """
    self.logger.debug(
        f"Encountered event(s) {', '.join(map(repr, events))} in final step"
    )

    old_state = state_features(last_game_state, use_target_aware_escape=not self.train)
    reward = reward_from_events(self, events) + potential_shaping(old_state, None)
    self.logger.debug(f"Reward for action {last_action}: {reward}")

    # (s, a, r, s')
    self.transitions.append(Transition(old_state, last_action, None, reward))

    # target = reward, there is no next state
    sgd_update(self, old_state, last_action, td_target=reward, round=last_game_state["round"])

    # finalize any in-progress escape tracking using this final transition --
    # end_of_round replaces game_events_occurred for exactly the last step of
    # the round, so this is the only chance to log it. Survival is read
    # directly from whether KILLED_SELF fired this step, not inferred.
    if self.bomb_tracker is not None:
        _track_escape_step(self, old_state, last_action)
        _finalize_bomb_tracker(self, survived=(e.KILLED_SELF not in events))
    elif e.KILLED_SELF in events:
        # sanity check: a self-kill with no bomb currently being tracked
        # should be structurally impossible in Task 2 (no opponents means
        # every explosion is self-owned, and at most one bomb is ever
        # pending), but record it explicitly rather than assume
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

    # append-only: write only the records added since the last flush, as
    # newline-delimited JSON, instead of re-serializing the entire
    # accumulated list every round (O(total records) per round, O(n^2)
    # over a full run). self.tracked_bomb_log itself is unchanged and
    # still grows in memory (the doomed-bomb gate in callbacks.py reads
    # its last N entries directly); only its on-disk persistence is
    # incremental. Format: tracked_bomb_log.jsonl, one JSON object per
    # line, not a single JSON array -- read it as
    # `[json.loads(line) for line in f]`.
    with open("tracked_bomb_log.jsonl", "a") as jf:
        for record in self.tracked_bomb_log[self.tracked_bomb_log_flushed:]:
            jf.write(json.dumps(record) + "\n")
    self.tracked_bomb_log_flushed = len(self.tracked_bomb_log)

    # Store the model
    with open("linear-model.pt", "wb") as file:
        pickle.dump(self.weights, file)

    # analysis-only checkpoint: a separate, round-numbered weight snapshot,
    # written alongside (not instead of) the line above -- purely so a long
    # run's weight trajectory can be reconstructed afterward. Never read
    # back during training, so it cannot affect learning or action
    # selection.
    round_num = int(last_game_state["round"])
    if round_num % CHECKPOINT_INTERVAL == 0:
        os.makedirs("checkpoints", exist_ok=True)
        checkpoint_path = os.path.join("checkpoints", f"weights_round_{round_num}.pt")
        with open(checkpoint_path, "wb") as cf:
            pickle.dump(self.weights, cf)


def sgd_update(self, old_state, action, td_target: float, round: int) -> None:
    """
    One semi-gradient TD(0) step on the squared error between td_target
    and the current prediction:

        weights += alpha(round) * (td_target - prediction) * feature_vector

    `round` is required (not read from self) so every call site -- the main
    TD update, the synthetic counterfactual update, and the terminal update
    in end_of_round -- uses the same decaying schedule consistently, keyed
    to whatever round that specific transition actually occurred in.
    """
    features = state_action_features(old_state, action)
    prediction = np.dot(self.weights, features)
    alpha = _alpha(round)
    self.weights += alpha * (td_target - prediction) * features


def reward_from_events(self, events: List[str]) -> float:
    """Sum of the sparse, per-event rewards for one step's events list."""
    game_rewards = {
        e.COIN_COLLECTED: 1.0,
        e.WAITED: -0.15,
        e.INVALID_ACTION: -0.1,
        e.BOMB_DROPPED: -0.05,
        e.CRATE_DESTROYED: 1.5,
        e.COIN_FOUND: 2.0,
        e.KILLED_SELF: KILLED_SELF_PENALTY,
    }
    reward_sum = 0.0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def potential_shaping(old_state, new_state) -> float:
    """
    Potential-based reward shaping: reward = gamma * phi(s') - phi(s),
    where phi = coin_potential + escape_potential + crate_potential is the
    sum of three independent potential functions. Ng, Harada & Russell
    (1999): any potential function preserves the optimal policy, and the
    sum of several valid potential functions is itself one.

    crate_hit_if_bomb's credit is handled separately, as a direct
    CRATE_HIT_BONUS on the BOMB transition itself (see
    game_events_occurred) rather than through a potential here -- a
    potential's 0->1->0 shape over a bomb's lifetime unavoidably decays
    away and inverts sign before detonation, landing its penalty on a
    later, unrelated action instead of the BOMB decision it's meant to
    reward.

    crate_potential pairs with the moves_to_crate feature the same way
    coin_potential pairs with moves_to_coin: it gives directional guidance
    toward the nearest crate-adjacent tile on the steps where no coin is
    visible.
    """
    old_potential = (
        coin_potential(old_state)
        + escape_potential(old_state)
        + crate_potential(old_state)
    )

    if new_state is None:
        new_potential = 0.0
    else:
        new_potential = (
            coin_potential(new_state)
            + escape_potential(new_state)
            + crate_potential(new_state)
        )

    return GAMMA * new_potential - old_potential
