"""
Fitted Q-Iteration training loop for the tree_agent (Ernst, Geurts &
Wehenkel, 2005): unlike linear_agent's per-step semi-gradient TD(0) update,
a tree ensemble can't be updated incrementally one sample at a time -- so
this instead accumulates transitions into a replay buffer and periodically
(every REFIT_INTERVAL rounds) batch-refits a fresh ExtraTreesRegressor on
the whole buffer, bootstrapping TD targets off the previous refit's model.
Only active when running with --train (see callbacks.py's setup(), which
calls setup_training() below before the first act() call).

Task 2/3 scope: reward shaping (potential-based coin/crate/escape/opponent
guidance, the direct CRATE_HIT_BONUS/OPPONENT_HIT_BONUS credit) is ported
from linear_agent essentially unchanged -- these are properties of the
reward signal, not the function approximator, so they transfer directly.
What does NOT transfer is linear_agent's per-step SGD update or its
synthetic counterfactual weight-nudges (Task 2's escape_exists_if_bomb one,
Task 3's escape_correct_move one -- see features.py's module docstring) --
there is no per-sample weight to nudge here; all learning happens in the
batch refit below.

Framework entry points (called automatically by the game engine):
    setup_training(self)                         -- once, before training starts
    game_events_occurred(self, s, a, s', events) -- once per non-terminal step
    end_of_round(self, s, a, events)              -- once per round, on the
                                                      final (possibly terminal) step

Everything else is a private helper.
"""

import json
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

# How many (phi, reward, next_phis, done) transitions to keep around. A
# sliding window (deque with maxlen), not an ever-growing list -- Fitted
# Q-Iteration classically refits on a fixed batch, but for an online variant
# a bounded window keeps each refit's cost from growing without limit over a
# long run, at the cost of "forgetting" the earliest (least-refined-policy)
# experience once the buffer fills up.
#
# 100,000 = 250 full rounds (was 20,000 = 50 rounds). With a 50-round
# window, the moment the greedy policy slips into a degenerate loop (WAIT
# next to a crate, a two-tile oscillation) the whole buffer is loop data
# within 50 rounds and every later refit only ever sees that -- the
# collapse seals itself. Screening the routing change (1,500 rounds,
# pure-greedy 50-round eval): 20k window + one-sided move bonus 36.9
# coins/round and 0 full clears; 100k + antisymmetric bonus (+ target-aware
# escape in training) 47-50 coins/round and 47-49 full clears. A refit on
# 100k x 16 is still a few seconds.
REPLAY_CAPACITY = 100_000

# Refit every this many rounds. Task 2's board has crates, so rounds run
# longer than Task 1's (episodes no longer end the instant all coins are
# grabbed) -- kept at Task 1's sweep-selected value throughout Tasks 2-4.
#
# A re-sweep (metrics/tree_agent/refit_interval_sweep.py: REFIT_INTERVAL in
# {10, 25, 50, 100}, 2000 rounds each on loot-crate) was tried, reading
# coins/round from the training-time --save-stats breakdown only (epsilon
# still at its 0.05 floor by round 2000, matching Task 1's own sweep
# methodology). REFIT_INTERVAL=10 looked like a clear win there (20.63
# coins/round in the final 200 training rounds vs. 25's 15.87), so a full
# 5,000-round model was trained at 10 and evaluated PURE-GREEDY -- which
# revealed the training-time read was actively misleading: 0.03 coins/round,
# 0.06 bombs/round (score 6 across 200 full rounds). A direct trace found
# the greedy policy locked into a deterministic two-tile corner oscillation,
# refusing BOMB even when valid and safe (Q(BOMB)=51.089 vs. Q(DOWN)=51.133,
# a near-tie it never won). More frequent refits apparently locked in this
# degenerate pattern from early training before enough diverse experience
# could correct it -- the training-time metric only looked better because
# epsilon-floor exploration noise was randomly kicking the agent out of the
# loop often enough to look like real progress, masking that the underlying
# greedy policy never actually learned to escape it. Same "training stats
# mislead, pure-greedy reveals the truth" lesson this project has hit
# repeatedly (Task 2's WAIT-collapse, Task 3's OPPONENT_POTENTIAL_SCALE
# bug), via a new mechanism: this time the sweep's OWN methodology (reading
# training-time metrics only, no per-config pure-greedy eval, mirroring
# Task 1's sweep) was the thing that got fooled, not a reward term. Reverted
# to 25, the only value validated end-to-end at pure-greedy evaluation.
REFIT_INTERVAL = 25

# Number of Fitted-Q-Iteration sweeps per refit call: each sweep refits a
# new regressor on targets bootstrapped off the PREVIOUS sweep's regressor
# (or off raw rewards alone, for the very first sweep of the very first
# refit, when no model exists yet). This is what "Fitted Q-Iteration" means
# -- repeated batch fits on the same data, not a single fit.
FQI_SWEEPS = 3

# Task 1's sweep-selected defaults. Diagnosing a WAIT-collapse problem (a
# pure-greedy Task 2 evaluation spent 388/400 steps of a round on WAIT) led
# to a test at N_ESTIMATORS=150/MIN_SAMPLES_LEAF=2 (Task 1's largest swept
# capacity) on the theory that WAIT was winning a widespread near-tie the
# same way the escape-direction bug did -- that test produced an
# essentially identical WAIT-collapse pattern (391/400), ruling out model
# capacity as the cause. Reverted back to these defaults; see
# MOVE_TOWARD_BONUS below for the fix that was actually diagnosed to work.
N_ESTIMATORS = 50
MIN_SAMPLES_LEAF = 5

# read-only instrumentation window: covers a self-placed bomb's full danger
# lifetime -- the BOMB_TIMER countdown phase (ESCAPE_BUDGET steps) plus the
# EXPLOSION_TIMER-only lingering-blast phase that follows detonation.
# Ported from linear_agent: feeds the doomed-bomb gate in callbacks.py.
TRACK_WINDOW = ESCAPE_BUDGET + s.EXPLOSION_TIMER

KILLED_SELF_PENALTY = -10.0

# Task 3: dying to an OPPONENT's bomb (GOT_KILLED) is scored the same as
# dying to our own (KILLED_SELF_PENALTY) -- death is death from the agent's
# perspective. Ported from linear_agent unchanged.
GOT_KILLED_PENALTY = -10.0

# Deliberately larger in magnitude than CRATE_HIT_BONUS: killing an opponent
# is meant to dominate the reward landscape once one is reachable, not just
# edge out crate-clearing. Ported from linear_agent unchanged.
KILLED_OPPONENT_REWARD = 10.0

# Direct, immediate reward for a bomb-drop that hits at least one crate,
# paid on the BOMB transition itself rather than waiting for
# CRATE_DESTROYED (which doesn't fire until ESCAPE_BUDGET steps later, once
# the bomb detonates). Ported from linear_agent unchanged: a reward that
# only arrives several steps later, through a chain of escape-phase states
# with no feature distinguishing "escaping after a productive bomb" from
# "escaping any other bomb", gives one-step-bootstrapped TD no
# representational bridge to carry that credit back to the BOMB action --
# this is a property of the multi-step credit-assignment gap itself, not of
# the linear model specifically, so the same fix is expected to be needed
# here too (see reward_from_events/game_events_occurred).
CRATE_HIT_BONUS = 5.0

# Additional direct credit per crate BEYOND THE FIRST that a bomb would
# destroy (features.py's crate_count_if_bomb). Diagnosed as the next lever
# after confirming tree_agent's Task 2 coins/round gap versus linear_agent
# is purely a step-budget problem, not a decision-quality one: given a
# generously extended step budget, every one of 30 test rounds reached a
# perfect 50/50 coins in 428-515 steps -- well under the extended budget,
# but over the real 400-step one. With ~40 bombs/round, each costing a
# multi-step mandatory escape window that can't be shortened (fixed by
# BOMB_TIMER/EXPLOSION_TIMER), the escape-window overhead itself is the
# dominant cost, not travel routing between bombs. Fewer, higher-yield
# bombs directly cuts that overhead -- this is what actually creates the
# INCENTIVE for it: CRATE_HIT_BONUS alone treats a 1-crate and a 3-crate
# bomb identically, giving the model no reason to prefer the latter.
# Deliberately smaller than CRATE_HIT_BONUS itself, and added on top of it
# rather than replacing it, so hitting at least one crate is never worth
# less than before -- this only adds upward pressure for hitting more.
#
# First tried at 1.5 (a 3-crate bomb worth 8.0 total vs. a 1-crate bomb's
# 5.0, a 60% premium) and reverted: a full 5,000-round retrain + pure-greedy
# evaluation showed a real regression, not an improvement -- bombs/round
# fell by more than half (34.6 -> 15.1) and crates-per-bomb rose sharply
# (1.2 -> 4.1), but TOTAL crates cleared per round still dropped (110.7 ->
# 62.6) and coins/round collapsed (44.97 -> 25.05). The agent was actively
# detouring to hunt rare high-yield clusters instead of opportunistically
# clearing whatever crate was nearest -- the travel-time cost of chasing
# yield outweighed the escape-overhead savings it was meant to buy.
#
# Then tried at 0.5 (a 20% premium instead of 60%), on the assumption the
# regression was purely a magnitude/overcorrection problem. It wasn't: a
# second full retrain + evaluation barely moved anything (coins/round
# 25.05 -> 25.55, bombs/round 15.1 -> 16.64) -- nowhere close to recovering
# toward the pre-bonus 44.97/34.6 baseline despite cutting the premium by
# 3x. That non-response falsifies the "magnitude was too aggressive"
# theory: whatever is suppressing bomb-dropping isn't scaling with this
# constant. Set to 0.0 here as a clean ablation -- this makes the reward
# function for BOMB identical to the pre-crate-bonus baseline (pure
# CRATE_HIT_BONUS, no crate-count grading at all) while leaving every other
# change from that period in place (the BOMB-validity escape-existence
# gate, the shared _safety_mask refactor covering the greedy branch). If
# bombs/round recovers to ~34/round here, the crate-count reward mechanism
# itself is the cause, at any nonzero weight. If it stays suppressed at
# ~16-17/round even at 0.0, the cause is one of those other concurrent
# changes, not this constant, and it isn't safe to reintroduce a nonzero
# value until that's found separately.
EXTRA_CRATE_BONUS = 0.0

# Direct, immediate reward for taking an action that moves toward the
# nearest visible coin or, if none is visible, the nearest crate
# (moves_to_coin==1 or moves_to_crate==1). Diagnosed as necessary, not
# assumed: pure-greedy evaluation showed the agent spending 388-391 of 400
# steps per round on WAIT, taking only a handful of bombs/moves total.
# Direct Q-value inspection at a representative stuck state found
# Q(WAIT)=28.993 vs. Q(the verified coin-toward move)=27.955 -- WAIT ahead
# by about 1.0, consistently, across many otherwise-different states,
# despite potential-based shaping's own arithmetic favoring a real
# distance-reducing move by nearly a full point over WAIT for any nonzero
# distance (reward = distance*(1-GAMMA) for WAIT vs.
# distance*(1-GAMMA) + GAMMA for a productive move -- see
# potential_shaping). A single-config capacity increase (N_ESTIMATORS 150,
# MIN_SAMPLES_LEAF 2) produced an unchanged pattern, ruling out coarse
# leaf-averaging as the cause. This pays the same kind of direct,
# single-transition credit CRATE_HIT_BONUS already pays for a productive
# bomb -- sidesteps whatever is diluting the shaped-reward differential
# rather than trying to first explain it.
MOVE_TOWARD_BONUS = 3.0

# How MOVE_TOWARD_BONUS is paid (see move_bonus). False = the original
# one-sided credit (+bonus for a step that reduces the distance to the
# navigation target, nothing otherwise). True = antisymmetric (+bonus
# toward, -bonus away), which closes the one-sided version's exploit: a
# two-tile oscillation next to a target collects +bonus every other step
# forever at zero risk, and an earlier 1,500-round ablation of this agent
# found that exact loop (or a WAIT-collapse cousin) freezing every
# pure-greedy policy, pre-routing code included, unless the credit was
# antisymmetric AND the replay window was long enough to keep pre-collapse
# experience. Both are screened again on top of the routing change rather
# than assumed. Screened True on the routing change: see REPLAY_CAPACITY.
MOVE_BONUS_ANTISYMMETRIC = True


# Task 3: same direct-credit mechanism as CRATE_HIT_BONUS, applied
# proactively instead of being discovered the hard way -- KILLED_OPPONENT
# wouldn't fire until several steps after the triggering bomb, through the
# same kind of feature-less escape-phase states that made CRATE_HIT_BONUS
# necessary in the first place. Paid immediately on the BOMB transition
# whenever bomb_hits_opponent_if_bomb == 1. Ported from linear_agent at the
# same starting magnitude (5.0, matching CRATE_HIT_BONUS) as a first guess,
# not yet independently swept for this architecture.
OPPONENT_HIT_BONUS = 5.0


def setup_training(self):
    """
    Initialise self for training. Called once, after callbacks.setup().
    """
    self.replay = deque(maxlen=REPLAY_CAPACITY)

    # pure read-only instrumentation: measures what actually happens in the
    # ESCAPE_BUDGET steps following a real (non-synthetic), escape_exists==1
    # bomb drop -- feeds callbacks.py's doomed-bomb gate. Never touches
    # replay/model selection.
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
    Called once per non-terminal step. Computes the reward for the
    transition just taken and appends it to the replay buffer -- no
    learning happens here directly, only in end_of_round's periodic refit.
    """
    self.logger.debug(
        f"Encountered game event(s) {', '.join(map(repr, events))} in step {new_game_state['step']}"
    )

    # use_target_aware_escape=False: train.py only ever runs during
    # training, and the smarter routing is validated only at evaluation
    # time on top of already-trained weights -- see features.py's
    # state_features docstring.
    old_state = state_features(
        old_game_state, use_target_aware_escape=False, opponent_has_bombed=self.opponent_has_bombed
    )
    new_state = state_features(
        new_game_state, use_target_aware_escape=False, opponent_has_bombed=self.opponent_has_bombed
    )
    reward = reward_from_events(self, events) + potential_shaping(old_state, new_state)

    phi = state_action_features(old_state, self_action)
    if self_action == "BOMB" and phi[1] == 1.0 and phi[6] >= 1.0:
        reward += CRATE_HIT_BONUS + EXTRA_CRATE_BONUS * max(0.0, phi[6] - 1.0)
    if self_action == "BOMB" and phi[1] == 1.0 and phi[10] == 1.0:
        reward += OPPONENT_HIT_BONUS
    reward += move_bonus(old_state, phi)

    self.logger.debug(f"Reward for action {self_action}: {reward}")

    next_phis = np.stack([state_action_features(new_state, a) for a in ACTIONS])
    self.replay.append((phi, reward, next_phis, False))

    # --- read-only bomb-danger-window instrumentation (ported from linear_agent) ---
    _track_escape_step(self, old_state, self_action)
    if self.bomb_tracker is not None and len(self.bomb_tracker["steps"]) >= TRACK_WINDOW:
        _finalize_bomb_tracker(self, survived=True)

    if self.bomb_tracker is None and self_action == "BOMB":
        if phi[1] == 1.0:
            self.bomb_tracker = {
                "round": int(old_game_state["round"]),
                "drop_step": int(old_game_state["step"]),
                "drop_position": [int(old_state["position"][0]), int(old_state["position"][1])],
                "escape_exists_at_drop": float(phi[5]),
                "crate_count_at_drop": float(phi[6]),
                "escape_distance_at_drop": new_state["escape_distance"],
                "steps": [],
            }


def _track_escape_step(self, old_state, action_taken):
    """
    Read-only: if a bomb escape window is currently being tracked, record
    whether this step's action moved toward the nearest coin, whether it was
    itself a move into danger, and whether a strictly safer alternative
    action existed but wasn't taken.
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

    # dangerous/safer_available (moves_into_avoidable_danger, index 4) are
    # structurally zeroed for EVERY action during in_escape_window (see
    # features.py) -- meaningless here, kept only for parity with
    # linear_agent's original instrumentation. escape_correct_move (index
    # 7) is what actually carries information during a tracked window: did
    # the action taken match the verified escape direction, and did one
    # exist at all this step (a real route can run out mid-escape even
    # though it existed at drop time, if this step is past its budget).
    in_escape_window = bool(old_state["in_escape_window"])
    escape_direction_exists = old_state["escape_direction"] is not None
    self.bomb_tracker["steps"].append(
        {
            "coinward": bool(taken[0] == 1.0),
            "dangerous": bool(taken[4] == 1.0),
            "safer_available": safer_available,
            "in_escape_window": in_escape_window,
            "escape_direction_exists": escape_direction_exists,
            "took_correct_escape_move": bool(taken[7] == 1.0),
        }
    )


def _finalize_bomb_tracker(self, survived):
    self.bomb_tracker["survived"] = survived
    if not survived:
        self.bomb_tracker["death_step_index"] = len(self.bomb_tracker["steps"])
    self.tracked_bomb_log.append(self.bomb_tracker)
    self.bomb_tracker = None


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called once per round, replacing game_events_occurred for the final
    (possibly terminal) step. Appends the final transition, finalizes any
    in-progress bomb-survival tracking, persists the model, and -- every
    REFIT_INTERVAL rounds -- runs a fresh batch of Fitted-Q-Iteration
    sweeps over the whole replay buffer.
    """
    self.logger.debug(f"Encountered event(s) {', '.join(map(repr, events))} in final step")

    old_state = state_features(
        last_game_state, use_target_aware_escape=False, opponent_has_bombed=self.opponent_has_bombed
    )
    reward = reward_from_events(self, events) + potential_shaping(old_state, None)

    phi = state_action_features(old_state, last_action)
    if last_action == "BOMB" and phi[1] == 1.0 and phi[6] >= 1.0:
        reward += CRATE_HIT_BONUS + EXTRA_CRATE_BONUS * max(0.0, phi[6] - 1.0)
    if last_action == "BOMB" and phi[1] == 1.0 and phi[10] == 1.0:
        reward += OPPONENT_HIT_BONUS
    reward += move_bonus(old_state, phi)

    self.logger.debug(f"Reward for action {last_action}: {reward}")

    # terminal: no next_phis to bootstrap off of -- the placeholder zeros
    # are never read, since fitted_q_iteration masks them out via `done`
    self.replay.append((phi, reward, np.zeros((len(ACTIONS), N_FEATURES)), True))

    if self.bomb_tracker is not None:
        _track_escape_step(self, old_state, last_action)
        _finalize_bomb_tracker(self, survived=(e.KILLED_SELF not in events))
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

    with open("tracked_bomb_log.jsonl", "a") as jf:
        for record in self.tracked_bomb_log[self.tracked_bomb_log_flushed:]:
            jf.write(json.dumps(record) + "\n")
    self.tracked_bomb_log_flushed = len(self.tracked_bomb_log)

    round_num = int(last_game_state["round"])
    if round_num % REFIT_INTERVAL == 0 and len(self.replay) > 0:
        self.model = fitted_q_iteration(self.replay, self.model, sweeps=FQI_SWEEPS)

    if self.model is not None:
        with open("tree-model.pt", "wb") as file:
            pickle.dump(self.model, file)


def fitted_q_iteration(replay, prev_model, sweeps: int):
    """
    Runs `sweeps` rounds of Fitted Q-Iteration over the entire replay
    buffer and returns the final regressor.

    Each sweep:
      1. target_i = reward_i                                     if done_i
                   = reward_i + GAMMA * max_a Q_model(next_s_i, a) otherwise
         using the model from the PREVIOUS sweep (or `prev_model`, which may
         be None on the very first call -- targets are then just the raw
         rewards, since there's nothing to bootstrap off yet).
      2. Fit a fresh ExtraTreesRegressor on (phi_i -> target_i) for the
         whole buffer.

    Unlike linear_agent's single SGD step per transition, this re-fits from
    scratch on every sweep -- that's what makes it "batch" reinforcement
    learning rather than online.
    """
    phis = np.stack([t[0] for t in replay])
    rewards = np.array([t[1] for t in replay])
    next_phis = np.stack([t[2] for t in replay])  # shape (N, len(ACTIONS), N_FEATURES)
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

        # n_jobs deliberately left at its default (1, no multiprocessing):
        # see features.py / this module's docstring for the measured
        # multiprocessing-overhead reason.
        model = ExtraTreesRegressor(
            n_estimators=N_ESTIMATORS,
            min_samples_leaf=MIN_SAMPLES_LEAF,
        )
        model.fit(phis, targets)

    return model


def move_bonus(old_state, phi):
    """
    Direct movement credit for one transition (see MOVE_TOWARD_BONUS and
    MOVE_BONUS_ANTISYMMETRIC). One-sided: +MOVE_TOWARD_BONUS iff the action
    was a valid step toward the navigation target (moves_to_coin or
    moves_to_crate == 1, exactly the original rule). Antisymmetric: the
    same +bonus toward, and -bonus for a valid step that moves away
    (target_distance_delta, feature 12, == +1). Never paid for WAIT, BOMB,
    invalid moves or moves inside an escape window (the forced escape route
    is not a navigation choice), and not when there is no target.
    """
    if phi[1] != 1.0 or phi[2] == 1.0 or phi[3] == 1.0:
        return 0.0
    if old_state["in_escape_window"] or old_state["target_kind"] is None:
        return 0.0
    # toward the selected target, whatever its kind (coin, bomb spot, or --
    # with OPPONENTS_AS_TARGETS -- an opponent): feature 12 is the signed
    # change in BFS distance to it, so this reproduces the original
    # moves_to_coin / moves_to_crate rule for coin and spot targets
    if phi[12] <= -1.0:
        return MOVE_TOWARD_BONUS
    if not MOVE_BONUS_ANTISYMMETRIC or phi[12] < 1.0:
        return 0.0
    if HUNT_EXEMPT_FROM_AWAY_PENALTY and phi[9] == 1.0:
        return 0.0
    return -MOVE_TOWARD_BONUS


def reward_from_events(self, events: List[str]) -> float:

    """
    Sum of the sparse, per-event rewards for one step's events list.
    Task 2 adds bombing/crate events on top of Task 1's coin-only set;
    Task 3 adds opponent-hunting events on top of that -- magnitudes ported
    from linear_agent's own values unchanged. OPPONENT_ELIMINATED is
    deliberately not rewarded here: it fires whenever *any* opponent dies,
    including from their own mistakes, not only when this agent caused it --
    KILLED_OPPONENT is the event that actually means "we did this."
    """
    game_rewards = {
        e.COIN_COLLECTED: 1.0,
        e.WAITED: -0.15,
        e.INVALID_ACTION: -0.1,
        e.BOMB_DROPPED: -0.05,
        e.CRATE_DESTROYED: 1.5,
        e.COIN_FOUND: 2.0,
        e.KILLED_SELF: KILLED_SELF_PENALTY,
        e.GOT_KILLED: GOT_KILLED_PENALTY,
        e.KILLED_OPPONENT: KILLED_OPPONENT_REWARD,
    }
    reward_sum = 0.0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def potential_shaping(old_state, new_state) -> float:
    """
    Potential-based reward shaping: reward = gamma * phi(s') - phi(s), where
    phi = coin_potential + escape_potential + crate_potential +
    opponent_potential is the sum of four independent potential functions
    (Ng, Harada & Russell 1999: any potential function preserves the
    optimal policy, and the sum of several valid potential functions is
    itself one). crate_count_if_bomb's and bomb_hits_opponent_if_bomb's
    credit are handled separately, as direct bonuses on the BOMB transition
    itself (see game_events_occurred/end_of_round) rather than through a
    potential here -- a potential's 0->1->0 shape over a bomb's lifetime
    would land its credit on a later, unrelated action instead of the BOMB
    decision it's meant to reward. opponent_potential pairs with
    moves_to_opponent the same way, but is always active (not gated to be
    mutually exclusive with the other three) -- see the N_FEATURES comment
    in features.py for why.
    """
    old_potential = (
        coin_potential(old_state)
        + escape_potential(old_state)
        + crate_potential(old_state)
        + opponent_potential(old_state)
    )

    if new_state is None:
        new_potential = 0.0
    else:
        new_potential = (
            coin_potential(new_state)
            + escape_potential(new_state)
            + crate_potential(new_state)
            + opponent_potential(new_state)
        )

    return GAMMA * new_potential - old_potential
