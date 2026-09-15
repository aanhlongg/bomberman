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
    opponent_potential,
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

# Task 3: dying to an OPPONENT's bomb (GOT_KILLED) is scored the same as
# dying to our own (KILLED_SELF) -- death is death from the agent's
# perspective, and there's no reason yet to treat the two differently.
GOT_KILLED_PENALTY = -10.0

# deliberately larger in magnitude than CRATE_DESTROYED: killing an
# opponent is meant to dominate the reward landscape once one is reachable
# (per the "keep coins/crates, weight kills much higher" design choice),
# not just edge it out
KILLED_OPPONENT_REWARD = 10.0

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

# Task 3: same direct-credit mechanism as CRATE_HIT_BONUS, applied
# proactively this time instead of being discovered the hard way --
# KILLED_OPPONENT wouldn't fire until several steps after the bomb that
# caused it (through the same kind of feature-less escape-phase states that
# made crate_hit_potential fail), so pay it immediately on the BOMB
# transition whenever bomb_hits_opponent_if_bomb == 1. Starting at the same
# magnitude as CRATE_HIT_BONUS as a first guess; may need its own sweep --
# this is a genuinely different, moving-target situation and the current
# bomb_hits_opponent_if_bomb feature is only an MVP (current-position)
# proxy, not one that accounts for the opponent moving before detonation.
OPPONENT_HIT_BONUS = 5.0

# Direct, immediate reward for taking the verified-correct escape step
# (escape_correct_move == 1), on top of escape_potential's existing dense
# shaping. escape_correct_move is ground truth (the first step of a BFS
# shortest path to safety), exactly the kind of "known-good decision" the
# other two direct bonuses above already pay for immediately rather than
# waiting on it. The reasoning for needing this at all: three independent
# 5,000-round Task 3 training runs, all otherwise identical, produced
# w_escape_correct_move values of 1.03, 0.73, and 0.28 -- and the 0.28 run
# self-killed 100% of the time, because moves_to_opponent (which typically
# settles around 2.6-3.7) then wins every escape-vs-chase disagreement.
# escape_potential alone is apparently too weak or too slow a signal for
# this weight to develop reliably within 5,000 rounds; this pays for the
# same ground truth directly, the same way CRATE_HIT_BONUS did for
# crate_hit_if_bomb.
#
# NOT currently wired into game_events_occurred -- tried at 8.0 and
# reverted. It worked exactly as intended for w_escape_correct_move's own
# magnitude (reached ~15.4, cleanly above moves_to_opponent), but unlike
# CRATE_HIT_BONUS/OPPONENT_HIT_BONUS (paid once, on a single BOMB
# transition), this one pays out on every correct step of a multi-step
# escape sequence -- which made deliberately bombing yourself into danger,
# then "correctly" escaping it repeatedly, net-profitable: w_is_bomb
# collapsed from its usual -8 to -11 down to -1.2, and self-kill rate rose
# to 87% despite the escape direction itself being more reliably correct
# than ever. A smaller magnitude, or paying it once per escape episode
# rather than per step, might avoid this -- not yet tried.
ESCAPE_MOVE_BONUS = 8.0

# see the synthetic-update comment in game_events_occurred for the full
# writeup. Named index into state_action_features' return vector (see
# features.py's N_FEATURES layout comment) -- a local constant here rather
# than importing callbacks.py's private one, since this module shouldn't
# depend on callbacks.py's internals.
_ESCAPE_CORRECT_MOVE_INDEX = 7

# Target value the synthetic escape update below pulls w_escape_correct_move
# toward. Set comfortably above moves_to_opponent's observed ceiling
# (~3.6-4.85 across every Task 3 run so far) so the correct escape
# direction wins the argmax outright rather than by a fragile margin.
ESCAPE_CORRECT_MOVE_SYNTHETIC_TARGET = 8.0

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
    if self_action == "BOMB" and bomb_feat[1] == 1.0 and bomb_feat[10] == 1.0:
        reward += OPPONENT_HIT_BONUS

    # ESCAPE_MOVE_BONUS is intentionally NOT wired in here -- see its
    # definition for why it was tried and reverted (it worked exactly as
    # intended for w_escape_correct_move's magnitude, but paying per
    # correct escape *step* made repeatedly bombing yourself into danger
    # and then escaping it net-profitable: w_is_bomb collapsed from its
    # usual -8 to -11 down to -1.2, and self-kill rate rose to 87%, despite
    # the escape *direction* being more reliably correct than ever).

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

    # Synthetic counterfactual update (Task 3): escape_direction, whenever
    # in_escape_window holds, is likewise deterministic ground truth (the
    # first step of features._compute_escape_direction's verified BFS
    # shortest path to safety) -- exactly the kind of known fact the update
    # above already injects directly rather than waiting for real
    # experience to teach it. Motivated by trace analysis across BOTH Task
    # 3 phases: escape_correct_move's weight (~1.0-1.5) reliably loses the
    # argmax to moves_to_opponent's (~3.6-4.85) during real escape windows,
    # and real experience alone isn't generating enough signal for it to
    # grow past that.
    #
    # Two different fixes were tried first and both failed differently.
    # Zeroing moves_to_opponent during escape windows (see its comment in
    # features.py) collapsed escape_correct_move into a permanent 0-valued
    # tie -- that competing pull turns out to have been providing the
    # reward variance the weight needed to learn from at all, not just
    # getting in its way. A per-step direct reward bonus (ESCAPE_MOVE_BONUS
    # above) enabled reward-hacking (repeatedly bombing yourself into
    # danger to "correctly" escape it for profit).
    #
    # This differs from both. Unlike ESCAPE_MOVE_BONUS, it never touches
    # the reward function -- it's a direct weight update, not a payment for
    # a real action, so there's nothing to hack. Unlike the update above
    # (which reuses the ordinary feature vector via sgd_update), it uses an
    # ISOLATED synthetic feature vector with only the escape_correct_move
    # slot set to 1: the ordinary feature vector for a directional action
    # also carries the "valid" feature (weight ~20-30, shared across every
    # valid action, escape-related or not) -- running the ordinary update
    # would drag that shared weight toward this target too, corrupting it
    # everywhere else in the model. Isolating the feature avoids that
    # entanglement entirely, and the ordinary bounded SGD rule means this
    # self-limits as w_escape_correct_move approaches the target, rather
    # than drifting unboundedly the way a plain additive nudge would.
    #
    # Fired unconditionally (not epsilon-scaled like the update above)
    # whenever in_escape_window holds with a real escape route -- the
    # diagnosed problem here is signal WEAKNESS, so throttling this by a
    # decaying epsilon would work against the fix rather than for it.
    if old_state is not None and old_state["in_escape_window"] and old_state["escape_direction"] is not None:
        escape_feature_vector = np.zeros(N_FEATURES)
        escape_feature_vector[_ESCAPE_CORRECT_MOVE_INDEX] = 1.0
        prediction = self.weights[_ESCAPE_CORRECT_MOVE_INDEX]
        alpha = _alpha(old_game_state["round"])
        self.weights += alpha * (ESCAPE_CORRECT_MOVE_SYNTHETIC_TARGET - prediction) * escape_feature_vector

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
    """
    Sum of the sparse, per-event rewards for one step's events list.

    OPPONENT_ELIMINATED is deliberately not rewarded here: it fires
    whenever *any* opponent dies, including from their own mistakes, not
    only when we caused it -- KILLED_OPPONENT is the event that actually
    means "we did this."
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
    Potential-based reward shaping: reward = gamma * phi(s') - phi(s),
    where phi = coin_potential + escape_potential + crate_potential +
    opponent_potential is the sum of four independent potential functions.
    Ng, Harada & Russell (1999): any potential function preserves the
    optimal policy, and the sum of several valid potential functions is
    itself one.

    crate_hit_if_bomb's and bomb_hits_opponent_if_bomb's credit are both
    handled separately, as direct bonuses on the BOMB transition itself
    (see game_events_occurred) rather than through a potential here -- a
    potential's 0->1->0 shape over a bomb's lifetime unavoidably decays
    away and inverts sign before detonation, landing its penalty on a
    later, unrelated action instead of the BOMB decision it's meant to
    reward.

    crate_potential pairs with the moves_to_crate feature the same way
    coin_potential pairs with moves_to_coin: it gives directional guidance
    toward the nearest crate-adjacent tile on the steps where no coin is
    visible. opponent_potential pairs with moves_to_opponent the same way,
    but is always active (not gated to be mutually exclusive with the
    other two) -- see the N_FEATURES comment in features.py for why.
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
