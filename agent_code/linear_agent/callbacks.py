"""
Inference entry point for the linear_agent: model loading and action
selection. Used both during training (with epsilon-greedy, safety-filtered
exploration) and standalone evaluation (pure greedy).

Framework entry points (called automatically by the game engine):
    setup(self)              -- once, before the first act() call
    act(self, game_state)    -- once per step, returns one of features.ACTIONS

setup_training() (train.py) is called once, right after setup(), only when
running with --train.
"""

import csv
import os
import pickle
import random

import numpy as np

from .features import (
    ACTIONS,
    GAMMA,
    N_FEATURES,
    coin_potential,
    q_values,
    state_action_features,
    state_features,
)

# decaying epsilon exploration for training
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.997

# The exploration filter's BOMB veto is normally absolute: a BOMB with
# escape_exists_if_bomb==0 is never offered as a safe exploration choice,
# which means that feature only ever takes the value 1 on every real
# bomb-drop the agent experiences -- with no variance, its weight and
# is_bomb's weight are only jointly identifiable through their sum.
# ALLOW_DOOMED_BOMB_PROB lets a small fraction of doomed bombs through so
# escape_exists_if_bomb finally sees real 0-valued samples with real
# outcomes. See _safe_random_action for the live-quality gate that decides
# *when* this is allowed to fire.
ALLOW_DOOMED_BOMB_PROB = 0.02

# the gate below requires this many tracked bomb-drops (see train.py's
# bomb_tracker/tracked_bomb_log) before it can open at all, and then
# evaluates survival over exactly this many of the most recent ones -- a
# sliding window, not an all-time average, so it stays responsive to
# CURRENT execution quality rather than being diluted by early history.
DOOMED_BOMB_GATE_WINDOW = 500
DOOMED_BOMB_GATE_SURVIVAL_THRESHOLD = 0.95


def setup(self):
    """
    Load the model (or initialise a fresh one) and open the per-run metrics
    CSV. Called once when the agent is loaded, before any act() call.

    This model is a linear function approximator: Q(s, a) = weights . state_action_features(s, a),
    where self.weights is a vector of length N_FEATURES.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # --seed only fixes the world's own RNG (crate/coin layout), not this
    # agent's random exploration -- the framework documentation is explicit
    # that "this will not affect the randomness of agents ... which use a
    # separate random state." AGENT_SEED closes that gap so repeated runs of
    # the same config are actually comparable instead of confounded by two
    # independent sources of randomness.
    agent_seed = os.environ.get("AGENT_SEED")
    if agent_seed is not None:
        random.seed(int(agent_seed))
        np.random.seed(int(agent_seed))

    if self.train or not os.path.isfile("linear-model.pt"):
        self.logger.info("Setting up model from scratch.")
        self.weights = np.zeros(N_FEATURES)
    else:
        self.logger.info("Loading model from saved state.")
        with open("linear-model.pt", "rb") as file:
            self.weights = pickle.load(file)

    # per-step weight/reward log, always written to agent_code/linear_agent/metrics/
    agent_dir = os.path.dirname(os.path.abspath(__file__))
    metrics_dir = os.path.join(agent_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    if self.train:
        metrics_path = os.path.join(metrics_dir, "training_log.csv")
    else:
        metrics_path = os.path.join(metrics_dir, "evaluation_log.csv")

    self.metrics_file = open(metrics_path, "w", newline="")
    self.metrics_writer = csv.writer(self.metrics_file)
    self.metrics_writer.writerow(
        [
            "round",
            "step",
            "valid",
            "synthetic_update",
            "doomed_bomb_allowed",
            "sparse_reward",
            "shaped_reward",
            "w_coin",
            "w_valid",
            "w_wait",
            "w_bomb",
            "w_moves_into_avoidable_danger",
            "w_escape_exists",
            "w_crate_hit",
            "w_escape_correct_move",
            "w_moves_to_crate",
            "w_moves_to_opponent",
            "w_bomb_hits_opponent",
        ]
    )

    self.log_round = None
    self.prev_state = None
    self.prev_valid = None
    self.prev_synthetic_update = False
    self.prev_doomed_bomb_allowed = False
    self.prev_score = 0


def act(self, game_state: dict) -> str:
    """
    Choose an action for the current game state: epsilon-greedy
    (safety-filtered) while training, purely greedy otherwise.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    state = state_features(game_state, use_target_aware_escape=not self.train)
    _log_previous_transition(self, game_state, state)

    epsilon = _epsilon(game_state["round"]) if self.train else 0.0
    if self.train and random.random() < epsilon:
        self.logger.debug("Choosing action purely at random (safety-filtered).")
        action = _safe_random_action(self, state)
    else:
        self.logger.debug("Querying model for action.")
        values = q_values(self.weights, state)
        if _adjacent_opponent_can_bomb(state):
            values[ACTIONS.index("BOMB")] = -np.inf
        best_move = np.argmax(values)
        action = ACTIONS[best_move]

    action_features = state_action_features(state, action)

    self.log_round = game_state["round"]
    self.prev_state = state
    self.prev_valid = bool(action_features[1])
    # tags samples where the doomed-bomb exploration gate (see
    # _safe_random_action) let a doomed BOMB through, identified by outcome
    # -- a valid BOMB with escape_exists_if_bomb==0 -- rather than a flag
    # threaded out of that function, so it can't drift out of sync with
    # what actually happened.
    self.prev_doomed_bomb_allowed = bool(
        action == "BOMB" and action_features[1] == 1.0 and action_features[_IDX_ESCAPE_EXISTS_IF_BOMB] == 0.0
    )
    self.prev_score = game_state["self"][1]

    return action


# exploration action weights: UP/RIGHT/DOWN/LEFT/WAIT/BOMB, 15% each
# direction, 10% wait, 30% bomb
RANDOM_ACTION_WEIGHTS = np.array([0.15, 0.15, 0.15, 0.15, 0.1, 0.3])

# named indices into state_action_features' return vector, so a future
# reordering of features.py's layout has something visible to update here
# instead of silently breaking a bare f[4]/f[7]
_IDX_MOVES_INTO_AVOIDABLE_DANGER = 4
_IDX_ESCAPE_EXISTS_IF_BOMB = 5
_IDX_ESCAPE_CORRECT_MOVE = 7

# Applying a hard, non-learned safety override to the GREEDY action
# selection (not just the exploration branch below) was tried and
# reverted, in two scopes, against coin_collector_agent:
#   1. Veto BOMB from the argmax if verified doomed (escape_exists_if_bomb
#      == 0) + always take escape_direction during an active escape
#      window: self-kills were essentially unchanged (the dominant
#      remaining causes are dynamic -- e.g. the opponent closing the last
#      exit, or mutual/simultaneous bombing -- not "the policy knew
#      better and picked wrong"), while the added conservatism nearly
#      doubled the stalemate rate (~32% -> 45.5%) by discouraging
#      otherwise-reasonable engagement.
#   2. Escape-direction override only (drop the BOMB veto): a single
#      5,000-round run showed the WORST self-kill rate of any config
#      tested (29.5%, vs. 18.5% baseline) -- though given this matchup's
#      already-wide seed-to-seed variance (15-30% suicides on the
#      unmodified baseline across 5 seeds), this wasn't distinguished
#      from noise before the idea was dropped in favor of keeping the
#      validated seed-401 baseline.
#
# A THIRD, narrower veto (_adjacent_opponent_can_bomb below, still active)
# targets rule_based_agent specifically: unlike the two attempts above
# (general "is this bomb doomed" checks), this unconditionally refuses to
# BOMB whenever an opponent is orthogonally adjacent AND has a bomb
# available, full stop -- no escape-existence calculus involved. Motivated
# by trace analysis showing rule_based_agent reliably bombs back the
# instant we bomb it while adjacent (its own rule-based logic checks
# exactly this condition), and making our own escape_exists_if_bomb
# retaliation-aware (see features.py) didn't change behavior at all (13
# kills / 139 suicides vs. 17 kills / 131 suicides without it) -- the
# learned weights just weren't discouraging that exchange even once the
# ground truth correctly flagged it as usually fatal. This veto removes
# the choice entirely rather than hoping the Q-values act on better
# information.
#
# A FOURTH override -- the escape-direction-forced-move from attempt #2
# above, RETRIED in the greedy branch of act() -- was tried and reverted.
# Motivation: re-tracing deaths against rule_based_agent with this
# session's shipped (seed-505) model showed 2 of 4 self-kill traces where
# escape_correct_move was genuinely 1.0 for a real, computed escape route,
# and the agent's own greedy policy still chose WAIT (or dropped a second
# bomb) instead of following it, breaking a partially-completed escape.
# Tested with a full retrain + eval against all three opponents: suicides
# vs. rule_based_agent were statistically unchanged (41/200 -> 40/200,
# both well inside the 5-seed sweep's 17-22% baseline range) and kills
# dropped (23 -> 14); peaceful_agent and coin_collector_agent were
# unaffected either way. Conclusion: forcing the verified-correct escape
# move does NOT reduce self-kills here, which means the 2 "policy ignored
# a real escape" trace examples were not representative of the dominant
# failure mode in aggregate -- the remaining suicides are overwhelmingly
# the genuine "no escape exists" case (attempt #3's motivating trace
# evidence, rounds 3 and 16: the opponent's own simultaneous, independent
# bombing closes the last exit one step after ours is committed), which no
# override on OUR action selection can prevent. Reverted; seed-505 remains
# the shipped model.
#
# A FIFTH, different-shaped override -- a _board_cleared_with_opponent
# veto, tried and reverted -- targeted not a self-kill but the "no urge to
# finish off the last opponent" stall: once no coin is visible and no
# crate is reachable, a direct trace (15 samples, seed-505 model vs
# peaceful_agent, opponent 8-19 tiles away, no danger active) showed
# WAIT's Q-value (~20.9) beats BOTH moving toward the opponent (~17.38)
# and BOMB (~19.94) in literally every sample -- the agent never even
# starts closing the distance, so it never gets adjacent enough to bomb.
# The fix vetoed WAIT outright in this situation (no coin, no reachable
# crate, an opponent exists, not mid-escape), forcing the existing
# weights' own ranking (BOMB > approach) to take over once distance
# started closing.
#
# Tested with a full retrain + eval against all three opponents: a clean
# win against peaceful_agent (kills 77-79.5% -> 88%, score up, suicides
# still low at 4%) but a real regression against BOTH bomb-capable
# opponents -- coin_collector_agent (score 1299->1209, kills 28.5%->8.5%,
# suicides 15.5%->17.5%) and rule_based_agent (score down, kills down,
# suicides 20-20.5%->25%, outside the previously-established 5-seed
# sweep's 17-22% range). Root cause: WAIT was also serving as a passive
# safety valve specifically against opponents that can retaliate --
# forcing constant pursuit once the board clears exposes the agent to
# more bomb-capable-opponent interactions than standing pat did, a
# downside that simply doesn't exist against peaceful_agent (which never
# bombs). Reverted; seed-505 remains the shipped model. A scoped version
# (veto WAIT only when the tracked opponent cannot bomb, mirroring
# _adjacent_opponent_can_bomb's own bomb-capability check) might capture
# the peaceful_agent win without the other two regressions -- not yet
# tried.


def _adjacent_opponent_can_bomb(state):
    """
    True if any opponent is orthogonally adjacent (Manhattan distance <=
    1) to our current position and currently has a bomb available. See
    the comment above this function's callers for why this exists.
    """
    position = state["position"]
    for opp_pos, opp_bomb_available in state["opponents_with_bomb"]:
        if opp_bomb_available and abs(opp_pos[0] - position[0]) + abs(opp_pos[1] - position[1]) <= 1:
            return True
    return False


def _recent_escape_survival_rate(tracked_bomb_log, window):
    """
    Fraction of the last `window` tracked bomb-drops (train.py's
    bomb_tracker/tracked_bomb_log -- every valid BOMB action, survived or
    not) that survived. Returns None if fewer than `window` bombs have
    been tracked yet, so callers can treat "not enough data" as its own
    case rather than an implicit 0% or 100%.
    """
    if len(tracked_bomb_log) < window:
        return None
    recent = tracked_bomb_log[-window:]
    survived = sum(1 for record in recent if record.get("survived"))
    return survived / window


def _safe_random_action(self, state):
    """
    Filters random exploration so it doesn't have to "discover" bomb danger
    by dying.

    Branches on in_escape_window because the same column (index 4,
    moves_into_avoidable_danger) means something different depending on
    context: outside an escape window it's the real "don't walk into
    danger" signal, but it is structurally forced to 0.0 for EVERY action
    while in_escape_window is True (see features.py) -- during an active
    escape, escape_correct_move (index 7) is the feature that actually
    carries information, so the filter has to read the right column for
    the context it's in.

    The BOMB/escape_exists_if_bomb veto is unaffected by this branch: BOMB
    is never a valid action while in_escape_window is True (bombs_left
    only refills a full step after the window has already closed), so the
    veto only ever has real work to do outside an escape window anyway.

    The veto itself is not absolute: with probability ALLOW_DOOMED_BOMB_PROB,
    a BOMB with escape_exists_if_bomb==0 is allowed through -- but only
    once the doomed-bomb gate below is open. The gate is keyed to live
    escape-execution QUALITY (tracked-bomb survival rate over the last
    DOOMED_BOMB_GATE_WINDOW drops), not to round number or epsilon: round
    count says nothing about whether escape execution is actually
    trustworthy yet, and injecting doomed-bomb samples before it is would
    confound the result -- a death following a doomed bomb needs to be
    attributable to the bomb being genuinely unsurvivable, not to
    execution failing on an escape that existed.
    """
    in_escape_window = state["in_escape_window"]
    features = {action: state_action_features(state, action) for action in ACTIONS}

    survival_rate = _recent_escape_survival_rate(self.tracked_bomb_log, DOOMED_BOMB_GATE_WINDOW)
    doomed_bomb_gate_open = (
        survival_rate is not None and survival_rate >= DOOMED_BOMB_GATE_SURVIVAL_THRESHOLD
    )

    def is_safe(action):
        f = features[action]
        if action == "BOMB" and _adjacent_opponent_can_bomb(state):
            return False
        if action == "BOMB" and f[_IDX_ESCAPE_EXISTS_IF_BOMB] == 0.0:
            return doomed_bomb_gate_open and random.random() < ALLOW_DOOMED_BOMB_PROB
        if in_escape_window:
            return f[_IDX_ESCAPE_CORRECT_MOVE] == 1.0
        return f[_IDX_MOVES_INTO_AVOIDABLE_DANGER] == 0.0

    safe = np.array([is_safe(action) for action in ACTIONS])

    if not safe.any():
        # genuinely trapped (or, during an escape, no verified path found
        # within budget): fall back to the original, unfiltered distribution
        return np.random.choice(ACTIONS, p=RANDOM_ACTION_WEIGHTS)

    weights = RANDOM_ACTION_WEIGHTS[safe]
    weights = weights / weights.sum()
    return np.random.choice(np.array(ACTIONS)[safe], p=weights)


def _log_previous_transition(self, game_state, state) -> None:
    """Write one row of the per-step metrics CSV for the *previous* step."""
    if self.log_round != game_state["round"] or self.prev_state is None:
        return
    sparse_reward = game_state["self"][1] - self.prev_score
    shaped_reward = GAMMA * coin_potential(state) - coin_potential(self.prev_state)
    self.metrics_writer.writerow([
        game_state["round"],
        game_state["step"] - 1,
        int(self.prev_valid),
        int(self.prev_synthetic_update),
        int(self.prev_doomed_bomb_allowed),
        sparse_reward,
        shaped_reward,
        *self.weights,
    ])
    # Flushing every step forces an OS-level sync on an ever-growing file,
    # which measurably slows training down over a long run (see README's
    # "Note on training speed"). Every row is still written to Python's
    # own buffer regardless -- this only defers when it's actually synced
    # to disk, so no data is lost in normal operation; only an ungraceful
    # process kill (not a risk here, since linear-model.pt itself is saved
    # separately every round in train.py) could lose the last <=200 rows.
    if game_state["step"] % 200 == 0:
        self.metrics_file.flush()


def _epsilon(round: int) -> float:
    return max(EPSILON_MIN, EPSILON_START * EPSILON_DECAY ** (round))
