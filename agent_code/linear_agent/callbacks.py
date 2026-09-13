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
    self.metrics_file.flush()


def _epsilon(round: int) -> float:
    return max(EPSILON_MIN, EPSILON_START * EPSILON_DECAY ** (round))
