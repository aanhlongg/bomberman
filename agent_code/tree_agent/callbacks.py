"""
Action selection for the tree_agent. Epsilon-greedy with a safety filter
while training, greedy otherwise. self.model is None until the first refit.
"""

import csv
import os
import pickle
import random

import numpy as np

from .features import (
    ACTIONS,
    F_ESCAPE_MOVE,
    F_INTO_DANGER,
    F_VALID,
    GAMMA,
    N_FEATURES,
    coin_potential,
    q_values,
    state_action_features,
    state_features,
)

EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.997
# TREE_AGENT_EPSILON pins the exploration rate, e.g. when collecting a replay
# window from an already trained model
_FIXED_EPSILON = float(os.environ["TREE_AGENT_EPSILON"]) if os.environ.get("TREE_AGENT_EPSILON") else None

# exploration weights per action, mild bias toward bombing
EXPLORE_WEIGHTS = np.array([0.15, 0.15, 0.15, 0.15, 0.1, 0.3])

# BOMB is vetoed when a bomb-capable opponent is within this Manhattan
# distance: 1 normally, 2 once that opponent has been seen bombing this
# round. A flat 2 halved the suicide rate against bombers but regressed
# against peaceful_agent, whose bomb flag is always up although it never bombs.
VETO_RADIUS_QUIET = 1
VETO_RADIUS_ACTIVE = 2


def setup(self):
    """
    Load the model (or start with none) and open the per-run metrics CSV.
    Called once when the agent is loaded, before any act() call.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    agent_seed = os.environ.get("AGENT_SEED")
    if agent_seed is not None:
        random.seed(int(agent_seed))
        np.random.seed(int(agent_seed))

    # TREE_AGENT_MODEL_PATH evaluates another model file (e.g. a checkpoint),
    # TREE_AGENT_INIT_MODEL starts a --train run from an existing model
    model_path = os.environ.get("TREE_AGENT_MODEL_PATH", "tree-model.pt")
    init_model = os.environ.get("TREE_AGENT_INIT_MODEL")
    if self.train and init_model:
        self.logger.info(f"Training from existing model {init_model}.")
        with open(init_model, "rb") as f:
            self.model = pickle.load(f)
    elif self.train or not os.path.isfile(model_path):
        self.logger.info("Setting up model from scratch.")
        self.model = None
    else:
        self.logger.info(f"Loading model from {model_path}.")
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        model_n_features = getattr(self.model, "n_features_in_", None)
        if model_n_features is not None and model_n_features != N_FEATURES:
            raise ValueError(
                f"{model_path} was trained on {model_n_features} features but "
                f"features.py now defines N_FEATURES = {N_FEATURES}; retrain with --train 1."
            )

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
            "has_model",
            "sparse_reward",
            "shaped_reward",
            "best_q",
        ]
    )

    self.log_round = None
    self.prev_state = None
    self.prev_valid = None
    self.prev_score = 0
    self.prev_best_q = None

    # which opponents have been seen bombing this round (reset per round in act)
    self.opponent_has_bombed = {}
    self.prev_opponent_positions = {}
    self.history_round = None


def act(self, game_state: dict) -> str:
    """
    Choose an action for the current game state: epsilon-greedy
    (safety-filtered) while training, purely greedy otherwise.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    _track_opponent_bombs(self, game_state)
    state = state_features(
        game_state, use_target_aware_escape=not self.train, opponent_has_bombed=self.opponent_has_bombed
    )
    _log_step(self, game_state, state)

    epsilon = _exploration_rate(game_state["round"]) if self.train else 0.0
    if self.train and _FIXED_EPSILON is not None:
        epsilon = _FIXED_EPSILON
    best_q = None
    if self.train and (self.model is None or random.random() < epsilon):
        self.logger.debug("Choosing action at random (safety-filtered).")
        action = _safe_random_action(self, state)
    elif state is not None and state["in_escape_window"] and state["escape_direction"] is not None:
        # the argmax has lost near-ties here; the verified route is safer
        self.logger.debug("Forcing verified escape direction.")
        action = state["escape_direction"]
    else:
        self.logger.debug("Querying model for action.")
        values = q_values(self.model, state)
        # argmax over structurally safe actions only; if none passes, fall
        # back to the merely valid ones rather than masking everything
        safe_mask = _safety_mask(self, state)
        if not safe_mask.any():
            safe_mask = np.array(
                [state_action_features(state, a)[F_VALID] == 1.0 for a in ACTIONS]
            )
        masked_values = np.where(safe_mask, values, -np.inf)
        best = np.argmax(masked_values)
        action = ACTIONS[best]
        best_q = float(values[best])

    action_features = state_action_features(state, action)

    self.log_round = game_state["round"]
    self.prev_state = state
    self.prev_valid = bool(action_features[F_VALID])
    self.prev_score = game_state["self"][1]
    self.prev_best_q = best_q

    return action


def _safety_mask(self, state):
    """
    Per action: is it structurally safe to consider at all? Valid; during an
    escape only the verified direction; otherwise no step into a blast; and
    never BOMB next to an opponent who can bomb back. Used by both the
    greedy branch of act() and the random exploration.
    """
    in_escape_window = state["in_escape_window"]
    features = {action: state_action_features(state, action) for action in ACTIONS}

    def is_safe(action):
        f = features[action]
        if f[F_VALID] == 0.0:
            return False
        if action == "BOMB" and _adjacent_opponent_can_bomb(self, state):
            return False
        if in_escape_window:
            return f[F_ESCAPE_MOVE] == 1.0
        return f[F_INTO_DANGER] == 0.0

    return np.array([is_safe(action) for action in ACTIONS])


def _track_opponent_bombs(self, game_state):
    round_num = game_state["round"]
    if round_num != self.history_round:
        self.opponent_has_bombed = {}
        self.prev_opponent_positions = {}
        self.history_round = round_num

    # bombs block movement, so a bomb on an opponent's previous tile is theirs
    bomb_positions = {pos for pos, _timer in game_state["bombs"]}
    for name, prev_pos in self.prev_opponent_positions.items():
        if prev_pos in bomb_positions:
            self.opponent_has_bombed[name] = True

    self.prev_opponent_positions = {other[0]: other[3] for other in game_state["others"]}


def _adjacent_opponent_can_bomb(self, state):
    position = state["position"]
    for name, opp_pos, opp_bomb_available in state["opponents"]:
        if not opp_bomb_available:
            continue
        radius = VETO_RADIUS_ACTIVE if self.opponent_has_bombed.get(name) else VETO_RADIUS_QUIET
        if abs(opp_pos[0] - position[0]) + abs(opp_pos[1] - position[1]) <= radius:
            return True
    return False


def _safe_random_action(self, state):
    safe = _safety_mask(self, state)

    if not safe.any():
        return np.random.choice(ACTIONS, p=EXPLORE_WEIGHTS)

    weights = EXPLORE_WEIGHTS[safe]
    weights = weights / weights.sum()
    return np.random.choice(np.array(ACTIONS)[safe], p=weights)


def _log_step(self, game_state, state) -> None:
    # one metrics row for the previous step
    if self.log_round != game_state["round"] or self.prev_state is None:
        return
    sparse_reward = game_state["self"][1] - self.prev_score
    shaped_reward = GAMMA * coin_potential(state) - coin_potential(self.prev_state)
    best_q = "" if self.prev_best_q is None else self.prev_best_q  # empty during exploration
    self.metrics_writer.writerow(
        [
            game_state["round"],
            game_state["step"] - 1,
            int(self.prev_valid),
            int(self.model is not None),
            sparse_reward,
            shaped_reward,
            best_q,
        ]
    )
    if game_state["step"] % 200 == 0:
        self.metrics_file.flush()


def _exploration_rate(round: int) -> float:
    return max(EPSILON_MIN, EPSILON_START * EPSILON_DECAY ** (round))
