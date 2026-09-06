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

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    This model is a linear function approximator: Q(s, a) = weights . state_action_features(s, a),
    where self.weights is a vector of length N_FEATURES

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train or not os.path.isfile("linear-model.pt"):
        self.logger.info("Setting up model from scratch.")
        self.weights = np.zeros(N_FEATURES)
    else:
        self.logger.info("Loading model from saved state.")
        with open("linear-model.pt", "rb") as file:
            self.weights = pickle.load(file)

    # logging metrics for plots at every step
    os.makedirs("metrics", exist_ok=True)

    if self.train:
        metrics_path = os.path.join("metrics", "training_log.csv")
    else:
        metrics_path = os.path.join("metrics", "evaluation_log.csv")

    self.metrics_file = open(metrics_path, "w", newline="")
    self.metrics_writer = csv.writer(self.metrics_file)
    self.metrics_writer.writerow(
        [
            "round",
            "step",
            "valid",
            "sparse_reward",
            "shaped_reward",
            "w_moves_to_coin",
            "w_valid",
            "w_wait",
            "w_bomb",
        ]
    )

    self.log_round = None
    self.prev_state = None
    self.prev_valid = None
    self.prev_score = 0


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    state = state_features(game_state)
    _log_previous_transition(self, game_state, state)

    random_prob = 0.1
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        action = np.random.choice(ACTIONS, p=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1])
    else:
        self.logger.debug("Querying model for action.")
        values = q_values(self.weights, state)
        best_move = np.argmax(values)
        action = ACTIONS[best_move]

    self.log_round = game_state["round"]
    self.prev_state = state
    self.prev_valid = bool(state_action_features(state, action)[1])
    self.prev_score = game_state["self"][1]

    return action


def _log_previous_transition(self, game_state, state) -> None:
    """
    log outcome of action chosen in previous act()
    """
    if self.log_round != game_state["round"] or self.prev_state is None:
        return

    sparse_reward = game_state["self"][1] - self.prev_score
    shaped_reward = GAMMA * coin_potential(state) - coin_potential(self.prev_state)

    self.metrics_writer.writerow(
        [
            game_state["round"],
            game_state["step"] - 1,
            int(self.prev_valid),
            sparse_reward,
            shaped_reward,
            *self.weights,
        ]
    )
