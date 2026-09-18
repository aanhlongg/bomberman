import os
import pickle
import random

import numpy as np

from .features import (
    ACTIONS,
    N_FEATURES,
    q_values,
    state_features,
)

# decaying epsilon exploration for training
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.997


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

    # read model from environment variable, fully trained model as default
    model_path = os.environ.get("LINEAR_AGENT_MODEL", "linear-model.pt")

    if self.train or not os.path.isfile(model_path):
        self.logger.info("Setting up model from scratch.")
        self.weights = np.zeros(N_FEATURES)
    else:
        self.logger.info(f"Loading model from {model_path}.")
        with open(model_path, "rb") as file:
            self.weights = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    state = state_features(game_state)

    epsilon = _epsilon(game_state["round"]) if self.train else 0.0
    if self.train and random.random() < epsilon:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        action = np.random.choice(ACTIONS, p=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1])
    else:
        self.logger.debug("Querying model for action.")
        values = q_values(self.weights, state)
        best_move = np.argmax(values)
        action = ACTIONS[best_move]

    return action


def _epsilon(round: int) -> float:
    return max(EPSILON_MIN, EPSILON_START * EPSILON_DECAY ** (round))
