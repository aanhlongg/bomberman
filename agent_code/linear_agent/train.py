import pickle
from collections import deque, namedtuple
from typing import List

import events as e
import numpy as np

from .features import (
    ACTIONS,
    GAMMA,
    N_FEATURES,
    bomb_escapable,
    bomb_hits,
    coin_potential,
    q_values,
    state_action_features,
    state_features,
)

# This is only an example!
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
ALPHA = 0.01  # learning rate

# custom events
BOMB_NO_ESCAPE = "BOMB_NO_ESCAPE"
BOMB_NO_TARGET = "BOMB_NO_TARGET"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


def game_events_occurred(
    self,
    old_game_state: dict,
    self_action: str,
    new_game_state: dict,
    events: List[str],
):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(
        f"Encountered game event(s) {', '.join(map(repr, events))} in step {new_game_state['step']}"
    )

    old_state = state_features(old_game_state)
    new_state = state_features(new_game_state)
    events = events + bad_bomb_events(old_state, events)
    reward = reward_from_events(self, events) + potential_shaping(old_state, new_state)
    self.logger.debug(f"Reward for action {self_action}: {reward}")

    # (s, a, r, s')
    self.transitions.append(Transition(old_state, self_action, new_state, reward))

    # td target = reward + gamma * max(Q(s',a')) (highest q value in next state)
    next_value = np.max(q_values(self.weights, new_state))
    td_target = reward + GAMMA * next_value
    sgd_update(self, old_state, self_action, td_target)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(
        f"Encountered event(s) {', '.join(map(repr, events))} in final step"
    )

    old_state = state_features(last_game_state)
    events = events + bad_bomb_events(old_state, events)
    reward = reward_from_events(self, events) + potential_shaping(old_state, None)
    self.logger.debug(f"Reward for action {last_action}: {reward}")

    # (s, a, r, s')
    self.transitions.append(Transition(old_state, last_action, None, reward))

    # target = reward, there is no next state
    sgd_update(self, old_state, last_action, td_target=reward)

    # Store the model
    with open("linear-model.pt", "wb") as file:
        pickle.dump(self.weights, file)


def sgd_update(self, old_state, action, td_target: float) -> None:
    """
    compute one gradient descent step on the squared error between
    td target and prediction

    weights = weights + alpha * (td_target - prediction) * feature_vector
    """
    features = state_action_features(old_state, action)
    prediction = np.dot(self.weights, features)
    self.weights += ALPHA * (td_target - prediction) * features


def reward_from_events(self, events: List[str]) -> float:
    """
    maps game events to rewards
    """
    game_rewards = {
        e.COIN_COLLECTED: 1.0,
        e.WAITED: -0.05,
        e.INVALID_ACTION: -0.1,
        e.KILLED_SELF: -5.0,
        BOMB_NO_ESCAPE: -5.0,
        BOMB_NO_TARGET: -2.0,
    }
    reward_sum = 0.0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def potential_shaping(old_state, new_state) -> float:
    """
    potential-based reward shaping: reward = gamma * phi(s') - phi(s),
    where phi is negative coin distance
    """
    old_potential = coin_potential(old_state)

    if new_state is None:
        new_potential = 0.0
    else:
        new_potential = coin_potential(new_state)

    return GAMMA * new_potential - old_potential


def bad_bomb_events(old_state, events):
    """
    returns a custom event if a bomb can't be escaped or hasn't hit anything
    """
    if e.BOMB_DROPPED not in events:
        return []

    bad_events = []
    if not bomb_escapable(old_state):
        bad_events.append(BOMB_NO_ESCAPE)
    if not bomb_hits(old_state):
        bad_events.append(BOMB_NO_TARGET)
    return bad_events
