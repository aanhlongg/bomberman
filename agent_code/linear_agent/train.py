import csv
import os
import pickle
import shutil
from typing import List

import events as e
import numpy as np

from .callbacks import _epsilon
from .features import (
    ACTIONS,
    FEATURE_NAMES,
    GAMMA,
    N_FEATURES,
    bomb_escapable,
    bomb_hits,
    coin_potential,
    crate_potential,
    escape_potential,
    state_action_features,
    state_features,
)

# Hyper parameters -- DO modify
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
ALPHA = 0.01  # learning rate

# experience replay
BUFFER_SIZE = 50000
BATCH_SIZE = 32

# start value for averaging
AVERAGE_FROM_ROUND = 1000

CHECKPOINT_INTERVAL = 100
CHECKPOINT_DIR = os.path.join("metrics", "checkpoints")

# custom events
BOMB_NO_ESCAPE = "BOMB_NO_ESCAPE"
BOMB_NO_TARGET = "BOMB_NO_TARGET"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    # feature vector of taken actions
    self.buffer_features = np.zeros((BUFFER_SIZE, N_FEATURES))

    # feature vector for all actions in next state
    self.buffer_next_features = np.zeros((BUFFER_SIZE, len(ACTIONS), N_FEATURES))

    # realized reward
    self.buffer_reward = np.zeros(BUFFER_SIZE)

    # stores if next state is empty (round ended)
    self.buffer_final = np.zeros(BUFFER_SIZE, dtype=bool)

    self.buffer_index = 0  # next row to be written
    self.buffer_filled = 0  # rows written

    self.weight_sum = np.zeros(N_FEATURES)
    self.weight_count = 0

    # log one row per round in training
    os.makedirs("metrics", exist_ok=True)
    self.log_file = open(os.path.join("metrics", "training_log.csv"), "w", newline="")
    self.log_writer = csv.writer(self.log_file)
    self.log_writer.writerow(
        ["round", "steps", *LOGGED_EVENTS.values(), "reward", "epsilon"]
        + [f"w_{name}" for name in FEATURE_NAMES]
    )
    self.round_totals = dict.fromkeys([*LOGGED_EVENTS.values(), "reward"], 0)

    # checkpoint of weight vector
    shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)
    os.makedirs(CHECKPOINT_DIR)


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
    count_for_training_log(self, events, reward)

    # (s, a, r, s'), with s' kept as the features of every action that follows it
    next_features = []
    for action in ACTIONS:
        next_features.append(state_action_features(new_state, action))

    write_replay_buffer(
        self, old_state, self_action, next_features, reward, final=False
    )
    replay_update(self)


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
    count_for_training_log(self, events, reward)

    # target = reward, since there is no next state
    final_features = np.zeros((len(ACTIONS), N_FEATURES))
    write_replay_buffer(
        self, old_state, last_action, final_features, reward, final=True
    )
    replay_update(self)

    # since convergence isn't guaranteed (weights will begin to oscillate),
    # store the average weight vector starting at episode 1000
    if last_game_state["round"] >= AVERAGE_FROM_ROUND:
        self.weight_sum += self.weights
        self.weight_count += 1

    model = self.weight_sum / self.weight_count if self.weight_count else self.weights
    with open("linear-model.pt", "wb") as file:
        pickle.dump(model, file)

    write_training_log(self, last_game_state)
    save_checkpoint(self, last_game_state["round"])


def write_replay_buffer(
    self, old_state, action, next_features, reward: float, final: bool
):
    """
    store one transition in the replay buffer, overwriting the oldest one once full
    """
    i = self.buffer_index

    self.buffer_features[i] = state_action_features(old_state, action)
    self.buffer_next_features[i] = next_features
    self.buffer_reward[i] = reward
    self.buffer_final[i] = final

    self.buffer_index = (i + 1) % BUFFER_SIZE
    self.buffer_filled = min(self.buffer_filled + 1, BUFFER_SIZE)


def replay_update(self):
    """
    compute one gradient descent step on the squared error between td target and
    prediction, averaged over a random minibatch of transitions

    weights = weights + alpha * mean((td_target - prediction) * feature_vector)
    """
    if self.buffer_filled < BATCH_SIZE:
        return

    # sample random batch
    batch = np.random.randint(0, self.buffer_filled, BATCH_SIZE)
    features = self.buffer_features[batch]

    # keep best action for each transition
    next_values = np.max(self.buffer_next_features[batch] @ self.weights, axis=1)

    # last step of round has no next state
    next_values[self.buffer_final[batch]] = 0.0

    # td target = reward + gamma * max(Q(s',a'))
    td_targets = self.buffer_reward[batch] + GAMMA * next_values

    errors = td_targets - features @ self.weights
    self.weights += ALPHA * (errors @ features) / BATCH_SIZE


def reward_from_events(self, events: List[str]) -> float:
    """
    maps game events to rewards
    """
    game_rewards = {
        e.COIN_COLLECTED: 1.0,
        e.COIN_FOUND: 1.0,
        e.CRATE_DESTROYED: 1.5,
        e.WAITED: -0.05,
        e.INVALID_ACTION: -2.0,
        e.KILLED_SELF: -10.0,
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
    where phi sums negative coin distance, negative distance to safety and the
    payoff of a pending crate hit
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


def bad_bomb_events(old_state, events):
    """
    returns a custom event if a bomb can't be escaped or hasn't hit anything
    """
    if e.BOMB_DROPPED not in events:
        return []

    bad_events = []
    if not bomb_escapable(
        old_state["field"], old_state["position"], old_state["occupied"]
    ):
        bad_events.append(BOMB_NO_ESCAPE)
    if not bomb_hits(old_state):
        bad_events.append(BOMB_NO_TARGET)
    return bad_events


# game events for training log
LOGGED_EVENTS = {
    e.COIN_COLLECTED: "coins",
    e.INVALID_ACTION: "invalid_actions",
    e.BOMB_DROPPED: "bombs",
    e.CRATE_DESTROYED: "crates",
    e.KILLED_SELF: "suicides",
}


def count_for_training_log(self, events, reward: float):
    """
    add number of (log-relevant) events and reward that occurred each step to a total
    """
    for event in events:
        if event in LOGGED_EVENTS:
            self.round_totals[LOGGED_EVENTS[event]] += 1
    self.round_totals["reward"] += reward


def write_training_log(self, last_game_state: dict):
    """
    write the total number of each occurred event (that is log-relevant) and the (non-averaged)
    weights at the end of a round to a row
    """
    round_number = last_game_state["round"]

    self.log_writer.writerow(
        [round_number, last_game_state["step"]]
        + [self.round_totals[name] for name in LOGGED_EVENTS.values()]
        + [self.round_totals["reward"], _epsilon(round_number)]
        + list(self.weights)
    )
    self.log_file.flush()
    self.round_totals = dict.fromkeys(self.round_totals, 0)


def save_checkpoint(self, round_number: int):
    """
    save the current weights (non-averaged) every CHECKPOINT_INTERVAL rounds
    """
    if round_number % CHECKPOINT_INTERVAL != 0:
        return

    path = os.path.join(CHECKPOINT_DIR, f"weights_{round_number}.pt")
    with open(path, "wb") as file:
        pickle.dump(self.weights.copy(), file)
