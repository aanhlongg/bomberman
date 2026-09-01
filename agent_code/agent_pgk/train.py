import os
import pickle
from collections import namedtuple, deque
from typing import List
import random
import numpy as np

import events as e
from .callbacks import ACTIONS, DIRECTION_VECTORS, state_to_features

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# Hyperparameters -- tune these!
TRANSITION_HISTORY_SIZE = 1000
ALPHA = 0.1   # learning rate
GAMMA = 0.9   # discount factor
REPLAY_BATCH_SIZE = 16     # number of past transitions to replay per step
REPLAY_EVERY_N_STEPS = 20  # how often to trigger a replay update
REPLAY_ALPHA = 0.02        # smaller LR for replay so it doesn't dominate the online update

# Custom events for reward shaping
MOVED_TOWARDS_COIN = "MOVED_TOWARDS_COIN"
MOVED_AWAY_FROM_COIN = "MOVED_AWAY_FROM_COIN"

# Maps a movement action name to its index within the first 4 feature dims.
_MOVE_TO_FEATURE_INDEX = {name: i for i, name in enumerate(DIRECTION_VECTORS)}
assert len(DIRECTION_VECTORS) == 4, (
    "_update_q assumes features[0:4] are coin-direction indicators and "
    "features[4:8] are walkability flags, matching ACTIONS[:4]. If this "
    "changes, the valid_next_mask logic in _update_q must be updated too."
)

def _valid_action_mask(features: np.array) -> np.array:
    """
    Returns a boolean mask over ACTIONS indicating which actions are valid
    from this state: movement actions require the corresponding tile to be
    walkable, WAIT is always valid, and BOMB is disabled entirely for this
    task (no crates/bombs needed to collect coins).
    """
    mask = np.ones(len(ACTIONS), dtype=bool)
    mask[:4] = features[4:8] == 1.0   # UP, RIGHT, DOWN, LEFT
    mask[4] = True                     # WAIT always allowed
    mask[5] = False                    # BOMB disabled for this task
    return mask

def setup_training(self):
    """
    Called once after setup() when training. Initializes a transition
    history (useful for debugging/inspection; the actual learning happens
    online in game_events_occurred/end_of_round below) and creates the
    episode-stats CSV log if it doesn't exist yet.
    """
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

    if not os.path.isfile("training_stats.csv"):
        with open("training_stats.csv", "w") as f:
            f.write("round,score,steps\n")
        self.episode_num = 1
    else:
        with open("training_stats.csv", "r") as f:
            self.episode_num = sum(1 for _ in f)  # header counts as row 0 offset


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    old_features = state_to_features(old_game_state)
    new_features = state_to_features(new_game_state)

    _add_movement_shaping_events(old_features, self_action, events)

    reward = reward_from_events(self, events)
    self.transitions.append(Transition(old_features, self_action, new_features, reward))
    _update_q(self, old_features, self_action, reward, new_features)

    if new_game_state["step"] % REPLAY_EVERY_N_STEPS == 0:
        _replay_batch(self)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    last_features = state_to_features(last_game_state)
    _add_movement_shaping_events(last_features, last_action, events)

    reward = reward_from_events(self, events)
    self.transitions.append(Transition(last_features, last_action, None, reward))
    _update_q(self, last_features, last_action, reward, None)
    _replay_batch(self)

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)

    _, score, _, _ = last_game_state['self']
    with open("training_stats.csv", "a") as f:
        f.write(f"{self.episode_num},{score},{last_game_state['step']}\n")
    self.episode_num += 1

def _add_movement_shaping_events(features: np.array, action: str, events: List[str]):
    """
    Rewards moving in the direction the coin-BFS feature recommends, and
    penalizes moving away from it, so the agent gets dense feedback
    instead of only the sparse COIN_COLLECTED event.
    """
    if features is None or action not in _MOVE_TO_FEATURE_INDEX:
        return
    idx = _MOVE_TO_FEATURE_INDEX[action]
    if features[idx] == 1.0:
        events.append(MOVED_TOWARDS_COIN)
    elif features[:4].sum() > 0:  # a coin direction exists, but we didn't take it
        events.append(MOVED_AWAY_FROM_COIN)


def _update_q(self, features: np.array, action: str, reward: float, next_features: np.array):
    if features is None or action is None:
        return

    action_idx = ACTIONS.index(action)
    q_current = self.model[action_idx] @ features

    if next_features is not None:
        valid_next_mask = _valid_action_mask(next_features)
        q_next = self.model @ next_features
        q_next_masked = np.where(valid_next_mask, q_next, -np.inf)
        q_next_max = np.max(q_next_masked)
    else:
        q_next_max = 0.0

    td_target = reward + GAMMA * q_next_max
    td_error = td_target - q_current
    
    # Weight update
    self.model[action_idx] += ALPHA * td_error * features

def _replay_batch(self):
    """
    Samples a random minibatch from transition history and applies ONE
    averaged gradient update per action (rather than many sequential
    per-sample updates). Sequential updates compound within a single
    call and, combined with online bootstrapping, can destabilize a
    linear model -- averaging keeps the replay step from dominating
    the online TD update.
    """
    if len(self.transitions) < REPLAY_BATCH_SIZE:
        return
    batch = random.sample(self.transitions, REPLAY_BATCH_SIZE)

    # Accumulate gradients per action, then apply once per action.
    grad_accum = np.zeros_like(self.model)
    counts = np.zeros(len(ACTIONS))

    for t in batch:
        if t.state is None or t.action is None:
            continue
        action_idx = ACTIONS.index(t.action)
        q_current = self.model[action_idx] @ t.state

        if t.next_state is not None:
            valid_next_mask = _valid_action_mask(t.next_state)
            q_next = self.model @ t.next_state
            q_next_masked = np.where(valid_next_mask, q_next, -np.inf)
            q_next_max = np.max(q_next_masked)
        else:
            q_next_max = 0.0

        td_target = t.reward + GAMMA * q_next_max
        td_error = td_target - q_current

        grad_accum[action_idx] += td_error * t.state
        counts[action_idx] += 1

    for action_idx in range(len(ACTIONS)):
        if counts[action_idx] > 0:
            self.model[action_idx] += REPLAY_ALPHA * (grad_accum[action_idx] / counts[action_idx])
        
def reward_from_events(self, events: List[str]) -> float:
    """
    Maps game events (and our custom shaping events) to numeric rewards.
    Balanced so that moving towards/away from a coin cancel out if the
    agent oscillates, discouraging back-and-forth farming behaviour.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1.0,
        e.KILLED_OPPONENT: 5.0,
        e.KILLED_SELF: -5.0,
        e.GOT_KILLED: -5.0,
        e.INVALID_ACTION: -0.2,
        e.WAITED: -0.05,
        MOVED_TOWARDS_COIN: 0.1,
        MOVED_AWAY_FROM_COIN: -0.1,
    }
    reward_sum = sum(game_rewards.get(event, 0.0) for event in events)
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum