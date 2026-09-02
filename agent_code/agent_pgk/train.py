import os
import pickle
import random
from collections import deque, namedtuple
from typing import List
import events as e
import numpy as np
from .callbacks import ACTIONS, DIRECTION_VECTORS, state_to_features, EVAL_MODE, MODEL_TAG, _model_filename

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))
TRANSITION_HISTORY_SIZE = 10000
ALPHA = 0.05  # Stable learning rate
GAMMA = 0.95  # High discount factor for pathing
REPLAY_BATCH_SIZE = 32
REPLAY_EVERY_N_STEPS = 5
MOVED_TOWARDS_COIN = "MOVED_TOWARDS_COIN"
MOVED_AWAY_FROM_COIN = "MOVED_AWAY_FROM_COIN"
_MOVE_TO_FEATURE_INDEX = {name: i for i, name in enumerate(DIRECTION_VECTORS)}

_kind = "test" if EVAL_MODE else "training"
STATS_FILE = f"{_kind}_stats{'_' + MODEL_TAG if MODEL_TAG else ''}.csv"


def setup_training(self):
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    if not os.path.isfile(STATS_FILE):
        with open(STATS_FILE, "w") as f:
            f.write("round,score,steps\n")
        self.episode_num = 1
    else:
        with open(STATS_FILE, "r") as f:
            self.episode_num = sum(1 for _ in f)


def game_events_occurred(
    self,
    old_game_state: dict,
    self_action: str,
    new_game_state: dict,
    events: List[str],
):
    if EVAL_MODE:
        return

    old_features = state_to_features(self, old_game_state)
    new_features = state_to_features(self, new_game_state)
    _add_movement_shaping_events(old_features, self_action, events)
    reward = reward_from_events(self, events)
    self.transitions.append(Transition(old_features, self_action, new_features, reward))
    _update_q(self, old_features, self_action, reward, new_features)
    if new_game_state["step"] % REPLAY_EVERY_N_STEPS == 0:
        _replay_batch(self)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    if not EVAL_MODE:
        last_features = state_to_features(self, last_game_state)
        _add_movement_shaping_events(last_features, last_action, events)
        reward = reward_from_events(self, events)
        self.transitions.append(Transition(last_features, last_action, None, reward))
        _update_q(self, last_features, last_action, reward, None)
        _replay_batch(self)

        with open(_model_filename(), "wb") as file:
            pickle.dump(self.model, file)

    _, score, _, _ = last_game_state["self"]
    with open(STATS_FILE, "a") as f:
        f.write(f"{self.episode_num},{score},{last_game_state['step']}\n")
    self.episode_num += 1

    self.current_coin_target = None  # fresh board next round


def _add_movement_shaping_events(features: np.array, action: str, events: List[str]):
    if features is None or action not in _MOVE_TO_FEATURE_INDEX:
        return
    idx = _MOVE_TO_FEATURE_INDEX[action]
    if features[idx] == 1.0:
        events.append(MOVED_TOWARDS_COIN)
    elif features[:4].sum() > 0:
        events.append(MOVED_AWAY_FROM_COIN)


def _update_q(
    self, features: np.array, action: str, reward: float, next_features: np.array
):
    if features is None or action is None:
        return
    action_idx = ACTIONS.index(action)
    q_current = self.model[action_idx] @ features
    if next_features is not None:
        valid_mask = np.ones(len(ACTIONS), dtype=bool)
        valid_mask[:4] = next_features[4:8] == 1.0
        valid_mask[4] = True   # WAIT -- must match act()'s masking in callbacks.py
        valid_mask[5] = False  # BOMB -- disabled, must match act()'s masking
        q_next = self.model @ next_features
        q_next[~valid_mask] = -1e9
        q_next_max = np.max(q_next)
    else:
        q_next_max = 0.0
    td_target = reward + GAMMA * q_next_max
    td_error = td_target - q_current
    self.model[action_idx] += ALPHA * td_error * features


def _replay_batch(self):
    if len(self.transitions) < REPLAY_BATCH_SIZE:
        return
    batch = random.sample(self.transitions, REPLAY_BATCH_SIZE)
    for t in batch:
        if t.state is None or t.action is None:
            continue
        _update_q(self, t.state, t.action, t.reward, t.next_state)


def reward_from_events(self, events: List[str]) -> float:
    game_rewards = {
        e.COIN_COLLECTED: 2.0,
        e.INVALID_ACTION: -0.2,
        e.WAITED: -0.2,
        MOVED_TOWARDS_COIN: 0.2,
        MOVED_AWAY_FROM_COIN: -0.2,
    }
    return sum(game_rewards.get(event, 0.0) for event in events)