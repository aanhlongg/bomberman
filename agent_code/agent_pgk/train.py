import os
import pickle
import random
from collections import deque, namedtuple
from typing import List
import events as e
import numpy as np
from .callbacks import ACTIONS, DIRECTION_VECTORS, state_to_features, EVAL_MODE, MODEL_TAG, _model_filename

# -----------------------------------------------------------------------------------
# DATA STRUCTURES & HYPERPARAMETERS
# -----------------------------------------------------------------------------------

# Replay buffer tuple storing state-action-reward transitions for experience replay
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

# Maximum number of past transitions retained in memory
TRANSITION_HISTORY_SIZE = 10000

# Q-Learning Hyperparameters
ALPHA = 0.05                # Learning rate (controls gradient step size during weight updates)
GAMMA = 0.95                # Discount factor (prioritizes long-term rewards for pathfinding)
REPLAY_BATCH_SIZE = 32      # Number of sampled transitions per replay batch
REPLAY_EVERY_N_STEPS = 5    # Step frequency for triggering experience replay updates

# Custom events used for auxiliary reward shaping
MOVED_TOWARDS_COIN = "MOVED_TOWARDS_COIN"
MOVED_AWAY_FROM_COIN = "MOVED_AWAY_FROM_COIN"

# Map movement actions (UP, RIGHT, DOWN, LEFT) to their feature indices (0, 1, 2, 3)
_MOVE_TO_FEATURE_INDEX = {name: i for i, name in enumerate(DIRECTION_VECTORS)}

# Determine target statistics CSV filename based on evaluation mode and custom model tags
_kind = "test" if EVAL_MODE else "training"
STATS_FILE = f"{_kind}_stats{'_' + MODEL_TAG if MODEL_TAG else ''}.csv"


# -----------------------------------------------------------------------------------
# TRAINING SETUP
# -----------------------------------------------------------------------------------

def setup_training(self):
    """
    Initializes the experience replay buffer and sets up the CSV tracking file.
    Executed once before the game loop starts when training is enabled.
    """
    # Fixed-size queue that automatically discards old transitions when full
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

    # Create CSV header if file doesn't exist; otherwise resume counting episode numbers
    if not os.path.isfile(STATS_FILE):
        with open(STATS_FILE, "w") as f:
            f.write("round,score,steps\n")
        self.episode_num = 1
    else:
        with open(STATS_FILE, "r") as f:
            # Set episode count based on existing rows in CSV
            self.episode_num = sum(1 for _ in f)


# -----------------------------------------------------------------------------------
# STEP-BY-STEP GAME EVENT HANDLER
# -----------------------------------------------------------------------------------

def game_events_occurred(
    self,
    old_game_state: dict,
    self_action: str,
    new_game_state: dict,
    events: List[str],
):
    """
    Called after every single step in the game environment.
    Converts states to features, appends custom rewards, updates Q-table weights,
    and samples experience replay.
    """
    # Skip weight updates and experience storing during evaluation mode
    if EVAL_MODE:
        return
    
    # Extract feature vectors for previous and current step
    old_features = state_to_features(self, old_game_state)
    new_features = state_to_features(self, new_game_state)

    # Inject auxiliary movement events (MOVED_TOWARDS_COIN / MOVED_AWAY_FROM_COIN)
    _add_movement_shaping_events(old_features, self_action, events)

    # Compute scalar reward based on game engine + custom shaping events
    reward = reward_from_events(self, events)

    # Store transition tuple in replay buffer
    self.transitions.append(Transition(old_features, self_action, new_features, reward))

    # Perform online Q-learning weight update
    _update_q(self, old_features, self_action, reward, new_features)

    # Periodically sample past transitions to stabilize weight convergence
    if new_game_state["step"] % REPLAY_EVERY_N_STEPS == 0:
        _replay_batch(self)


# -----------------------------------------------------------------------------------
# END OF ROUND HANDLER
# -----------------------------------------------------------------------------------

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called when an episode finishes (e.g., all coins collected or max steps reached).
    Processes the final transition, saves the model weights, and logs statistics.
    """
    if not EVAL_MODE:
        # Extract features for the final board state
        last_features = state_to_features(self, last_game_state)
        _add_movement_shaping_events(last_features, last_action, events)
        reward = reward_from_events(self, events)

        # Terminal transition has next_state = None (no future Q-value)
        self.transitions.append(Transition(last_features, last_action, None, reward))
        _update_q(self, last_features, last_action, reward, None)

        # Extra replay step at episode end
        _replay_batch(self)

        # Save updated model weight matrix to disk
        with open(_model_filename(), "wb") as file:
            pickle.dump(self.model, file)

    # Log episode summary stats to CSV
    _, score, _, _ = last_game_state["self"]
    with open(STATS_FILE, "a") as f:
        f.write(f"{self.episode_num},{score},{last_game_state['step']}\n")
    self.episode_num += 1

    # Clear target tracking state for fresh episode setup
    self.current_coin_target = None  # fresh board next round


# -----------------------------------------------------------------------------------
# HELPER FUNCTIONS & Q-LEARNING CORE
# -----------------------------------------------------------------------------------

def _add_movement_shaping_events(features: np.array, action: str, events: List[str]):
    """
    Analyzes the directional features in `old_features` to infer if the agent's
    chosen action moved it closer to or further from its targeted coin.
    """
    if features is None or action not in _MOVE_TO_FEATURE_INDEX:
        return
    idx = _MOVE_TO_FEATURE_INDEX[action]
    # If the action chosen matches the optimal BFS step direction (feature value == 1.0)
    if features[idx] == 1.0:
        events.append(MOVED_TOWARDS_COIN)
    # If a valid path to a coin exists (sum > 0) but the agent chose a different direction
    elif features[:4].sum() > 0:
        events.append(MOVED_AWAY_FROM_COIN)


def _update_q(
    self, features: np.array, action: str, reward: float, next_features: np.array
):
    """
    Performs Semi-Gradient Q-Learning update for linear function approximation:
    W_a <- W_a + ALPHA * (reward + GAMMA * max_a'(Q(s', a')) - Q(s, a)) * features
    """
    if features is None or action is None:
        return
    action_idx = ACTIONS.index(action)

    # Current predicted Q-value for chosen action: Q(s, a) = W_a . features
    q_current = self.model[action_idx] @ features

    # Compute target Q-value using Bellman equation
    if next_features is not None:
        # Build action mask for next state (match walkability rules from callbacks.py)
        valid_mask = np.ones(len(ACTIONS), dtype=bool)
        valid_mask[:4] = next_features[4:8] == 1.0  # Movement walkability
        valid_mask[4] = True                        # WAIT permitted
        valid_mask[5] = False                       # BOMB disabled for coin collection task

        # Compute Q-values across all actions for the next state
        q_next = self.model @ next_features
        q_next[~valid_mask] = -1e9  # Suppress invalid moves with large negative value
        q_next_max = np.max(q_next)
    else:
        # Terminal state has zero expected future rewards
        q_next_max = 0.0

    # Temporal Difference (TD) target and error calculations
    td_target = reward + GAMMA * q_next_max
    td_error = td_target - q_current

    # Weight update step for the row corresponding to the executed action
    self.model[action_idx] += ALPHA * td_error * features


def _replay_batch(self):
    """
    Samples a mini-batch from the experience buffer to re-train weights,
    breaking correlation between consecutive training steps.
    """
    if len(self.transitions) < REPLAY_BATCH_SIZE:
        return
    batch = random.sample(self.transitions, REPLAY_BATCH_SIZE)
    for t in batch:
        if t.state is None or t.action is None:
            continue
        _update_q(self, t.state, t.action, t.reward, t.next_state)


def reward_from_events(self, events: List[str]) -> float:
    """
    Maps game events to scalar rewards used for training.
    """
    game_rewards = {
        e.COIN_COLLECTED: 2.0,      # Primary target objective
        e.INVALID_ACTION: -0.2,     # Penalty for running into walls/obstacles
        e.WAITED: -0.2,             # Penalty for standing still needlessly
        MOVED_TOWARDS_COIN: 0.2,    # Positive reward shaping for following BFS path
        MOVED_AWAY_FROM_COIN: -0.2, # Negative reward shaping for deviating from BFS path
    }
    return sum(game_rewards.get(event, 0.0) for event in events)