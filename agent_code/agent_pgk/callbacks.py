import os
import pickle
import random
from collections import deque

import numpy as np

# -----------------------------------------------------------------------------------
# CONSTANTS & CONFIGURATION
# -----------------------------------------------------------------------------------

# Full action set available in the Bomberman framework
ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]

# Coordinate offset mapping for grid navigation (x, y)
DIRECTION_VECTORS = {
    "UP": (0, -1),
    "RIGHT": (1, 0),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
}

# Map directional coordinate vectors to standard action indices (0: UP, 1: RIGHT, 2: DOWN, 3: LEFT)
VECTOR_TO_INDEX = {v: i for i, v in enumerate(DIRECTION_VECTORS.values())}

# Total size of feature vector fed to linear model:
# 4 (directional step to target) + 4 (immediate walkability) + 1 (inverse distance) + 1 (bias)
FEATURE_DIM = 10

# Flag to disable exploration (epsilon=0) and model updates when testing performance
EVAL_MODE = os.environ.get("EVAL_MODE") == "1"

# Tag to organize distinct model save files and prevent overwriting baseline runs
MODEL_TAG = os.environ.get("MODEL_TAG", "")


def _model_filename() -> str:
    """
    Returns the formatted pickle filename based on current environment settings.
    """
    return f"my-saved-model{'_' + MODEL_TAG if MODEL_TAG else ''}.pt"


# -----------------------------------------------------------------------------------
# AGENT INITIALIZATION & SETUP
# -----------------------------------------------------------------------------------

def setup(self):
    """
    Called once when initializing the agent process.
    Loads trained Q-learning weights if present, or initializes a new model matrix.
    """
    # Track targeted coin position across steps to avoid target oscillation
    self.current_coin_target = None  # used by sticky coin-targeting in state_to_features

    model_file = _model_filename()
    if os.path.isfile(model_file):
        self.logger.info(f"Loading model from saved state ({model_file}).")
        with open(model_file, "rb") as file:
            self.model = pickle.load(file)
    else:
        self.logger.info(f"Setting up model from scratch ({model_file}).")
        # Initialize weight matrix W of shape (num_actions, feature_dim) with small gaussian noise
        self.model = np.random.normal(0, 0.01, size=(len(ACTIONS), FEATURE_DIM))


# -----------------------------------------------------------------------------------
# ACTION SELECTION (POLICY)
# -----------------------------------------------------------------------------------

def act(self, game_state: dict) -> str:
    """
    Determines the next move given the game state.
    Employs epsilon-greedy exploration during training and purely greedy selection during eval.
    """
    # Transform raw game state into normalized feature array
    features = state_to_features(self, game_state)

    # Construct mask restricting invalid actions to avoid wall collisions and unnecessary bombs
    valid_mask = np.ones(len(ACTIONS), dtype=bool)
    valid_mask[:4] = features[4:8] == 1.0   # UP, RIGHT, DOWN, LEFT walkability
    valid_mask[4] = True                    # WAIT allowed (must match mask in train.py)
    valid_mask[5] = False                   # BOMB disabled (not required for Task 1)

    # Compute epsilon decay for exploration schedule
    if self.train and not EVAL_MODE:
        ep = getattr(self, "episode_num", 1)
        epsilon = max(0.05, 0.5 * (0.99**ep))
    else:
        epsilon = 0.0

    # Exploration: select a random valid action
    if self.train and not EVAL_MODE and random.random() < epsilon:
        valid_indices = np.where(valid_mask)[0]
        return ACTIONS[np.random.choice(valid_indices)]

    # Exploitation: compute predicted Q-values Q(s, a) = W @ features
    q_values = self.model @ features

    # Suppress illegal/invalid actions by assigning negative infinity Q-values
    q_values[~valid_mask] = -1e9

    # Select action with highest Q-value (break ties randomly among top values)
    max_q = np.max(q_values)
    best_actions = np.where(q_values == max_q)[0]

    return ACTIONS[np.random.choice(best_actions)]


# -----------------------------------------------------------------------------------
# PATHFINDING & BOARD NAVIGATION (BFS)
# -----------------------------------------------------------------------------------

def _get_blocked_tiles(game_state: dict) -> set:
    """
    Extracts locations occupied by other players or active bombs.
    (Maintained for system compatibility and future task extensions).
    """
    blocked = set()
    for _, _, _, pos in game_state.get("others", []):
        blocked.add(pos)
    for bomb_pos, _timer in game_state.get("bombs", []):
        blocked.add(bomb_pos)
    return blocked


def _bfs_nearest_coin(field: np.array, start: tuple, coins: list, blocked: set = frozenset()):
    """
    Breadth-First Search to locate the closest coin from `start`.
    Returns: (target_coin_pos, first_step_direction_tile, distance_in_steps)
    """
    if not coins:
        return None, None, None

    coin_set = set(coins)
    if start in coin_set:
        return start, None, 0

    visited = {start: None}
    depths = {start: 0}
    queue = deque([start])

    while queue:
        current = queue.popleft()
        if current in coin_set:
            # Trace parent pointers back to find the first step taken from start
            step = current
            while visited[step] != start:
                step = visited[step]
            return current, step, depths[current]

        x, y = current
        for dx, dy in DIRECTION_VECTORS.values():
            n = (x + dx, y + dy)
            # Verify coordinates stay inside map bounds and navigate unblocked free tiles (field == 0)
            nx, ny = n
            if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]:
                if field[nx, ny] == 0 and n not in blocked and n not in visited:
                    visited[n] = current
                    depths[n] = depths[current] + 1
                    queue.append(n)

    return None, None, None


def _bfs_distance_and_step(field: np.array, start: tuple, target: tuple, blocked: set = frozenset()):
    """
    Breadth-First Search to pathfind from `start` to a specific `target` tile.
    Returns: (first_step_direction_tile, distance_in_steps)
    """
    if target is None:
        return None, None
    if start == target:
        return None, 0

    visited = {start: None}
    depths = {start: 0}
    queue = deque([start])

    while queue:
        current = queue.popleft()
        if current == target:
            step = current
            while visited[step] != start:
                step = visited[step]
            return step, depths[current]

        x, y = current
        for dx, dy in DIRECTION_VECTORS.values():
            n = (x + dx, y + dy)
            nx, ny = n
            if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]:
                if field[nx, ny] == 0 and n not in blocked and n not in visited:
                    visited[n] = current
                    depths[n] = depths[current] + 1
                    queue.append(n)

    return None, None


# Hysteresis threshold to lock targeting onto a single coin unless an alternative is significantly closer
STICKY_MARGIN = 2


# -----------------------------------------------------------------------------------
# FEATURE ENGINEERING
# -----------------------------------------------------------------------------------

def state_to_features(self, game_state: dict) -> np.array:
    """
    Converts raw game board dictionary into a 10-dimensional real-valued feature vector:
    - [0:4]: One-hot directional indicator for the optimal BFS move towards the chosen coin.
    - [4:8]: Walkability binary flags for adjacent tiles [UP, RIGHT, DOWN, LEFT].
    - [8]: Normalized inverse distance to targeted coin (1.0 = on coin, ~0.0 = far away).
    - [9]: Constant bias term (1.0) for linear function approximation.
    """
    if game_state is None:
        return None

    field = game_state["field"]
    _, _, _, (x, y) = game_state["self"]
    coins = game_state["coins"]
    coin_set = set(coins)
    blocked = _get_blocked_tiles(game_state)

    # 1. Target persistence (Sticky coin targeting):
    # Maintain tracking of current target unless collected or another coin becomes distinctly closer
    sticky_target = getattr(self, "current_coin_target", None)
    if sticky_target is not None and sticky_target not in coin_set:
        sticky_target = None

    nearest_coin, nearest_step, nearest_dist = _bfs_nearest_coin(field, (x, y), coins, blocked)

    if sticky_target is not None and sticky_target != nearest_coin:
        sticky_step, sticky_dist = _bfs_distance_and_step(field, (x, y), sticky_target, blocked)
        if sticky_dist is not None and (nearest_dist is None or nearest_dist >= sticky_dist - STICKY_MARGIN):
            target, next_step = sticky_target, sticky_step
        else:
            target, next_step = nearest_coin, nearest_step
    else:
        target, next_step = nearest_coin, nearest_step

    self.current_coin_target = target

    # 2. Optimal step direction vector (4 dims)
    coin_features = np.zeros(4)
    if next_step is not None:
        vector = (next_step[0] - x, next_step[1] - y)
        idx = VECTOR_TO_INDEX.get(vector)
        if idx is not None:
            coin_features[idx] = 1.0

    # 3. Surrounding tile walkability vector (4 dims)
    walkable_features = np.zeros(4)
    for i, (dx, dy) in enumerate(DIRECTION_VECTORS.values()):
        nx, ny = x + dx, y + dy
        if (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]
                and field[nx, ny] == 0 and (nx, ny) not in blocked):
            walkable_features[i] = 1.0

    # 4. Normalized distance feature to target coin (1 dim)
    dist_feature = np.array([0.0])
    if coins:
        ref_dist = nearest_dist if target == nearest_coin else _bfs_distance_and_step(field, (x, y), target, blocked)[1]
        if ref_dist is not None:
            dist_feature[0] = ref_dist / 30.0

    # 5. Intercept / Bias term (1 dim)
    bias_feature = np.array([1.0])

    return np.concatenate(
        [coin_features, walkable_features, dist_feature, bias_feature]
    )