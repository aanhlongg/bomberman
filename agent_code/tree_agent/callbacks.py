"""
Inference entry point for the tree_agent: model loading and action
selection. Used both during training (with epsilon-greedy, safety-filtered
exploration) and standalone evaluation (pure greedy).

Structured the same way as linear_agent's callbacks.py, but the underlying
model is a tree ensemble (see train.py's Fitted Q-Iteration loop) instead of
a weight vector -- the main practical consequence is that self.model can be
None (no Fitted-Q-Iteration refit has happened yet), which q_values()
already handles by returning all zeros.

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

# decaying epsilon exploration for training -- same schedule as linear_agent's
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.997
# Analysis hook: TREE_AGENT_EPSILON pins the training exploration rate (e.g.
# 0.05 when collecting a replay buffer from an already-trained model via
# TREE_AGENT_INIT_MODEL, so the buffer reflects that policy's states rather
# than the round-1 random walk). Unset in normal use.
_EPSILON_OVERRIDE = float(os.environ["TREE_AGENT_EPSILON"]) if os.environ.get("TREE_AGENT_EPSILON") else None


# exploration action weights: UP/RIGHT/DOWN/LEFT/WAIT/BOMB, matching
# linear_agent's own Task 2 weights (mild bias toward productive bombing)
RANDOM_ACTION_WEIGHTS = np.array([0.15, 0.15, 0.15, 0.15, 0.1, 0.3])

# named indices into state_action_features' return vector (see features.py's
# N_FEATURES layout comment)
_IDX_MOVES_INTO_AVOIDABLE_DANGER = 4
_IDX_ESCAPE_CORRECT_MOVE = 7


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

    # Analysis hooks (both environment variables, both unset in normal use):
    # TREE_AGENT_MODEL_PATH evaluates a specific model file (e.g. a checkpoint)
    # instead of tree-model.pt; TREE_AGENT_INIT_MODEL starts a --train run
    # from an existing model instead of from None (fine-tuning, or collecting
    # a replay buffer under an already-trained policy).
    model_path = os.environ.get("TREE_AGENT_MODEL_PATH", "tree-model.pt")
    init_model = os.environ.get("TREE_AGENT_INIT_MODEL")
    if self.train and init_model:
        self.logger.info(f"Training from existing model {init_model}.")
        with open(init_model, "rb") as file:
            self.model = pickle.load(file)
    elif self.train or not os.path.isfile(model_path):
        self.logger.info("Setting up model from scratch (no Fitted-Q-Iteration refit yet).")
        self.model = None
    else:
        self.logger.info(f"Loading model from {model_path}.")
        with open(model_path, "rb") as file:
            self.model = pickle.load(file)
        # The feature layout is part of the model: a regressor fit on an
        # older N_FEATURES silently produces garbage (or raises deep inside
        # sklearn) if fed today's vectors. Fail here, with a message that
        # says what to do, rather than "play" with a mismatched model.
        model_n_features = getattr(self.model, "n_features_in_", None)
        if model_n_features is not None and model_n_features != N_FEATURES:
            raise ValueError(
                f"{model_path} was trained on {model_n_features} features but "

                f"features.py now defines N_FEATURES = {N_FEATURES}. Serialized "
                "tree models are not compatible across feature-layout changes; "
                "retrain with --train 1."
            )


    # per-step reward log, always written to agent_code/tree_agent/metrics/
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

    # per-opponent bomb-history tracking (Task 4 radius scoping, see
    # _update_opponent_bomb_history) -- reset per round, not here, since
    # setup() only runs once per process while rounds repeat
    self.opponent_has_bombed = {}
    self.prev_opponent_positions = {}
    self._tracked_bomb_history_round = None


def act(self, game_state: dict) -> str:
    """
    Choose an action for the current game state: epsilon-greedy
    (safety-filtered) while training, purely greedy otherwise.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    _update_opponent_bomb_history(self, game_state)
    state = state_features(
        game_state, use_target_aware_escape=not self.train, opponent_has_bombed=self.opponent_has_bombed
    )
    _log_previous_transition(self, game_state, state)

    epsilon = _epsilon(game_state["round"]) if self.train else 0.0
    if self.train and _EPSILON_OVERRIDE is not None:
        epsilon = _EPSILON_OVERRIDE  # analysis runs (see setup): fixed exploration rate
    best_q = None
    if self.train and (self.model is None or random.random() < epsilon):
        self.logger.debug("Choosing action at random (safety-filtered).")
        action = _safe_random_action(self, state)
    elif state is not None and state["in_escape_window"] and state["escape_direction"] is not None:
        # Force the verified-correct escape direction rather than trusting
        # the argmax during an active escape window. Diagnosed empirically:
        # death-trace analysis of the shipped model found the greedy policy
        # picking the WRONG direction on a genuine razor-thin near-tie
        # (e.g. Q(correct)=23.30 vs Q(wrong)=23.45, a 0.15 margin) -- the
        # tree ensemble's coarse leaf-averaged Q-values aren't separating
        # escape_correct_move's two cases cleanly enough to be trusted on a
        # decision this safety-critical. linear_agent tried forcing the
        # escape direction this same way in Task 3/4 and it didn't help
        # there, but those matchups' dominant failure mode was a SECOND,
        # opponent-controlled bomb invalidating the plan mid-execution --
        # Task 2 has no opponents, so a verified escape route here can never
        # be invalidated by anything but our own already-accounted-for
        # bomb, making this a much lower-risk place to force it.
        self.logger.debug("Forcing verified escape direction.")
        action = state["escape_direction"]
    else:
        self.logger.debug("Querying model for action.")
        values = q_values(self.model, state)
        # Mask unsafe actions out of the argmax entirely, rather than
        # trusting the model to rank them low -- see _safety_mask's
        # docstring for the two real failure modes this closes (walking
        # into a wall via a tied invalid action; walking into a bomb about
        # to detonate via a tied dangerous-but-structurally-valid one).
        # Falls back to a validity-only mask if nothing passes the fuller
        # safety criteria (mirrors _safe_random_action's own "genuinely
        # trapped" fallback) rather than masking every action to -inf.
        safe_mask = _safety_mask(self, state)
        if not safe_mask.any():
            safe_mask = np.array(
                [state_action_features(state, a)[1] == 1.0 for a in ACTIONS]
            )
        masked_values = np.where(safe_mask, values, -np.inf)
        best_move = np.argmax(masked_values)
        action = ACTIONS[best_move]
        best_q = float(values[best_move])

    action_features = state_action_features(state, action)

    self.log_round = game_state["round"]
    self.prev_state = state
    self.prev_valid = bool(action_features[1])
    self.prev_score = game_state["self"][1]
    self.prev_best_q = best_q

    return action


def _safety_mask(self, state):
    """
    True/False per action in ACTIONS: is this action structurally safe to
    consider at all (not "does the model think it's good"). Shared between
    _safe_random_action (exploration) and the greedy branch of act() --
    diagnosed as necessary after a real death trace found the greedy branch
    trusted raw Q-values for exactly the case this mask exists to rule out.
    Q(UP)=71.386 tied with Q(DOWN)=71.386 at a step where the agent's
    CURRENT tile was safe (in_escape_window==False) but UP walked directly
    into a bomb about to detonate (moves_into_avoidable_danger==1.0 for
    that specific action) -- argmax's index tie-break picked UP anyway,
    since nothing outside the exploration filter had ever checked this
    feature. Before this fix, only _safe_random_action enforced it, so the
    greedy policy (the one actually used at evaluation time, and for most
    of training once epsilon decays) had a real gap the filtered random
    branch didn't.

    Branches on in_escape_window because the same column
    (moves_into_avoidable_danger) means something different depending on
    context (see features.py) -- during an active escape,
    escape_correct_move is the feature that actually carries information.

    Unlike linear_agent, BOMB with no verified escape route is simply never
    a *valid* action at all here (see features.py's state_action_features)
    -- diagnosed empirically to be necessary for this architecture, not
    ported as a design choice. There is therefore no separate doomed-bomb
    veto branch needed here: the general `f[1] == 0.0` check below already
    excludes it, the same way it excludes any other structurally invalid
    action.

    Task 4: also vetoes BOMB whenever _adjacent_opponent_can_bomb holds --
    see that function's docstring. This is layered ON TOP OF, not instead
    of, features.py's retaliation-aware _escape_exists_after_bomb: that
    extension already makes BOMB structurally invalid in many
    adjacent-retaliation cases (if the combined blast leaves no escape,
    valid is already 0 before this check ever runs), but linear_agent found
    the escape feature alone didn't change self-kill behavior at all against
    rule_based_agent -- an escape can still technically exist post-retaliation
    in some geometries, without the exchange being one the agent should ever
    choose to start. The unconditional veto below removes the choice
    entirely rather than trusting the model to weigh that residual risk
    correctly.
    """
    in_escape_window = state["in_escape_window"]
    features = {action: state_action_features(state, action) for action in ACTIONS}

    def is_safe(action):
        f = features[action]
        if f[1] == 0.0:
            return False
        if action == "BOMB" and _adjacent_opponent_can_bomb(self, state):
            return False
        if in_escape_window:
            return f[_IDX_ESCAPE_CORRECT_MOVE] == 1.0
        return f[_IDX_MOVES_INTO_AVOIDABLE_DANGER] == 0.0

    return np.array([is_safe(action) for action in ACTIONS])


# Manhattan-distance radius for _adjacent_opponent_can_bomb's veto.
# Ported from linear_agent at 1 (strictly orthogonally adjacent) and
# validated there -- catches an opponent only once they're already
# touching us. A first widening experiment to 2 (unconditionally, for
# every opponent) cut the suicide rate against every bomb-capable opponent
# roughly in half and even improved kill rate -- but cost a real
# regression against peaceful_agent, whose bomb_available flag stays True
# the entire game (it simply never chooses BOMB by construction, see its
# own callbacks.py) even though it can never actually retaliate. The wider
# radius couldn't tell "opponent that could bomb but structurally never
# will" from "opponent that could bomb and reliably does" -- both looked
# identical from bomb_available alone.
#
# _RADIUS_IF_HAS_BOMBED / _RADIUS_IF_NEVER_BOMBED below scope the two
# values to that distinction directly, using per-opponent bombing history
# (_update_opponent_bomb_history) rather than applying one radius to
# everyone: an opponent only gets the wider, more cautious radius once
# THEY THEMSELVES have been directly observed dropping a bomb this round.
# peaceful_agent never triggers this (it never bombs, so it never gets
# flagged, so it keeps the narrower radius=1 baseline that was already
# validated as safe and effective against it); rule_based_agent and
# coin_collector_agent typically get flagged within their first few bombs
# each round, well before most close encounters, so they get the wider,
# safer radius for the vast majority of the round.
_RADIUS_IF_NEVER_BOMBED = 1
_RADIUS_IF_HAS_BOMBED = 2


def _update_opponent_bomb_history(self, game_state):
    """
    Tracks, per opponent name, whether that opponent has been directly
    observed dropping a bomb at least once this round -- read by
    _adjacent_opponent_can_bomb to decide which radius applies to which
    opponent. Reset at the start of every new round (bombing history from
    a previous round/game says nothing about this one).

    Attribution is exact, not a heuristic: bombs block movement (an agent
    cannot walk onto a tile that already has a ticking bomb on it), so the
    only way an opponent's OWN previous-step position can have a bomb on it
    now is if that opponent placed it there themselves before moving away
    (or staying put) -- no other agent could have passed through or landed
    on that tile while the bomb already existed.
    """
    round_num = game_state["round"]
    if round_num != self._tracked_bomb_history_round:
        self.opponent_has_bombed = {}
        self.prev_opponent_positions = {}
        self._tracked_bomb_history_round = round_num

    bomb_positions = {pos for pos, _timer in game_state["bombs"]}
    for name, prev_pos in self.prev_opponent_positions.items():
        if prev_pos in bomb_positions:
            self.opponent_has_bombed[name] = True

    self.prev_opponent_positions = {other[0]: other[3] for other in game_state["others"]}


def _adjacent_opponent_can_bomb(self, state):
    """
    True if any opponent is within that specific opponent's own radius
    (_RADIUS_IF_HAS_BOMBED if they've been observed bombing this round,
    else _RADIUS_IF_NEVER_BOMBED -- see _update_opponent_bomb_history) of
    our current position, Manhattan distance, and currently has a bomb
    available. Ported from linear_agent's Task 4 fix: rule_based_agent
    reliably bombs back the instant it is bombed while adjacent (its own
    rule-based logic checks exactly this condition), and two simultaneous
    blasts from adjacent tiles routinely leave no escape within the fixed
    budget. Applied via _safety_mask, so it covers both the greedy branch
    of act() and _safe_random_action uniformly -- linear_agent needed two
    separate call sites for the same check; tree_agent's shared mask needs
    only this one.
    """
    position = state["position"]
    for name, opp_pos, opp_bomb_available in state["opponents_with_bomb"]:
        if not opp_bomb_available:
            continue
        radius = _RADIUS_IF_HAS_BOMBED if self.opponent_has_bombed.get(name) else _RADIUS_IF_NEVER_BOMBED
        if abs(opp_pos[0] - position[0]) + abs(opp_pos[1] - position[1]) <= radius:
            return True
    return False


def _safe_random_action(self, state):
    """
    Filters random exploration so it doesn't have to "discover" bomb danger
    by dying. See _safety_mask for the actual criteria.
    """
    safe = _safety_mask(self, state)

    if not safe.any():
        # genuinely trapped (or, during an escape, no verified path found
        # within budget): fall back to the original, unfiltered distribution
        return np.random.choice(ACTIONS, p=RANDOM_ACTION_WEIGHTS)

    weights = RANDOM_ACTION_WEIGHTS[safe]
    weights = weights / weights.sum()
    return np.random.choice(np.array(ACTIONS)[safe], p=weights)


def _log_previous_transition(self, game_state, state) -> None:
    """
    Write one row of the per-step metrics CSV for the *previous* step.

    best_q is whatever act() already computed for that step (None during
    exploration, since no prediction happens there) -- deliberately NOT
    recomputed here via a fresh q_values() call (see linear_agent's
    equivalent note: recomputing was a measured slowdown for a tree
    ensemble, unlike linear_agent's near-free dot product).
    """
    if self.log_round != game_state["round"] or self.prev_state is None:
        return
    sparse_reward = game_state["self"][1] - self.prev_score
    shaped_reward = GAMMA * coin_potential(state) - coin_potential(self.prev_state)
    best_q = "" if self.prev_best_q is None else self.prev_best_q
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


def _epsilon(round: int) -> float:
    return max(EPSILON_MIN, EPSILON_START * EPSILON_DECAY ** (round))
