"""
td_agent.py — a reinforcement-learning chess agent.

The agent holds a LINEAR value function over the features in chess_env:

    V(s) = w . features(s)        (estimates White's expected return)

It chooses a move by 1-ply lookahead ("afterstate" evaluation): it looks at the
position resulting from every legal move and, when it is White, picks the move
maximising V; when Black, the move minimising V. An immediate checkmate is
always taken (and being mated always avoided when an alternative exists).

The weights are trained by TD(0) during self-play (see train_chess.py):

    delta = (reward + gamma * V(s')) - V(s)
    w    += alpha * delta * features(s)

This is real reinforcement learning: no supervised data, no engine to copy —
the agent improves only from the +capture / -captured / +checkmate rewards.
"""

from __future__ import annotations

import pickle
import random

import chess

from chess_env import N_FEATURES, feature_terms

# Sentinel value used so the policy always grabs a mate-in-1 and never walks
# into being mated when it can avoid it.
_MATE_VALUE = 1e6

# Clip threshold for the TD error (in the same pawn units as the reward) so the
# rare, large checkmate target cannot destabilise the linear weights.
_MATE_CLIP = 60.0


class TDAgent:
    def __init__(
        self,
        alpha: float = 0.01,
        gamma: float = 0.99,
        epsilon: float = 0.9,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.9995,
        rng: random.Random | None = None,
    ):
        self.w = [0.0] * N_FEATURES
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = rng or random.Random()

    # ----- value ---------------------------------------------------------
    def value(self, board):
        """Return (V(board), active_terms).

        V is the signed sum of the active piece-square weights — O(32 pieces)."""
        terms = feature_terms(board)
        w = self.w
        v = 0.0
        for i, sign in terms:
            v += sign * w[i]
        return v, terms

    @staticmethod
    def _has_mate_in_one(board):
        """Does the side to move have a checkmate available right now?"""
        for m in board.legal_moves:
            board.push(m)
            mate = board.is_check() and board.is_checkmate()
            board.pop()
            if mate:
                return True
        return False

    def _afterstate_value(self, board, move, safe=False):
        """White-perspective value of the position AFTER `move`, with hard
        overrides for immediate checkmate (and, if `safe`, for moves that let the
        opponent mate in one)."""
        board.push(move)
        try:
            # is_check() is cheap; only pay for the expensive is_checkmate()
            # (which generates all replies) when the move actually gives check.
            if board.is_check() and board.is_checkmate():
                # Whoever just moved delivered mate. After the push it is the
                # other side to move, so White moved iff it is now Black's turn.
                return _MATE_VALUE if board.turn == chess.BLACK else -_MATE_VALUE
            if safe and self._has_mate_in_one(board):
                # our move hangs a mate-in-one for the opponent — avoid it
                return -_MATE_VALUE if board.turn == chess.BLACK else _MATE_VALUE
            v, _ = self.value(board)
            return v
        finally:
            board.pop()

    # ----- policy --------------------------------------------------------
    def select_move(self, board, explore: bool = True, safe: bool = False):
        legal = list(board.legal_moves)
        if not legal:
            return None
        if explore and self.rng.random() < self.epsilon:
            return self.rng.choice(legal)

        white_to_move = board.turn == chess.WHITE
        scored = [(self._afterstate_value(board, m, safe), m) for m in legal]
        best_val = max(s for s, _ in scored) if white_to_move else min(s for s, _ in scored)
        best_moves = [m for s, m in scored if s == best_val]
        return self.rng.choice(best_moves)

    # ----- learning ------------------------------------------------------
    def td_update(self, active_terms, v_estimate, target):
        """One TD(0) gradient step. Only the active piece-square weights move
        (each by step * its sign)."""
        delta = target - v_estimate
        # clip to keep the (rare) huge checkmate target from destabilising w
        if delta > _MATE_CLIP:
            delta = _MATE_CLIP
        elif delta < -_MATE_CLIP:
            delta = -_MATE_CLIP
        step = self.alpha * delta
        w = self.w
        for i, sign in active_terms:
            w[i] += step * sign

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def clone(self, rng=None):
        """A frozen, greedy copy with the same weights — the self-play opponent."""
        twin = TDAgent(
            alpha=self.alpha,
            gamma=self.gamma,
            epsilon=0.0,
            epsilon_min=0.0,
            epsilon_decay=1.0,
            rng=rng or random.Random(),
        )
        twin.w = list(self.w)
        return twin

    def copy_weights_from(self, other):
        """Re-sync this (frozen opponent) to another agent's current weights."""
        self.w = list(other.w)

    # ----- persistence ---------------------------------------------------
    def save(self, path, meta=None):
        with open(path, "wb") as f:
            pickle.dump({"w": self.w, "meta": meta or {}}, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        agent = cls(epsilon=0.0, epsilon_min=0.0)
        agent.w = data["w"]
        return agent, data.get("meta", {})
