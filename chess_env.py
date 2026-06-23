"""
chess_env.py — a thin RL layer over python-chess.

python-chess provides the "hardcoded rules of where it can play": legal-move
generation, check / checkmate / stalemate / draw detection. This module adds
the pieces an RL agent needs:

    * feature_terms(board) -> the raw board encoding the agent sees (no chess
                              knowledge — just which piece sits on which square)
    * material_balance -> White material minus Black material, in pawns
    * terminal_reward  -> +5 if White checkmates ("captures the king"), -5 if mated
    * reward design     = final material captured (capture +, get captured -)
                          plus the +5 king-capture (checkmate) reward.

The agent starts tabula rasa: it is given ONLY the rules (via python-chess) and
this reward. It is never told piece values or any heuristic — it discovers them.
Everything is expressed from White's point of view; the agent's value function
V(s) estimates White's expected return, and Black simply tries to minimise it.
"""

from __future__ import annotations

import chess

# Standard piece values, in pawns.
PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}

# Reward for "capturing the king". A king is never literally captured in chess —
# the game ends at checkmate — so delivering checkmate IS the king capture, worth
# +5 (and -5 to the side that gets mated), on top of the material captured.
KING_CAPTURE_REWARD = 5.0

# ---------------------------------------------------------------------------
# State representation: a PURE piece-square encoding (tabula rasa).
#
# The agent is given ONLY the raw board — it learns one value per
# (piece-type, square) = 6 * 64 = 384 weights, forming a piece-square table per
# piece type. There is NO hand-coded chess knowledge (no "material", "centre" or
# "bishop pair" hints): the agent must discover piece values AND where pieces
# belong entirely from reward.
#
# Black's table is TIED to White's mirror image (square ^ 56 flips the rank):
# a white piece on `sq` adds +W[type][sq]; a black piece on `sq` subtracts
# W[type][mirror(sq)]. This bakes in the game's colour symmetry (not chess
# knowledge — just the rules' invariance), so the start position scores exactly
# 0 and every game trains both colours' weights at once.
# ---------------------------------------------------------------------------
N_PIECE_TYPES = 6                       # pawn..king
N_FEATURES = N_PIECE_TYPES * 64         # 384


def feature_terms(board: chess.Board):
    """The active (index, sign) terms for this position, from White's view.
    V(board) = sum(sign * w[index]); only ~32 terms (one per piece)."""
    terms = []
    for square, piece in board.piece_map().items():
        base = (piece.piece_type - 1) * 64
        if piece.color == chess.WHITE:
            terms.append((base + square, 1.0))
        else:
            terms.append((base + (square ^ 56), -1.0))  # mirror rank, negate
    return terms


def material_balance(board: chess.Board) -> float:
    """White material minus Black material, in pawns. Used only for the terminal
    reward and for reporting — NOT shown to the agent as a feature."""
    bal = 0.0
    for piece in board.piece_map().values():
        v = PIECE_VALUES[piece.piece_type]
        bal += v if piece.color == chess.WHITE else -v
    return bal


def terminal_reward(board: chess.Board) -> float:
    """White-perspective reward for a finished game (0 for any draw).

    Delivering checkmate = "capturing the opponent's king" = +5 (the mated side
    gets -5)."""
    if board.is_checkmate():
        # The side to move is the one that has been checkmated.
        return -KING_CAPTURE_REWARD if board.turn == chess.WHITE else KING_CAPTURE_REWARD
    return 0.0


def is_terminal(board: chess.Board) -> bool:
    return board.is_game_over(claim_draw=False)


def render(board: chess.Board) -> str:
    try:
        return board.unicode(borders=False, empty_square=".")
    except Exception:
        return str(board)
