"""
play_chess.py — play against the trained RL agent, or watch it play.

    # Play against it yourself (you are White by default):
    python3 play_chess.py --human
    python3 play_chess.py --human --agent-color white   # let the agent move first

    # Watch it:
    python3 play_chess.py                      # agent (White) vs random (Black)
    python3 play_chess.py --color black
    python3 play_chess.py --games 3
    python3 play_chess.py --opponent self      # trained agent vs itself
    python3 play_chess.py --opponent stockfish # vs the Stockfish engine (a strong AI)
    python3 play_chess.py --opponent stockfish --elo 2000 --engine-ms 200

Games run to a real chess result (checkmate / stalemate / draw) — nothing stops
them early. In human mode you don't need to know chess notation: your legal
moves are shown as a numbered menu and you just type the number you want.
"""

from __future__ import annotations

import argparse
import random
import shutil

import chess
import chess.engine

from chess_env import material_balance, render
from td_agent import TDAgent


def random_move(board, rng):
    return rng.choice(list(board.legal_moves))


def open_stockfish(path, elo):
    """Launch Stockfish (a strong, well-trained engine) over UCI, capped to `elo`.
    Returns an engine handle, or None with a helpful message if not installed."""
    path = path or shutil.which("stockfish") or "/opt/homebrew/bin/stockfish"
    try:
        engine = chess.engine.SimpleEngine.popen_uci(path)
    except (FileNotFoundError, PermissionError):
        print("Stockfish engine not found.\n"
              "Install it with:  brew install stockfish\n"
              "or pass its location with --engine-path /path/to/stockfish")
        return None
    try:
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": max(1320, elo)})
    except Exception:
        pass  # some builds lack strength-limiting; just play full strength
    return engine


def report_result(board, agent_is_white):
    outcome = board.outcome(claim_draw=False)
    if outcome and outcome.winner is not None:
        winner = "White" if outcome.winner == chess.WHITE else "Black"
        agent_won = (outcome.winner == chess.WHITE) == agent_is_white
        who = "AGENT" if agent_won else "opponent"
        print(f"Result: {winner} wins by {outcome.termination.name.lower()} -> {who} wins.")
    else:
        bal = material_balance(board)
        reason = outcome.termination.name.lower() if outcome else "stopped"
        print(f"Result: draw ({reason}); final material balance {bal:+.0f} (White's view).")
    print()


# --------------------------------------------------------------------------
# Watch mode (agent vs random / self)
# --------------------------------------------------------------------------
def play_one(agent, opponent, agent_is_white, rng, max_plies, engine=None, engine_limit=None):
    board = chess.Board()
    print(render(board) + "\n")
    ply = 0
    while not board.is_game_over(claim_draw=False) and (max_plies <= 0 or ply < max_plies):
        agents_turn = board.turn == (chess.WHITE if agent_is_white else chess.BLACK)
        if agents_turn or opponent == "self":
            mover, move = "agent", agent.select_move(board, explore=False, safe=True)
        elif opponent == "stockfish":
            mover, move = "stockfish", engine.play(board, engine_limit).move
        else:
            mover, move = "random", random_move(board, rng)

        san = board.san(move)
        is_cap = board.is_capture(move)
        board.push(move)
        ply += 1
        flag = "  (capture)" if is_cap else ""
        if board.is_checkmate():
            flag += "  CHECKMATE"
        elif board.is_check():
            flag += "  (check)"
        side = "White" if (board.turn == chess.BLACK) else "Black"  # side that just moved
        print(f"{ply:>3}. {side:<5} [{mover}] {san}{flag}")
        print(render(board) + "\n")

    report_result(board, agent_is_white)


# --------------------------------------------------------------------------
# Human mode (you vs the agent)
# --------------------------------------------------------------------------
PIECE_NAMES = {
    chess.PAWN: "Pawn", chess.KNIGHT: "Knight", chess.BISHOP: "Bishop",
    chess.ROOK: "Rook", chess.QUEEN: "Queen", chess.KING: "King",
}


def render_for_human(board):
    """Board with file letters and rank numbers, so you can read coordinates."""
    rows = []
    for rank in range(7, -1, -1):
        cells = []
        for file in range(8):
            piece = board.piece_at(chess.square(file, rank))
            cells.append(piece.unicode_symbol() if piece else ".")
        rows.append(f"  {rank + 1}  " + " ".join(cells))
    rows.append("     " + " ".join("abcdefgh"))
    return "\n".join(rows)


def describe_move(board, move):
    """A plain-English description like 'Knight g1->f3 (check)'."""
    piece = board.piece_at(move.from_square)
    desc = (f"{PIECE_NAMES.get(piece.piece_type, 'Piece')} "
            f"{chess.square_name(move.from_square)}->{chess.square_name(move.to_square)}")
    extras = []
    if board.is_capture(move):
        extras.append("takes")
    if move.promotion:
        extras.append(f"promotes to {PIECE_NAMES[move.promotion]}")
    board.push(move)
    if board.is_checkmate():
        extras.append("CHECKMATE")
    elif board.is_check():
        extras.append("check")
    board.pop()
    return desc + (" — " + ", ".join(extras) if extras else "")


def show_menu(board):
    """Print the legal moves as a numbered, two-column menu. Returns the list."""
    moves = sorted(board.legal_moves,
                   key=lambda m: (board.piece_at(m.from_square).piece_type,
                                  m.from_square, m.to_square))
    items = [describe_move(board, m) for m in moves]
    print("\nYour move — pick a number:")
    col_w = max((len(s) for s in items), default=0) + 5
    half = (len(items) + 1) // 2
    for r in range(half):
        left = f"{r + 1:>2}) {items[r]}"
        ri = r + half
        right = f"{ri + 1:>2}) {items[ri]}" if ri < len(items) else ""
        print(f"  {left:<{col_w}}{right}")
    return moves


def play_vs_human(agent, agent_is_white):
    board = chess.Board()
    human_color = chess.BLACK if agent_is_white else chess.WHITE
    human_name = "White" if human_color == chess.WHITE else "Black"
    agent_name = "Black" if human_color == chess.WHITE else "White"

    print(f"You are {human_name} ({'♙' if human_color == chess.WHITE else '♟'}); "
          f"the agent is {agent_name}.")
    print("Just type the number of the move you want. (0 = quit, u = take back)\n")
    print(render_for_human(board))

    while not board.is_game_over(claim_draw=False):
        if board.turn != human_color:
            move = agent.select_move(board, explore=False, safe=True)
            desc = describe_move(board, move)
            board.push(move)
            print(f"\nAgent plays: {desc}\n")
            print(render_for_human(board))
            continue

        moves = show_menu(board)
        try:
            choice = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nStopped.")
            return

        if choice in ("0", "q", "quit", "exit"):
            print("Stopped.")
            return
        if choice in ("u", "undo"):
            if len(board.move_stack) >= 2:
                board.pop()
                board.pop()
                print("\nTook back your last move.\n")
                print(render_for_human(board))
            else:
                print("Nothing to take back yet.")
            continue
        if not choice.isdigit() or not (1 <= int(choice) <= len(moves)):
            print("Please type one of the numbers shown (or 0 to quit).")
            continue

        move = moves[int(choice) - 1]
        desc = describe_move(board, move)
        board.push(move)
        print(f"\nYou play: {desc}\n")
        print(render_for_human(board))

    print()
    report_result(board, agent_is_white)


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Play against or watch the trained RL chess agent.")
    p.add_argument("--model", default="chess_weights.pkl")
    p.add_argument("--human", action="store_true", help="play against the agent yourself")
    p.add_argument("--agent-color", choices=["white", "black"], default="black",
                   help="(human mode) which side the agent plays; you get the other")
    p.add_argument("--color", choices=["white", "black"], default="white",
                   help="(watch mode) which side the agent plays")
    p.add_argument("--opponent", choices=["random", "self", "stockfish"], default="random",
                   help="(watch mode) the agent's opponent: random, itself, or the Stockfish engine")
    p.add_argument("--games", type=int, default=1)
    p.add_argument("--max-plies", type=int, default=None,
                   help="0 = no limit; default uses the value from training")
    p.add_argument("--elo", type=int, default=1350,
                   help="(stockfish) cap the engine's strength to this Elo (min ~1320)")
    p.add_argument("--engine-ms", type=int, default=100,
                   help="(stockfish) thinking time per move, in milliseconds")
    p.add_argument("--engine-path", default=None,
                   help="(stockfish) path to the engine binary (auto-detected if omitted)")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    agent, meta = TDAgent.load(args.model)
    print(f"Loaded agent trained for {meta.get('episodes', '?')} self-play games.\n")

    if args.human:
        play_vs_human(agent, agent_is_white=(args.agent_color == "white"))
        return

    max_plies = args.max_plies if args.max_plies is not None else meta.get("max_plies", 0)
    rng = random.Random(args.seed)

    engine, engine_limit = None, None
    if args.opponent == "stockfish":
        engine = open_stockfish(args.engine_path, args.elo)
        if engine is None:
            return
        engine_limit = chess.engine.Limit(time=args.engine_ms / 1000.0)
        print(f"Opponent: Stockfish (capped to ~{args.elo} Elo, {args.engine_ms} ms/move)\n")

    try:
        for g in range(args.games):
            if args.games > 1:
                print(f"===== Game {g + 1} =====")
            play_one(agent, args.opponent, agent_is_white=(args.color == "white"),
                     rng=rng, max_plies=max_plies, engine=engine, engine_limit=engine_limit)
    finally:
        if engine is not None:
            engine.quit()


if __name__ == "__main__":
    main()
