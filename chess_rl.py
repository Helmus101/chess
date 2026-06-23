"""
chess_rl.py — single entry point for the RL chess project.

    python3 chess_rl.py train [options]     # train the agent (see train_chess.py)
    python3 chess_rl.py play  [options]     # play / watch it     (see play_chess.py)

Examples:
    python3 chess_rl.py train                         # train continuously (Ctrl-C to stop)
    python3 chess_rl.py train --episodes 3000
    python3 chess_rl.py train --opponent stockfish    # difficulty curriculum
    python3 chess_rl.py play  --human                 # play against it
    python3 chess_rl.py play  --opponent stockfish    # watch it vs Stockfish

Everything for the project lives in this one folder:
    chess_env.py  td_agent.py  train_chess.py  play_chess.py  chess_rl.py
"""

from __future__ import annotations

import sys


def main():
    cmds = {"train", "play"}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        print("usage: python3 chess_rl.py {train|play} [options]")
        print("  train   train the agent (self-play or vs Stockfish)")
        print("  play    play against the agent, or watch it")
        sys.exit(0 if len(sys.argv) >= 2 and sys.argv[1] in ("-h", "--help") else 1)

    cmd = sys.argv.pop(1)  # remove subcommand so the sub-command's argparse is clean
    if cmd == "train":
        import train_chess
        train_chess.main()
    else:
        import play_chess
        play_chess.main()


if __name__ == "__main__":
    main()
