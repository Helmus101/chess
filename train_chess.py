"""
train_chess.py — train the RL chess agent by self-play.

    python3 train_chess.py                       # defaults
    python3 train_chess.py --episodes 4000
    python3 train_chess.py --episodes 6000 --max-plies 80 --alpha 0.01

How it works:
    * The agent plays full games against ITSELF (both sides use the same value
      function; White maximises V, Black minimises it), with epsilon-greedy
      exploration that decays over training.
    * After every move it makes a TD(0) update toward
          reward + gamma * V(next_state)
      where reward = material captured this move (+) / lost (-) plus the large
      terminal reward for checkmate.
    * Periodically it is evaluated (greedily) against a uniform-random opponent
      so we can watch the win-rate climb. A random-vs-random baseline is printed
      first for context.

Outputs a learning curve and saves the learned weights to chess_weights.pkl.
"""

from __future__ import annotations

import argparse
import random
import shutil
import signal

import chess
import chess.engine

from chess_env import material_balance, terminal_reward
from td_agent import TDAgent


def learned_piece_values(agent):
    """Recover the agent's learned piece values by averaging each piece type's
    learned piece-square table (white plane minus black plane). The agent was
    NEVER told these — it inferred them purely from reward."""
    values = {}
    for t in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING):
        base = (t - 1) * 64
        values[t] = sum(agent.w[base:base + 64]) / 64  # average over the table
    return values


# --------------------------------------------------------------------------
# Opponents
# --------------------------------------------------------------------------
def random_move(board, rng):
    return rng.choice(list(board.legal_moves))


# --------------------------------------------------------------------------
# Training episode (one full game, online TD(0))
# --------------------------------------------------------------------------
def training_episode(learner, opponent_move, learner_is_white, max_plies, mat_scale):
    """One training game with online TD(0).

    The learner plays one colour (ε-greedy) and `opponent_move(board) -> move`
    plays the other — that opponent can be a frozen snapshot of the learner
    (self-play) or a chess engine like Stockfish. Only the learner's value
    function is updated, and it learns from every position regardless of who moved.

    This is a terminal-reward MDP: the only reward arrives at game end and is
        mat_scale * final_material_balance  +  (+/-5 for checkmate)
    i.e. the user's "+1 per captured opponent / -1 per piece lost" summed over
    the whole game, plus the +5 king-capture (checkmate) reward. Material is a feature, so
    the value function learns a (correctly signed, positive) material weight,
    and the policy then prefers winning material and avoiding losses move-to-move.
    """
    board = chess.Board()
    learner_color = chess.WHITE if learner_is_white else chess.BLACK
    v_prev, f_prev = learner.value(board)

    ply = 0
    while True:
        if board.turn == learner_color:
            move = learner.select_move(board, explore=True)
        else:
            move = opponent_move(board)
        if move is None:
            break
        board.push(move)
        ply += 1

        capped = max_plies > 0 and ply >= max_plies
        if board.is_game_over(claim_draw=False) or capped:
            reward = mat_scale * material_balance(board) + terminal_reward(board)
            learner.td_update(f_prev, v_prev, reward)  # V(terminal) = 0
            break

        v_now, f_now = learner.value(board)
        target = learner.gamma * v_now  # no intermediate reward
        learner.td_update(f_prev, v_prev, target)
        v_prev, f_prev = v_now, f_now

    return board, ply


def open_training_engine(path):
    """Open Stockfish as a training opponent. Returns the engine, or None with a
    message if the binary is missing."""
    path = path or shutil.which("stockfish") or "/opt/homebrew/bin/stockfish"
    try:
        return chess.engine.SimpleEngine.popen_uci(path)
    except (FileNotFoundError, PermissionError):
        print("Stockfish not found. Install it (brew install stockfish) or pass "
              "--engine-path /path/to/stockfish.")
        return None


def difficulty_ladder(skill_max, depth_max):
    """Difficulty rungs from very weak to strong, each (label, skill, Limit).
    The first rungs node-limit Stockfish to near-random play (so even a weak
    agent can win and start climbing); then Skill Level ramps 0..skill_max at
    depth 1; finally the search depth deepens up to depth_max."""
    rungs = [("nodes=1", 0, chess.engine.Limit(nodes=1)),
             ("nodes=30", 0, chess.engine.Limit(nodes=30)),
             ("nodes=200", 0, chess.engine.Limit(nodes=200))]
    for s in range(0, skill_max + 1):
        rungs.append((f"skill {s}", s, chess.engine.Limit(depth=1)))
    for d in range(2, depth_max + 1):
        rungs.append((f"skill {skill_max} depth {d}", skill_max, chess.engine.Limit(depth=d)))
    return rungs


# --------------------------------------------------------------------------
# Evaluation vs a random opponent
# --------------------------------------------------------------------------
def play_vs_random(agent, agent_is_white, rng, max_plies, adjudicate=2.0):
    """Greedy agent vs uniform-random opponent. Returns 'win'/'loss'/'draw'
    from the agent's perspective. Games hitting the ply cap are adjudicated by
    material (a clear material edge counts as the win)."""
    board = chess.Board()
    ply = 0
    while not board.is_game_over(claim_draw=False) and (max_plies <= 0 or ply < max_plies):
        agents_turn = board.turn == (chess.WHITE if agent_is_white else chess.BLACK)
        move = agent.select_move(board, explore=False) if agents_turn else random_move(board, rng)
        board.push(move)
        ply += 1

    if board.is_checkmate():
        loser_is_white = board.turn == chess.WHITE
        agent_lost = loser_is_white == agent_is_white
        return "loss" if agent_lost else "win"

    # draw by rule, or adjudicate the capped game by material
    bal = material_balance(board)  # White-perspective
    edge = bal if agent_is_white else -bal
    if edge >= adjudicate:
        return "win"
    if edge <= -adjudicate:
        return "loss"
    return "draw"


def evaluate(agent, games, rng, max_plies):
    wins = losses = draws = 0
    for g in range(games):
        res = play_vs_random(agent, agent_is_white=(g % 2 == 0), rng=rng, max_plies=max_plies)
        wins += res == "win"
        losses += res == "loss"
        draws += res == "draw"
    return wins / games, draws / games, losses / games


class _RandomAgent:
    """Wraps random play to reuse evaluate()/play_vs_random()."""

    def __init__(self, rng):
        self.rng = rng

    def select_move(self, board, explore=False):
        return random_move(board, self.rng)


# --------------------------------------------------------------------------
# ASCII learning curve
# --------------------------------------------------------------------------
def ascii_plot(series, height=12, title=""):
    if not series:
        return "(no data)"
    pts = list(series)
    lo, hi = min(pts), max(pts)
    if hi - lo < 1e-9:
        hi = lo + 1e-9
    out = [title] if title else []
    for row in range(height, 0, -1):
        thresh = lo + (hi - lo) * (row - 0.5) / height
        line = "".join("#" if p >= thresh else " " for p in pts)
        out.append(f"{thresh:5.2f} |{line}")
    out.append(" " * 6 + "+" + "-" * len(pts))
    return "\n".join(out)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def train(args):
    forever = args.episodes <= 0
    anneal = args.anneal or (args.episodes if not forever else 2000)
    agent = TDAgent(
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=0.9,
        epsilon_min=0.05,
        epsilon_decay=(0.05 / 0.9) ** (1.0 / max(1, int(0.7 * anneal))),
        rng=random.Random(args.seed + 1),
    )

    if args.init:
        pretrained, _ = TDAgent.load(args.init)
        agent.w = list(pretrained.w)
        print(f"Warm-started from already-trained model: {args.init}")

    eval_rng = random.Random(args.seed + 2)
    base_w, base_d, base_l = evaluate(_RandomAgent(random.Random(args.seed + 3)),
                                      args.eval_games, eval_rng, args.max_plies)
    print(f"Random-vs-random baseline: win {base_w:.0%} | draw {base_d:.0%} | loss {base_l:.0%}")
    limit_txt = "no move limit — games end by chess rules" if args.max_plies <= 0 \
        else f"max {args.max_plies} plies each"

    # Set up the training opponent: self-play snapshot, or the Stockfish engine
    # at progressively higher difficulty (a curriculum).
    engine = None
    opponent = None
    ladder = None
    level = 0           # current rung of the difficulty ladder
    ramp_every = 0

    if args.opponent == "stockfish":
        engine = open_training_engine(args.engine_path)
        if engine is None:
            return
        ladder = difficulty_ladder(args.sf_skill_max, args.sf_depth_max)
        ramp_every = args.ramp_every or (max(1, args.episodes // len(ladder))
                                         if not forever else 200)

        def set_level(i):
            try:
                engine.configure({"Skill Level": ladder[i][1]})
            except Exception:
                pass
        set_level(0)

        def opponent_move(b):
            return engine.play(b, ladder[level][2]).move

        opp_txt = (f"Stockfish CURRICULUM — difficulty steps up every {ramp_every} games "
                   f"across {len(ladder)} rungs, '{ladder[0][0]}' -> '{ladder[-1][0]}'")
    else:
        # self-play: a frozen snapshot of the learner, re-synced every --refresh games
        opponent = agent.clone(rng=random.Random(args.seed + 7))
        opponent_move = lambda b: opponent.select_move(b, explore=False)  # noqa: E731
        opp_txt = f"self-play snapshot (refreshed every {args.refresh} games)"

    how_many = "continuously (Ctrl-C to stop)" if forever else f"{args.episodes} games"
    print(f"Training {how_many} vs {opp_txt} ({limit_txt}); "
          f"autosaving to {args.out} every {args.save_every} games...\n")

    # Stop cleanly on Ctrl-C: finish the current game, then save and report.
    stop = {"requested": False}

    def _request_stop(*_):
        if not stop["requested"]:
            print("\n(stop requested — finishing this game, then saving...)")
        stop["requested"] = True
    try:
        signal.signal(signal.SIGINT, _request_stop)
    except (ValueError, OSError):
        pass  # not in the main thread; KeyboardInterrupt fallback still applies

    meta = {
        "max_plies": 0,  # playback is always uncapped (games run to a real result)
        "representation": "piece-square-tied-384",
    }

    curve = []
    ep = 0
    try:
        while (forever or ep < args.episodes) and not stop["requested"]:
            ep += 1
            # Curriculum: step Stockfish up the difficulty ladder as training proceeds.
            if ladder is not None:
                want = min(len(ladder) - 1, (ep - 1) // ramp_every)
                if want != level:
                    level = want
                    set_level(level)
                    print(f"  ↑ difficulty up (ep {ep}): Stockfish now '{ladder[level][0]}'")

            training_episode(agent, opponent_move, learner_is_white=(ep % 2 == 0),
                             max_plies=args.max_plies, mat_scale=args.mat_scale)
            agent.decay_epsilon()
            if opponent is not None and ep % args.refresh == 0:
                opponent.copy_weights_from(agent)  # snapshot the improved learner

            if ep % args.eval_every == 0:
                win, draw, loss = evaluate(agent, args.eval_games,
                                           random.Random(args.seed + 100), args.max_plies)
                curve.append(win)
                extra = f" | SF '{ladder[level][0]}'" if ladder is not None else ""
                print(f"ep {ep:>6} | eps {agent.epsilon:.3f} | "
                      f"vs random: win {win:5.0%} draw {draw:4.0%} loss {loss:4.0%}{extra}")

            if ep % args.save_every == 0:
                meta["episodes"] = ep
                agent.save(args.out, meta)
    except KeyboardInterrupt:
        print(f"\nStopping at episode {ep} (Ctrl-C).")
    finally:
        if engine is not None:
            engine.quit()

    if curve:
        print("\nLearning curve — win-rate vs a random opponent:")
        print(ascii_plot(curve, title=f"  (each column = eval every {args.eval_every} games)"))

    print("\nPiece values the agent INFERRED from reward alone (never told these):")
    names = {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
             chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king"}
    for t, v in learned_piece_values(agent).items():
        print(f"  {names[t]:<7} {v:+6.2f}")
    print("  (classic reference values: pawn 1, knight 3, bishop 3, rook 5, queen 9)")

    meta["episodes"] = ep
    agent.save(args.out, meta)
    print(f"\nSaved weights ({ep} games) -> {args.out}")
    print(f"Watch it play:  python3 chess_rl.py play --model {args.out}")


def main():
    p = argparse.ArgumentParser(description="Train an RL chess agent by self-play.")
    p.add_argument("--episodes", type=int, default=0,
                   help="number of games to train; 0 = train continuously until Ctrl-C")
    p.add_argument("--save-every", type=int, default=200,
                   help="autosave the model to --out every N games")
    p.add_argument("--anneal", type=int, default=None,
                   help="games over which exploration (epsilon) decays to its floor")
    p.add_argument("--ramp-every", type=int, default=0,
                   help="(stockfish) step difficulty up one rung every N games "
                        "(0 = auto)")
    p.add_argument("--max-plies", type=int, default=0,
                   help="0 = no limit (games end by chess rules: checkmate, "
                        "stalemate, 75-move, fivefold repetition)")
    p.add_argument("--alpha", type=float, default=0.03)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--mat-scale", type=float, default=1.0,
                   help="scale on the final-material (capture) reward")
    p.add_argument("--opponent", choices=["self", "stockfish"], default="self",
                   help="train by self-play (default) or against Stockfish (curriculum)")
    p.add_argument("--refresh", type=int, default=50,
                   help="(self-play) re-sync the frozen opponent every N games")
    p.add_argument("--sf-skill-max", type=int, default=20,
                   help="(stockfish) top Skill Level the curriculum ramps up to (0-20)")
    p.add_argument("--sf-depth-max", type=int, default=1,
                   help="(stockfish) deepen search up to this depth once skill is maxed")
    p.add_argument("--engine-path", default=None,
                   help="(stockfish) path to the engine binary (auto-detected if omitted)")
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--eval-games", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init", default=None,
                   help="warm-start from an already-trained model and keep improving it")
    p.add_argument("--out", default="chess_weights.pkl")
    train(p.parse_args())


if __name__ == "__main__":
    main()
