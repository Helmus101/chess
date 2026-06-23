# RL chess — an agent that learns to play from capture/checkmate rewards

A reinforcement-learning agent that learns to play **chess** by self-play. It is
given only the rules and a reward signal — no opening book, no engine to copy,
no human games:

> **+1 for every opponent piece it captures, −1 for every piece it loses**
> (weighted by piece value), plus a large bonus for delivering **checkmate**.

From that alone it discovers piece values, learns to win material, avoid losing
its own pieces, and go for mate — and it beats a random opponent ~100%.

## Why this design (and its honest limits)

Full chess has ~10⁴⁰ positions, so a plain lookup table ("tabular" RL) is
impossible, and there's no GPU/PyTorch here. So the agent uses **linear value
function approximation** trained by **TD(0) self-play** — the same family of
method behind TD-Gammon. It learns sound, material-and-position-driven chess and
crushes a random opponent, but it is *not* a strong engine (a few hand-built
features + 1-ply lookahead). It's a clear, fully-from-scratch demonstration of RL
learning chess, not a Stockfish competitor.

## How it works

| Piece | What it does |
|-------|--------------|
| [`chess_env.py`](chess_env.py) | Wraps `python-chess` (the rules / legal moves / checkmate detection). Defines the **board encoding** the agent sees and the **reward**. |
| [`td_agent.py`](td_agent.py) | The agent: a linear value function `V(s) = w · features(s)`, 1-ply move selection, and the TD(0) learning update. |
| [`train_chess.py`](train_chess.py) | Self-play training loop (learner vs a frozen snapshot of itself), evaluation vs a random opponent, learning curve, saves weights. |
| [`play_chess.py`](play_chess.py) | Play against the trained agent, or watch it. |

**Tabula rasa — only rules + reward.** The agent is given *no* chess knowledge.
Its value function `V(s)` is a weighted sum over a **pure piece-square encoding**:
one learned weight per (piece-type, square) — 6 × 64 = 384 weights forming a
piece-square table per piece type. It is never told that a queen is worth more
than a pawn, or that the centre matters; it must infer all of that from reward.
(Black's table is tied to White's mirror image, which just encodes the game's
colour symmetry and roughly halves what must be learned.) After training, the
script prints the **piece values it inferred** — they come out in the right order
(queen > rook > bishop > knight > pawn), learned purely from the +capture /
−captured / +5-checkmate signal.

**Policy (move choice).** The agent looks one ply ahead: it evaluates the
position after each legal move and picks the best one — White maximises `V`,
Black minimises it (ε-greedy, with ε decaying over training). An immediate
checkmate is always taken.

**Self-play.** Training is pure self-play: the learner plays a **frozen snapshot
of itself** (re-synced to the learner's latest weights every `--refresh` games,
default 50), alternating colours each game. Only the learner's value function is
updated. Freezing the opponent for a while keeps it from chasing a moving target
— it's noticeably more stable than having both sides share the live weights.

**Reward & learning.** This is a *terminal-reward* problem: the reward arrives at
game end and equals `final_material_balance + (±5 for checkmate)` — exactly
your "+1 captured / −1 lost" summed over the whole game, plus a **+5 reward for
"capturing the king"** (a king is never literally taken in chess, so delivering
checkmate is the king capture). Because material is also a feature, the TD(0) update

```
w  ←  w + α · ( reward + γ·V(s′) − V(s) ) · features(s)
```

teaches the agent correct, positive piece values (and where pieces belong), and
the greedy policy then prefers winning material and avoiding losses on every move.

> Note: an earlier version rewarded the per-move material *change*, which makes a
> state's future reward *anti-correlated* with its current material and taught the
> agent *negative* piece values. Using the **final** material as a terminal reward
> fixes this — a nice illustration of why reward design matters in RL.

## Setup

```bash
pip3 install python-chess          # the only dependency
```

## Train

```bash
python3 train_chess.py                       # ~1500 self-play games
python3 train_chess.py --episodes 3000       # train longer for sharper play
python3 train_chess.py --refresh 100          # freeze the self-play opponent longer
python3 train_chess.py --max-plies 120        # re-impose a ply cap (default 0 = none)
python3 train_chess.py --init chess_weights.pkl --out v2.pkl   # keep improving an already-trained model
```

`--init` warm-starts from an existing model and keeps training it (the self-play
opponent also starts from that model) — so you can train in rounds, each one
playing against an already-trained version of itself.

### Training against Stockfish (difficulty curriculum)

You can train *against the Stockfish engine* instead of self-play, at
**progressively higher difficulty**:

```bash
python3 train_chess.py --opponent stockfish                          # ramp up to skill 20
python3 train_chess.py --opponent stockfish --sf-skill-max 8 --episodes 3000
python3 train_chess.py --opponent stockfish --sf-depth-max 4         # also deepen search at the top
python3 train_chess.py --init chess_weights.pkl --opponent stockfish --out vs_sf.pkl
```

Stockfish starts near-random and steps up a difficulty **ladder** over the course
of training:

```
nodes=1 -> nodes=30 -> nodes=200 -> skill 0 -> skill 1 -> ... -> skill <sf-skill-max>
        -> (then deepen search up to --sf-depth-max)
```

The ramp is spread evenly across `--episodes`, so the agent meets each rung in
turn and you'll see `↑ difficulty up` messages as it climbs. The first rungs are
deliberately near-random because this 1-ply agent *hangs pieces* (its evaluation
doesn't see recaptures), so even Stockfish with a 1-node search punishes it — it
needs an easy floor to get any positive signal.

**Honest notes.** Training vs Stockfish is slower than self-play (the engine
thinks on every move), and once the ladder climbs past what this weak agent can
handle, those late games are near-certain losses (uniformly negative reward), so
they teach little. **Self-play remains the better default** for actually getting
stronger; the curriculum is here because you asked to face progressively tougher
engines. A good combo is the last command: train by self-play first, then ramp
against Stockfish.

By default there is **no move limit** — every game (training, evaluation, and
playback) runs until a real chess result: checkmate, stalemate, insufficient
material, the 75-move rule, or fivefold repetition. Because the agent wins
material easily but has no dedicated *mating technique*, converting that into
checkmate can take a long time (occasionally a few hundred moves), so games — and
training — run slower than with a cap. Pass `--max-plies N` to cap them again.

You'll see a random-vs-random baseline, the win-rate vs a random opponent rising
over training, an ASCII learning curve, and finally **the piece values the agent
inferred from reward alone** — they come out in the right order
(queen > rook > bishop > knight > pawn), discovered with zero chess knowledge.
(The absolute magnitudes are compressed; the ordering is the point.)

## Play against it

```bash
python3 play_chess.py --human                    # you are White, agent is Black
python3 play_chess.py --human --agent-color white  # agent moves first; you are Black
```

**No chess notation needed.** On your turn it lists every legal move as a
numbered menu, in plain English, and you just type the number:

```
Your move — pick a number:
   1) Pawn a2->a3    10) Pawn e2->e4    19) Knight g1->f3
   ...
> 10
You play: Pawn e2->e4
```

The board is drawn with file letters and rank numbers so the coordinates
(`e2->e4`) are easy to read. Type `0` to quit or `u` to take back your last move.
The game runs until a real chess result — checkmate, stalemate, or a draw — it
never stops early.

## Watch it play

```bash
python3 play_chess.py                      # agent (White) vs random (Black)
python3 play_chess.py --color black
python3 play_chess.py --games 3
python3 play_chess.py --opponent self      # the trained agent vs itself
```

It prints the board after every move (in algebraic notation), flags captures and
checks, and reports the result.

### Against a strong engine (Stockfish)

You can pit the agent against **Stockfish**, a world-class engine — a real
"AI that knows how to play well":

```bash
brew install stockfish                                  # one-time
python3 play_chess.py --opponent stockfish              # capped to ~1350 Elo, 100 ms/move
python3 play_chess.py --opponent stockfish --elo 2200 --engine-ms 300
python3 play_chess.py --opponent stockfish --engine-path /path/to/stockfish
```

`--elo` caps Stockfish's strength (min ~1320) and `--engine-ms` sets its thinking
time. Be realistic: this linear, 1-ply agent will lose to Stockfish even when it
is heavily handicapped — Stockfish is a different league. It's there so you can
*see* the agent tested against genuinely strong play.

## Ideas to take it further

- Add features: pawn structure, king safety, passed pawns, mobility.
- Deeper search (2–3 ply / alpha-beta) on top of the learned value function.
- TD(λ) with eligibility traces, or a small neural net value function (if you add
  NumPy/PyTorch) for a non-linear evaluator.
- Grow the single frozen opponent into a full *league* of many past checkpoints
  (sample an opponent each game) for more robust self-play.
