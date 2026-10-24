# Solitaire RL — legal-action Double DQN

A reinforcement-learning baseline for **draw-three Klondike solitaire**. The
original prototype had game initialization bugs, an inconsistent action
encoding, invalid action selection, and no reliable evaluation. This revision
replaces the environment and training loop rather than merely tuning the old
hyperparameters.

## Rules

- Standard seven-column tableau; descending ranks with alternating colors.
  Any exposed descending run can be moved as a unit.
- Foundations are built from ace to king by suit. Only the exposed tableau,
  current waste card, or a foundation top can be moved.
- Draw three cards from the stock per draw, with two stock redeals by default.
  `--draw-count 1` and `--max-redeals -1` are available, but those constitute
  **different game variants** and should not be compared to draw-three results.
- An episode ends when the foundations are complete, no moves remain, or a
  fixed step/stall limit is reached. Time/stall truncation counts as a loss.
- Cards in the stock and face-down tableau are hidden from observations and
  move features. The actual deck is used by the simulator, not the policy.

## Setup

Python 3.10+; install a compatible version of PyTorch and:

```sh
pip install -r requirements.txt
python -m unittest -v test
```

## Train

```sh
python network.py --mode train --episodes 10000
# To watch metrics:
tensorboard --logdir runs/
```

The default model is a state-action MLP trained with **Double DQN**, Huber
loss, replay, target network, gradient clipping, and epsilon-greedy selection
**among legal actions only**. The encoded observation is 642-dimensional;
legal moves are described by 49 features. This avoids the enormous fixed
permutation action space and ensures unselectable actions receive no Q value.

Training writes `checkpoints/last.pt` and `checkpoints/best.pt` (best according
to validation win rate, then foundation progress) and TensorBoard metrics.
To resume from a prior checkpoint:

```sh
python network.py --mode train --episodes 20000 --resume checkpoints/last.pt
```

Resuming restores the model, target, optimizer and counters, **not** replay
samples or RNG state, so continuation is not bit-for-bit identical.

## Evaluate honestly

```sh
python network.py --mode evaluate --checkpoint checkpoints/best.pt --eval-games 1000
```

This compares a trained greedy policy to random-legal and deterministic
heuristic policies on **identical held-out deals** (seeds 1,000,000+). Report
win rate, mean cards in foundation, face-down cards revealed, and moves.
Validation during training uses a separate fixed set (seeds 100,000+).
Do not tune on the held-out evaluation set. Repeat with multiple train seeds
and report uncertainty (e.g., confidence intervals for win rate).

## Important limitations

This is a sounder **experimental baseline**, not a guaranteed high-win-rate
Klondike solver. Full Klondike has hidden information, long-horizon credit
assignment, nontrivial draw order, and many losing deals. Pure DQN may still
converge to weak or myopic play. The deterministic heuristic is only a simple
comparison and is not an expert solver.

Meaningful research improvements to evaluate next:

1. Use **curriculum training** (draw-one or partially solved deals before
   draw-three full games), retaining held-out draw-three as the real target.
2. Try imitation learning from a search-based or heuristic expert, followed
   by offline-to-online RL fine-tuning.
3. Add action history/belief features for hidden-stock inference and recurrent
   policy networks; current observation is not perfectly Markov.
4. Validate rule variants, episode truncation, and win percentage against
   trusted Klondike implementations using independently seeded test deals.

Reward is progress-based: +0.7 per foundation card, +0.3 per uncovered tableau
card, -0.02 per step, a repeated-position penalty, and a win bonus. Backward
moves undo their earlier progress reward; this reduces simple reversal loops.
Such shaping may still favor foundation progress over a strategically better
move, so evaluate **actual wins**, not training return alone.

## Files

- `solitaire_environment.py`: legal game engine, public observation encoding,
  action generation, and reward/termination rules.
- `network.py`: legal-action Double DQN training, checkpointing, baseline
  comparisons, and holdout evaluation.
- `main.py`: compatibility training entry point.
- `test.py`: deterministic mechanics/property/smoke tests.
- `.github/workflows/tests.yml`: CI rule checks and tiny training smoke test.

Old TensorBoard runs and PDFs in the original repository are historical and
not compatible with the new metrics.
