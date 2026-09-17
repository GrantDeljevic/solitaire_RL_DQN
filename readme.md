# Solitaire RL DQN

An experimental Deep Q-Network agent for Klondike solitaire, built with PyTorch. The project combines a custom solitaire environment with reinforcement-learning infrastructure for representing game state, selecting moves, training, and evaluating an agent.

## Implementation

- Custom Klondike environment and move logic
- Encoded card-state representation with a 110-action space
- DQN policy and target networks
- Experience replay and epsilon-greedy exploration
- TensorBoard logging and saved experiment data

## Running

Install PyTorch, NumPy, and TensorBoard, then run:

```bash
python network.py
```

Training metrics are written under `runs/` and can be viewed with:

```bash
tensorboard --logdir runs
```

## Status

This is an incomplete research prototype rather than a finished solitaire solver. The environment and DQN training pipeline are implemented, but the model and training strategy still require tuning and validation.

The repository also includes the project report and supporting experimental results.
