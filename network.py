"""Legal-action Double DQN for draw-three Klondike.

Train: python network.py --mode train --episodes 10000
Evaluate: python network.py --mode evaluate --checkpoint checkpoints/best.pt
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from solitaire_environment import ACTION_SIZE, OBS_SIZE, Move, SolitaireEnvironment


class QNetwork(nn.Module):
    """Q(s, a), for any legal action; no huge, mostly-invalid action head."""

    def __init__(self, hidden_size: int = 256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(OBS_SIZE + ACTION_SIZE, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.layers(torch.cat((states.float(), actions.float()), dim=-1)).squeeze(-1)


@dataclass
class Transition:
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    next_actions: np.ndarray
    done: bool


class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory: deque[Transition] = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self.memory.append(transition)

    def sample(self, batch_size: int, rng: random.Random) -> list[Transition]:
        return rng.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


def action_values(model: QNetwork, state: np.ndarray,
                  feats: np.ndarray, device: torch.device) -> np.ndarray:
    if len(feats) == 0:
        return np.empty(0, dtype=np.float32)
    with torch.no_grad():
        s = torch.from_numpy(np.broadcast_to(state, (len(feats), OBS_SIZE)).copy()).to(device)
        a = torch.from_numpy(feats).to(device)
        return model(s, a).cpu().numpy()


def choose_action(env: SolitaireEnvironment, model: QNetwork | None,
                  epsilon: float, rng: random.Random, device: torch.device,
                  strategy: str = "learned") -> tuple[Move, np.ndarray]:
    moves = env.legal_actions()
    if not moves:
        raise RuntimeError("No legal actions")
    features = np.stack([env.action_features(m) for m in moves])
    if strategy == "random" or rng.random() < epsilon:
        index = rng.randrange(len(moves))
    elif strategy == "heuristic":
        priority = {
            "tableau_foundation": 8.0, "waste_foundation": 8.0,
            "tableau_tableau": 1.0, "waste_tableau": 2.0,
            "draw": 0.0, "foundation_tableau": -5.0,
        }
        scores = [priority[m.kind] +
                  (5.0 if m.kind == "tableau_tableau" and
                   len(env.tableau[m.src]) > m.count and
                   not env.tableau[m.src][-m.count - 1].face_up else 0.0)
                  for m in moves]
        index = int(np.argmax(scores))
    else:
        if model is None:
            raise ValueError("A trained model is required for learned selection")
        index = int(np.argmax(action_values(model, env.observe(), features, device)))
    return moves[index], features[index].copy()


def optimize(model: QNetwork, target: QNetwork, optimizer: torch.optim.Optimizer,
             memory: ReplayMemory, batch_size: int, gamma: float,
             device: torch.device, rng: random.Random) -> float | None:
    if len(memory) < batch_size:
        return None
    batch = memory.sample(batch_size, rng)
    states = torch.tensor(np.stack([t.state for t in batch]), device=device)
    actions = torch.tensor(np.stack([t.action for t in batch]), device=device)
    rewards = torch.tensor([t.reward for t in batch], device=device, dtype=torch.float32)
    dones = torch.tensor([t.done for t in batch], device=device, dtype=torch.bool)
    q = model(states, actions)

    # Double DQN: online network selects the best *legal* successor action;
    # target network estimates its value. Terminal states have no bootstrap.
    with torch.no_grad():
        next_values = torch.zeros(batch_size, device=device)
        active = [i for i, t in enumerate(batch) if not t.done and len(t.next_actions)]
        if active:
            lengths = [len(batch[i].next_actions) for i in active]
            next_states = np.concatenate([
                np.repeat(batch[i].next_state[None, :], n, axis=0)
                for i, n in zip(active, lengths)
            ])
            next_actions = np.concatenate([batch[i].next_actions for i in active])
            ns = torch.tensor(next_states, device=device)
            na = torch.tensor(next_actions, device=device)
            online_q = model(ns, na)
            cursor = 0
            selected_state = []
            selected_action = []
            for i, n in zip(active, lengths):
                j = int(torch.argmax(online_q[cursor:cursor + n]).item())
                selected_state.append(batch[i].next_state)
                selected_action.append(batch[i].next_actions[j])
                cursor += n
            target_q = target(torch.tensor(np.stack(selected_state), device=device),
                              torch.tensor(np.stack(selected_action), device=device))
            next_values[torch.tensor(active, device=device)] = target_q
        td_target = rewards + gamma * next_values * (~dones).float()

    loss = F.smooth_l1_loss(q, td_target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    optimizer.step()
    return float(loss.item())


def environment(args: argparse.Namespace, seed: int) -> SolitaireEnvironment:
    return SolitaireEnvironment(seed=seed, draw_count=args.draw_count,
                                max_redeals=args.max_redeals,
                                max_steps=args.max_steps, max_stall=args.max_stall)


def evaluate(model: QNetwork | None, args: argparse.Namespace,
             seeds: list[int], strategy: str = "learned") -> dict[str, float]:
    device = torch.device(args.device)
    if model is not None:
        model.eval()
    wins = foundations = reveals = moves = 0
    for seed in seeds:
        env = environment(args, seed)
        rng = random.Random(seed + 900_000)
        while not env.done:
            action, _ = choose_action(env, model, 0.0, rng, device, strategy)
            _, done, info = env.step(action)
            if done:
                wins += int(info["won"])
                foundations += info["foundation_cards"]
                reveals += 21 - info["face_down"]
                moves += info["steps"]
    n = max(1, len(seeds))
    return {"win_rate": wins / n, "foundation": foundations / n,
            "revealed": reveals / n, "steps": moves / n}


def save_checkpoint(path: Path, policy: QNetwork, target: QNetwork,
                    optimizer: torch.optim.Optimizer, episode: int,
                    steps: int, best_score: float, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"policy": policy.state_dict(), "target": target.state_dict(),
                "optimizer": optimizer.state_dict(), "episode": episode,
                "steps": steps, "best_score": best_score,
                "settings": {k: getattr(args, k) for k in
                             ("draw_count", "max_redeals", "max_steps", "max_stall", "hidden_size")}}, path)


def load_checkpoint(path: str, policy: QNetwork, device: torch.device) -> dict:
    # Only load checkpoints you trust. PyTorch state_dicts use pickle storage.
    obj = torch.load(path, map_location=device, weights_only=True)
    policy.load_state_dict(obj["policy"])
    return obj


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    rng = random.Random(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = QNetwork(args.hidden_size).to(device)
    target = QNetwork(args.hidden_size).to(device)
    target.load_state_dict(model.state_dict())
    target.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    memory = ReplayMemory(args.replay_size)
    writer = None
    if not args.no_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(args.logdir)
        except ImportError:
            print("TensorBoard unavailable; logging to console only")

    steps = 0
    start_episode = 0
    best_score = float("-inf")
    if args.resume:
        ckpt = load_checkpoint(args.resume, model, device)
        if ckpt.get("settings") != {k: getattr(args, k) for k in
                ("draw_count", "max_redeals", "max_steps", "max_stall", "hidden_size")}:
            raise ValueError("Checkpoint environment/model settings don't match")
        target.load_state_dict(ckpt["target"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_episode = ckpt["episode"]
        steps = ckpt["steps"]
        best_score = ckpt["best_score"]
        print("Resuming weights/optimizer; replay buffer starts empty")

    env = environment(args, args.seed)
    last_losses: list[float] = []
    try:
        for episode in range(start_episode + 1, args.episodes + 1):
            obs = env.reset(seed=args.seed + episode)
            total_reward = 0.0
            while not env.done:
                eps = args.eps_end + (args.eps_start - args.eps_end) * \
                      max(0.0, 1.0 - steps / max(1, args.eps_decay_steps))
                action, features = choose_action(env, model, eps, rng, device)
                reward, done, info = env.step(action)
                next_obs = env.observe()
                next_feats = (np.stack([env.action_features(m) for m in env.legal_actions()])
                              if not done else np.zeros((0, ACTION_SIZE), dtype=np.float32))
                memory.push(Transition(obs.copy(), features, float(reward),
                                       next_obs.copy(), next_feats, done))
                obs = next_obs
                total_reward += reward
                steps += 1
                if steps >= args.warmup and steps % args.train_every == 0:
                    loss = optimize(model, target, optimizer, memory, args.batch_size,
                                    args.gamma, device, rng)
                    if loss is not None:
                        last_losses.append(loss)
                        if len(last_losses) > 100:
                            last_losses.pop(0)
                if steps % args.target_every == 0:
                    target.load_state_dict(model.state_dict())
            if writer:
                writer.add_scalar("train/reward", total_reward, episode)
                writer.add_scalar("train/foundation", info["foundation_cards"], episode)
                writer.add_scalar("train/win", int(info["won"]), episode)
                writer.add_scalar("train/epsilon", eps, episode)
                if last_losses:
                    writer.add_scalar("train/loss", float(np.mean(last_losses)), episode)
            if episode % args.log_every == 0:
                print(f"episode={episode} steps={steps} epsilon={eps:.3f} "
                      f"foundation={info['foundation_cards']} won={info['won']} "
                      f"reward={total_reward:.2f} "
                      f"loss={np.mean(last_losses) if last_losses else float('nan'):.4f}", flush=True)
            if episode % args.eval_every == 0:
                # Fixed validation deals allow paired comparisons, not a proof
                # of generalization. Use disjoint unseen test deals later.
                results = evaluate(model, args, [100_000+i for i in range(args.eval_games)])
                score = results["win_rate"] * 100 + results["foundation"] / 52.0
                print("validation", episode, json.dumps(results), flush=True)
                if writer:
                    for name, val in results.items():
                        writer.add_scalar("validation/" + name, val, episode)
                if score > best_score:
                    best_score = score
                    save_checkpoint(Path(args.checkpoint_dir)/"best.pt", model,
                                    target, optimizer, episode, steps, best_score, args)
                save_checkpoint(Path(args.checkpoint_dir)/"last.pt", model,
                                target, optimizer, episode, steps, best_score, args)
        save_checkpoint(Path(args.checkpoint_dir)/"last.pt", model, target,
                        optimizer, args.episodes, steps, best_score, args)
    finally:
        if writer:
            writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "evaluate"), default="train")
    parser.add_argument("--checkpoint", help="Trained model checkpoint for evaluation")
    parser.add_argument("--resume", help="Resume a training checkpoint")
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--draw-count", type=int, choices=(1, 3), default=3)
    parser.add_argument("--max-redeals", type=int, default=2,
                        help="-1 permits unlimited redeals")
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--max-stall", type=int, default=150)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-size", type=int, default=50000)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--train-every", type=int, default=4)
    parser.add_argument("--target-every", type=int, default=2000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay-steps", type=int, default=200000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--logdir", default="runs/solitaire_ddqn")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--no-tensorboard", action="store_true")
    args = parser.parse_args()
    args.max_redeals = None if args.max_redeals == -1 else args.max_redeals
    if args.mode == "train":
        train(args)
    else:
        model = None
        if args.checkpoint:
            saved = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
            settings = saved.get("settings", {})
            for key in ("draw_count", "max_redeals", "max_steps", "max_stall", "hidden_size"):
                if key in settings:
                    setattr(args, key, settings[key])
            model = QNetwork(args.hidden_size).to(args.device)
            model.load_state_dict(saved["policy"])
        seeds = [1_000_000+i for i in range(args.eval_games)]
        for strategy in (["random", "heuristic", "learned"] if model else
                         ["random", "heuristic"]):
            print(strategy, json.dumps(evaluate(model, args, seeds, strategy)))


if __name__ == "__main__":
    main()
