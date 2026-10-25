"""Rule invariants and RL smoke tests: python -m unittest -v test"""
import random
import unittest

import numpy as np
import torch

from solitaire_environment import ACTION_SIZE, OBS_SIZE, Move, SolitaireEnvironment
from network import QNetwork, ReplayMemory, Transition, optimize, evaluate


class TestKlondike(unittest.TestCase):
    def test_seed_and_initial_deal(self):
        a, b = SolitaireEnvironment(seed=17), SolitaireEnvironment(seed=17)
        self.assertEqual(a._signature(), b._signature())
        self.assertEqual(len(a.stock), 24)
        self.assertEqual([len(c) for c in a.tableau], list(range(1, 8)))
        self.assertEqual(sum(not card.face_up for pile in a.tableau for card in pile), 21)
        a.assert_valid()
        self.assertEqual(a.observe().shape, (OBS_SIZE,))

    def test_hidden_cards_are_not_exposed(self):
        env = SolitaireEnvironment(seed=17)
        expected = env.observe().copy()
        env.stock[0], env.stock[1] = env.stock[1], env.stock[0]
        np.testing.assert_array_equal(env.observe(), expected)
        # Neither the identity nor ordering of tableau face-down cards leaks.
        hidden = [card for pile in env.tableau for card in pile if not card.face_up]
        hidden[0].suit, hidden[1].suit = hidden[1].suit, hidden[0].suit
        hidden[0].rank, hidden[1].rank = hidden[1].rank, hidden[0].rank
        np.testing.assert_array_equal(env.observe(), expected)

    def test_draw_three_and_redeal(self):
        env = SolitaireEnvironment(seed=0)
        original_stock = [(c.suit, c.rank) for c in env.stock]
        for _ in range(8):
            self.assertIn(Move("draw"), env.legal_actions())
            env.step(Move("draw"))
        self.assertEqual(len(env.stock), 0)
        self.assertEqual(len(env.waste), 24)
        env.step(Move("draw"))
        self.assertEqual(env.redeals, 1)
        self.assertEqual(len(env.stock), 21)
        self.assertEqual([(c.suit, c.rank) for c in env.waste],
                         list(reversed(original_stock[-3:])))
        env.assert_valid()

    def test_random_legal_play_preserves_cards_and_order(self):
        for seed in range(25):
            env = SolitaireEnvironment(seed=seed, max_steps=130)
            rng = random.Random(seed)
            while not env.done:
                action = rng.choice(env.legal_actions())
                self.assertEqual(env.action_features(action).shape, (ACTION_SIZE,))
                env.step(action)
                env.assert_valid()

    def test_illegal_moves_cannot_change_state(self):
        env = SolitaireEnvironment(seed=1)
        observation = env.observe().copy()
        with self.assertRaises(ValueError):
            env.step(Move("tableau_tableau", src=0, dst=0))
        np.testing.assert_array_equal(env.observe(), observation)

    def test_tableau_reveal_and_foundation(self):
        # Find an opening with an exposed ace and verify legal placement.
        for seed in range(1000):
            env = SolitaireEnvironment(seed=seed)
            moves = [m for m in env.legal_actions() if m.kind == "tableau_foundation"]
            if moves:
                move = moves[0]
                card = env.tableau[move.src][-1]
                reward, _, _ = env.step(move)
                self.assertEqual(env.foundation[card.suit][-1], card)
                self.assertGreater(reward, 0)
                env.assert_valid()
                return
        self.fail("No exposed ace in 1000 opening hands")


class TestLearning(unittest.TestCase):
    def test_q_model_and_gradient_update(self):
        torch.manual_seed(0)
        env = SolitaireEnvironment(seed=0)
        model = QNetwork(32)
        target = QNetwork(32)
        target.load_state_dict(model.state_dict())
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        memory = ReplayMemory(32)
        for _ in range(8):
            obs = env.observe()
            action = env.legal_actions()[0]
            feat = env.action_features(action)
            reward, done, _ = env.step(action)
            after = env.observe()
            possible = env.legal_actions()
            next_feats = (np.stack([env.action_features(m) for m in possible]) if possible
                          else np.zeros((0, ACTION_SIZE), dtype=np.float32))
            memory.push(Transition(obs, feat, reward, after, next_feats, done))
            if done:
                env.reset()
        previous = [p.detach().clone() for p in model.parameters()]
        loss = optimize(model, target, opt, memory, 4, 0.99,
                        torch.device("cpu"), random.Random(0))
        self.assertIsNotNone(loss)
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(any(not torch.equal(a, b) for a, b in
                            zip(previous, model.parameters())))


if __name__ == "__main__":
    unittest.main()
