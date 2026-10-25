"""Draw-three Klondike with observable (not omniscient) agent inputs.

Tableau positions increase from bottom to top. The final card is the playable
end of a pile; only the exposed portion of the tableau is observable.
"""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Optional

import numpy as np

SUITS = 4
TABLEAU = 7
MAX_CARDS = 13
ACTION_TYPES = (
    "draw", "tableau_tableau", "tableau_foundation",
    "waste_tableau", "waste_foundation", "foundation_tableau",
)
ACTION_SIZE = len(ACTION_TYPES) + 12 + 11 + 13 + 4 + 3
OBS_SIZE = TABLEAU * (1 + MAX_CARDS * 5) + SUITS + 4 + 24 * 5 + 52


@dataclass
class Card:
    suit: int  # 0 spades, 1 hearts, 2 clubs, 3 diamonds
    rank: int  # 1 ace through 13 king
    face_up: bool = False

    @property
    def color(self) -> int:
        return self.suit % 2


@dataclass(frozen=True)
class Move:
    kind: str
    src: int = -1
    dst: int = -1
    count: int = 1


class SolitaireEnvironment:
    """Klondike, draw-three, with a configurable limit on stock redeals.

    None for max_redeals permits unlimited recycling (episodes still have a
    finite step limit). Rewards are potential differences + a per-move cost;
    moving cards backwards cannot repeatedly generate positive reward.
    """

    def __init__(self, seed: Optional[int] = None, draw_count: int = 3,
                 max_redeals: Optional[int] = 2, max_steps: int = 400,
                 max_stall: int = 150):
        if draw_count not in (1, 3):
            raise ValueError("draw_count must be 1 or 3")
        if max_redeals is not None and max_redeals < 0:
            raise ValueError("max_redeals must be nonnegative or None")
        self.rng = random.Random(seed)
        self.draw_count = draw_count
        self.max_redeals = max_redeals
        self.max_steps = max_steps
        self.max_stall = max_stall
        self.reset()

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng.seed(seed)
        cards = [Card(s, r) for s in range(4) for r in range(1, 14)]
        self.rng.shuffle(cards)
        self.tableau: list[list[Card]] = [[] for _ in range(TABLEAU)]
        k = 0
        for col in range(TABLEAU):
            self.tableau[col] = cards[k:k + col + 1]
            k += col + 1
            self.tableau[col][-1].face_up = True
        self.stock: list[Card] = cards[k:]
        self.waste: list[Card] = []
        self.foundation: list[list[Card]] = [[] for _ in range(SUITS)]
        self.redeals = 0
        self.steps = 0
        self.stall = 0
        self.done = False
        self._best_progress = self._progress()
        self._seen = {self._signature(): 1}
        return self.observe()

    @staticmethod
    def _tableau_fit(card: Card, destination: list[Card]) -> bool:
        if not destination:
            return card.rank == 13
        top = destination[-1]
        return top.face_up and card.rank + 1 == top.rank and card.color != top.color

    def _foundation_fit(self, card: Card) -> bool:
        return card.rank == len(self.foundation[card.suit]) + 1

    def legal_actions(self) -> list[Move]:
        if self.done:
            return []
        moves: list[Move] = []
        for src, pile in enumerate(self.tableau):
            if not pile:
                continue
            for i, card in enumerate(pile):
                if not card.face_up:
                    continue
                count = len(pile) - i
                for dst, dest in enumerate(self.tableau):
                    if src != dst and self._tableau_fit(card, dest):
                        # Moving a fully visible pile onto an empty column is a
                        # pointless column permutation, not useful exploration.
                        if dest or i > 0:
                            moves.append(Move("tableau_tableau", src, dst, count))
            if self._foundation_fit(pile[-1]):
                moves.append(Move("tableau_foundation", src, pile[-1].suit))
        if self.waste:
            card = self.waste[-1]
            for dst, dest in enumerate(self.tableau):
                if self._tableau_fit(card, dest):
                    moves.append(Move("waste_tableau", -1, dst))
            if self._foundation_fit(card):
                moves.append(Move("waste_foundation", -1, card.suit))
        for suit, pile in enumerate(self.foundation):
            if pile:
                for dst, dest in enumerate(self.tableau):
                    if self._tableau_fit(pile[-1], dest):
                        moves.append(Move("foundation_tableau", suit, dst))
        if self.stock or (len(self.waste) > 1 and
                          (self.max_redeals is None or self.redeals < self.max_redeals)):
            moves.append(Move("draw"))
        return moves

    def _source_card(self, action: Move) -> Optional[Card]:
        if action.kind == "draw":
            return None
        if action.kind.startswith("tableau_"):
            return self.tableau[action.src][-action.count]
        if action.kind.startswith("waste_"):
            return self.waste[-1]
        return self.foundation[action.src][-1]

    def action_features(self, action: Move) -> np.ndarray:
        """Features of an actually legal move. No hidden-card identities."""
        f = np.zeros(ACTION_SIZE, dtype=np.float32)
        f[ACTION_TYPES.index(action.kind)] = 1.0
        offset = len(ACTION_TYPES)
        if action.kind.startswith("tableau_"):
            f[offset + action.src] = 1.0
        elif action.kind.startswith("foundation_"):
            f[offset + 7 + action.src] = 1.0
        elif action.kind.startswith("waste_"):
            f[offset + 11] = 1.0
        offset += 12
        if action.kind.endswith("tableau"):
            f[offset + action.dst] = 1.0
        elif action.kind.endswith("foundation"):
            f[offset + 7 + action.dst] = 1.0
        offset += 11
        card = self._source_card(action)
        if card is not None:
            f[offset + card.rank - 1] = 1.0
            f[offset + 13 + card.suit] = 1.0
        offset += 17
        f[offset] = action.count / 13.0
        f[offset + 1] = float(action.kind == "tableau_tableau" and
                               len(self.tableau[action.src]) > action.count and
                               not self.tableau[action.src][-action.count - 1].face_up)
        f[offset + 2] = (1.0 if action.kind.endswith("foundation") else
                         -1.0 if action.kind.startswith("foundation_") else 0.0)
        return f

    def _progress(self) -> float:
        foundation_count = sum(map(len, self.foundation))
        exposed = sum(sum(c.face_up for c in pile) for pile in self.tableau)
        return 0.7 * foundation_count + 0.3 * exposed

    def _signature(self) -> tuple:
        # Public-state cycle detection: stock contents intentionally omitted.
        tableau = tuple(tuple((c.suit, c.rank) if c.face_up else (-1, -1)
                              for c in pile) for pile in self.tableau)
        return (tableau, tuple(map(len, self.foundation)),
                tuple((c.suit, c.rank) for c in self.waste),
                len(self.stock), self.redeals)

    def step(self, action: Move) -> tuple[float, bool, dict]:
        if self.done:
            raise RuntimeError("Episode ended; call reset")
        # Validity is checked before modifying anything.
        if action not in self.legal_actions():
            raise ValueError(f"Illegal action: {action}")
        before = self._progress()
        if action.kind == "draw":
            if not self.stock:
                self.stock = list(reversed(self.waste))
                self.waste.clear()
                for c in self.stock:
                    c.face_up = False
                self.redeals += 1
            for _ in range(min(self.draw_count, len(self.stock))):
                card = self.stock.pop()
                card.face_up = True
                self.waste.append(card)
        elif action.kind == "tableau_tableau":
            src = self.tableau[action.src]
            moving = src[-action.count:]
            del src[-action.count:]
            self.tableau[action.dst].extend(moving)
            if src:
                src[-1].face_up = True
        elif action.kind == "tableau_foundation":
            src = self.tableau[action.src]
            self.foundation[action.dst].append(src.pop())
            if src:
                src[-1].face_up = True
        elif action.kind == "waste_tableau":
            self.tableau[action.dst].append(self.waste.pop())
        elif action.kind == "waste_foundation":
            self.foundation[action.dst].append(self.waste.pop())
        elif action.kind == "foundation_tableau":
            self.tableau[action.dst].append(self.foundation[action.src].pop())
        self.steps += 1
        progress = self._progress()
        if progress > self._best_progress:
            self.stall = 0
            self._best_progress = progress
        else:
            self.stall += 1
        signature = self._signature()
        old_visits = self._seen.get(signature, 0)
        self._seen[signature] = old_visits + 1
        reward = progress - before - 0.02 - (0.05 if old_visits else 0.0)
        won = all(len(pile) == 13 for pile in self.foundation)
        if won:
            reward += 15.0
        no_actions = not self.legal_actions() if not won else False
        truncated = self.steps >= self.max_steps or self.stall >= self.max_stall
        self.done = won or no_actions or truncated
        if self.done and not won:
            reward -= 1.0
        return reward, self.done, {
            "won": won, "truncated": truncated,
            "foundation_cards": sum(map(len, self.foundation)),
            "face_down": sum(not c.face_up for p in self.tableau for c in p),
            "steps": self.steps,
        }

    def observe(self) -> np.ndarray:
        result = np.zeros(OBS_SIZE, dtype=np.float32)
        at = 0
        for pile in self.tableau:
            result[at] = sum(not c.face_up for c in pile) / 6.0
            at += 1
            visible = [c for c in pile if c.face_up]
            for slot, card in enumerate(visible):
                start = at + slot * 5
                result[start] = card.rank / 13.0
                result[start + 1 + card.suit] = 1.0
            at += MAX_CARDS * 5
        for pile in self.foundation:
            result[at] = len(pile) / 13.0
            at += 1
        result[at:at+4] = [len(self.stock) / 24.0, len(self.waste) / 24.0,
                            (self.redeals / max(1, self.max_redeals)
                             if self.max_redeals is not None else 0.0),
                            self.steps / self.max_steps]
        at += 4
        for i, card in enumerate(self.waste[-24:]):
            start = at + i * 5
            result[start] = card.rank / 13.0
            result[start + 1 + card.suit] = 1.0
        at += 24 * 5
        if self.waste:
            top = self.waste[-1]
            result[at + top.suit * 13 + top.rank - 1] = 1.0
        return result

    def assert_valid(self) -> None:
        """Debug invariant checks, not used inside the training hot loop."""
        piles = self.tableau + self.foundation + [self.stock, self.waste]
        cards = [c for pile in piles for c in pile]
        assert len(cards) == 52
        assert len({(c.suit, c.rank) for c in cards}) == 52
        for suit, pile in enumerate(self.foundation):
            assert [(c.suit, c.rank) for c in pile] == [(suit, r) for r in range(1, len(pile)+1)]
        for pile in self.tableau:
            seen_up = False
            for c in pile:
                if c.face_up:
                    seen_up = True
                else:
                    assert not seen_up
            visible = [c for c in pile if c.face_up]
            for below, above in zip(visible, visible[1:]):
                assert below.rank == above.rank + 1 and below.color != above.color
