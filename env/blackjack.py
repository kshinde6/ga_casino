# ===================== env/blackjack.py =====================
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np

# --- Simple Blackjack Engine (MVP) ---
# Rules (configurable later):
# - 6-deck shoe, shuffle when penetration exceeds 75%
# - Dealer stands on all 17 (S17)
# - Blackjack pays 3:2, no insurance/surrender (MVP)
# - Double on any two cards (one card draw), one split allowed (max 2 hands), no resplit aces
# - Aces count 1 or 11; soft totals tracked

CARD_VALUES = {**{str(v): min(v,10) for v in range(2,11)}, 'J':10, 'Q':10, 'K':10, 'A':1}
RANKS = ['2','3','4','5','6','7','8','9','10','J','Q','K','A']

@dataclass
class BJOutcome:
    profit: float
    action_log: List[str]  # human-readable action sequence
    player_hands: List[List[str]]
    dealer_up: str
    dealer_final: List[str]

class Shoe:
    def __init__(self, rng: np.random.Generator, n_decks: int = 6):
        self.rng = rng
        self.n_decks = n_decks
        self.cards: List[str] = []
        self.running_count = 0  # Hi-Lo
        self._build_and_shuffle()

    def _build_and_shuffle(self):
        self.cards = []
        for _ in range(self.n_decks):
            for r in RANKS:
                for _s in range(4):
                    self.cards.append(r)
        self.rng.shuffle(self.cards)
        self.running_count = 0

    def penetration(self) -> float:
        used = self.n_decks*52 - len(self.cards)
        return used / float(self.n_decks*52)

    def draw(self) -> str:
        if not self.cards:
            self._build_and_shuffle()
        c = self.cards.pop()
        # Hi-Lo count update (2-6:+1, 7-9:0, 10-A:-1)
        if c in ['2','3','4','5','6']:
            self.running_count += 1
        elif c in ['10','J','Q','K','A']:
            self.running_count -= 1
        return c

# Hand utilities

def hand_value(cards: List[str]) -> Tuple[int,bool]:
    total = sum(CARD_VALUES[c] for c in cards)
    soft = False
    if 'A' in cards and total + 10 <= 21:
        total += 10
        soft = True
    return total, soft

class BlackjackTable:
    def __init__(self, rng: np.random.Generator, n_decks: int = 6, shuffle_penetration: float = 0.75):
        self.rng = rng
        self.shoe = Shoe(rng, n_decks)
        self.shuffle_penetration = shuffle_penetration

    def maybe_shuffle(self):
        if self.shoe.penetration() >= self.shuffle_penetration:
            self.shoe._build_and_shuffle()

    def deal_initial(self) -> Tuple[List[str], List[str]]:
        p = [self.shoe.draw(), self.shoe.draw()]
        d = [self.shoe.draw(), self.shoe.draw()]
        return p, d

    def dealer_play(self, dealer: List[str]) -> List[str]:
        while True:
            val, soft = hand_value(dealer)
            if val < 17 or (val == 17 and soft is True and False):  # S17; change last False->True for H17
                dealer.append(self.shoe.draw())
            else:
                break
        return dealer

    def resolve(self, player: List[str], dealer_final: List[str]) -> float:
        pv, _ = hand_value(player)
        dv, _ = hand_value(dealer_final)
        if pv > 21:
            return -1.0
        if dv > 21:
            return 1.0
        if pv > dv:
            return 2.0  # NOTE: win: +1, temprarily doubled for incentive
        if pv < dv:
            return -0.5 # NOTE: loss: -1 unit, temporarily halved for incentive
        return 0.0

    def is_blackjack(self, hand: List[str]) -> bool:
        return sorted(hand_value(hand))[0] != -1 and set(hand) and len(hand) == 2 and (('A' in hand) and any(c in hand for c in ['10','J','Q','K']))

    def play_hand(self, bankroll: float, bet: float, policy_fn, state_extra: Optional[dict] = None) -> BJOutcome:
        """
        policy_fn(state) -> one of {'hit','stand','double','split'} where legal.
        Returns BJOutcome with profit in chips (can be fractional)
        """
        self.maybe_shuffle()
        action_log: List[str] = []
        player, dealer = self.deal_initial()
        dealer_up = dealer[0]
        dealer_final: List[str] = []

        # natural check
        player_bj = self.is_blackjack(player)
        dealer_bj = self.is_blackjack(dealer)
        if player_bj or dealer_bj:
            if player_bj and dealer_bj:
                return BJOutcome(0.0, ["push:blackjack"], [player], dealer_up, dealer)
            elif player_bj:
                return BJOutcome(1.5 * bet, ["win:blackjack"], [player], dealer_up, dealer)
            else:
                return BJOutcome(-bet, ["loss:dealer_blackjack"], [player], dealer_up, dealer)

        # splitting: allow at most one split (two hands max)
        hands: List[List[str]] = [player]
        bets: List[float] = [bet]
        i = 0
        while i < len(hands):
            h = hands[i]
            # per-hand action loop
            doubled = False
            while True:
                total, soft = hand_value(h)
                legal = {'hit','stand'}
                if len(h) == 2 and bankroll - sum(bets) >= bet:  # allow double if bankroll covers another bet
                    legal.add('double')
                if len(h) == 2 and len(hands) == 1 and h[0] == h[1]:
                    legal.add('split')
                state = {
                    'player_total': total,
                    'player_soft': soft,
                    'dealer_up': dealer_up,
                    'running_count': self.shoe.running_count,
                    'can_double': 'double' in legal,
                    'can_split': 'split' in legal,
                    'hand_index': i,
                    'n_hands': len(hands)
                }
                if state_extra:
                    state.update(state_extra)
                action = policy_fn(state, legal)
                if action not in legal:
                    action = 'stand'
                action_log.append(f"H{i}:{action}({total}{'s' if soft else ''})")

                if action == 'hit':
                    h.append(self.shoe.draw())
                    if hand_value(h)[0] > 21:
                        action_log.append(f"H{i}:bust")
                        break
                    continue
                elif action == 'double':
                    bets[i] += bet
                    h.append(self.shoe.draw())
                    doubled = True
                    break
                elif action == 'split':
                    # Only one split allowed
                    new_hand = [h.pop()]  # move one card to new hand
                    h.append(self.shoe.draw())
                    new_hand.append(self.shoe.draw())
                    hands.append(new_hand)
                    bets.append(bet)
                    action_log.append(f"split->H{len(hands)-1}")
                    # continue playing current hand (i unchanged)
                    continue
                else:  # stand
                    break
            i += 1

        # Dealer plays once for all hands
        dealer_final = self.dealer_play(dealer)
        total_profit = 0.0
        for hi, h in enumerate(hands):
            unit = self.resolve(h, dealer_final)
            total_profit += unit * bets[hi]
        return BJOutcome(total_profit, action_log, hands, dealer_up, dealer_final)

