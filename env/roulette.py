
# ===================== env/roulette.py =====================
import numpy as np
from typing import Optional
from dataclasses import dataclass

@dataclass
class RouletteOutcome:
    number: int  # 0..36 (single zero wheel)
    color: str   # 'green', 'red', 'black'

class RouletteTable:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        # European single-zero wheel. Reds/Blacks per standard layout.
        self.red_numbers = {1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36}

    def spin(self) -> RouletteOutcome:
        n = int(self.rng.integers(0, 37))  # 0..36
        if n == 0:
            color = 'green'
        elif n in self.red_numbers:
            color = 'red'
        else:
            color = 'black'
        return RouletteOutcome(number=n, color=color)

    def resolve_bet(self, bet_type: str, amount: float, chosen_number:  Optional[int] = None) -> float:
        out = self.spin()
        # returns net profit (can be negative)
        if bet_type == 'red_black':
            # Even-money bet on red (we fix to 'red' for MVP). Payout 1:1, zero loses.
            if out.color == 'red':
                return amount  # win +amount
            else:
                return -amount
        elif bet_type == 'straight':
            # Straight-up number bet. Payout 35:1 (net +35*amount on hit)
            if chosen_number is None:
                # Pick a random number each time if not provided
                chosen_number = int(self.rng.integers(0, 37))
            if out.number == chosen_number:
                return 35.0 * amount
            else:
                return -amount
        else:
            raise ValueError(f"Unknown bet_type {bet_type}")

