
# ===================== agents/gambler.py =====================
from dataclasses import dataclass, field
import numpy as np
from typing import List, Dict
from env.grid import Position

@dataclass
class StepLog:
    bankroll: float
    bet: float
    profit: float

@dataclass
class Gambler:
    id: int
    pos: Position
    bankroll: float
    rng: np.random.Generator
    genome_dim: int = 10
    genome: np.ndarray = field(default_factory=lambda: np.zeros(10, dtype=np.float32))
    history: List[StepLog] = field(default_factory=list)

    # recent signals
    recent_profit_ema: float = 0.0

    def clone_with_mutation(self, new_id: int, mutation_std: float) -> "Gambler":
        child = Gambler(
            id=new_id,
            pos=self.pos,
            bankroll=self.bankroll,
            rng=self.rng,
            genome_dim=self.genome_dim,
            genome=self.genome.copy(),
        )
        child.genome += self.rng.normal(0.0, mutation_std, size=self.genome_dim).astype(np.float32)
        return child

    def reset_for_generation(self, pos: Position, bankroll: float):
        self.pos = pos
        self.bankroll = bankroll
        self.history = []
        self.recent_profit_ema = 0.0

    # ---------- Policy ----------
    def _features(self, in_table_radius: bool, min_bet: float, max_bet: float) -> np.ndarray:
        # very small feature vector for MVP
        b = self.bankroll
        feat = np.array([
            1.0,                       # bias
            np.tanh((b - 100.0) / 50), # bankroll centered
            1.0 if in_table_radius else 0.0,
            np.tanh(self.recent_profit_ema / max(min_bet, 1e-6)),
            (min_bet / max(1.0, b)),
            (max_bet / max(1.0, b)),
            self.rng.standard_normal(),  # exploration noise
            0.0, 0.0, 0.0               # padding to 10 dims
        ], dtype=np.float32)
        if feat.shape[0] < self.genome_dim:
            feat = np.pad(feat, (0, self.genome_dim - feat.shape[0]))
        return feat[: self.genome_dim]

    def act(self, in_table_radius: bool, min_bet: float, max_bet: float) -> Dict:
        x = self._features(in_table_radius, min_bet, max_bet)
        w = self.genome
        # Linear heads for (bet_type, bet_fraction)
        # Split genome into two heads of equal size
        h = x  # identity (MVP). Could add hidden layer later.
        mid = len(w) // 2
        logits_type = float(np.dot(h[:mid], w[:mid]))
        cont_raw = float(np.dot(h[mid:], w[mid:]))

        bet_type = 'red_black' if logits_type < 0 else 'straight'
        # map cont_raw -> (0,1) via sigmoid, then to bet fraction
        frac = 1.0 / (1.0 + np.exp(-cont_raw))
        # discretize to curb tiny bets
        frac = max(0.05, min(1.0, frac))
        # respect table limits and bankroll
        bet_amt = frac * self.bankroll
        bet_amt = max(min_bet, min(bet_amt, max_bet, self.bankroll))
        return {"bet_type": bet_type, "bet_amount": bet_amt}

    def update_after_bet(self, profit: float, metabolic_cost: float):
        self.bankroll += profit
        self.bankroll -= metabolic_cost
        # EMA for mild momentum signal
        self.recent_profit_ema = 0.9 * self.recent_profit_ema + 0.1 * profit
        self.history.append(StepLog(self.bankroll, self.history[-1].bet if self.history else 0.0, profit))

