# ===================== evolution/ga.py =====================
import numpy as np
from typing import List, Tuple
from agents.gambler import Gambler

class Evolution:
    def __init__(self, keep_frac: float, mutation_std: float):
        self.keep_frac = keep_frac
        self.mutation_std = mutation_std

    @staticmethod
    def risk_adjusted_fitness(bankroll_series: List[float]) -> float:
        if len(bankroll_series) < 2:
            return -1e9
        returns = np.diff(bankroll_series)
        mean = np.mean(returns)
        vol = np.std(returns) + 1e-6
        return float(mean / vol)

    def select_and_reproduce(self, gamblers: List[Gambler], rng: np.random.Generator) -> List[Gambler]:
        # Compute fitness
        fitness = []
        for g in gamblers:
            series = [log.bankroll for log in g.history]
            f = self.risk_adjusted_fitness(series) if series else -1e9
            # bankruptcy harsh penalty
            if g.bankroll <= 0:
                f -= 5.0
            fitness.append(f)
        idx_sorted = np.argsort(fitness)[::-1]
        keep_n = max(2, int(len(gamblers) * self.keep_frac))
        keep_ids = idx_sorted[:keep_n]

        survivors = [gamblers[i] for i in keep_ids]
        # refill
        new_list: List[Gambler] = []
        gid = 0
        while len(new_list) < len(gamblers):
            parent = survivors[rng.integers(0, len(survivors))]
            child = parent.clone_with_mutation(new_id=gid, mutation_std=self.mutation_std)
            new_list.append(child)
            gid += 1
        return new_list
