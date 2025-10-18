# viz/model_mesa.py
from __future__ import annotations
from typing import Tuple, List, Optional

import numpy as np
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

from config import SimConfig
from env.roulette import RouletteTable
from env.blackjack import BlackjackTable


def dist2(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """Squared Euclidean distance (no torus adjustment)."""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


class TableMarker(Agent):
    """
    Static overlay to tint cells that belong to a table's radius.

    kind: 'roulette' or 'blackjack' for coloring in the UI.
    """
    def __init__(self, unique_id, model, kind: str):
        super().__init__(unique_id, model)
        self.kind = kind

    def step(self):
        pass


class GamblerAgent(Agent):
    """
    Evolvable gambler with:
      - genome controlling bet type/size and movement bias
      - bankroll & short rolling action history for the inspector
      - death: removed from sim when bankroll <= 0
    """
    def __init__(
        self,
        unique_id,
        model: "CasinoModel",
        rng: np.random.Generator,
        genome: Optional[np.ndarray] = None,
        genome_dim: int = 10,
    ):
        super().__init__(unique_id, model)
        self.rng = rng
        self.bankroll = model.cfg.starting_bankroll
        self.recent_profit_ema = 0.0
        self.genome_dim = genome_dim
        self.genome = (
            genome
            if genome is not None
            else self.rng.normal(0, 0.5, size=self.genome_dim).astype(np.float32)
        )

        # GA tracking
        self.gen_series: List[float] = []  # bankroll trace within the current generation
        self.time_in_radius = 0            # roulette-time counter this gen (kept for legacy chart)
        self.dead = False

        # Agent inspector: (tick, in_radius, label, amount, profit, post_bankroll)
        self.history: List[tuple] = []
        self.history_maxlen = 12

    # -------- policy (reuse simple linear heads) --------
    def _features(self, in_table_radius: bool, min_bet: float, max_bet: float) -> np.ndarray:
        b = self.bankroll
        feat = np.array(
            [
                1.0,                                         # bias
                np.tanh((b - 100.0) / 50.0),                 # bankroll centered
                1.0 if in_table_radius else 0.0,             # proximity flag
                np.tanh(self.recent_profit_ema / max(min_bet, 1e-6)),
                (min_bet / max(1.0, b)),
                (max_bet / max(1.0, b)),
                self.rng.standard_normal(),                  # exploration noise
                0.0, 0.0, 0.0,                               # padding to 10 dims
            ],
            dtype=np.float32,
        )
        if feat.shape[0] < self.genome_dim:
            feat = np.pad(feat, (0, self.genome_dim - feat.shape[0]))
        return feat[: self.genome_dim]

    def act(self, in_table_radius: bool, min_bet: float, max_bet: float) -> dict:
        """Return {'bet_type': str, 'bet_amount': float} for roulette OR bet sizing for BJ."""
        x = self._features(in_table_radius, min_bet, max_bet)
        w = self.genome
        mid = len(w) // 2
        logits_type = float(np.dot(x[:mid], w[:mid]))
        cont_raw = float(np.dot(x[mid:], w[mid:]))

        bet_type = "red_black" if logits_type < 0 else "straight"
        frac = 1.0 / (1.0 + np.exp(-cont_raw))  # sigmoid
        frac = max(0.05, min(1.0, frac))
        bet_amt = frac * self.bankroll
        bet_amt = max(min_bet, min(bet_amt, max_bet, self.bankroll))
        return {"bet_type": bet_type, "bet_amount": bet_amt}

    # -------- movement (genome-controlled drift; sign = toward/away) --------
    def _movement_step(self, x: int, y: int) -> tuple[int, int]:
        """
        Attraction/repulsion to roulette center:
          bias = tanh(genome[0]) in [-1, 1]
          |bias| = probability to follow the preferred step; sign picks toward/away.
        """
        bias = np.tanh(float(self.genome[0]))  # [-1, 1]
        p_follow = abs(bias)
        toward = bias >= 0.0

        tx, ty = self.model.table_center
        to_dx = 0 if tx == x else (1 if tx > x else -1)
        to_dy = 0 if ty == y else (1 if ty > y else -1)
        away_dx, away_dy = -to_dx, -to_dy

        def sample_axis(pref: int) -> int:
            if self.rng.random() < p_follow and pref != 0:
                return pref
            return int(self.rng.integers(-1, 2))

        pref_dx = to_dx if toward else away_dx
        pref_dy = to_dy if toward else away_dy

        dx = sample_axis(pref_dx)
        dy = sample_axis(pref_dy)
        if dx == 0 and dy == 0:
            if pref_dx != 0:
                dx = pref_dx
            elif pref_dy != 0:
                dy = pref_dy
            else:
                dx = int(self.rng.choice([-1, 1]))

        nx = (x + dx) % self.model.grid.width
        ny = (y + dy) % self.model.grid.height
        return nx, ny

    # -------- one tick --------
    def step(self):
        if self.dead:
            return

        # Move
        x, y = self.pos
        nx, ny = self._movement_step(x, y)
        self.model.grid.move_agent(self, (nx, ny))

        # Check table zones
        in_roulette = dist2((nx, ny), self.model.table_center) <= self.model.cfg.table_radius ** 2
        in_blackjack = dist2((nx, ny), self.model.bj_center) <= self.model.cfg.bj_table_radius ** 2
        if in_roulette:
            self.time_in_radius += 1

        profit = 0.0
        bet_amt = 0.0

        # Priority: Blackjack if overlapping both
        if in_blackjack and self.bankroll > 0 and self.bankroll >= self.model.cfg.bj_min_bet:
            # Size bet using same fraction head
            max_bj = min(self.model.cfg.bj_max_bet, self.model.cfg.max_bet_frac_of_bankroll * self.bankroll)
            decision = self.act(True, self.model.cfg.bj_min_bet, max_bj)
            bet_amt = decision["bet_amount"]

            # Tiny genome-based BJ policy
            def policy_fn(state: dict, legal: set) -> str:
                # Features: [bias, centered total, soft flag, dealer up index, normalized running count]
                ranks = ['2','3','4','5','6','7','8','9','10','J','Q','K','A']
                f = np.array(
                    [
                        1.0,
                        (state["player_total"] - 12) / 10.0,
                        1.0 if state["player_soft"] else -1.0,
                        (ranks.index(state["dealer_up"]) - 6) / 6.0,
                        np.tanh(state["running_count"] / 5.0),
                    ],
                    dtype=np.float32,
                )
                w = self.genome[:5]
                score = float(np.dot(f, w))
                if "split" in legal and score > 1.2:
                    return "split"
                if "double" in legal and 0.6 < score <= 1.2:
                    return "double"
                if score > 0.0 and "hit" in legal:
                    return "hit"
                return "stand"

            outcome = self.model.bj_table.play_hand(self.bankroll, bet_amt, policy_fn)
            profit = outcome.profit

            # Inspector log (two compact lines)
            self.history.append(
                (self.model.tick, True, f"BJ up={outcome.dealer_up}", float(bet_amt), float(profit), float(self.bankroll + profit))
            )
            self.history.append(
                (self.model.tick, True, "BJ:" + ";".join(outcome.action_log), 0.0, 0.0, float(self.bankroll + profit))
            )

        elif in_roulette and self.bankroll > 0 and self.bankroll >= self.model.cfg.table_min_bet:
            decision = self.act(
                True,
                self.model.cfg.table_min_bet,
                min(self.model.cfg.table_max_bet, self.model.cfg.max_bet_frac_of_bankroll * self.bankroll),
            )
            bet_type = decision["bet_type"]
            bet_amt = decision["bet_amount"]
            if bet_type == "straight":
                chosen_num = int(self.unique_id % 37)
                profit = self.model.table.resolve_bet("straight", bet_amt, chosen_number=chosen_num)
            else:
                profit = self.model.table.resolve_bet("red_black", bet_amt)

            self.history.append(
                (self.model.tick, True, bet_type, float(bet_amt), float(profit), float(self.bankroll + profit))
            )
        else:
            # Idle tick
            self.history.append((self.model.tick, False, "-", 0.0, 0.0, float(self.bankroll)))

        # Bankroll update + logs
        self.bankroll += profit
        self.bankroll -= self.model.cfg.metabolic_cost
        self.recent_profit_ema = 0.9 * self.recent_profit_ema + 0.1 * profit
        self.gen_series.append(self.bankroll)
        if len(self.history) > self.history_maxlen:
            self.history.pop(0)

        # Death
        if self.bankroll <= 0 and not self.dead:
            self.dead = True
            self.model.to_remove.append(self)


class CasinoModel(Model):
    """
    Mesa model with GA evolution every `ticks_per_gen`:
      - Select top `selection_keep_frac` (alive only), clone+mutate genomes
      - Remove bankrupt agents during the gen
      - Reset bankrolls/positions after each generation
    """
    def __init__(self, cfg: SimConfig):
        super().__init__()
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        self.grid = MultiGrid(cfg.width, cfg.height, torus=True)
        self.schedule = RandomActivation(self)

        # Games
        self.table = RouletteTable(self.rng)
        self.table_center = (cfg.table_x, cfg.table_y)
        self.bj_table = BlackjackTable(self.rng)
        self.bj_center = (cfg.bj_table_x, cfg.bj_table_y)

        # GA state
        self.tick = 0
        self.generation = 1
        self.to_remove: List[GamblerAgent] = []

        # Overlays for both tables
        uid = 0
        for x in range(cfg.width):
            for y in range(cfg.height):
                if dist2((x, y), self.table_center) <= self.cfg.table_radius ** 2:
                    m = TableMarker(f"roulette-{uid}", self, kind="roulette")
                    self.grid.place_agent(m, (x, y))
                    uid += 1
                if dist2((x, y), self.bj_center) <= self.cfg.bj_table_radius ** 2:
                    m2 = TableMarker(f"blackjack-{uid}", self, kind="blackjack")
                    self.grid.place_agent(m2, (x, y))
                    uid += 1

        # Agents
        for i in range(cfg.n_agents):
            genome = self.rng.normal(0, 0.5, size=10).astype(np.float32)
            a = GamblerAgent(i, self, self.rng, genome=genome)
            rx = int(self.rng.integers(0, cfg.width))
            ry = int(self.rng.integers(0, cfg.height))
            self.grid.place_agent(a, (rx, ry))
            self.schedule.add(a)

        # Metrics
        self.datacollector = DataCollector(
            model_reporters={
                "MeanBankroll": lambda m: float(np.mean([ag.bankroll for ag in m._g()] or [0.0])),
                "BankruptPct": lambda m: 100.0
                * float(np.mean([1.0 if ag.bankroll <= 0 else 0.0 for ag in m._g()] or [0.0])),
                "InRadiusPct": lambda m: 100.0
                * float(np.mean([ag.time_in_radius for ag in m._g()] or [0.0]) / max(1, m.cfg.ticks_per_gen)),
                "Population": lambda m: len(m._g()),
                "Generation": lambda m: m.generation,
            }
        )

    # --- helpers ---
    def _g(self) -> List[GamblerAgent]:
        """Alive gamblers only (skip markers and dead)."""
        return [ag for ag in self.schedule.agents if isinstance(ag, GamblerAgent) and not ag.dead]

    @staticmethod
    def _risk_adjusted_fitness(series: List[float]) -> float:
        if len(series) < 2:
            return -1e9
        returns = np.diff(series)
        mean = float(np.mean(returns))
        vol = float(np.std(returns) + 1e-6)
        return mean / vol

    # --- GA ---
    def _evolve_generation(self):
        gamblers = self._g()
        if not gamblers:
            # Repopulate if everyone died
            next_genomes = [
                self.rng.normal(0, 0.5, size=10).astype(np.float32) for _ in range(self.cfg.n_agents)
            ]
        else:
            fitness = [
                self._risk_adjusted_fitness(ag.gen_series) if ag.gen_series else -1e9 for ag in gamblers
            ]
            idx = np.argsort(fitness)[::-1]
            keep_n = max(2, int(len(gamblers) * self.cfg.selection_keep_frac))
            survivors = [gamblers[i] for i in idx[:keep_n]]

            next_genomes: List[np.ndarray] = []
            while len(next_genomes) < self.cfg.n_agents:
                p = survivors[int(self.rng.integers(0, len(survivors)))]
                child = p.genome.copy()
                child += self.rng.normal(0.0, self.cfg.mutation_std, size=child.shape).astype(np.float32)
                next_genomes.append(child)

        # purge remaining alive agents
        for ag in list(self._g()):
            try:
                self.grid.remove_agent(ag)
            except Exception:
                pass
            try:
                self.schedule.remove(ag)
            except Exception:
                pass

        # spawn new generation
        for i, genome in enumerate(next_genomes):
            ag = GamblerAgent(i, self, self.rng, genome=genome)
            rx = int(self.rng.integers(0, self.grid.width))
            ry = int(self.rng.integers(0, self.grid.height))
            self.grid.place_agent(ag, (rx, ry))
            self.schedule.add(ag)

        self.generation += 1

    def _reset_for_next_generation(self):
        for ag in self._g():
            ag.bankroll = self.cfg.starting_bankroll
            ag.recent_profit_ema = 0.0
            ag.gen_series = []
            ag.time_in_radius = 0
            ag.dead = False
            rx = int(self.rng.integers(0, self.grid.width))
            ry = int(self.rng.integers(0, self.grid.height))
            self.grid.move_agent(ag, (rx, ry))

    # --- main step ---
    def step(self):
        # advance agents
        self.schedule.step()

        # remove newly-dead this tick
        if self.to_remove:
            for ag in self.to_remove:
                try:
                    self.grid.remove_agent(ag)
                except Exception:
                    pass
                try:
                    self.schedule.remove(ag)
                except Exception:
                    pass
            self.to_remove = []

        # collect & advance time
        self.datacollector.collect(self)
        self.tick += 1

        # generation boundary
        if self.tick % self.cfg.ticks_per_gen == 0:
            self._evolve_generation()
            self._reset_for_next_generation()
