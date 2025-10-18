# ===================== viz/model_mesa.py =====================
from __future__ import annotations
from typing import Tuple, List, Optional
import numpy as np
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

from config import SimConfig
from env.roulette import RouletteTable


def dist2(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


class TableMarker(Agent):
    """Static agent used to tint cells inside the table radius."""
    def step(self):
        pass


class GamblerAgent(Agent):
    def __init__(
        self,
        unique_id,
        model: 'CasinoModel',
        rng: np.random.Generator,
        genome: Optional[np.ndarray] = None,
        genome_dim: int = 10,
    ):
        super().__init__(unique_id, model)
        self.rng = rng
        self.bankroll = model.cfg.starting_bankroll
        self.recent_profit_ema = 0.0
        self.genome_dim = genome_dim
        self.genome = genome if genome is not None else self.rng.normal(0, 0.5, size=self.genome_dim).astype(np.float32)
        self.gen_series: List[float] = []
        self.time_in_radius = 0
        self.dead = False
        # keep a short rolling action history for inspection: (tick, in_radius, bet_type, amount, profit)
        self.history: List[tuple] = []
        self.history_maxlen = 12

    # -------- policy --------
    def _features(self, in_table_radius: bool, min_bet: float, max_bet: float) -> np.ndarray:
        b = self.bankroll
        feat = np.array([
            1.0,
            np.tanh((b - 100.0) / 50),
            1.0 if in_table_radius else 0.0,
            np.tanh(self.recent_profit_ema / max(min_bet, 1e-6)),
            (min_bet / max(1.0, b)),
            (max_bet / max(1.0, b)),
            self.rng.standard_normal(),
            0.0, 0.0, 0.0
        ], dtype=np.float32)
        if feat.shape[0] < self.genome_dim:
            feat = np.pad(feat, (0, self.genome_dim - feat.shape[0]))
        return feat[: self.genome_dim]

    def act(self, in_table_radius: bool, min_bet: float, max_bet: float):
        x = self._features(in_table_radius, min_bet, max_bet)
        w = self.genome
        mid = len(w) // 2
        logits_type = float(np.dot(x[:mid], w[:mid]))
        cont_raw = float(np.dot(x[mid:], w[mid:]))
        bet_type = 'red_black' if logits_type < 0 else 'straight'
        frac = 1.0 / (1.0 + np.exp(-cont_raw))
        frac = max(0.05, min(1.0, frac))
        bet_amt = frac * self.bankroll
        bet_amt = max(min_bet, min(bet_amt, max_bet, self.bankroll))
        return {"bet_type": bet_type, "bet_amount": bet_amt}

    # movement: allow attraction (+) or repulsion (-) to table
    def _movement_step(self, x: int, y: int) -> tuple[int, int]:
        # --- Geometry helpers (torus-aware dx, dy to center) ---
        W, H = self.model.grid.width, self.model.grid.height
        tx, ty = self.model.table_center
        # torus-aware shortest vector from (x,y) to center
        def torus_delta(a, b, size):
            d = b - a
            if d > size // 2:  d -= size
            if d < -size // 2: d += size
            return d
        dx_c = torus_delta(x, tx, W)
        dy_c = torus_delta(y, ty, H)
        dist2_to_center = dx_c*dx_c + dy_c*dy_c
        R = self.model.cfg.table_radius

        # --- Tapered bias band around the radius ---
        # Only bias movement when within [R - inner_margin, R + outer_margin].
        inner_margin = 2   # you can tune
        outer_margin = 4   # you can tune
        R2 = R*R

        # Signed distance^2 from radius^2 (negative = inside radius)
        delta2 = dist2_to_center - R2

        # Compute a 0..1 weight for how much to apply bias
        #  - inside radius to R+outer_margin: linearly taper down
        #  - from R-inner_margin inward: cap at 1
        #  - beyond R+outer_margin: 0 (pure random walk)
        def clamp01(z): return 0.0 if z < 0 else (1.0 if z > 1.0 else z)
        # Approximate a band using distance (no sqrt): compare dist to (R +/- margin)^2
        band_inner2 = max(0, (R - inner_margin) * (R - inner_margin))
        band_outer2 = (R + outer_margin) * (R + outer_margin)
        if dist2_to_center <= band_inner2:
            bias_weight = 1.0
        elif dist2_to_center >= band_outer2:
            bias_weight = 0.0
        else:
            # linear taper between inner and outer boundary
            bias_weight = 1.0 - (dist2_to_center - band_inner2) / (band_outer2 - band_inner2)
            bias_weight = clamp01(bias_weight)

        # Movement gene in [-1,1]
        bias = float(np.tanh(self.genome[0]))
        # Probability to follow the preferred direction scaled by band weight
        p_follow = abs(bias) * bias_weight
        toward = (bias >= 0.0)

        # Preferred unit step toward/away center (sign only)
        pref_dx = 0 if dx_c == 0 else (1 if dx_c > 0 else -1)
        pref_dy = 0 if dy_c == 0 else (1 if dy_c > 0 else -1)
        if not toward:
            pref_dx, pref_dy = -pref_dx, -pref_dy

        def sample_axis(pref):
            if self.rng.random() < p_follow and pref != 0:
                return pref
            return int(self.rng.integers(-1, 2))

        sx = sample_axis(pref_dx)
        sy = sample_axis(pref_dy)
        if sx == 0 and sy == 0:
            # ensure motion
            sx = pref_dx if pref_dx != 0 else int(self.rng.choice([-1, 1]))
        nx = (x + sx) % W
        ny = (y + sy) % H
        return nx, ny


    def step(self):
        if self.dead:
            return
        x, y = self.pos
        nx, ny = self._movement_step(x, y)
        self.model.grid.move_agent(self, (nx, ny))

        in_radius = dist2((nx, ny), self.model.table_center) <= self.model.cfg.table_radius ** 2
        if in_radius:
            self.time_in_radius += 1
        profit = 0.0
        bet_type = None
        bet_amt = 0.0
        if in_radius and self.bankroll > 0 and self.bankroll >= self.model.cfg.table_min_bet:
            decision = self.act(True, self.model.cfg.table_min_bet,
                                min(self.model.cfg.table_max_bet,
                                    self.model.cfg.max_bet_frac_of_bankroll * self.bankroll))
            bet_type = decision["bet_type"]
            bet_amt = decision["bet_amount"]
            if bet_type == 'straight':
                chosen_num = int(self.unique_id % 37)
                profit = self.model.table.resolve_bet('straight', bet_amt, chosen_number=chosen_num)
            else:
                profit = self.model.table.resolve_bet('red_black', bet_amt)
        self.bankroll += profit
        self.bankroll -= self.model.cfg.metabolic_cost
        self.recent_profit_ema = 0.9 * self.recent_profit_ema + 0.1 * profit
        self.gen_series.append(self.bankroll)
        # push to short action history
        # NOTE: -------------------------------------------
        # Updated version will show bet ('red', 'black', 'green' instead of bet_type)
        # self.history.append((self.model.tick, in_radius, bet_type or '-', float(bet_amt), float(profit), float(self.bankroll)))
        self.history.append((self.model.tick, in_radius, bet_type or '-', float(bet_amt), float(profit), float(self.bankroll)))
        if len(self.history) > self.history_maxlen:
            self.history.pop(0)

        if self.bankroll <= 0 and not self.dead:
            self.dead = True
            self.model.to_remove.append(self)


class CasinoModel(Model):
    def __init__(self, cfg: SimConfig):
        super().__init__()
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.grid = MultiGrid(cfg.width, cfg.height, torus=True)
        self.schedule = RandomActivation(self)
        self.table = RouletteTable(self.rng)
        self.table_center = (cfg.table_x, cfg.table_y)

        self.tick = 0
        self.generation = 1
        self.to_remove: List[GamblerAgent] = []

        # table overlay
        uid = 0
        for x in range(cfg.width):
            for y in range(cfg.height):
                if dist2((x, y), self.table_center) <= cfg.table_radius ** 2:
                    m = TableMarker(f"table-{uid}", self)
                    self.grid.place_agent(m, (x, y))
                    uid += 1

        # agents
        for i in range(cfg.n_agents):
            genome = self.rng.normal(0, 0.5, size=10).astype(np.float32)
            a = GamblerAgent(i, self, self.rng, genome=genome)
            rx = int(self.rng.integers(0, cfg.width))
            ry = int(self.rng.integers(0, cfg.height))
            self.grid.place_agent(a, (rx, ry))
            self.schedule.add(a)

        self.datacollector = DataCollector(model_reporters={
            "MeanBankroll": lambda m: float(np.mean([ag.bankroll for ag in m._g()] or [0.0])),
            "BankruptPct": lambda m: 100.0 * float(np.mean([1.0 if ag.bankroll <= 0 else 0.0 for ag in m._g()] or [0.0])),
            "InRadiusPct": lambda m: 100.0 * float(np.mean([ag.time_in_radius for ag in m._g()] or [0.0]) / max(1, m.cfg.ticks_per_gen)),
            "Population": lambda m: len(m._g()),
            "Generation": lambda m: m.generation
        })

    def _g(self) -> List[GamblerAgent]:
        return [ag for ag in self.schedule.agents if isinstance(ag, GamblerAgent) and not ag.dead]

    @staticmethod
    def _risk_adjusted_fitness(series: List[float]) -> float:
        if len(series) < 2:
            return -1e9
        returns = np.diff(series)
        mean = float(np.mean(returns))
        vol = float(np.std(returns) + 1e-6)
        return mean / vol

    def _evolve_generation(self):
        gamblers = self._g()
        if not gamblers:
            next_genomes = [self.rng.normal(0, 0.5, size=10).astype(np.float32) for _ in range(self.cfg.n_agents)]
        else:
            fitness = [self._risk_adjusted_fitness(ag.gen_series) if ag.gen_series else -1e9 for ag in gamblers]
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

    def step(self):
        self.schedule.step()
        # remove bankrupt agents after all have stepped this tick
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
        self.datacollector.collect(self)
        self.tick += 1
        if self.tick % self.cfg.ticks_per_gen == 0:
            self._evolve_generation()
            self._reset_for_next_generation()
