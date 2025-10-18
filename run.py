
# ===================== run.py =====================
from typing import List
import numpy as np

from config import SimConfig
from env.grid import GridWorld, Position
from env.roulette import RouletteTable
from agents.gambler import Gambler
from evolution.ga import Evolution


def run_generation(cfg: SimConfig, rng: np.random.Generator, gamblers: List[Gambler]) -> List[Gambler]:
    grid = GridWorld(cfg.width, cfg.height, rng)
    table = RouletteTable(rng)
    table_pos = Position(cfg.table_x, cfg.table_y)

    # Reset everyone
    for g in gamblers:
        g.reset_for_generation(pos=grid.random_position(), bankroll=cfg.starting_bankroll)

    for t in range(cfg.ticks_per_gen):
        for g in gamblers:
            # move
            g.pos = grid.step_random_move(g.pos)

            # check table radius
            in_radius = grid.in_radius(g.pos, table_pos, cfg.table_radius)
            profit = 0.0
            if in_radius and g.bankroll >= cfg.table_min_bet:
                decision = g.act(
                    in_table_radius=True,
                    min_bet=cfg.table_min_bet,
                    max_bet=min(cfg.table_max_bet, cfg.max_bet_frac_of_bankroll * g.bankroll),
                )
                bet_type = decision["bet_type"]
                amt = decision["bet_amount"]
                if amt > 0:
                    if bet_type == 'straight':
                        # pick a deterministic favorite number per agent for MVP (g.id % 37)
                        chosen_num = int(g.id % 37)
                        profit = table.resolve_bet('straight', amt, chosen_number=chosen_num)
                    else:
                        profit = table.resolve_bet('red_black', amt)
            # metabolic cost applied every tick
            g.update_after_bet(profit=profit, metabolic_cost=cfg.metabolic_cost)

    return gamblers


def main():
    cfg = SimConfig()
    rng = np.random.default_rng(cfg.seed)

    # init population
    gamblers: List[Gambler] = []
    for i in range(cfg.n_agents):
        genome_dim = 10
        genome = rng.normal(0, 0.5, size=genome_dim).astype(np.float32)
        gamblers.append(
            Gambler(
                id=i,
                pos=Position(0, 0),
                bankroll=cfg.starting_bankroll,
                rng=rng,
                genome_dim=genome_dim,
                genome=genome,
            )
        )

    evo = Evolution(keep_frac=cfg.selection_keep_frac, mutation_std=cfg.mutation_std)

    for gen in range(cfg.generations):
        gamblers = run_generation(cfg, rng, gamblers)
        # Evaluate + evolve
        gamblers = evo.select_and_reproduce(gamblers, rng)

        # logging summary
        bank_end = np.array([g.bankroll for g in gamblers], dtype=float)
        mean_bank = float(np.mean(bank_end))
        med_bank = float(np.median(bank_end))
        bankrupt = int(np.sum(bank_end <= 0))
        print(f"Gen {gen+1:02d}: mean={mean_bank:6.2f}  median={med_bank:6.2f}  bankrupt={bankrupt:4d}/{len(gamblers)}")

    # TODO: save trajectories for visualization later (positions heatmap, etc.)
    # You can pickle per-tick positions, or aggregate counts per cell per generation.

if __name__ == "__main__":
    main()
