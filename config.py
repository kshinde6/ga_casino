
# ===================== config.py =====================
from dataclasses import dataclass

@dataclass
class SimConfig:
    seed: int = 42
    width: int = 40
    height: int = 24

    n_agents: int = 200
    generations: int = 30
    ticks_per_gen: int = 500 # 1200

    starting_bankroll: float = 100.0
    metabolic_cost: float = 0.1  # per tick, discourages idling

    # Roulette table
    table_x: int = 10
    table_y: int = 12
    table_radius: int = 0
    table_min_bet: float = 1.0
    table_max_bet: float = 20.0
    max_bet_frac_of_bankroll: float = 0.25

    # Blackjack table
    bj_table_x: int = 40
    bj_table_y: int = 24
    bj_table_radius: int = 30
    bj_min_bet: float = 1.0
    bj_max_bet: float = 20.0

    selection_keep_frac: float = 0.5
    mutation_std: float = 0.05  # gaussian noise on genome
    table_min_bet: float = 1.0
    table_max_bet: float = 20.0
    max_bet_frac_of_bankroll: float = 0.25
