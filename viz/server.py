# ===================== viz/server.py =====================
from mesa.visualization import CanvasGrid, ChartModule, ModularServer
from mesa.visualization.modules import TextElement
from viz.model_mesa import CasinoModel, GamblerAgent, dist2
from config import SimConfig


def agent_portrayal(agent):
    if isinstance(agent, GamblerAgent):
        return {"Shape": "circle", "Filled": "true", "r": 0.6, "Color": "#1f77b4", "Layer": 1}
    else:  # TableMarker
        color = "#e0e0e0"
        if getattr(agent, 'kind', 'roulette') == 'blackjack':
            color = "#f2e8c9"  # light tan for BJ
        return {"Shape": "rect", "Filled": "true", "Layer": 0, "w": 1, "h": 1, "Color": color}


class GenReadout(TextElement):
    def render(self, model):
        return f"Generation: {model.generation} | Tick: {model.tick} | Pop: {len(model._g())}"


class AgentLogPanel(TextElement):
    """Shows top-K agents by bankroll with recent actions.
    Clickless, always-updated sidebar.
    """
    def __init__(self, k: int = 10):
        self.k = k
    def render(self, model):
        agents = sorted(model._g(), key=lambda a: a.bankroll, reverse=True)[: self.k]
        lines = ["<b>Top agents</b> (id | bankroll | in_radius | last actions)"]
        for ag in agents:
            inrad = dist2(ag.pos, model.table_center) <= model.cfg.table_radius ** 2
            # last 5 actions compact: (t, R/Y, type, amt, profit)
            tail = ag.history[-5:]
            tail_s = ", ".join(
                f"t{t}:{'R' if ir else '-'}:{bt}:{amt:.1f}:{p:+.1f}" for (t, ir, bt, amt, p, bk) in tail
            ) if tail else "-"
            lines.append(f"{ag.unique_id:3d} | {ag.bankroll:7.2f} | {'Y' if inrad else 'N'} | {tail_s}")
        return "<br>".join(lines)


def main():
    cfg = SimConfig()
    grid_viz = CanvasGrid(agent_portrayal, cfg.width, cfg.height, 800, 480)
    chart_bank = ChartModule([{ "Label": "MeanBankroll", "Color": "#1f77b4" }], data_collector_name='datacollector')
    # chart_bankrupt = ChartModule([{ "Label": "BankruptPct", "Color": "#d62728" }], data_collector_name='datacollector')
    chart_inradius = ChartModule([{ "Label": "InRadiusPct", "Color": "#2ca02c" }], data_collector_name='datacollector')
    chart_pop = ChartModule([{ "Label": "Population", "Color": "#9467bd" }], data_collector_name='datacollector')
    gen_text = GenReadout()
    agent_panel = AgentLogPanel(k=10)
    server = ModularServer(
        CasinoModel,
        [gen_text, grid_viz, chart_bank, chart_inradius, chart_pop, agent_panel],
        "GA Casino – Roulette (Mesa with GA & Agent Panel)",
        {"cfg": cfg}
    )
    server.port = 8521
    server.launch()

if __name__ == "__main__":
    main()
