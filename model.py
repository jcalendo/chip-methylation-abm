import mesa
import numpy as np
from agents import Cell, Clone
from metrics import (
    mean_methylation,
    jsd_unmeth,
    mean_shannon,
    beta_var,
    gini,
    population_jsd,
    subpopulation_jsd,
    population_jsd_unmeth,
)


class AgingModel(mesa.Model):
    def __init__(
        self,
        n_agents,
        n_genes,
        n_cpgs,
        p_meth,
        p_unmeth,
        p_meth_clone,
        p_unmeth_clone,
        chip_time,
        p_duplicate,
        p_die,
    ):
        super().__init__()

        self.n_genes = n_genes
        self.n_cpgs = n_cpgs
        self.p_meth = p_meth
        self.p_unmeth = p_unmeth
        self.p_meth_clone = p_meth_clone
        self.p_unmeth_clone = p_unmeth_clone
        self.p_duplicate = p_duplicate
        self.p_die = p_die
        self.chip_time = chip_time

        Cell.create_agents(
            model=self,
            n=n_agents,
            n_genes=n_genes,
            n_cpgs=n_cpgs,
            p_meth=p_meth,
            p_unmeth=p_unmeth,
        )

        self.schedule_event(self.trigger_chip_highest_meth, at=chip_time)

        self.datacollector = mesa.DataCollector(
            agent_reporters={
                "Agent_Type": lambda a: type(a).__name__,
                "Mean_Methylation": mean_methylation,
                "JSD_Unmeth": jsd_unmeth,
                "Shannon": mean_shannon,
                "Beta_Variance": beta_var,
                "Gini": gini,
            },
            model_reporters={
                "Cell_Count": lambda m: len(m.agents.select(agent_type=Cell)),
                "Clone_Count": lambda m: len(m.agents.select(agent_type=Clone)),
                "Overall_Mean_Methylation": lambda m: (
                    np.mean(m.agents.get("cpgs")) if len(m.agents) > 0 else np.nan
                ),
                "Cell_Mean_Methylation": lambda m: (
                    np.mean(m.agents.select(agent_type=Cell).get("cpgs"))
                    if len(m.agents.select(agent_type=Cell)) > 0
                    else np.nan
                ),
                "Clone_Mean_Methylation": lambda m: (
                    np.mean(m.agents.select(agent_type=Clone).get("cpgs"))
                    if len(m.agents.select(agent_type=Clone)) > 0
                    else np.nan
                ),
                "Overall_Methylation_Variance": lambda m: (
                    np.var([mean_methylation(a) for a in m.agents], ddof=1)
                    if len(m.agents) > 1
                    else np.nan
                ),
                "Cell_Methylation_Variance": lambda m: (
                    np.var(
                        [mean_methylation(a) for a in m.agents.select(agent_type=Cell)],
                        ddof=1,
                    )
                    if len(m.agents.select(agent_type=Cell)) > 1
                    else np.nan
                ),
                "Clone_Methylation_Variance": lambda m: (
                    np.var(
                        [
                            mean_methylation(a)
                            for a in m.agents.select(agent_type=Clone)
                        ],
                        ddof=1,
                    )
                    if len(m.agents.select(agent_type=Clone)) > 1
                    else np.nan
                ),
                "Overall_Population_JSD": population_jsd,
                "Cell_Population_JSD": lambda m: subpopulation_jsd(
                    m.agents.select(agent_type=Cell)
                ),
                "Clone_Population_JSD": lambda m: subpopulation_jsd(
                    m.agents.select(agent_type=Clone)
                ),
                "Overall_Drift_JSD": lambda m: population_jsd_unmeth(m.agents),
                "Cell_Drift_JSD": lambda m: population_jsd_unmeth(
                    m.agents.select(agent_type=Cell)
                ),
                "Clone_Drift_JSD": lambda m: population_jsd_unmeth(
                    m.agents.select(agent_type=Clone)
                ),
            },
        )

    def trigger_chip_random(self):
        """Selects a random Cell, copies its state to a new Clone, and removes the Cell."""
        cell_agents = self.agents.select(agent_type=Cell)

        if cell_agents and len(cell_agents) > 0:
            target_cell = self.random.choice(cell_agents)
            new_clone = Clone(
                self,
                self.n_genes,
                self.n_cpgs,
                self.p_meth_clone,
                self.p_unmeth_clone,
                self.p_duplicate,
                self.p_die,
            )
            new_clone.cpgs = target_cell.cpgs.copy()
            target_cell.remove()

    def trigger_chip_highest_meth(self):
        """Select the highest methylated Cell as the new Clone"""
        cell_agents = self.agents.select(agent_type=Cell)

        if cell_agents and len(cell_agents) > 0:
            sorted_cells = cell_agents.sort(key=mean_methylation, ascending=False)
            target_cell = sorted_cells[0]
            new_clone = Clone(
                self,
                self.n_genes,
                self.n_cpgs,
                self.p_meth_clone,
                self.p_unmeth_clone,
                self.p_duplicate,
                self.p_die,
            )
            new_clone.cpgs = target_cell.cpgs.copy()
            target_cell.remove()

    def step(self):
        """Standard step executed every time unit."""
        self.datacollector.collect(self)
        self.agents.shuffle_do("methylate")

        if Clone in self.agents_by_type:
            self.agents_by_type[Clone].shuffle_do("divide")
