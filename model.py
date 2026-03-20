import math

import mesa

from agents import Cell, Clone
from metrics import (
    mean_methylation,
    population_meth_mean,
    population_meth_var,
    population_jsd,
)


class AgingModel(mesa.Model):
    def __init__(
        self,
        n_agents,
        n_genes,
        n_cpgs,
        p_meth,
        p_unmeth,
        init_meth,
        p_meth_clone,
        p_unmeth_clone,
        chip_time,
        prop_chip,
        p_duplicate,
        p_die,
        rng=None,
    ):
        super().__init__(rng=rng)

        self.n_genes = n_genes
        self.n_cpgs = n_cpgs
        self.p_meth = p_meth
        self.p_unmeth = p_unmeth
        self.init_meth = init_meth
        self.p_meth_clone = p_meth_clone
        self.p_unmeth_clone = p_unmeth_clone
        self.p_duplicate = p_duplicate
        self.p_die = p_die
        self.chip_time = chip_time
        self.target_population = n_agents
        self.prop_chip = prop_chip

        # Compute growth phase steps needed to reach target population size
        self.growth_steps = math.ceil(math.log2(self.target_population))
        self.is_growing = True

        # Ensure CHIP doesn't trigger while still developing
        if chip_time <= self.growth_steps:
            raise ValueError(
                f"chip_time ({chip_time}) must be > growth_steps ({self.growth_steps})"
            )

        # Founder cell
        Cell(
            model=self,
            n_genes=n_genes,
            n_cpgs=n_cpgs,
            p_meth=p_meth,
            p_unmeth=p_unmeth,
            init_meth=init_meth,
        )

        self.schedule_event(self.end_growth_phase, at=self.growth_steps)
        self.schedule_event(self.trigger_chip_random, at=chip_time)

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Cell_Count": lambda m: len(m.agents.select(agent_type=Cell)),
                "Clone_Count": lambda m: len(m.agents.select(agent_type=Clone)),
                "Overall_Mean_Methylation": lambda m: population_meth_mean(m.agents),
                "Cell_Mean_Methylation": lambda m: population_meth_mean(
                    m.agents.select(agent_type=Cell)
                ),
                "Clone_Mean_Methylation": lambda m: population_meth_mean(
                    m.agents.select(agent_type=Clone)
                ),
                "Overall_Methylation_Var": lambda m: population_meth_var(m.agents),
                "Cell_Methylation_Var": lambda m: population_meth_var(
                    m.agents.select(agent_type=Cell)
                ),
                "Clone_Methylation_Var": lambda m: population_meth_var(
                    m.agents.select(agent_type=Clone)
                ),
                "Overall_Mean_JSD": lambda m: population_jsd(m.agents),
                "Cell_Mean_JSD": lambda m: population_jsd(
                    m.agents.select(agent_type=Cell)
                ),
                "Clone_Mean_JSD": lambda m: population_jsd(
                    m.agents.select(agent_type=Clone)
                ),
            }
        )

    def end_growth_phase(self):
        """Prunes the population back down to a fixed number of agents"""
        current_cells = self.agents.select(agent_type=Cell)
        excess = len(current_cells) - self.target_population

        if excess > 0:
            to_remove = self.random.sample(list(current_cells), excess)
            for cell in to_remove:
                cell.remove()

        self.is_growing = False

    def trigger_chip_random(self):
        """
        Selects a random Cell and then copies it N times into a new CLonal population until 
        the desired proportion of Cloness is created.
        """
        cell_agents = self.agents.select(agent_type=Cell)
        n_cells = len(cell_agents)
        n_clones = int(self.prop_chip * n_cells)

        if n_clones == 0 and self.prop_chip > 0:
            n_clones = 1
    
        targets = self.random.sample(list(cell_agents), n_clones)
        clone_template = targets[0]

        for target in targets:
            new_clone = Clone(
                self,
                self.n_genes,
                self.n_cpgs,
                self.p_meth_clone,
                self.p_unmeth_clone,
                self.p_duplicate,
                self.p_die,
            )
            new_clone.cpgs = clone_template.cpgs.copy()
            target.remove()

    def trigger_chip_highest_meth(self):
        """
        Selects the Cell with the greatest mean methylation and then copies it N times into a new 
        CLonal population until the desired proportion of Cloness is created.
        """
        cell_agents = self.agents.select(agent_type=Cell)
        n_cells = len(cell_agents)
        n_clones = int(self.prop_chip * n_cells)

        if n_clones == 0 and self.prop_chip > 0:
            n_clones = 1
    
        targets = self.random.sample(list(cell_agents), n_clones)   
        sorted_cells = cell_agents.sort(key=mean_methylation, ascending=False)
        clone_template = sorted_cells[0]

        for target in targets:
            new_clone = Clone(
                self,
                self.n_genes,
                self.n_cpgs,
                self.p_meth_clone,
                self.p_unmeth_clone,
                self.p_duplicate,
                self.p_die,
            )
            new_clone.cpgs = clone_template.cpgs.copy()
            target.remove()

    def step(self):
        """Advances the simulation by one step."""
        self.datacollector.collect(self)

        if self.is_growing:
            self.agents.shuffle_do("duplicate")
            self.agents.shuffle_do("methylate")
        else:
            self.agents.shuffle_do("methylate")
            self.agents.select(agent_type=Clone).shuffle_do("divide")
