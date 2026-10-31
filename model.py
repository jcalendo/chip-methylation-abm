import math

import mesa
import numpy as np

from agents import Cell, Clone
from metrics import (
    mean_methylation,
    population_meth_mean,
    population_meth_var,
    population_jsd,
)


def _check_prob(name, value):
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} ({value}) must be in [0, 1]")


def assert_population_ok(model):
    """
    Check the population bookkeeping invariants. After the growth phase the population size must
    equal the target, every agent id must be unique, and Cells + Clones must account for everyone.
    """
    agents = model.agents.to_list()
    ids = [a.unique_id for a in agents]
    assert len(ids) == len(set(ids)), "duplicate agent ids"
    n_cells = sum(isinstance(a, Cell) for a in agents)
    n_clones = sum(isinstance(a, Clone) for a in agents)
    assert n_cells + n_clones == len(agents), "agents that are neither Cell nor Clone"
    if not model.is_growing:
        assert len(agents) == model.target_population, (
            f"population size {len(agents)} != target {model.target_population}"
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
        debug=False,
        rng=None,
    ):
        super().__init__(rng=rng)

        for name, value in [
            ("p_meth", p_meth),
            ("p_unmeth", p_unmeth),
            ("init_meth", init_meth),
            ("p_meth_clone", p_meth_clone),
            ("p_unmeth_clone", p_unmeth_clone),
            ("p_duplicate", p_duplicate),
            ("p_die", p_die),
            ("prop_chip", prop_chip),
        ]:
            _check_prob(name, value)
        if p_duplicate + p_die > 1.0:
            raise ValueError(
                f"p_duplicate ({p_duplicate}) + p_die ({p_die}) must be <= 1.0"
            )
        if n_agents < 1:
            raise ValueError(f"n_agents ({n_agents}) must be >= 1")

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
        self.debug = debug

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
            cpgs=self._founder_pattern(),
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

    def _founder_pattern(self):
        """
        Initial CpG state of the founder: exactly round(init_meth * total) methylated sites in random
        positions. No random numbers are drawn when init_meth == 0.
        """
        shape = (self.n_genes, self.n_cpgs)
        if self.init_meth == 0:
            return np.zeros(shape, dtype=int)
        total_cpgs = self.n_genes * self.n_cpgs
        num_ones = int(round(self.init_meth * total_cpgs))
        arr = np.concatenate((np.ones(num_ones), np.zeros(total_cpgs - num_ones)))
        self.rng.shuffle(arr)
        return arr.reshape(shape).astype(int)

    def end_growth_phase(self):
        """Prunes the population back down to a fixed number of agents"""
        current_cells = self.agents.select(agent_type=Cell)
        excess = len(current_cells) - self.target_population

        if excess > 0:
            to_remove = self.random.sample(list(current_cells), excess)
            for cell in to_remove:
                cell.remove()

        self.is_growing = False

    def _n_clones_at_onset(self, n_cells):
        """Number of Cells converted at CHIP onset: prop_chip * n_cells, at least 1 if prop_chip > 0."""
        n_clones = int(self.prop_chip * n_cells)
        if n_clones == 0 and self.prop_chip > 0:
            n_clones = 1
        return n_clones

    def _replace_with_clones(self, targets, clone_template):
        """Replace each target Cell with a Clone carrying a copy of the template's CpGs."""
        for target in targets:
            Clone(
                self,
                self.n_genes,
                self.n_cpgs,
                self.p_meth_clone,
                self.p_unmeth_clone,
                self.p_duplicate,
                self.p_die,
                cpgs=clone_template.cpgs.copy(),
            )
            target.remove()

    def trigger_chip_random(self):
        """
        Selects a random Cell and then copies it N times into a new Clonal population until
        the desired proportion of Clones is created.
        """
        cell_agents = self.agents.select(agent_type=Cell)
        n_clones = self._n_clones_at_onset(len(cell_agents))
        if n_clones == 0:
            return

        targets = self.random.sample(list(cell_agents), n_clones)
        self._replace_with_clones(targets, clone_template=targets[0])

    def trigger_chip_highest_meth(self):
        """
        Selects the Cell with the greatest mean methylation and then copies it N times into a new
        Clonal population until the desired proportion of Clones is created.
        """
        cell_agents = self.agents.select(agent_type=Cell)
        n_clones = self._n_clones_at_onset(len(cell_agents))
        if n_clones == 0:
            return

        targets = self.random.sample(list(cell_agents), n_clones)
        sorted_cells = cell_agents.sort(key=mean_methylation, ascending=False)
        self._replace_with_clones(targets, clone_template=sorted_cells[0])

    def step(self):
        """Advances the simulation by one step."""
        self.datacollector.collect(self)

        if self.is_growing:
            self.agents.shuffle_do("duplicate")
            self.agents.shuffle_do("methylate")
        else:
            self.agents.shuffle_do("methylate")
            self.agents.select(agent_type=Clone).shuffle_do("divide")

        if self.debug:
            assert_population_ok(self)
