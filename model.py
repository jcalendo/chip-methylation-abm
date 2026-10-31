import math

import mesa
import numpy as np

from agents import Cell, Clone
from metrics import (
    bulk_beta,
    mean_methylation,
    population_meth_mean,
    population_meth_var,
    population_jsd,
    population_largest_lineage_fraction,
    population_lineage_count,
)
from rates import RateProfile

INIT_MODES = ("founder", "independent")


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
        p_turnover=0.0,
        rate_profile=None,
        init_mode="founder",
        sample_steps=(),
        debug=False,
        rng=None,
    ):
        """
        Args (beyond the scalar rates):
            p_turnover: per-step probability that each agent (Cell or Clone) divides after the
                growth phase, its daughter replacing a random agent (neutral Moran turnover).
                Clones divide with p_turnover + p_duplicate, so s = p_duplicate - p_die is the
                advantage relative to normal cells.
            rate_profile: None to broadcast the scalar rate arguments to every CpG, or the path of a
                per-CpG rate CSV (see rates.py), in which case the scalar rate arguments are ignored.
                Must be a string, not an array: mesa.batch_run sweeps iterable parameters.
            init_mode: "founder" (the founder's pattern is copied to every cell) or "independent"
                (at the end of growth every cell redraws each site from init_meth).
            sample_steps: steps at which to store per-site bulk beta in `site_snapshots`, keyed by
                the same Step label as the DataCollector rows. Used by runner.py, not batch_run.
            debug: check population invariants at the end of every step.
        """
        super().__init__(rng=rng)

        if init_mode not in INIT_MODES:
            raise ValueError(f"init_mode ({init_mode!r}) must be one of {INIT_MODES}")
        for name, value in [
            ("p_duplicate", p_duplicate),
            ("p_die", p_die),
            ("p_turnover", p_turnover),
            ("prop_chip", prop_chip),
        ]:
            _check_prob(name, value)
        if p_turnover + p_duplicate + p_die > 1.0:
            raise ValueError(
                f"p_turnover ({p_turnover}) + p_duplicate ({p_duplicate}) + p_die ({p_die}) "
                "must be <= 1.0"
            )
        if n_agents < 1:
            raise ValueError(f"n_agents ({n_agents}) must be >= 1")

        if rate_profile is None:
            self.rates = RateProfile.from_scalars(
                n_genes, n_cpgs, p_meth, p_unmeth, p_meth_clone, p_unmeth_clone, init_meth
            )
        else:
            self.rates = RateProfile.from_csv(rate_profile, n_genes, n_cpgs)

        self.n_genes = n_genes
        self.n_cpgs = n_cpgs
        self.init_mode = init_mode
        self.p_duplicate = p_duplicate
        self.p_die = p_die
        self.p_turnover = p_turnover
        self.chip_time = chip_time
        self.target_population = n_agents
        self.prop_chip = prop_chip
        self.debug = debug
        self.sample_steps = frozenset(sample_steps)
        self.site_snapshots = {}
        self.clone_lineage = None  # lineage id shared by the Clones created at CHIP onset
        self.clone_parent_lineage = None  # lineage id of the Cell used as the clone template

        # Compute growth phase steps needed to reach target population size
        self.growth_steps = math.ceil(math.log2(self.target_population))
        self.is_growing = True

        # Ensure CHIP doesn't trigger while still developing
        if chip_time <= self.growth_steps:
            raise ValueError(
                f"chip_time ({chip_time}) must be > growth_steps ({self.growth_steps})"
            )

        # Founder cell
        Cell(self, cpgs=self._founder_pattern())

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
                "Lineage_Count": lambda m: population_lineage_count(m.agents),
                "Largest_Lineage_Fraction": lambda m: population_largest_lineage_fraction(
                    m.agents
                ),
            }
        )

    def _founder_pattern(self):
        """
        Initial CpG state of the founder. With a scalar init_meth the founder has exactly
        round(init_meth * n_sites) methylated sites in random positions (the original behaviour);
        with a per-site init_meth each site is drawn independently. No random numbers are drawn
        when init_meth is zero everywhere.
        """
        rates = self.rates
        if rates.is_zero("init_meth"):
            return np.zeros(rates.shape, dtype=int)
        if rates.init_is_scalar:
            num_ones = int(round(float(rates.init_meth.flat[0]) * rates.n_sites))
            arr = np.concatenate((np.ones(num_ones), np.zeros(rates.n_sites - num_ones)))
            self.rng.shuffle(arr)
            return arr.reshape(rates.shape).astype(int)
        return self._bernoulli_pattern()

    def _bernoulli_pattern(self):
        """Draw each site independently: methylated with probability init_meth at that site."""
        rates = self.rates
        if rates.is_zero("init_meth"):
            return np.zeros(rates.shape, dtype=int)
        return (self.rng.random(rates.shape) < rates.init_meth).astype(int)

    def snapshot(self):
        """Per-site bulk beta for all agents, Cells and Clones, plus the clone fraction."""
        n_sites = self.rates.n_sites
        cells = self.agents.select(agent_type=Cell)
        clones = self.agents.select(agent_type=Clone)
        n_total = len(self.agents)
        return {
            "all": bulk_beta(self.agents, n_sites),
            "cell": bulk_beta(cells, n_sites),
            "clone": bulk_beta(clones, n_sites),
            "clone_frac": len(clones) / n_total if n_total else np.nan,
        }

    def end_growth_phase(self):
        """Prunes the population back down to a fixed number of agents"""
        current_cells = self.agents.select(agent_type=Cell)
        excess = len(current_cells) - self.target_population

        if excess > 0:
            to_remove = self.random.sample(list(current_cells), excess)
            for cell in to_remove:
                cell.remove()

        if self.init_mode == "independent":
            for cell in self.agents:
                cell.cpgs = self._bernoulli_pattern()

        # Every surviving cell founds its own lineage
        for lineage_id, cell in enumerate(self.agents):
            cell.lineage_id = lineage_id

        self.is_growing = False

    # Moran bookkeeping ----------------------------------------------------------------------------

    def random_agent(self):
        """An agent chosen uniformly from the current population."""
        agents = self.agents.to_list()
        return agents[self.rng.integers(len(agents))]

    def birth(self, parent, victim=None):
        """
        `parent` divides: a daughter of the same class with a copy of its CpGs and lineage replaces
        `victim`. If no victim is given, it is chosen uniformly from the current population (the
        parent included) before the daughter exists, so the newborn is never removed. The
        population size is unchanged.
        """
        n_before = len(self.agents)
        if victim is None:
            victim = self.random_agent()
        daughter = type(parent)(self, cpgs=parent.cpgs.copy(), lineage_id=parent.lineage_id)
        victim.remove()
        if self.debug:
            assert len(self.agents) == n_before, "birth changed the population size"
        return daughter

    def death(self, agent):
        """
        `agent` dies and a copy of a parent chosen uniformly from the current population replaces
        it. If the agent picks itself as the parent nothing changes, which keeps the parent pool
        the same size as birth's victim pool (a neutral clone with p_duplicate == p_die does not
        drift).
        """
        parent = self.random_agent()
        if parent is agent:
            return None
        return self.birth(parent, victim=agent)

    def _n_clones_at_onset(self, n_cells):
        """Number of Cells converted at CHIP onset: prop_chip * n_cells, at least 1 if prop_chip > 0."""
        n_clones = int(self.prop_chip * n_cells)
        if n_clones == 0 and self.prop_chip > 0:
            n_clones = 1
        return n_clones

    def _replace_with_clones(self, targets, clone_template):
        """
        Replace each target Cell with a Clone carrying a copy of the template's CpGs. All Clones
        created at onset share one new lineage id.
        """
        self.clone_lineage = self.target_population
        self.clone_parent_lineage = clone_template.lineage_id
        for target in targets:
            Clone(self, cpgs=clone_template.cpgs.copy(), lineage_id=self.clone_lineage)
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

    def _divide_all(self):
        """
        Every agent present at the start of this phase acts once, in random order. Agents removed
        earlier in the step are skipped and newborns wait until the next step. Without turnover
        only Clones act; that list is built and shuffled exactly as
        `agents.select(agent_type=Clone).shuffle_do("divide")` would, so turnover-free runs keep
        their original random stream.
        """
        if self.p_turnover == 0:
            actors = [a for a in self.agents if isinstance(a, Clone)]
        else:
            actors = self.agents.to_list()
        self.random.shuffle(actors)
        for agent in actors:
            if agent in self.agents:
                agent.divide()

    def step(self):
        """Advances the simulation by one step."""
        self.datacollector.collect(self)
        if self.steps in self.sample_steps:
            self.site_snapshots[self.steps] = self.snapshot()

        if self.is_growing:
            self.agents.shuffle_do("duplicate")
            self.agents.shuffle_do("methylate")
        else:
            self.agents.shuffle_do("methylate")
            self._divide_all()

        if self.debug:
            assert_population_ok(self)
