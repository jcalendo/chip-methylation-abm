import mesa
import numpy as np


class Cell(mesa.Agent):
    """
    Initialize a Cell with a 2D gene x CpG array of all 0s. Every step determine if a CpG will flip
    states based on the probability of methylating (p_meth) or unmethylating (p_unmeth).
    """

    def __init__(self, model, n_genes, n_cpgs, p_meth, p_unmeth):
        super().__init__(model)
        self.n_genes = n_genes
        self.n_cpgs = n_cpgs
        self.p_meth = p_meth
        self.p_unmeth = p_unmeth
        self.cpgs = np.zeros((n_genes, n_cpgs), dtype=int)

    def methylate(self):
        """Apply random methylation to all CpGs and genes"""
        probs = np.random.rand(self.n_genes, self.n_cpgs)
        do_meth = (probs < self.p_meth) & (self.cpgs == 0)
        do_unmeth = (probs < self.p_unmeth) & (self.cpgs == 1)
        self.cpgs ^= do_meth | do_unmeth


class Clone(mesa.Agent):
    """
    A Clone divides according to cell division probabilities.
    """

    def __init__(self, model, n_genes, n_cpgs, p_meth, p_unmeth, p_duplicate, p_die):
        super().__init__(model)
        self.n_genes = n_genes
        self.n_cpgs = n_cpgs
        self.p_meth = p_meth
        self.p_unmeth = p_unmeth
        self.cpgs = np.zeros((n_genes, n_cpgs), dtype=int)
        self.p_duplicate = p_duplicate
        self.p_die = p_die
        self.p_neutral = 1.0 - (p_duplicate + p_die)

        if self.p_neutral < 0.0 or self.p_neutral > 1.0:
            raise ValueError(
                f"Invalid probabilities: p_duplicate ({p_duplicate}) + p_die ({p_die}) must be <= 1.0"
            )

    def methylate(self):
        probs = np.random.rand(self.n_genes, self.n_cpgs)
        do_meth = (probs < self.p_meth) & (self.cpgs == 0)
        do_unmeth = (probs < self.p_unmeth) & (self.cpgs == 1)
        self.cpgs ^= do_meth | do_unmeth

    def divide(self):
        """Execute one of the three cell division types."""
        division_type = np.random.choice(
            ["neutral", "duplicate", "die"],
            p=[self.p_neutral, self.p_duplicate, self.p_die],
        )

        # No net change
        if division_type == "neutral":
            pass

        # Randomly chosen Cell gets replaced by a copy of the Clone
        elif division_type == "duplicate":
            cell_agents = self.model.agents_by_type.get(Cell)

            if cell_agents and len(cell_agents) > 0:
                daughter = Clone(
                    self.model,
                    self.n_genes,
                    self.n_cpgs,
                    self.p_meth,
                    self.p_unmeth,
                    self.p_duplicate,
                    self.p_die,
                )
                daughter.cpgs = self.cpgs.copy()

                target = self.random.choice(cell_agents)
                target.remove()

        # Clone 'dies' and gets replaced with a randomly chosen Cell
        elif division_type == "die":
            cell_agents = self.model.agents_by_type.get(Cell)

            if cell_agents and len(cell_agents) > 0:
                parent_cell = self.random.choice(cell_agents)

                daughter_cell = Cell(
                    self.model, self.n_genes, self.n_cpgs, self.p_meth, self.p_unmeth
                )

                daughter_cell.cpgs = parent_cell.cpgs.copy()
                self.remove()
