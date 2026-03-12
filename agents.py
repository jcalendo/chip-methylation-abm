import mesa
import numpy as np


class Cell(mesa.Agent):
    """
    Basic Cell class consists of a 2D gene x CpG numpy array populated with 0 or 1

    Args:
        n_genes: Number of genes (rows)
        n_cpgs: Number of CpG sites per gene (columns)
        p_meth: Probability of switching from a methylated state to an unmethylated CpG
        p_unmeth: Probability of switching from an unmethylated state to a methylated CpG
        init_meth: Proportion of methylated CpG sites in the initial state
    """

    def __init__(self, model, n_genes, n_cpgs, p_meth, p_unmeth, init_meth):
        super().__init__(model)
        self.n_genes = n_genes
        self.n_cpgs = n_cpgs
        self.p_meth = p_meth
        self.p_unmeth = p_unmeth
        self.init_meth = init_meth

        # Determine the starting state of the Cell
        if init_meth == 0:
            self.cpgs = np.zeros((n_genes, n_cpgs), dtype=int)
        else:
            total_cpgs = n_genes * n_cpgs
            num_ones = int(round(init_meth * total_cpgs))
            num_zeros = total_cpgs - num_ones
            arr = np.concatenate((np.ones(num_ones), np.zeros(num_zeros)))
            np.random.shuffle(arr)
            self.cpgs = arr.reshape((n_genes, n_cpgs)).astype(int)

    def methylate(self):
        """Apply random methylation to all CpGs and genes"""
        probs = np.random.rand(self.n_genes, self.n_cpgs)
        do_meth = (probs < self.p_meth) & (self.cpgs == 0)
        do_unmeth = (probs < self.p_unmeth) & (self.cpgs == 1)
        self.cpgs ^= do_meth | do_unmeth


class Clone(mesa.Agent):
    """
    A Clone is a Cell that divides according to cell division probabilities.

    Args:
        n_genes: Number of genes (rows)
        n_cpgs: Number of CpG sites per gene (columns)
        p_meth: Probability of switching from a methylated state to an unmethylated CpG
        p_unmeth: Probability of switching from an unmethylated state to a methylated CpG
        p_duplicate: Probability of a clone creating a copy of itself and removing a random Cell
        p_die: Probability of a clone dying and being replaced by a random Cell
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
                
                # Register the new Cell 
                Cell(
                    self.model,
                    parent_cell.n_genes,
                    parent_cell.n_cpgs,
                    parent_cell.p_meth,
                    parent_cell.p_unmeth,
                    parent_cell.init_meth,
                )

                self.remove()
