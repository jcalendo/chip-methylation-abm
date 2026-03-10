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
    A Clone divides according to stem cell division probabilities.
    """

    def __init__(
        self, model, n_genes, n_cpgs, p_meth, p_unmeth, p_sym_renew, p_sym_diff
    ):
        super().__init__(model)
        self.n_genes = n_genes
        self.n_cpgs = n_cpgs
        self.p_meth = p_meth
        self.p_unmeth = p_unmeth
        self.cpgs = np.zeros((n_genes, n_cpgs), dtype=int)
        self.p_sym_renew = p_sym_renew
        self.p_sym_diff = p_sym_diff
        self.p_asym = 1.0 - (p_sym_renew + p_sym_diff)

        if self.p_asym < 0.0 or self.p_asym > 1.0:
            raise ValueError(
                f"Invalid probabilities: p_sym_renew ({p_sym_renew}) + p_sym_diff ({p_sym_diff}) must be <= 1.0"
            )

    def methylate(self):
        probs = np.random.rand(self.n_genes, self.n_cpgs)
        do_meth = (probs < self.p_meth) & (self.cpgs == 0)
        do_unmeth = (probs < self.p_unmeth) & (self.cpgs == 1)
        self.cpgs ^= do_meth | do_unmeth

    def divide(self):
        """Execute one of the three stem cell division types."""
        division_type = np.random.choice(
            ["asymmetric", "symmetric_renewal", "symmetric_differentiation"],
            p=[self.p_asym, self.p_sym_renew, self.p_sym_diff],
        )

        if division_type == "asymmetric":
            pass

        elif division_type == "symmetric_renewal":
            cell_agents = self.model.agents_by_type.get(Cell)

            if cell_agents and len(cell_agents) > 0:
                daughter = Clone(
                    self.model,
                    self.n_genes,
                    self.n_cpgs,
                    self.p_meth,
                    self.p_unmeth,
                    self.p_sym_renew,
                    self.p_sym_diff,
                )
                daughter.cpgs = self.cpgs.copy()

                target = self.random.choice(cell_agents)
                target.remove()

        elif division_type == "symmetric_differentiation":
            cell_agents = self.model.agents_by_type.get(Cell)

            if cell_agents and len(cell_agents) > 0:
                parent_cell = self.random.choice(cell_agents)

                daughter_cell = Cell(
                    self.model, self.n_genes, self.n_cpgs, self.p_meth, self.p_unmeth
                )

                daughter_cell.cpgs = parent_cell.cpgs.copy()
                self.remove()
