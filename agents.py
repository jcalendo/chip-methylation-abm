"""
Basic Cell agent class. Creates a 2d array of genes x CpGs all initialized to 0. Stochastically 
methylates or unmethylates based on the current methylation state and the probability of flipping
from one state to another (p_meth and p_unmeth, respectively).

 TODO:
 
- Implement 'clone' method for copying state into a new Cell
- Should there be a concept of 'fitness' that makes it more or less likely for a Cell to divide
- Likewise, should there be Cell death?
"""

import mesa
import numpy as np


class CellAgent(mesa.Agent):
    """
    Initialize a Cell with a 2D gene x CpG array of all 0s. Every step determine if a CpG will flip 
    states based on the probability of methylating (p_meth) or unmenthylating (p_unmeth).
    """
    def __init__(self, model, n_genes, n_cpgs, p_meth, p_unmeth):
        super().__init__(model)
        self.n_genes = n_genes
        self.n_cpgs = n_cpgs
        self.p_meth = p_meth
        self.p_unmeth = p_unmeth
        self.cpgs = np.zeros((n_genes, n_cpgs), dtype=int)

    def step(self):
        probs = np.random.rand(self.n_genes, self.n_cpgs)
        do_meth = (probs < self.p_meth) & (self.cpgs == 0)
        do_unmeth = (probs < self.p_unmeth) & (self.cpgs == 1)
        self.cpgs ^= (do_meth | do_unmeth)