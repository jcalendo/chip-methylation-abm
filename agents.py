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
        self.cpgs ^= (do_meth | do_unmeth)


class Clone(mesa.Agent):
    """
    A Clone is basically the same as a Cell but clones divide and copy their information into 
    daughter cells, replacing a randomly selected Cell in the process. 
    """
    def __init__(self, model, n_genes, n_cpgs, p_meth, p_unmeth):
        super().__init__(model)
        self.n_genes = n_genes
        self.n_cpgs = n_cpgs
        self.p_meth = p_meth
        self.p_unmeth = p_unmeth
        self.cpgs = np.zeros((n_genes, n_cpgs), dtype=int)

    def methylate(self):
        probs = np.random.rand(self.n_genes, self.n_cpgs)
        do_meth = (probs < self.p_meth) & (self.cpgs == 0)
        do_unmeth = (probs < self.p_unmeth) & (self.cpgs == 1)
        self.cpgs ^= (do_meth | do_unmeth)
    
    def divide(self):
        """Create a copy of the Clone and replace it with a randomly chosen Cell"""
        cell_agents = self.model.agents_by_type.get(Cell)
            
        if cell_agents and len(cell_agents) > 0:
            daughter = Clone(self.model, self.n_genes, self.n_cpgs, self.p_meth, self.p_unmeth)
            daughter.cpgs = self.cpgs.copy()
            
            target = self.random.choice(cell_agents)
            target.remove()
        