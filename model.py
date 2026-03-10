import mesa
from agents import Cell, Clone
from metrics import mean_methylation, jsd_unmeth, mean_shannon, beta_var, gini


class AgingModel(mesa.Model):
    def __init__(self, n_agents, n_genes, n_cpgs, p_meth, p_unmeth, chip_time):
        super().__init__()
        
        self.n_genes = n_genes
        self.n_cpgs = n_cpgs
        self.p_meth = p_meth
        self.p_unmeth = p_unmeth

        Cell.create_agents(
            model=self, 
            n=n_agents, 
            n_genes=n_genes, 
            n_cpgs=n_cpgs, 
            p_meth=p_meth, 
            p_unmeth=p_unmeth
        )

        self.schedule_event(self.trigger_chip, at=chip_time)
        
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
                "Cell_Count": lambda m: len(m.agents_by_type.get(Cell, [])),
                "Clone_Count": lambda m: len(m.agents_by_type.get(Clone, []))
            }
        )

    def trigger_chip(self):
        """Selects a random Cell, copies its state to a new Clone, and removes the Cell."""
        cell_agents = self.agents_by_type.get(Cell)
        if cell_agents and len(cell_agents) > 0:
            target_cell = self.random.choice(cell_agents)
            new_clone = Clone(self, self.n_genes, self.n_cpgs, self.p_meth, self.p_unmeth)
            new_clone.cpgs = target_cell.cpgs.copy()
            target_cell.remove()
            print(f"CHIP triggered at t={self.time:.1f}. First clone created!")

    def step(self):
        """Standard step executed every time unit."""
        self.datacollector.collect(self)
        
        self.agents.shuffle_do("methylate")

        if Clone in self.agents_by_type:
            self.agents_by_type[Clone].shuffle_do("divide")