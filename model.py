"""
Basic linear aging/methylation model and data collection for testing.

TODO:

- At a given age induce CHIP by cloning Cell agent(s). 
- Need to decide how to pick which Cell to clone. JP suggested by methylation percentage initially. 
  Could also add 'fitness' into the mix. 
- Need to decide how to replace current cells (add or replace, i.e. fixed population or growth). 
  Keeping a fixed population and replacing with clones seems the easiest. I know that later they 
  want to test different mixtures of populations though, for example sampling 40% clones and 60% 
  remaining cells.    
"""

import mesa
from agents import CellAgent
from metrics import mean_methylation, jsd_unmeth, mean_shannon, mean_pdr, beta_var, gini, alpha_diversity


class AgingModel(mesa.Model):
    def __init__(self, n_agents, n_genes, n_cpgs, p_meth, p_unmeth):
        super().__init__()        
        self.datacollector = mesa.DataCollector(
            agent_reporters={
                "Mean_Methylation": mean_methylation,
                "JSD_Unmeth": jsd_unmeth,
                "Mean_Shannon": mean_shannon,
                "Mean_PDR": mean_pdr,
                "Beta_Variance": beta_var,
                "Gini_Mean_Diff": gini,
                "Alpha_Diversity": alpha_diversity
            }
        )
        
        CellAgent.create_agents(
            self, 
            n_agents, 
            n_genes=n_genes, 
            n_cpgs=n_cpgs, 
            p_meth=p_meth, 
            p_unmeth=p_unmeth
            )
        
        self.datacollector.collect(self)

    def step(self):
        self.agents.do("step")
        self.datacollector.collect(self)