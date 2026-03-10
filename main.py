"""
Author: Gennaro Calendo
Date: 2026-03-10
Title: CHIP aging model


The ABM simulates a population of Cells and Clones. Each Cell/Clone is an agent with n_genes and
n_cpgs per gene. All CpGs are initialized as being completely unmethylated (all 0) at the
start of the simulation. With each step of the simulation, every CpG has a p_meth probability of
randomly gaining methylation (being flipped from a 0 -> 1). At a certain age (chip_time), one of
the Cells is randomly chosen to become a Clone. At every step after, the Clones can divide one of
three mutaully exclusive ways; symmetric self renewal, symmetric differentiation, or asymmetric
division. NOTE: the probability of asymetric division is computed from
p_asym = 1.0 - (p_sym_renew + p_sym_diff) and not given explicitly. When a new Clone is created
by one of the division strategies a Cell is randomly chosen to be removed from the simulation. Thus,
the population size throughout the simulation stays constant. During the simulation heterogeniety
metrics are computed for each agent and across the entire model. These metrics are saved as
compressed .csv files.


TODO:

- Figure out population level metrics. Added some from Gemini but need to actually check them for
    correctness
- Create plots of metrics using computed results (might just do this in R)
- Parameter sweeps
- Simulation replication
- Induce CHIP at multiple times
- Select Cells for CHIP induction (i.e. Pick the Cell with the highest methylation at chip_time)

"""

from datetime import datetime
from model import AgingModel


def main():

    N_AGENTS = 1_000
    N_GENES = 500
    N_CPGS = 10
    P_METH = 1e-3
    P_UNMETH = 1e-6
    CHIP_TIME = 50
    STEPS = 100
    P_RENEW = 0.12
    P_DIFF = 0.08

    print("Initializing the Aging Model...")
    model = AgingModel(
        n_agents=N_AGENTS,
        n_genes=N_GENES,
        n_cpgs=N_CPGS,
        p_meth=P_METH,
        p_unmeth=P_UNMETH,
        chip_time=CHIP_TIME,
        p_sym_renew=P_RENEW,
        p_sym_diff=P_DIFF,
    )

    print(f"Running simulation for {STEPS} steps...")
    start_time = datetime.now()
    model.run_for(STEPS)
    end_time = datetime.now()
    time_difference = (end_time - start_time).total_seconds()
    print(f"Simulation Complete!\nExecution time: {time_difference:.1f}s")

    model_df = model.datacollector.get_model_vars_dataframe()
    agent_df = model.datacollector.get_agent_vars_dataframe()

    model_df.to_csv("model_results.csv.gz", compression="gzip")
    agent_df.to_csv("agent_results.csv.gz", compression="gzip")


if __name__ == "__main__":
    main()
