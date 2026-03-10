from model import AgingModel


def main():
    N_AGENTS = 100
    N_GENES = 100
    N_CPGS = 10
    P_METH = 1e-3
    P_UNMETH = 1e-6
    CHIP_TIME = 50
    STEPS = 100

    print("Initializing the Aging Model...")
    model = AgingModel(
        n_agents=N_AGENTS,
        n_genes=N_GENES,
        n_cpgs=N_CPGS,
        p_meth=P_METH,
        p_unmeth=P_UNMETH,
        chip_time=CHIP_TIME
    )

    print(f"Running simulation for {STEPS} steps...")
    model.run_for(STEPS)
    print("Simulation complete!\n")

    model_df = model.datacollector.get_model_vars_dataframe()
    agent_df = model.datacollector.get_agent_vars_dataframe()

    model_df.to_csv("model_results.csv.gz", compression="gzip")
    agent_df.to_csv("agent_results.csv.gz", compression="gzip")

if __name__ == "__main__":
    main()