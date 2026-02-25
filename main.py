"""
Run a test simulation. Save results in a gzipped compressed csv file. I'd rather plot data in R...
"""

from model import AgingModel


def main():
    model = AgingModel(n_agents=10_000, n_cpgs=10, n_genes=1000, p_meth=1e-3, p_unmeth=1e-7)
    model.run_for(100)
    df = model.datacollector.get_agent_vars_dataframe()
    df.to_csv("results.csv.gz", compression="gzip")

if __name__ == "__main__":
    main()