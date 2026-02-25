"""
Run a test simulation. Save results in a gzipped compressed csv file. I'd rather plot data in R...
"""

import gzip

from datetime import datetime
from model import AgingModel

 
CELLS = 1_000
CPGS = 10
GENES = 1_000
P_METH = 1e-3
P_UNMETH = 1e-6
STEPS = 100


def main():
    model = AgingModel(n_agents=CELLS, n_cpgs=CPGS, n_genes=GENES, p_meth=P_METH, p_unmeth=P_UNMETH)
    model.run_for(STEPS)

    df = model.datacollector.get_agent_vars_dataframe()

    # Write agent-level results to compressed csv file
    # Add a comment line to the csv header indicating run parameters
    fname = f"results/{datetime.now().strftime("%Y%m%d-%H%M%S")}_results.csv.gz"
    comment = f"# Cells: {CELLS}, CpGs: {CPGS}, Genes: {GENES}, P_meth: {P_METH}, P_unmeth: {P_UNMETH}\n"
    with gzip.open(fname, "wt") as outfile:
        outfile.write(comment)

    with gzip.open(fname, "at") as outfile:
        df.to_csv(outfile)


if __name__ == "__main__":
    main()