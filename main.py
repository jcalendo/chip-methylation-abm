import argparse

from datetime import datetime
from model import AgingModel


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run the CHIP Methylation Agent-Based Model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model arguments
    parser.add_argument(
        "--n_agents",
        type=int,
        default=1_000,
        help="Total number of cells (constant population size).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Total number of steps (years) to run the simulation.",
    )
    parser.add_argument(
        "--chip_time",
        type=int,
        default=50,
        help="Step (year) at which the first mutant Clone is introduced.",
    )

    # Cell arguments
    parser.add_argument(
        "--n_genes", type=int, default=500, help="Number of genes (rows of CpG matrix) per agent."
    )
    parser.add_argument(
        "--n_cpgs", type=int, default=10, help="Number of CpGs (columns od CpG matrix) per gene."
    )
    parser.add_argument(
        "--p_meth",
        type=float,
        default=0.01,
        help="Probability of an unmethylated CpG becoming methylated per step.",
    )
    parser.add_argument(
        "--p_unmeth",
        type=float,
        default=1e-6,
        help="Probability of a methylated CpG becoming unmethylated per step.",
    )

    # Clone arguments
    parser.add_argument(
        "--p_meth_clone",
        type=float,
        default=0.01,
        help="Probability of an unmethylated CpG becoming methylated per step in Clones.",
    )
    parser.add_argument(
        "--p_unmeth_clone",
        type=float,
        default=1e-6,
        help="Probability of a methylated CpG becoming unmethylated per step in Clones.",
    )
    parser.add_argument(
        "--p_duplicate",
        type=float,
        default=1.0,
        help="Probability of a Clone duplicating.",
    )
    parser.add_argument(
        "--p_die",
        type=float,
        default=0.0,
        help="Probability of a Clone being replaced by a Cell.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    print("Initializing the Aging Model...")
    model = AgingModel(
        n_agents=args.n_agents,
        n_genes=args.n_genes,
        n_cpgs=args.n_cpgs,
        p_meth=args.p_meth,
        p_unmeth=args.p_unmeth,
        p_meth_clone=args.p_meth_clone,
        p_unmeth_clone=args.p_unmeth_clone,        
        chip_time=args.chip_time,
        p_duplicate=args.p_duplicate,
        p_die=args.p_die,
    )

    print(f"Running simulation for {args.steps} steps...")
    start_time = datetime.now()
    model.run_for(args.steps)
    end_time = datetime.now()
    time_difference = (end_time - start_time).total_seconds()
    print(f"Simulation Complete!\nExecution time: {time_difference:.1f}s")

    model_df = model.datacollector.get_model_vars_dataframe()
    agent_df = model.datacollector.get_agent_vars_dataframe()

    model_df.to_csv("results/model_results.csv.gz", compression="gzip")
    agent_df.to_csv("results/agent_results.csv.gz", compression="gzip")


if __name__ == "__main__":
    main()
