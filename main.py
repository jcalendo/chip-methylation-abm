import argparse
import mesa
import sys
import numpy as np
import pandas as pd

from model import AgingModel


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run the CHIP Methylation Agent-Based Model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Batch parameters
    parser.add_argument(
        "--runs",
        type=int,
        default=100,
        help="Number of times to repeat the experiment.",
    )
    parser.add_argument(
        "--n_proc",
        type=int,
        default=1,
        help="Number of processes used for multiprocessing.",
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
        "--n_genes",
        type=int,
        default=500,
        help="Number of genes (rows of CpG matrix) per agent.",
    )
    parser.add_argument(
        "--n_cpgs",
        type=int,
        default=10,
        help="Number of CpGs (columns od CpG matrix) per gene.",
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

    # Program args
    parser.add_argument(
        "--out_file",
        type=str,
        default="run_results.csv.gz",
        help="Filename to save run results DataFrame.",
    )

    return parser.parse_args()


def print_row(label, value):
    print(f"│ {label:<35} | {str(value):<15} │")


def main():
    args = parse_arguments()

    width = 55
    print("┌" + "─" * width + "┐")
    print("│" + "BEGINNING SIMULATION".center(width) + "│")
    print("├" + "─" * width + "┤")
    print_row("Number of replicate simulations", args.runs)
    print_row("Agents", args.n_agents)
    print_row("Steps", args.steps)
    print_row("Genes", args.n_genes)
    print_row("CpGs per Gene", args.n_cpgs)
    print_row("Cell Methylation probability", args.p_meth)
    print_row("Cell Unmethylation probability", args.p_unmeth)
    print_row("Step at CHIP onset", args.chip_time)
    print_row("Clone Methylation probability", args.p_meth_clone)
    print_row("Clone Unmethylation probability", args.p_unmeth_clone)
    print_row("Clone duplication probability", args.p_duplicate)
    print_row("Clone removal probability", args.p_die)
    print("└" + "─" * width + "┘\n")

    params = {
        "n_agents": args.n_agents,
        "n_genes": args.n_genes,
        "n_cpgs": args.n_cpgs,
        "p_meth": args.p_meth,
        "p_unmeth": args.p_unmeth,
        "p_meth_clone": args.p_meth_clone,
        "p_unmeth_clone": args.p_unmeth_clone,
        "chip_time": args.chip_time,
        "p_duplicate": args.p_duplicate,
        "p_die": args.p_die,
    }

    # Initialize RNGs for all runs -- this is how iterations are specified
    rng = np.random.default_rng(12345)
    rng_values = rng.integers(0, sys.maxsize, size=(args.runs,)).tolist()

    results = mesa.batch_run(
        model_cls=AgingModel,
        parameters=params,
        number_processes=args.n_proc,
        rng=rng_values,
        data_collection_period=1,
        max_steps=args.steps,
        display_progress=True,
    )
    print("\nSimulation Complete!")
    print(f"Writing results to {args.out_file}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(args.out_file, index=False, compression="gzip")

    print("Done.")


if __name__ == "__main__":
    main()
