import argparse
import mesa
import sys
import numpy as np
import pandas as pd

from model import AgingModel
from visualization import plot_simulation_metrics


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
        "--n-proc",
        type=int,
        default=4,
        help="Number of processes used for multiprocessing.",
    )

    # Model arguments
    parser.add_argument(
        "--n-agents",
        type=int,
        default=1_000,
        help="Total number of cells/agents. Constant population size after growth phase.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Total number of steps (years) to run the simulation.",
    )
    parser.add_argument(
        "--chip-time",
        type=int,
        default=50,
        help="Step (year) at which the first mutant Clone is introduced.",
    )
    parser.add_argument(
        "--prop-chip",
        type=float,
        default=0.02,
        help="Proportion of total population that will be turned into Clones at chip_time."
    )

    # Cell arguments
    parser.add_argument(
        "--n-genes",
        type=int,
        default=100,
        help="Number of genes (rows of CpG matrix) per agent.",
    )
    parser.add_argument(
        "--n-cpgs",
        type=int,
        default=10,
        help="Number of CpGs (columns od CpG matrix) per gene.",
    )
    parser.add_argument(
        "--p-meth",
        type=float,
        default=0.005,
        help="Probability of an unmethylated CpG becoming methylated per step.",
    )
    parser.add_argument(
        "--p-unmeth",
        type=float,
        default=1e-6,
        help="Probability of a methylated CpG becoming unmethylated per step.",
    )
    parser.add_argument(
        "--init-meth",
        type=float,
        default=0.0,
        help="Initial proportion of CpGs that are methylated at the start of the simulation.",
    )

    # Clone arguments
    parser.add_argument(
        "--p-meth-clone",
        type=float,
        default=0.005,
        help="Probability of an unmethylated CpG becoming methylated per step in Clones.",
    )
    parser.add_argument(
        "--p-unmeth-clone",
        type=float,
        default=1e-6,
        help="Probability of a methylated CpG becoming unmethylated per step in Clones.",
    )
    parser.add_argument(
        "--p-duplicate",
        type=float,
        default=0.0,
        help="Probability of a Clone duplicating.",
    )
    parser.add_argument(
        "--p-die",
        type=float,
        default=0.0,
        help="Probability of a Clone being replaced by a Cell.",
    )

    # Program args
    parser.add_argument(
        "--outfile",
        type=str,
        default="run_results.csv.gz",
        help="Filename to save run results DataFrame to.",
    )
    parser.add_argument(
        "--quiet",
        type=bool,
        default=False,
        help="Do not print program messages.",
        action=argparse.BooleanOptionalAction,
    )

    return parser.parse_args()


def print_row(label, value):
    """Helper funciton for formatting rows of banner"""
    print(f"│ {label:<35} | {str(value):<15} │")


def print_banner(args, width=55):
    """Print a nice looking startup banner"""

    print("┌" + "─" * width + "┐")
    print("│" + "SIMULATION PARAMETERS".center(width) + "│")
    print("├" + "─" * width + "┤")
    print_row("Runs", args.runs)
    print_row("Steps per run", args.steps)
    print_row("Agents per run", args.n_agents)
    print_row("Genes per agent", args.n_genes)
    print_row("CpGs per gene", args.n_cpgs)
    print_row("Initial methylation proportion", args.init_meth)
    print_row("Cell methylation probability", args.p_meth)
    print_row("Cell unmethylation probability", args.p_unmeth)
    print_row("Step of CHIP onset", args.chip_time)
    print_row("Proportion of Clones at CHIP", args.prop_chip)
    print_row("Clone methylation probability", args.p_meth_clone)
    print_row("Clone unmethylation probability", args.p_unmeth_clone)
    print_row("Clone duplication probability", args.p_duplicate)
    print_row("Clone removal probability", args.p_die)
    print("└" + "─" * width + "┘\n")


def main():
    args = parse_arguments()

    if not args.quiet:
        print_banner(args)

    params = {
        "n_agents": args.n_agents,
        "n_genes": args.n_genes,
        "n_cpgs": args.n_cpgs,
        "init_meth": args.init_meth,
        "p_meth": args.p_meth,
        "p_unmeth": args.p_unmeth,
        "p_meth_clone": args.p_meth_clone,
        "p_unmeth_clone": args.p_unmeth_clone,
        "chip_time": args.chip_time,
        "prop_chip": args.prop_chip,
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
        display_progress=not args.quiet,
    )

    results_df = pd.DataFrame(results)
    results_df.to_csv(args.outfile, index=False, compression="gzip")
    
    if not args.quiet:
        print(f"Writing results to {args.outfile}")
    
    plot_outfile = args.outfile.replace(".csv.gz", ".png")
    plot_simulation_metrics(df=results_df, output_path=plot_outfile)
    
    if not args.quiet:
        print(f"Plot saved to {plot_outfile}")
        print("Simulation Complete!")


if __name__ == "__main__":
    main()
