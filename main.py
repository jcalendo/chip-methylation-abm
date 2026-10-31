import argparse
import sys

import mesa
import numpy as np
import pandas as pd

from model import AgingModel
from rates import RateProfile
from visualization import plot_simulation_metrics

# Scalar rate flags. They default to None so we can tell whether the user set them; the values
# below are filled in after parsing. They are ignored when --rate-profile is given.
SCALAR_RATE_DEFAULTS = {
    "p_meth": 0.005,
    "p_unmeth": 1e-6,
    "init_meth": 0.0,
    "p_meth_clone": 0.005,
    "p_unmeth_clone": 1e-6,
}


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Show defaults, except for flags whose default is filled in after parsing."""

    def _get_help_string(self, action):
        if action.default is None:
            return action.help
        return super()._get_help_string(action)


def _int_list(text):
    try:
        return tuple(int(x) for x in text.split(",") if x.strip())
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected comma-separated integers, got {text!r}")


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the CHIP Methylation Agent-Based Model.",
        formatter_class=_HelpFormatter,
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
        help="Proportion of total population that will be turned into Clones at chip_time.",
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
        help="Number of CpGs (columns of CpG matrix) per gene.",
    )
    parser.add_argument(
        "--p-meth",
        type=float,
        default=None,
        help="Probability of an unmethylated CpG becoming methylated per step. (default: 0.005)",
    )
    parser.add_argument(
        "--p-unmeth",
        type=float,
        default=None,
        help="Probability of a methylated CpG becoming unmethylated per step. (default: 1e-06)",
    )
    parser.add_argument(
        "--init-meth",
        type=float,
        default=None,
        help="Initial proportion of CpGs that are methylated at the start of the simulation. "
        "(default: 0.0)",
    )

    # Clone arguments
    parser.add_argument(
        "--p-meth-clone",
        type=float,
        default=None,
        help="Probability of an unmethylated CpG becoming methylated per step in Clones. "
        "(default: 0.005)",
    )
    parser.add_argument(
        "--p-unmeth-clone",
        type=float,
        default=None,
        help="Probability of a methylated CpG becoming unmethylated per step in Clones. "
        "(default: 1e-06)",
    )
    parser.add_argument(
        "--p-duplicate",
        type=float,
        default=0.0,
        help="Extra per-step probability (on top of --p-turnover) that a Clone divides and its "
        "daughter replaces a random agent. The selective advantage s = p_duplicate - p_die is "
        "relative to normal cells.",
    )
    parser.add_argument(
        "--p-die",
        type=float,
        default=0.0,
        help="Per-step probability that a Clone dies and is replaced by a copy of a random agent "
        "(Cell or Clone).",
    )
    parser.add_argument(
        "--p-turnover",
        type=float,
        default=0.0,
        help="Per-step probability that each agent (Cell or Clone) divides after the growth "
        "phase, its daughter replacing a random agent (neutral Moran turnover). "
        "p_turnover + p_duplicate + p_die must be <= 1.",
    )
    parser.add_argument(
        "--drift-mode",
        choices=("time", "division", "both"),
        default="time",
        help="'time': every agent drifts once per step. 'division': methylation changes only as "
        "replication errors when a cell divides, and the rate flags/profile are per-division "
        "error probabilities. 'both': per-step drift plus replication errors with the *_div "
        "columns of the rate profile.",
    )
    parser.add_argument(
        "--errors-on",
        choices=("both", "daughter"),
        default="both",
        help="Which copies get replication errors at a division: parent and daughter, or the "
        "daughter only.",
    )

    # Per-CpG rates and per-site output
    parser.add_argument(
        "--rate-profile",
        type=str,
        default=None,
        help="CSV of per-CpG rates (one row per site, see rates.py). Overrides the scalar rate "
        "flags (--p-meth, --p-unmeth, --init-meth, --p-meth-clone, --p-unmeth-clone).",
    )
    parser.add_argument(
        "--init-mode",
        choices=("founder", "independent"),
        default="founder",
        help="'founder': every cell copies the founder's initial pattern. 'independent': at the "
        "end of growth every cell redraws each site from init_meth.",
    )
    parser.add_argument(
        "--sample-steps",
        type=_int_list,
        default=(),
        help="Comma-separated Step labels at which to save per-site bulk beta (requires "
        "--site-beta-out). The final state is always saved as step 'final'.",
    )
    parser.add_argument(
        "--site-beta-out",
        type=str,
        default=None,
        help="Path of a .npz file for per-site bulk beta snapshots.",
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

    args = parser.parse_args(argv)

    if args.sample_steps and not args.site_beta_out:
        parser.error("--sample-steps requires --site-beta-out")

    # Record which scalar rate flags were given explicitly, then fill in defaults
    args.explicit_rate_flags = [
        "--" + name.replace("_", "-")
        for name in SCALAR_RATE_DEFAULTS
        if getattr(args, name) is not None
    ]
    for name, default in SCALAR_RATE_DEFAULTS.items():
        if getattr(args, name) is None:
            setattr(args, name, default)

    return args


def print_row(label, value):
    """Helper function for formatting rows of banner"""
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
    if args.rate_profile:
        print_row("Rate profile", args.rate_profile)
    else:
        print_row("Initial methylation proportion", args.init_meth)
        print_row("Cell methylation probability", args.p_meth)
        print_row("Cell unmethylation probability", args.p_unmeth)
        print_row("Clone methylation probability", args.p_meth_clone)
        print_row("Clone unmethylation probability", args.p_unmeth_clone)
    print_row("Initialization mode", args.init_mode)
    print_row("Step of CHIP onset", args.chip_time)
    print_row("Proportion of Clones at CHIP", args.prop_chip)
    print_row("Clone duplication probability", args.p_duplicate)
    print_row("Clone removal probability", args.p_die)
    print_row("Turnover probability (all agents)", args.p_turnover)
    print_row("Drift mode", args.drift_mode)
    print_row("Replication errors on", args.errors_on)
    print("└" + "─" * width + "┘\n")


def build_params(args):
    """Model keyword arguments. New options are only added when they differ from the default, so
    default runs keep the original CSV columns."""
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
    if args.rate_profile:
        params["rate_profile"] = args.rate_profile
    if args.init_mode != "founder":
        params["init_mode"] = args.init_mode
    if args.p_turnover != 0.0:
        params["p_turnover"] = args.p_turnover
    if args.drift_mode != "time":
        params["drift_mode"] = args.drift_mode
    if args.errors_on != "both":
        params["errors_on"] = args.errors_on
    return params


def warn(message):
    print(f"WARNING: {message}", file=sys.stderr)


def main(argv=None):
    args = parse_arguments(argv)

    rate_profile = None
    if args.rate_profile:
        # Validate the CSV (shape, ranges) before launching any worker processes
        rate_profile = RateProfile.from_csv(args.rate_profile, args.n_genes, args.n_cpgs)
        if args.explicit_rate_flags:
            warn(
                "--rate-profile is set, so these flags are ignored: "
                + ", ".join(args.explicit_rate_flags)
            )

    if args.drift_mode == "division" and args.p_turnover == 0:
        warn(
            "--drift-mode division with --p-turnover 0: normal cells stop drifting after the "
            "growth phase, except when a cell is copied to replace a dying Clone (--p-die)."
        )

    if not args.quiet:
        print_banner(args)

    params = build_params(args)

    # Initialize RNGs for all runs -- this is how iterations are specified
    rng = np.random.default_rng(12345)
    rng_values = rng.integers(0, sys.maxsize, size=(args.runs,)).tolist()

    if args.site_beta_out:
        from runner import run_many, snapshots_to_npz_dict

        results_df, run_results = run_many(
            params,
            rng_values,
            max_steps=args.steps,
            n_proc=args.n_proc,
            sample_steps=args.sample_steps,
        )
        site_class = (
            rate_profile.site_class
            if rate_profile is not None
            else np.zeros(args.n_genes * args.n_cpgs, dtype=int)
        )
        np.savez_compressed(args.site_beta_out, **snapshots_to_npz_dict(run_results, site_class))
        if not args.quiet:
            print(f"Writing per-site snapshots to {args.site_beta_out}")
    else:
        results = mesa.batch_run(
            model_cls=AgingModel,
            parameters=params,
            number_processes=args.n_proc,
            rng=rng_values,
            data_collection_period=1,
            max_steps=args.steps,
            display_progress=not args.quiet,
        )
        # batch_run collects runs as they finish, so sort for a deterministic row order
        results_df = pd.DataFrame(results).sort_values(["RunId", "Step"], kind="stable")

    results_df.to_csv(args.outfile, index=False, compression="gzip")

    if not args.quiet:
        print(f"Writing results to {args.outfile}")

    try:
        plot_outfile = args.outfile.replace(".csv.gz", ".png")
        plot_simulation_metrics(df=results_df, output_path=plot_outfile)
        if not args.quiet:
            print(f"Plot saved to {plot_outfile}")
            print("Simulation Complete!")
    except Exception as err:
        print("Could not create plots from simulated data. Skipping...")
        print(err)


if __name__ == "__main__":
    main()
