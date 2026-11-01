"""Phase 0: seeded reproducibility, golden output, population invariant, and bug fixes."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agents import Cell, Clone
from helpers import (
    GOLDEN_1,
    GOLDEN_1_INIT,
    assert_matches_golden,
    cpgs_hash,
    make_model,
    run_model,
)
from metrics import (
    jsd_unmeth,
    population_beta_bray_curtis,
    population_epigenetic_burden,
)
from model import AgingModel, assert_population_ok

REPO = Path(__file__).resolve().parent.parent


# T0.1 ---------------------------------------------------------------------------------------------
def test_same_seed_same_output():
    a, b = make_model(prop_chip=0.1, rng=1), make_model(prop_chip=0.1, rng=1)
    df_a, df_b = run_model(a, 40), run_model(b, 40)
    pd.testing.assert_frame_equal(df_a, df_b, check_exact=True)
    assert cpgs_hash(a) == cpgs_hash(b)


def test_different_seed_different_output():
    a, b = make_model(prop_chip=0.1, rng=1), make_model(prop_chip=0.1, rng=2)
    run_model(a, 40), run_model(b, 40)
    assert cpgs_hash(a) != cpgs_hash(b)


# T0.2 ---------------------------------------------------------------------------------------------
LEGACY_RNG = re.compile(
    r"np\.random\.(rand|randn|random|choice|shuffle|seed|randint|normal|uniform|"
    r"binomial|permutation|sample|random_sample)\b"
)


def test_no_legacy_global_rng():
    offenders = []
    for path in REPO.glob("*.py"):
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if LEGACY_RNG.search(line):
                offenders.append(f"{path.name}:{lineno}: {line.strip()}")
    assert not offenders, "legacy global numpy RNG used:\n" + "\n".join(offenders)


# T0.3 ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(GOLDEN_1))
def test_golden_1(name):
    assert_matches_golden(name, GOLDEN_1[name])


@pytest.mark.parametrize("name", sorted(GOLDEN_1_INIT))
def test_golden_1_scalar_init_meth(name):
    """Captured from the Phase 0 code: pins the exact-count founder pattern for init_meth > 0."""
    assert_matches_golden(name, GOLDEN_1_INIT[name])


# T0.4 ---------------------------------------------------------------------------------------------
# The four README example configurations, scaled to N=100 and an earlier CHIP onset.
README_CONFIGS = {
    "constant_2pct": dict(prop_chip=0.02),
    "single_exponential": dict(prop_chip=1e-6, p_duplicate=1.0, p_die=0.0),
    "selective_advantage": dict(prop_chip=0.01, p_duplicate=0.25, p_die=0.2),
    "different_drift": dict(
        init_meth=0.25,
        prop_chip=0.1,
        p_meth=0.001,
        p_unmeth=1e-6,
        p_meth_clone=0.01,
        p_unmeth_clone=0.001,
    ),
}


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(README_CONFIGS))
def test_readme_configs_population_invariant(name):
    model = make_model(n_agents=100, chip_time=10, debug=True, **README_CONFIGS[name])
    run_model(model, 30)  # debug=True asserts the invariant at the end of every step
    assert_population_ok(model)


def test_population_invariant_with_selection():
    model = make_model(prop_chip=0.1, p_duplicate=0.3, p_die=0.2, debug=True)
    run_model(model, 40)


# T0.5 ---------------------------------------------------------------------------------------------
def test_prop_chip_zero_runs():
    model = make_model(prop_chip=0.0)
    df = run_model(model, 40)
    assert (df["Clone_Count"] == 0).all()
    assert df["Cell_Count"].iloc[-1] == 64


def test_highest_meth_trigger_handles_zero_clones():
    model = make_model(prop_chip=0.0)
    run_model(model, 10)
    model.trigger_chip_highest_meth()
    assert len(model.agents.select(agent_type=Clone)) == 0


# T0.6 ---------------------------------------------------------------------------------------------
def test_single_clone_mean_is_defined():
    model = make_model(prop_chip=0.02)  # int(0.02 * 64) = 1 clone
    df = run_model(model, 30)
    after_onset = df.iloc[model.chip_time + 1 :]
    assert (after_onset["Clone_Count"] == 1).all()
    assert after_onset["Clone_Mean_Methylation"].notna().all()
    assert (
        after_onset["Clone_Methylation_Var"].isna().all()
    )  # variance needs >= 2 agents


def test_jsd_unmeth_returns_scalar():
    model = make_model(p_meth=0.2)
    run_model(model, 15)
    value = jsd_unmeth(next(iter(model.agents)))
    assert np.ndim(value) == 0
    assert 0.0 <= value <= 1.0


@pytest.mark.parametrize("baseline", [0.0, 0.25, 0.5, 0.95])
def test_bray_curtis_baseline_bin(baseline):
    # No drift: after growth every cell is an exact copy of the founder, whose mean equals baseline
    model = make_model(init_meth=baseline, p_meth=0.0, p_unmeth=0.0)
    run_model(model, 10)
    assert population_beta_bray_curtis(model.agents) == 0.0


def test_metrics_work_when_first_agent_is_clone():
    model = make_model(prop_chip=1.0)
    run_model(model, 22)
    assert isinstance(next(iter(model.agents)), Clone)
    assert 0.0 <= population_epigenetic_burden(model.agents) <= 1.0
    assert 0.0 <= population_beta_bray_curtis(model.agents) <= 1.0


def test_founder_init_exact_count():
    model = make_model(init_meth=0.25)
    founder = next(iter(model.agents))
    assert isinstance(founder, Cell)
    assert founder.cpgs.sum() == round(0.25 * 20 * 10)


# T0.7 ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "overrides",
    [
        dict(p_duplicate=0.6, p_die=0.5),
        dict(prop_chip=1.5),
        dict(p_meth=1.2),
        dict(init_meth=-0.1),
        dict(chip_time=3),
    ],
)
def test_invalid_parameters_raise_at_construction(overrides):
    with pytest.raises(ValueError):
        make_model(**overrides)


def test_model_signature_matches_cli_params():
    """main.py passes these keys to batch_run; they must all be accepted by AgingModel."""
    import inspect

    accepted = set(inspect.signature(AgingModel.__init__).parameters)
    cli_keys = {
        "n_agents", "n_genes", "n_cpgs", "init_meth", "p_meth", "p_unmeth", "p_meth_clone",
        "p_unmeth_clone", "chip_time", "prop_chip", "p_duplicate", "p_die",
    }  # fmt: skip
    assert cli_keys <= accepted
