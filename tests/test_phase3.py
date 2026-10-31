"""Phase 3: methylation drift coupled to cell division."""

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agents import Clone
from helpers import (
    GOLDEN_1,
    GOLDEN_2,
    GOLDEN_DIR,
    assert_matches_golden,
    cpgs_hash,
    make_model,
    run_model,
)

REPO = Path(__file__).resolve().parent.parent


def expected_beta(beta0, a, b, k):
    total = a + b
    beta_inf = a / total
    return beta_inf + (beta0 - beta_inf) * (1.0 - total) ** k


# T3.1 ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["g1_prop0.1_seed1", "g1_prop0.02_seed2"])
def test_time_mode_reproduces_golden_1(name):
    assert_matches_golden(name, dict(GOLDEN_1[name], drift_mode="time", errors_on="both"))


@pytest.mark.parametrize("name", sorted(GOLDEN_2))
def test_time_mode_reproduces_golden_2(name):
    assert_matches_golden(name, dict(GOLDEN_2[name], drift_mode="time"))


def test_invalid_modes_raise():
    with pytest.raises(ValueError):
        make_model(drift_mode="sometimes")
    with pytest.raises(ValueError):
        make_model(errors_on="parent")


# T3.2 ---------------------------------------------------------------------------------------------
def test_no_division_no_drift():
    model = make_model(drift_mode="division", p_meth=0.05, p_unmeth=0.01, chip_time=10_000)
    run_model(model, model.growth_steps)  # end_growth_phase runs right after this step
    state = {a.unique_id: a.cpgs.copy() for a in model.agents}
    assert any(s.any() for s in state.values())  # growth divisions did introduce errors
    run_model(model, 100 - model.growth_steps)
    assert {a.unique_id for a in model.agents} == set(state)
    for agent in model.agents:
        np.testing.assert_array_equal(agent.cpgs, state[agent.unique_id])


# T3.3 ---------------------------------------------------------------------------------------------
def test_drift_tracks_each_cells_error_rounds():
    a, b, runs = 0.02, 0.01, 30
    residuals, ks = [], set()
    for seed in range(runs):
        model = make_model(
            n_agents=200, drift_mode="division", p_meth=a, p_unmeth=b, p_turnover=0.5,
            chip_time=10_000, rng=seed,
        )  # fmt: skip
        run_model(model, 40)
        cell_beta = np.array([ag.cpgs.mean() for ag in model.agents])
        k = np.array([ag.n_error_rounds for ag in model.agents])
        ks.update(k.tolist())
        residuals.append(np.mean(cell_beta - expected_beta(0.0, a, b, k)))
    residuals = np.array(residuals)
    se = residuals.std(ddof=1) / np.sqrt(runs)  # between-run SE: cells within a run are related
    assert abs(residuals.mean()) < 4 * se
    assert len(ks) > 5  # cells really do differ in their division history


# T3.4 ---------------------------------------------------------------------------------------------
def test_division_counts_after_growth():
    model = make_model(drift_mode="division")
    run_model(model, model.growth_steps)
    assert {a.n_divisions for a in model.agents} == {model.growth_steps}


@pytest.mark.parametrize("drift_mode", ["time", "division"])
def test_birth_bookkeeping_is_exact(drift_mode):
    model = make_model(
        drift_mode=drift_mode, p_turnover=0.5, prop_chip=0.2, p_duplicate=0.2, p_die=0.2,
        chip_time=12, debug=True,
    )  # fmt: skip
    original = model.birth
    checked = {"births": 0}

    def checked_birth(parent, victim=None):
        before = {a.unique_id: a.n_divisions for a in model.agents}
        p = parent.n_divisions
        daughter = original(parent, victim)
        after = {a.unique_id: a.n_divisions for a in model.agents}
        (removed,) = set(before) - set(after)
        r = p + 1 if removed == parent.unique_id else before[removed]
        assert daughter.n_divisions == p + 1
        assert parent.n_divisions == p + 1
        assert sum(after.values()) - sum(before.values()) == 2 + p - r
        checked["births"] += 1
        return daughter

    model.birth = checked_birth
    run_model(model, 40)
    assert checked["births"] > 100


# T3.5 ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("errors_on", ["daughter", "both"])
def test_errors_on(errors_on):
    model = make_model(
        drift_mode="division", errors_on=errors_on, p_meth=0.5, p_unmeth=0.5, chip_time=10_000
    )
    run_model(model, model.growth_steps)
    parent, victim = model.agents.to_list()[:2]
    before_cpgs, before_rounds = parent.cpgs.copy(), parent.n_error_rounds
    daughter = model.birth(parent, victim=victim)
    assert daughter.n_error_rounds == before_rounds + 1
    assert not np.array_equal(daughter.cpgs, before_cpgs)
    if errors_on == "daughter":
        np.testing.assert_array_equal(parent.cpgs, before_cpgs)
        assert parent.n_error_rounds == before_rounds
    else:
        assert not np.array_equal(parent.cpgs, before_cpgs)
        assert parent.n_error_rounds == before_rounds + 1
    assert parent.n_divisions == daughter.n_divisions


# T3.6 ---------------------------------------------------------------------------------------------
def final_clone_beta(drift_mode, p_duplicate, p_die, runs):
    out = []
    for seed in range(runs):
        model = make_model(
            drift_mode=drift_mode, p_meth=0.0, p_unmeth=0.0, p_meth_clone=0.01,
            p_unmeth_clone=0.001, prop_chip=0.25, chip_time=8, p_duplicate=p_duplicate,
            p_die=p_die, rng=seed,
        )  # fmt: skip
        run_model(model, 40)
        clones = model.agents.select(agent_type=Clone)
        if len(clones):  # condition on clone survival
            out.append(np.mean([c.cpgs.mean() for c in clones]))
    return np.array(out)


def two_sample_z(x, y):
    return (y.mean() - x.mean()) / np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))


@pytest.mark.slow
def test_faster_division_faster_drift():
    slow_div = final_clone_beta("division", 0.0, 0.0, 100)
    fast_div = final_clone_beta("division", 0.4, 0.4, 100)
    # a neutral clone can be lost by drift; enough runs must survive for a stable SE
    assert len(slow_div) >= 50 and len(fast_div) >= 50
    assert two_sample_z(slow_div, fast_div) > 4

    slow_time = final_clone_beta("time", 0.0, 0.0, 100)
    fast_time = final_clone_beta("time", 0.4, 0.4, 100)
    assert abs(two_sample_z(slow_time, fast_time)) < 4


# T3.7 ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["g1_prop0.1_seed1", "g2_seed2"])
def test_both_with_zero_division_rates_matches_golden(name):
    params = {**GOLDEN_1, **GOLDEN_2}[name]
    assert_matches_golden(name, dict(params, drift_mode="both"))


def test_both_equals_time_with_turnover():
    kwargs = dict(p_turnover=0.3, prop_chip=0.2, p_duplicate=0.2, p_die=0.1, rng=9)
    a, b = make_model(drift_mode="time", **kwargs), make_model(drift_mode="both", **kwargs)
    pd.testing.assert_frame_equal(run_model(a, 40), run_model(b, 40), check_exact=True)
    assert cpgs_hash(a) == cpgs_hash(b)


def test_both_mode_applies_division_rates(tmp_path):
    from rates import RateProfile

    shape = (20, 10)
    zero = np.zeros(shape)
    RateProfile(
        p_meth=zero, p_unmeth=zero, init_meth=zero, p_meth_clone=zero, p_unmeth_clone=zero,
        p_meth_div=np.full(shape, 0.05),
    ).to_csv(tmp_path / "r.csv")  # fmt: skip
    model = make_model(rate_profile=str(tmp_path / "r.csv"), drift_mode="both", chip_time=10_000)
    run_model(model, 20)
    # no per-step drift at all, so every methylated site came from a growth division
    assert np.mean([a.cpgs.mean() for a in model.agents]) > 0


# T3.8 ---------------------------------------------------------------------------------------------
def run_cli(tmp_path, *extra):
    import main

    main.main(
        [
            "--runs", "1", "--n-proc", "1", "--n-agents", "16", "--n-genes", "5", "--n-cpgs",
            "2", "--steps", "8", "--chip-time", "6", "--outfile", str(tmp_path / "o.csv.gz"),
            "--quiet", *extra,
        ]
    )  # fmt: skip


def test_division_without_turnover_warns(tmp_path, capsys):
    run_cli(tmp_path, "--drift-mode", "division", "--p-turnover", "0")
    err = capsys.readouterr().err
    assert "WARNING" in err and "division" in err


def test_division_with_turnover_does_not_warn(tmp_path, capsys):
    run_cli(tmp_path, "--drift-mode", "division", "--p-turnover", "0.2")
    assert "WARNING" not in capsys.readouterr().err


# TC.1 ---------------------------------------------------------------------------------------------
@pytest.mark.slow
def test_snakefile_command_runs(tmp_path):
    """Run the exact flags from scripts/Snakefile (one grid point) with fewer runs and steps."""
    import main

    shell = (REPO / "scripts" / "Snakefile").read_text().split('"""')[1]
    shell = shell.replace("{wildcards.prop}", "0.01").replace("{wildcards.meth}", "0.005")
    shell = shell.replace("{threads}", "1").replace("{output.csv}", str(tmp_path / "s.csv.gz"))
    argv = [tok for tok in re.split(r"\s+|\\", shell) if tok and tok not in ("uv", "run")]
    assert argv[0].endswith("main.py")
    main.main(argv[1:] + ["--runs", "2", "--steps", "60"])
    df = pd.read_csv(tmp_path / "s.csv.gz")
    assert len(df) == 2 * 60
    assert {"Cell_Count", "Clone_Count", "Overall_Mean_JSD"} <= set(df.columns)
    assert (tmp_path / "s.png").exists()


# TP.1 ---------------------------------------------------------------------------------------------
@pytest.mark.slow
def test_runtime_against_phase0_baseline():
    baseline = json.loads((GOLDEN_DIR / "baseline_runtime.json").read_text())["phase0_seconds"]
    times = []
    for seed in (1, 2, 3):
        model = make_model(
            n_agents=1000, n_genes=805, n_cpgs=1, chip_time=40, prop_chip=0.1,
            p_duplicate=0.2, p_die=0.2, p_turnover=0.5, drift_mode="division", rng=seed,
        )  # fmt: skip
        start = time.perf_counter()
        run_model(model, 80)
        times.append(time.perf_counter() - start)
    print(f"\nTP.1 runtime {min(times):.2f}s vs Phase 0 baseline {baseline:.2f}s")
    assert min(times) < 3 * baseline
