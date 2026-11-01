"""Phase 1: per-CpG rate profiles, initialization modes, per-site output, runner."""

import json
from pathlib import Path

import mesa
import numpy as np
import pandas as pd
import pytest
from mesa.batchrunner import _make_model_kwargs
from scipy.stats import spearmanr

from agents import Clone
from helpers import BASE_PARAMS, cpgs_hash, make_model, run_model
from metrics import bulk_beta
from model import AgingModel
from rates import RateProfile, generate_profile
from runner import run_many


def write_profile(path, n_genes, n_cpgs, **arrays):
    """Write a rate CSV; any array not given is zero. Scalars are broadcast."""
    shape = (n_genes, n_cpgs)
    fields = ("p_meth", "p_unmeth", "init_meth", "p_meth_clone", "p_unmeth_clone")
    values = {
        f: np.broadcast_to(np.asarray(arrays.get(f, 0.0), float), shape) for f in fields
    }
    RateProfile(**values, site_class=arrays.get("site_class")).to_csv(path)
    return str(path)


# T1.1 ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("init_meth", [0.0, 0.25])
def test_scalar_and_uniform_csv_identical(tmp_path, init_meth):
    rates = dict(p_meth=0.01, p_unmeth=0.002, p_meth_clone=0.03, p_unmeth_clone=0.001)
    path = write_profile(tmp_path / "r.csv", 20, 10, init_meth=init_meth, **rates)
    a = make_model(prop_chip=0.1, init_meth=init_meth, **rates)
    b = make_model(prop_chip=0.1, rate_profile=path)
    pd.testing.assert_frame_equal(run_model(a, 40), run_model(b, 40), check_exact=True)
    assert cpgs_hash(a) == cpgs_hash(b)


# T1.2 ---------------------------------------------------------------------------------------------
def test_csv_wrong_row_count(tmp_path):
    path = write_profile(tmp_path / "r.csv", 10, 1)
    with pytest.raises(ValueError, match="rows"):
        RateProfile.from_csv(path, 20, 1)


@pytest.mark.parametrize("bad_value", [1.5, -0.1, np.nan])
def test_csv_bad_value_names_column_and_row(tmp_path, bad_value):
    path = write_profile(tmp_path / "r.csv", 10, 1)
    df = pd.read_csv(path)
    df.loc[6, "p_unmeth"] = bad_value
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match=r"column 'p_unmeth', row 7"):
        RateProfile.from_csv(path, 10, 1)


def test_csv_missing_column(tmp_path):
    path = write_profile(tmp_path / "r.csv", 10, 1)
    pd.read_csv(path).drop(columns="init_meth").to_csv(path, index=False)
    with pytest.raises(ValueError, match="init_meth"):
        RateProfile.from_csv(path, 10, 1)


@pytest.mark.parametrize("bad_label", [np.nan, 1.7, -1])
def test_csv_bad_site_class(tmp_path, bad_label):
    path = write_profile(tmp_path / "r.csv", 10, 1)
    df = pd.read_csv(path)
    df["site_class"] = df["site_class"].astype(float)
    df.loc[3, "site_class"] = bad_label
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match=r"column 'site_class', row 4"):
        RateProfile.from_csv(path, 10, 1)


def test_profile_arrays_are_read_only():
    profile = RateProfile.from_scalars(2, 2, 0.1, 0.1, 0.1, 0.1, 0.0)
    with pytest.raises(ValueError):
        profile.p_meth[0, 0] = 0.5


# T1.3 ---------------------------------------------------------------------------------------------
def test_rate_profile_path_is_not_swept():
    kwargs = _make_model_kwargs({"rate_profile": "x.csv", "n_agents": 64})
    assert kwargs == [{"rate_profile": "x.csv", "n_agents": 64}]


# T1.4 ---------------------------------------------------------------------------------------------
def test_edge_rates(tmp_path):
    # site 0: no gain/loss; site 1: gain 1, loss 0; others: moderate drift
    n = 10
    p_meth = np.full((n, 1), 0.05)
    p_unmeth = np.full((n, 1), 0.05)
    p_meth[0], p_unmeth[0] = 0.0, 0.0
    p_meth[1], p_unmeth[1] = 1.0, 0.0
    path = write_profile(
        tmp_path / "r.csv", n, 1, p_meth=p_meth, p_unmeth=p_unmeth, init_meth=0.5
    )
    model = make_model(n_genes=n, n_cpgs=1, rate_profile=path, init_mode="independent")
    run_model(model, model.growth_steps)
    model.step()  # this step's drift runs, then end_growth_phase redraws every cell
    frozen = {a.unique_id: a.cpgs[0, 0] for a in model.agents}
    model.step()  # one drift step after the redraw
    assert all(a.cpgs[1, 0] == 1 for a in model.agents)
    run_model(model, 20)
    assert {a.unique_id: a.cpgs[0, 0] for a in model.agents} == frozen


# T1.5 ---------------------------------------------------------------------------------------------
def expected_beta(beta0, a, b, k):
    """Single-site two-state chain: gain a, loss b per round, starting at beta0, after k rounds."""
    total = a + b
    beta_inf = np.divide(a, total, out=np.zeros_like(total), where=total > 0)
    return beta_inf + (beta0 - beta_inf) * (1.0 - total) ** k


def test_per_site_drift_matches_theory(tmp_path):
    n_sites, n_agents, steps = 200, 500, 40
    rng = np.random.default_rng(2024)
    a = np.exp(rng.uniform(np.log(1e-4), np.log(1e-1), n_sites))
    b = np.exp(rng.uniform(np.log(1e-4), np.log(1e-1), n_sites))
    init = rng.uniform(0.1, 0.9, n_sites)
    path = write_profile(
        tmp_path / "r.csv", n_sites, 1,
        p_meth=a[:, None], p_unmeth=b[:, None], init_meth=init[:, None],
    )  # fmt: skip
    model = make_model(
        n_agents=n_agents, n_genes=n_sites, n_cpgs=1, rate_profile=path,
        init_mode="independent", chip_time=steps + 10, rng=7,
    )  # fmt: skip
    run_model(model, steps)

    # The redraw happens after step growth_steps, so drift has acted for steps - growth_steps rounds
    k = steps - model.growth_steps
    expected = expected_beta(init, a, b, k)
    observed = bulk_beta(model.agents, n_sites)
    se = np.sqrt(
        expected * (1 - expected) / n_agents
    )  # cells are independent after the redraw
    within = np.abs(observed - expected) <= 4 * se
    assert within.mean() >= 0.95
    assert spearmanr(expected, observed).statistic > 0.95

    # mean(z^2) is about 1 at the right k and grows quickly when k is off, so this pins the number
    # of drift rounds. It must reject k = steps (ignoring the growth phase) and k - 6.
    def mean_z2(k_):
        e = expected_beta(init, a, b, k_)
        return np.mean(((observed - e) / np.sqrt(e * (1 - e) / n_agents)) ** 2)

    limit = 1 + 4 * np.sqrt(2 / n_sites)
    assert mean_z2(k) < limit
    assert mean_z2(steps) > limit and mean_z2(k - 6) > limit


# T1.6 ---------------------------------------------------------------------------------------------
def test_clone_rates_apply_only_to_clones():
    model = make_model(
        prop_chip=0.25, p_meth=0.02, p_meth_clone=0.0, p_unmeth_clone=0.0
    )
    run_model(model, model.chip_time + 1)  # clones exist after step chip_time
    clones = list(model.agents.select(agent_type=Clone))
    assert clones
    frozen = {c.unique_id: c.cpgs.copy() for c in clones}
    df = run_model(model, 30)
    for clone in model.agents.select(agent_type=Clone):
        np.testing.assert_array_equal(clone.cpgs, frozen[clone.unique_id])
    cell_beta = df["Cell_Mean_Methylation"]
    assert cell_beta.iloc[-1] > cell_beta.iloc[model.chip_time + 1]


# T1.7 ---------------------------------------------------------------------------------------------
def test_founder_mode_sites_are_all_or_nothing(tmp_path):
    rng = np.random.default_rng(3)
    init = rng.uniform(0.1, 0.9, (200, 1))
    path = write_profile(tmp_path / "r.csv", 200, 1, init_meth=init)
    model = make_model(n_agents=128, n_genes=200, n_cpgs=1, rate_profile=path)
    run_model(model, model.growth_steps + 2)
    beta = bulk_beta(model.agents, 200)
    assert np.all((beta == 0) | (beta == 1))
    assert 0 < beta.mean() < 1


def test_independent_mode_matches_init_meth(tmp_path):
    n_sites, n_agents = 200, 256
    rng = np.random.default_rng(4)
    init = rng.uniform(0.1, 0.9, n_sites)
    path = write_profile(tmp_path / "r.csv", n_sites, 1, init_meth=init[:, None])
    model = make_model(
        n_agents=n_agents,
        n_genes=n_sites,
        n_cpgs=1,
        rate_profile=path,
        init_mode="independent",
    )
    run_model(model, model.growth_steps + 2)
    beta = bulk_beta(model.agents, n_sites)
    z = (beta - init) / np.sqrt(init * (1 - init) / n_agents)
    assert abs(z.mean()) < 4 / np.sqrt(n_sites)
    assert np.abs(z).max() < 5


# T1.8 ---------------------------------------------------------------------------------------------
def test_founder_effect_identity(tmp_path):
    path = write_profile(tmp_path / "r.csv", 20, 10, init_meth=0.5)
    model = make_model(rate_profile=path, init_mode="independent", prop_chip=0.25)
    n_sites, n = model.rates.n_sites, model.target_population
    captured = {}
    original = model._replace_with_clones

    def wrapped(targets, clone_template):
        captured["pre"] = bulk_beta(model.agents, n_sites)
        captured["replaced"] = np.mean([t.cpgs.ravel() for t in targets], axis=0)
        captured["template"] = clone_template.cpgs.ravel().astype(float)
        captured["n_clones"] = len(targets)
        original(targets, clone_template)
        captured["post"] = bulk_beta(model.agents, n_sites)

    model._replace_with_clones = wrapped
    run_model(model, model.chip_time + 1)
    c = captured
    assert c["n_clones"] == 16
    predicted = c["pre"] + (c["n_clones"] / n) * (c["template"] - c["replaced"])
    np.testing.assert_allclose(c["post"], predicted, atol=1e-12)


# T1.9 ---------------------------------------------------------------------------------------------
def test_snapshots_and_npz_round_trip(tmp_path):
    import main

    out_csv, out_npz = tmp_path / "res.csv.gz", tmp_path / "beta.npz"
    main.main(
        [
            "--runs", "2", "--n-proc", "1", "--n-agents", "64", "--n-genes", "20",
            "--n-cpgs", "10", "--steps", "40", "--chip-time", "20", "--prop-chip", "0.25",
            "--p-meth-clone", "0.03", "--sample-steps", "10,30", "--site-beta-out", str(out_npz),
            "--outfile", str(out_csv), "--quiet",
        ]
    )  # fmt: skip
    data = np.load(out_npz)
    assert data["site_class"].shape == (200,)
    for run in (0, 1):
        for step in ("10", "30", "final"):
            assert data[f"run{run}_step{step}_all"].shape == (200,)
        f = float(data[f"run{run}_stepfinal_clone_frac"])
        assert 0 < f < 1
        np.testing.assert_allclose(
            data[f"run{run}_stepfinal_all"],
            f * data[f"run{run}_stepfinal_clone"]
            + (1 - f) * data[f"run{run}_stepfinal_cell"],
        )
        assert np.isnan(data[f"run{run}_step10_clone"]).all()  # before CHIP onset
    df = pd.read_csv(out_csv)
    assert sorted(df["RunId"].unique()) == [0, 1]


def test_snapshot_label_matches_collector_row():
    model = make_model(sample_steps=(15,))
    df = run_model(model, 20)
    # Row index 14 has Step label 15 (the collector runs at the start of each step)
    assert model.site_snapshots[15]["all"].mean() == pytest.approx(
        df.loc[14, "Overall_Mean_Methylation"]
    )


# T1.10 --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("n_proc", [1, 2])
def test_runner_matches_batch_run(n_proc):
    params = dict(BASE_PARAMS, prop_chip=0.1)
    seeds = [11, 22, 33]
    expected = pd.DataFrame(
        mesa.batch_run(
            AgingModel, params, number_processes=1, rng=seeds, data_collection_period=1,
            max_steps=30, display_progress=False,
        )
    ).sort_values(["RunId", "Step"], kind="stable").reset_index(drop=True)  # fmt: skip
    got, _ = run_many(params, seeds, max_steps=30, n_proc=n_proc)
    pd.testing.assert_frame_equal(got, expected, check_exact=True)


# T1.11 --------------------------------------------------------------------------------------------
SPEC = {
    "classes": [
        {
            "site_class": 1,
            "fraction": 191,
            "p_meth": {"dist": "loguniform", "low": 1e-3, "high": 1e-2},
            "p_unmeth": {"dist": "fixed", "value": 1e-6},
            "init_meth": {"dist": "beta", "a": 1, "b": 9},
        },
        {
            "site_class": 2,
            "fraction": 614,
            "p_meth": {"dist": "fixed", "value": 1e-6},
            "p_unmeth": {"dist": "lognormal", "median": 5e-3, "sigma": 0.5},
            "init_meth": {"dist": "uniform", "low": 0.7, "high": 0.9},
            "p_meth_clone": {"dist": "fixed", "value": 0.0},
        },
    ]
}


def test_generator_is_deterministic(tmp_path):
    import rates

    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(SPEC))
    for name in ("a.csv", "b.csv"):
        rates.main([
            "generate", str(spec_path), "--seed", "5", "--n-genes", "805", "--n-cpgs", "1",
            "-o", str(tmp_path / name),
        ])  # fmt: skip
    a, b = (tmp_path / "a.csv").read_text(), (tmp_path / "b.csv").read_text()
    assert a == b
    profile = RateProfile.from_csv(tmp_path / "a.csv", 805, 1)
    counts = np.bincount(profile.site_class.ravel())
    assert abs(counts[1] - 191) <= 1 and abs(counts[2] - 614) <= 1
    hypo = profile.site_class == 2
    assert np.all(profile.p_meth_clone[hypo] == 0)
    np.testing.assert_array_equal(profile.p_meth_clone[~hypo], profile.p_meth[~hypo])
    assert np.all(profile.p_meth_div == 0)


def test_generator_clone_div_rates_fall_back_to_cell_div_rates():
    spec = {
        "classes": [
            {
                "fraction": 1,
                "p_meth": {"dist": "fixed", "value": 0.01},
                "p_unmeth": {"dist": "fixed", "value": 0.0},
                "init_meth": {"dist": "fixed", "value": 0.0},
                "p_meth_div": {"dist": "uniform", "low": 0.001, "high": 0.01},
            }
        ]
    }
    profile = generate_profile(spec, 50, 1, seed=3)
    np.testing.assert_array_equal(profile.p_meth_div_clone, profile.p_meth_div)
    assert profile.p_meth_div.min() > 0
    assert np.all(profile.p_unmeth_div_clone == 0)


def test_example_spec_generates():
    spec = json.loads(
        Path(__file__)
        .resolve()
        .parent.parent.joinpath("docs/example_spec.json")
        .read_text()
    )
    profile = generate_profile(spec, 805, 1, seed=1)
    assert np.bincount(profile.site_class.ravel()).tolist()[1:] == [191, 614]


def test_generator_seed_changes_rates():
    a = generate_profile(SPEC, 805, 1, seed=1)
    b = generate_profile(SPEC, 805, 1, seed=2)
    assert not np.array_equal(a.p_meth, b.p_meth)


# TC.2 ---------------------------------------------------------------------------------------------
def test_rate_profile_warns_about_ignored_flags(tmp_path, capsys):
    import main

    path = write_profile(tmp_path / "r.csv", 20, 10)
    main.main(
        [
            "--runs", "1", "--n-proc", "1", "--n-agents", "16", "--n-genes", "20", "--n-cpgs",
            "10", "--steps", "8", "--chip-time", "6", "--rate-profile", path, "--p-meth", "0.1",
            "--outfile", str(tmp_path / "o.csv.gz"), "--quiet",
        ]
    )  # fmt: skip
    err = capsys.readouterr().err
    assert "--p-meth" in err and "ignored" in err
    assert "--p-unmeth" not in err
    # the ignored scalar rates are recorded as NaN, not as values that were never used
    df = pd.read_csv(tmp_path / "o.csv.gz")
    assert df["p_meth"].isna().all() and df["init_meth"].isna().all()


@pytest.mark.parametrize("steps", ["0,10", "10,41"])
def test_sample_steps_outside_run_rejected(tmp_path, steps):
    import main

    with pytest.raises(SystemExit):
        main.parse_arguments(
            [
                "--steps",
                "40",
                "--sample-steps",
                steps,
                "--site-beta-out",
                str(tmp_path / "b.npz"),
            ]
        )
