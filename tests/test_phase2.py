"""Phase 2: normal-cell turnover, unified Moran bookkeeping, lineage tracking."""

import numpy as np
import pytest

from agents import Cell, Clone
from helpers import GOLDEN_2, assert_matches_golden, make_model, run_model
from metrics import bulk_beta

# Tiny genomes for the population-genetics tests, which do not look at methylation
TINY = dict(n_genes=1, n_cpgs=1)


def clone_fraction(model):
    return len(model.agents.select(agent_type=Clone)) / len(model.agents)


# T2.2 ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("p_turnover", [0.1, 0.5, 1.0])
@pytest.mark.parametrize("prop_chip", [0.0, 0.2])
def test_population_invariant_with_turnover(p_turnover, prop_chip):
    p_dup, p_die = (0.0, 0.0) if p_turnover == 1.0 else (0.2 * (1 - p_turnover),) * 2
    model = make_model(
        p_turnover=p_turnover, prop_chip=prop_chip, p_duplicate=p_dup, p_die=p_die, debug=True
    )
    df = run_model(model, 40)  # debug=True checks the invariant every step and in every birth
    assert (df["Cell_Count"] + df["Clone_Count"]).iloc[model.growth_steps + 1 :].eq(64).all()


def test_turnover_validation():
    with pytest.raises(ValueError):
        make_model(p_turnover=0.5, p_duplicate=0.3, p_die=0.3)
    with pytest.raises(ValueError):
        make_model(p_turnover=1.5)


# T2.3 ---------------------------------------------------------------------------------------------
def test_birth_never_removes_newborn():
    model = make_model(n_agents=2, chip_time=5, **TINY)
    run_model(model, 2)
    removed = {"parent": 0, "other": 0}
    for _ in range(1000):
        parent, other = model.agents.to_list()
        daughter = model.birth(parent)
        assert daughter in model.agents
        assert len(model.agents) == 2
        removed["parent" if parent not in model.agents else "other"] += 1
    assert removed["parent"] > 300 and removed["other"] > 300


# T2.4 ---------------------------------------------------------------------------------------------
def test_moran_pools_are_symmetric():
    """
    Record the pool each Moran pick draws from. Both birth's victim and death's parent are drawn
    from all N current agents (the actor included, the newborn excluded), so
    E[dC | duplicate] = 1 - C/N and E[dC | die] = -(1 - C/N) exactly.
    """
    model = make_model(n_agents=5, chip_time=4, prop_chip=0.4, **TINY)
    run_model(model, 5)
    clones = model.agents.select(agent_type=Clone).to_list()
    n, c = len(model.agents), len(clones)
    assert (n, c) == (5, 2)

    pools = []
    original = model.random_agent

    def recording_random_agent():
        pools.append(model.agents.to_list())
        return original()

    model.random_agent = recording_random_agent
    actor = clones[0]
    before = model.agents.to_list()

    daughter = model.birth(actor)
    assert pools[-1] == before and daughter not in pools[-1]
    expected_dup = np.mean([0 if isinstance(v, Clone) else 1 for v in pools[-1]])
    assert expected_dup == pytest.approx(1 - c / n)

    survivor = next(a for a in model.agents.select(agent_type=Clone) if a is not daughter)
    state = model.agents.to_list()
    model.death(survivor)
    pool = pools[-1]
    assert pool == state and survivor in pool
    c_now = sum(isinstance(a, Clone) for a in pool)
    expected_die = np.mean(
        [0 if p is survivor else (-1 + isinstance(p, Clone)) for p in pool]
    )
    assert expected_die == pytest.approx(-(1 - c_now / len(pool)))


# T2.5 ---------------------------------------------------------------------------------------------
@pytest.mark.slow
def test_neutral_lineage_fixation_probability():
    n, runs, focal = 20, 1000, 5
    fixed = 0
    for seed in range(runs):
        model = make_model(n_agents=n, chip_time=10_000, p_turnover=1.0, rng=seed, **TINY)
        run_model(model, model.growth_steps + 1)
        for cell in model.agents.to_list()[:focal]:
            cell.lineage_id = -2
        for _ in range(2000):
            k = sum(a.lineage_id == -2 for a in model.agents)
            if k in (0, n):
                break
            model.step()
        assert k in (0, n), "replicate not absorbed within the step cap"
        fixed += k == n
    p, expected = fixed / runs, focal / n
    se = np.sqrt(expected * (1 - expected) / runs)
    assert abs(p - expected) < 4 * se


# T2.6 ---------------------------------------------------------------------------------------------
def test_lineage_count_never_increases_and_coalesces():
    model = make_model(n_agents=10, chip_time=10_000, p_turnover=1.0, **TINY)
    df = run_model(model, 600)
    counts = df["Lineage_Count"].dropna()
    assert counts.iloc[0] == 10
    assert (counts.diff().dropna() <= 0).all()
    assert counts.iloc[-1] == 1
    assert df["Largest_Lineage_Fraction"].iloc[-1] == 1.0


# T2.7 ---------------------------------------------------------------------------------------------
def final_clone_fractions(runs, steps_after, **params):
    out = []
    for seed in range(runs):
        model = make_model(chip_time=8, prop_chip=0.25, rng=seed, **TINY, **params)
        run_model(model, model.chip_time + 1 + steps_after)
        out.append(clone_fraction(model))
    return np.array(out)


@pytest.mark.slow
@pytest.mark.parametrize(
    "params",
    [dict(p_turnover=0.3), dict(p_turnover=0.3, p_duplicate=0.2, p_die=0.2)],
    ids=["exchangeable", "dup_equals_die"],
)
def test_neutral_clone_keeps_its_fraction(params):
    fractions = final_clone_fractions(500, 30, **params)
    se = fractions.std(ddof=1) / np.sqrt(len(fractions))
    assert abs(fractions.mean() - 0.25) < 4 * se


@pytest.mark.slow
def test_advantageous_clone_expands():
    neutral = final_clone_fractions(500, 30, p_turnover=0.3)
    advantaged = final_clone_fractions(500, 30, p_turnover=0.3, p_duplicate=0.1)
    se = np.sqrt(neutral.var(ddof=1) / 500 + advantaged.var(ddof=1) / 500)
    assert advantaged.mean() - neutral.mean() > 4 * se


# T2.8 ---------------------------------------------------------------------------------------------
def per_site_run_means(p_turnover, runs):
    betas = []
    for seed in range(runs):
        model = make_model(
            p_meth=0.02, p_unmeth=0.005, p_turnover=p_turnover, chip_time=10_000, rng=seed
        )
        run_model(model, 40)
        betas.append(bulk_beta(model.agents, model.rates.n_sites))
    return np.array(betas)


def test_turnover_does_not_bias_expected_beta():
    runs = 40
    a, b = per_site_run_means(0.0, runs), per_site_run_means(0.5, runs)
    se = np.sqrt(a.var(axis=0, ddof=1) / runs + b.var(axis=0, ddof=1) / runs)
    z = (a.mean(axis=0) - b.mean(axis=0)) / se
    assert abs(z.mean()) < 4 / np.sqrt(z.size)
    assert (np.abs(z) < 4).mean() >= 0.99
    # turnover makes cells related, so per-site beta varies more between runs
    assert b.var(axis=0, ddof=1).mean() > a.var(axis=0, ddof=1).mean()


# T2.9 ---------------------------------------------------------------------------------------------
def test_lineage_ids():
    model = make_model(prop_chip=0.1, p_turnover=0.2)
    run_model(model, model.growth_steps)  # end_growth_phase runs right after step growth_steps
    ids = [a.lineage_id for a in model.agents]
    assert sorted(ids) == list(range(64))
    run_model(model, model.chip_time - model.growth_steps)
    clones = model.agents.select(agent_type=Clone)
    assert len(clones) == 6
    assert {c.lineage_id for c in clones} == {64}
    assert model.clone_parent_lineage in range(64)
    assert all(isinstance(a, Cell) for a in model.agents if a.lineage_id != 64)


def test_daughters_inherit_lineage():
    model = make_model(p_turnover=0.5, chip_time=10_000)
    run_model(model, 30)
    ids = [a.lineage_id for a in model.agents]
    assert len(set(ids)) < 64  # some lineages were lost to turnover
    assert all(0 <= i < 64 for i in ids)


# T2.10 --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(GOLDEN_2))
def test_golden_2(name):
    assert_matches_golden(name, GOLDEN_2[name])
