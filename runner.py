"""
Run AgingModel replicates and keep per-site snapshots.

mesa.batch_run only returns DataCollector rows, so it cannot carry the per-site bulk beta snapshots.
run_many produces the same row schema as batch_run (RunId, iteration, Step, <params>, rng,
<reporters>) plus a snapshots dict per run. main.py uses it only when per-site output is requested.
"""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from multiprocessing import get_context

import pandas as pd
from mesa.batchrunner import _collect_data

from model import AgingModel


@dataclass
class RunResult:
    run_id: int
    seed: int
    rows: list
    snapshots: dict = field(default_factory=dict)  # {step label or "final": snapshot dict}


def run_one(run, params, max_steps, sample_steps=()):
    """
    Run one replicate. `run` is (run_id, seed). Mirrors mesa.batchrunner._model_run_func so the
    collected rows are identical to batch_run's for the same parameters and seed.
    """
    run_id, seed = run
    kwargs = dict(params, rng=seed)
    model = AgingModel(**kwargs, sample_steps=tuple(sample_steps))
    while model.running and model.steps < max_steps:
        model.step()

    rows = []
    for step in model.datacollector._collection_steps:
        model_data, _ = _collect_data(model, step)
        rows.append(
            {"RunId": run_id, "iteration": run_id, "Step": step, **kwargs, **model_data}
        )

    snapshots = dict(model.site_snapshots)
    snapshots["final"] = model.snapshot()
    return RunResult(run_id=run_id, seed=seed, rows=rows, snapshots=snapshots)


def run_many(params, seeds, max_steps, n_proc=1, sample_steps=()):
    """Run one replicate per seed. Returns (DataFrame sorted by RunId/Step, list of RunResult)."""
    runs = list(enumerate(seeds))
    func = partial(run_one, params=params, max_steps=max_steps, sample_steps=sample_steps)
    if n_proc == 1:
        results = [func(run) for run in runs]
    else:
        with ProcessPoolExecutor(max_workers=n_proc, mp_context=get_context("spawn")) as ex:
            results = list(ex.map(func, runs))

    df = pd.DataFrame([row for result in results for row in result.rows])
    df = df.sort_values(["RunId", "Step"], kind="stable").reset_index(drop=True)
    return df, results


def snapshots_to_npz_dict(results, site_class):
    """Flatten snapshots into arrays keyed run{r}_step{s}_{all|cell|clone|clone_frac}."""
    out = {"site_class": site_class.ravel()}
    for result in results:
        for step, snap in result.snapshots.items():
            label = step if step == "final" else int(step)
            prefix = f"run{result.run_id}_step{label}"
            for key, value in snap.items():
                out[f"{prefix}_{key}"] = value
    return out
