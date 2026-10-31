"""Shared helpers for the test suite and the golden-capture script."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from model import AgingModel

GOLDEN_DIR = Path(__file__).parent / "golden"

# Small, fast configuration. Rates match the CLI defaults; CHIP is off unless overridden.
BASE_PARAMS = dict(
    n_agents=64,
    n_genes=20,
    n_cpgs=10,
    p_meth=0.005,
    p_unmeth=1e-6,
    init_meth=0.0,
    p_meth_clone=0.005,
    p_unmeth_clone=1e-6,
    chip_time=20,
    prop_chip=0.0,
    p_duplicate=0.0,
    p_die=0.0,
)

GOLDEN_STEPS = 60

# Golden-1: CLI-default dynamics (no selection) with one clone (prop_chip 0.02 of 64) and several.
GOLDEN_1 = {
    f"g1_prop{prop}_seed{seed}": dict(BASE_PARAMS, prop_chip=prop, rng=seed)
    for prop in (0.02, 0.1)
    for seed in (1, 2, 3)
}


def make_model(**overrides):
    params = dict(BASE_PARAMS)
    params.update(overrides)
    params.setdefault("rng", 1)
    return AgingModel(**params)


def run_model(model, steps):
    for _ in range(steps):
        model.step()
    return model.datacollector.get_model_vars_dataframe()


def cpgs_hash(model):
    """Hash of every agent's type and CpG state, in unique_id order."""
    h = hashlib.sha256()
    for agent in sorted(model.agents, key=lambda a: a.unique_id):
        h.update(type(agent).__name__.encode())
        h.update(np.ascontiguousarray(agent.cpgs, dtype=np.int64).tobytes())
    return h.hexdigest()


def run_config(params, steps=GOLDEN_STEPS):
    model = AgingModel(**params)
    df = run_model(model, steps)
    return df, cpgs_hash(model)


def save_golden(name, df, digest):
    GOLDEN_DIR.mkdir(exist_ok=True)
    df.to_csv(GOLDEN_DIR / f"{name}.csv", float_format="%.17g")
    hashes_path = GOLDEN_DIR / "hashes.json"
    hashes = json.loads(hashes_path.read_text()) if hashes_path.exists() else {}
    hashes[name] = digest
    hashes_path.write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")


def load_golden(name):
    df = pd.read_csv(GOLDEN_DIR / f"{name}.csv", index_col=0, float_precision="round_trip")
    digest = json.loads((GOLDEN_DIR / "hashes.json").read_text())[name]
    return df, digest


def assert_matches_golden(name, params, steps=GOLDEN_STEPS):
    golden_df, golden_digest = load_golden(name)
    df, digest = run_config(params, steps)
    pd.testing.assert_frame_equal(
        df[golden_df.columns].reset_index(drop=True),
        golden_df.reset_index(drop=True),
        check_exact=True,
        check_dtype=False,
    )
    assert digest == golden_digest, f"final CpG state differs from golden {name}"
