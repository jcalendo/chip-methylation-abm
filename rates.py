"""
Per-CpG rate profiles.

A RateProfile holds one value per CpG site (shape n_genes x n_cpgs) for every rate the model uses.
Scalar parameters are broadcast to that shape, so the default scalar CLI behaves exactly as before.

Rate profiles are loaded from a CSV (one row per site, row-major order). The `generate` command
below draws a CSV from distributions described in a small JSON spec:

    uv run rates.py generate spec.json --seed 1 --n-genes 805 --n-cpgs 1 -o rates.csv
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Time-based rates (applied once per step) and initial methylation probability
BASE_FIELDS = ("p_meth", "p_unmeth", "init_meth", "p_meth_clone", "p_unmeth_clone")
# Per-division error rates, only used when drift_mode == "both" (default 0)
DIV_FIELDS = ("p_meth_div", "p_unmeth_div", "p_meth_div_clone", "p_unmeth_div_clone")
PROB_FIELDS = BASE_FIELDS + DIV_FIELDS
CSV_COLUMNS = ("site_id",) + BASE_FIELDS + ("site_class",) + DIV_FIELDS


@dataclass(frozen=True, eq=False)
class RateProfile:
    """
    Per-site rates, each an array of shape (n_genes, n_cpgs).

    Attributes:
        p_meth, p_unmeth: per-step gain/loss probabilities for normal Cells
        p_meth_clone, p_unmeth_clone: per-step gain/loss probabilities for Clones
        init_meth: per-site probability that a site starts methylated
        site_class: integer label per site (0 unassigned, 1 hypermethylating, 2 hypomethylating),
            used only for reporting
        p_meth_div, p_unmeth_div, p_meth_div_clone, p_unmeth_div_clone: per-division error
            probabilities used by drift_mode="both" (zeros when not given)
        init_is_scalar: True when init_meth is one value for every site (a scalar, or a constant
            CSV column). The founder then gets exactly round(init_meth * n_sites) methylated sites
            (the original behaviour) rather than an independent Bernoulli draw per site.
    """

    p_meth: np.ndarray
    p_unmeth: np.ndarray
    init_meth: np.ndarray
    p_meth_clone: np.ndarray
    p_unmeth_clone: np.ndarray
    site_class: np.ndarray = None
    p_meth_div: np.ndarray = None
    p_unmeth_div: np.ndarray = None
    p_meth_div_clone: np.ndarray = None
    p_unmeth_div_clone: np.ndarray = None
    init_is_scalar: bool = False
    _all_zero: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self):
        shape = np.shape(self.p_meth)
        if len(shape) != 2:
            raise ValueError(f"p_meth must be 2-D (n_genes, n_cpgs), got shape {shape}")

        for name in PROB_FIELDS:
            value = getattr(self, name)
            arr = np.zeros(shape) if value is None else np.array(value, dtype=float)
            if arr.shape != shape:
                raise ValueError(f"{name} has shape {arr.shape}, expected {shape}")
            flat = arr.ravel()
            bad = np.flatnonzero(np.isnan(flat) | (flat < 0.0) | (flat > 1.0))
            if bad.size:
                i = bad[0]
                raise ValueError(
                    f"column '{name}', row {i + 1} (site index {i}): value {flat[i]} "
                    "is not a probability in [0, 1]"
                )
            arr.flags.writeable = False
            object.__setattr__(self, name, arr)
            self._all_zero[name] = not arr.any()

        raw = np.zeros(shape) if self.site_class is None else np.asarray(self.site_class)
        flat = raw.ravel().astype(float)
        bad = np.flatnonzero(np.isnan(flat) | (flat < 0) | (flat != np.round(flat)))
        if bad.size:
            i = bad[0]
            raise ValueError(
                f"column 'site_class', row {i + 1} (site index {i}): value {flat[i]} "
                "is not a non-negative integer label"
            )
        site_class = raw.astype(int)
        if site_class.shape != shape:
            raise ValueError(f"site_class has shape {site_class.shape}, expected {shape}")
        site_class.flags.writeable = False
        object.__setattr__(self, "site_class", site_class)

    @property
    def shape(self):
        return self.p_meth.shape

    @property
    def n_sites(self):
        return self.p_meth.size

    def is_zero(self, *names):
        """True when every named rate array is zero everywhere (so no draws are needed)."""
        return all(self._all_zero[name] for name in names)

    @classmethod
    def from_scalars(
        cls,
        n_genes,
        n_cpgs,
        p_meth,
        p_unmeth,
        p_meth_clone,
        p_unmeth_clone,
        init_meth,
        p_meth_div=0.0,
        p_unmeth_div=0.0,
        p_meth_div_clone=0.0,
        p_unmeth_div_clone=0.0,
    ):
        shape = (n_genes, n_cpgs)
        values = dict(
            p_meth=p_meth,
            p_unmeth=p_unmeth,
            init_meth=init_meth,
            p_meth_clone=p_meth_clone,
            p_unmeth_clone=p_unmeth_clone,
            p_meth_div=p_meth_div,
            p_unmeth_div=p_unmeth_div,
            p_meth_div_clone=p_meth_div_clone,
            p_unmeth_div_clone=p_unmeth_div_clone,
        )
        for name, value in values.items():
            if not 0.0 <= value <= 1.0:  # NaN fails this comparison too
                raise ValueError(f"{name} ({value}) must be in [0, 1]")
        return cls(
            **{name: np.full(shape, value, dtype=float) for name, value in values.items()},
            init_is_scalar=True,
        )

    @classmethod
    def from_csv(cls, path, n_genes, n_cpgs):
        """
        Load one row per site, filling the (n_genes, n_cpgs) arrays in row-major order. Required
        columns are the base rates; site_id, site_class and the *_div columns are optional.
        """
        df = pd.read_csv(path)
        missing = [c for c in BASE_FIELDS if c not in df.columns]
        if missing:
            raise ValueError(f"{path}: missing required column(s) {missing}")
        expected = n_genes * n_cpgs
        if len(df) != expected:
            raise ValueError(
                f"{path}: has {len(df)} rows but n_genes * n_cpgs = {n_genes} * {n_cpgs} = {expected}"
            )
        shape = (n_genes, n_cpgs)
        kwargs = {}
        for name in PROB_FIELDS:
            if name in df.columns:
                kwargs[name] = df[name].to_numpy(dtype=float).reshape(shape)
        if "site_class" in df.columns:
            kwargs["site_class"] = df["site_class"].to_numpy(dtype=float).reshape(shape)
        # A constant init_meth column means the same as the scalar --init-meth flag
        init = kwargs["init_meth"]
        kwargs["init_is_scalar"] = bool(np.all(init == init.flat[0]))
        return cls(**kwargs)

    def to_csv(self, path):
        data = {"site_id": np.arange(self.n_sites)}
        for name in BASE_FIELDS:
            data[name] = getattr(self, name).ravel()
        data["site_class"] = self.site_class.ravel()
        for name in DIV_FIELDS:
            data[name] = getattr(self, name).ravel()
        pd.DataFrame(data, columns=CSV_COLUMNS).to_csv(path, index=False)


# Generator ----------------------------------------------------------------------------------------


def _draw(spec, n, rng):
    """Draw n values from a distribution spec such as {"dist": "loguniform", "low": 1e-4, "high": 1e-1}."""
    dist = spec["dist"]
    if dist == "fixed":
        values = np.full(n, float(spec["value"]))
    elif dist == "uniform":
        values = rng.uniform(spec["low"], spec["high"], n)
    elif dist == "loguniform":
        values = np.exp(rng.uniform(np.log(spec["low"]), np.log(spec["high"]), n))
    elif dist == "lognormal":
        # Parameterised by the median and the sd of log(value)
        values = np.exp(rng.normal(np.log(spec["median"]), spec["sigma"], n))
    elif dist == "beta":
        values = rng.beta(spec["a"], spec["b"], n)
    else:
        raise ValueError(f"unknown distribution '{dist}'")
    return np.clip(values, 0.0, 1.0)


def _class_counts(fractions, n_sites):
    """Split n_sites across classes by largest remainder so counts sum exactly to n_sites."""
    fractions = np.asarray(fractions, dtype=float)
    fractions = fractions / fractions.sum()
    raw = fractions * n_sites
    counts = np.floor(raw).astype(int)
    order = np.argsort(-(raw - counts), kind="stable")
    counts[order[: n_sites - counts.sum()]] += 1
    return counts


def generate_profile(spec, n_genes, n_cpgs, seed):
    """
    Draw a RateProfile from a JSON-style spec:

        {"classes": [
            {"site_class": 1, "fraction": 191,
             "p_meth": {"dist": "loguniform", "low": 1e-3, "high": 1e-2},
             "p_unmeth": {"dist": "fixed", "value": 1e-6},
             "init_meth": {"dist": "beta", "a": 1, "b": 9}},
            {"site_class": 2, "fraction": 614, ...}
        ]}

    Required keys per class: fraction (a relative weight), p_meth, p_unmeth, init_meth. Optional:
    site_class (default 0), p_meth_clone / p_unmeth_clone (default: the cell values drawn for the
    same site), p_meth_div / p_unmeth_div (default 0) and p_meth_div_clone / p_unmeth_div_clone
    (default: the cell *_div values for the same site). Distributions: fixed (value), uniform
    (low, high), loguniform (low, high), lognormal (median, sigma of log), beta (a, b); values
    are clipped to [0, 1]. Sites are assigned to classes in a random order.
    """
    rng = np.random.default_rng(seed)
    n_sites = n_genes * n_cpgs
    classes = spec["classes"]
    counts = _class_counts([c["fraction"] for c in classes], n_sites)

    columns = {name: np.zeros(n_sites) for name in PROB_FIELDS}
    site_class = np.zeros(n_sites, dtype=int)
    positions = rng.permutation(n_sites)
    start = 0
    for cls_spec, count in zip(classes, counts):
        idx = positions[start : start + count]
        start += count
        site_class[idx] = cls_spec.get("site_class", 0)
        drawn = {}
        for name in ("p_meth", "p_unmeth", "init_meth"):
            drawn[name] = _draw(cls_spec[name], count, rng)
        for name, fallback in (("p_meth_clone", "p_meth"), ("p_unmeth_clone", "p_unmeth")):
            drawn[name] = (
                _draw(cls_spec[name], count, rng) if name in cls_spec else drawn[fallback]
            )
        for name in ("p_meth_div", "p_unmeth_div"):
            drawn[name] = _draw(cls_spec[name], count, rng) if name in cls_spec else 0.0
        for name, fallback in (
            ("p_meth_div_clone", "p_meth_div"),
            ("p_unmeth_div_clone", "p_unmeth_div"),
        ):
            drawn[name] = (
                _draw(cls_spec[name], count, rng) if name in cls_spec else drawn[fallback]
            )
        for name, values in drawn.items():
            columns[name][idx] = values

    shape = (n_genes, n_cpgs)
    return RateProfile(
        **{name: values.reshape(shape) for name, values in columns.items()},
        site_class=site_class.reshape(shape),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Per-CpG rate profile tools.")
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="Draw a rate-profile CSV from a JSON spec.")
    gen.add_argument("spec", type=Path, help="JSON spec describing site classes and distributions.")
    gen.add_argument("--seed", type=int, required=True, help="Seed for the rate draws.")
    gen.add_argument("--n-genes", type=int, required=True)
    gen.add_argument("--n-cpgs", type=int, required=True)
    gen.add_argument("-o", "--out", type=Path, required=True, help="Output CSV path.")
    args = parser.parse_args(argv)

    spec = json.loads(args.spec.read_text())
    profile = generate_profile(spec, args.n_genes, args.n_cpgs, args.seed)
    profile.to_csv(args.out)
    print(f"Wrote {profile.n_sites} sites to {args.out}")


if __name__ == "__main__":
    main()
