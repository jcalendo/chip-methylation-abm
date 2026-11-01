"""
Capture golden outputs. Run once at the phase boundary named in the engineering plan:

    uv run python tests/make_golden.py g1   # after Phase 0
    uv run python tests/make_golden.py g2   # after Phase 2

Never regenerate an existing golden to make a failing test pass.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helpers import GOLDEN_1, run_config, save_golden

SETS = {"g1": GOLDEN_1}

try:
    from helpers import GOLDEN_2

    SETS["g2"] = GOLDEN_2
except ImportError:
    pass


if __name__ == "__main__":
    for set_name in sys.argv[1:]:
        for name, params in SETS[set_name].items():
            df, digest = run_config(params)
            save_golden(name, df, digest)
            print(f"saved {name}: {len(df)} rows, hash {digest[:12]}")
