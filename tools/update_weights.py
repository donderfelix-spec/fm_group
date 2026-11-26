"""Refreshes data/acwi_region_weights.csv, the snapshot the Streamlit app reads so it starts quickly.

The notebook always downloads the filings itself; the snapshot is produced by the same function.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from regional_momentum.core import load_acwi_region_weights  # noqa: E402

out = ROOT / "data" / "acwi_region_weights.csv"
out.parent.mkdir(exist_ok=True)
weights = load_acwi_region_weights(ROOT / "sec_cache")
weights.to_csv(out)
print(f"Written {out.relative_to(ROOT)}: {len(weights)} reporting dates, last {weights.index[-1].date()}")
