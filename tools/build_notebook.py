"""Builds notebook/regional_momentum.ipynb from regional_momentum/core.py plus the analysis cells below.

The strategy code lives in one place (core.py) and is used by both the notebook and the Streamlit app.
After changing core.py or this file, run:  python tools/build_notebook.py
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "regional_momentum" / "core.py"
OUT = ROOT / "notebook" / "regional_momentum.ipynb"

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

INTRO = """# Regional Momentum on the MSCI ACWI

**2270 Financial Modeling, Group Assignment**

- Moodle group: *TODO*
- Members: *TODO*
- Streamlit app: *TODO (URL)*
- GitHub repository: *TODO (URL)*

## Setup
Open this notebook in Google Colab and choose *Runtime > Run all*. No data files are needed: all data are
downloaded through APIs (Yahoo Finance via `yfinance`, FRED, SEC EDGAR). A full run takes a few minutes,
mostly for downloading the SEC holdings filings. Missing packages are installed in the first code cell.

## Strategy in one paragraph
The portfolio rebuilds the MSCI ACWI from six regional ETFs (SPY, IEV, EEM, EWJ, EWC, EPP). At every
rebalancing date (last trading day of June and December) it starts from the current ACWI region weights,
ranks the regions by their total return over the past 12 months and tilts the weights by
+6/+4/+2/-2/-4/-6 percentage points. No region may deviate by more than 50% from its ACWI weight. If a cap
binds, the opposite side is scaled down pro rata so that weights sum to 100%. Trading costs are 15 bp of
traded volume.

## Generative AI disclosure
*TODO: tools, purpose, how outputs were checked.*
"""

SETUP = """import importlib.util, subprocess, sys

missing = [pkg for pkg in ("yfinance", "statsmodels") if importlib.util.find_spec(pkg) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])"""

ANALYSIS = [
    md("""# Analysis
## 1. Parameters
Start and end dates are set explicitly. The start (30 June 2008) is the first rebalancing date for which an
ACWI holdings filing was already public (holdings as of 30 April 2008, filed 26 June 2008). The default
parameters are the base case of the prospectus; the Streamlit app uses the same defaults."""),
    code("""from pathlib import Path
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from IPython.display import display

pd.set_option("display.float_format", "{:.4f}".format)
OUTPUT = Path("output")
OUTPUT.mkdir(exist_ok=True)
SEC_CACHE = Path("sec_cache")

PARAMS = Params(
    start="2008-06-30",
    end="2026-10-31",
    rebalance="S",          # semi-annual: last trading day of June and December
    lookback_months=12,
    tilt_pp=6.0,
    max_rel_deviation=0.5,
    cost_bps=15.0,
)
PARAMS"""),
    md("""## 2. Data
- **ETF prices:** Yahoo Finance via `yfinance`, daily closing prices adjusted for splits and dividends
  (total return, USD). Running ETF costs (TER) are already deducted in these prices.
- **Risk-free rate:** 3-month US Treasury bill (FRED series DTB3), converted to a monthly return.
- **ACWI region weights:** quarterly holdings of the iShares MSCI ACWI ETF from SEC EDGAR (forms N-Q
  until 2018, NPORT-EX for April 2019, NPORT-P since October 2019), aggregated from countries to the six
  regions. A filing is only used once it was public (filing date before the rebalancing date)."""),
    code("""prices = load_prices(PARAMS.start, PARAMS.end)
rf = load_riskfree(PARAMS.start, PARAMS.end)
acwi_weights = load_acwi_region_weights(SEC_CACHE)

print(f"Prices: {prices.index[0].date()} to {prices.index[-1].date()}, {len(prices)} trading days, tickers: {list(prices.columns)}")
print(f"Risk-free rate: {rf.index[0].date()} to {rf.index[-1].date()}, mean {rf.mean() * 12:.2%} p.a.")
print(f"ACWI holdings filings: {len(acwi_weights)} reporting dates, "
      f"{acwi_weights.index[0].date()} to {acwi_weights.index[-1].date()}, forms: {acwi_weights['form'].value_counts().to_dict()}")
acwi_weights.to_csv(OUTPUT / "acwi_region_weights.csv")
display(acwi_weights[["filed", "form"] + REGIONS].iloc[::4])"""),
    md("""### ACWI region weights over time and quality of the drift approximation
Between a reporting date and a rebalancing date the weights are rolled forward with the regional ETF
returns. The check below rolls each reporting date forward to the next one and compares the result with
the weights actually reported (half the sum of absolute differences, i.e. the share of the portfolio that
would be misallocated)."""),
    code("""rp = region_prices(prices)
dates = acwi_weights.index
drift_error = pd.Series({
    cur: ((acwi_weights.loc[prev, REGIONS].astype(float) * rp.asof(cur) / rp.asof(prev))
          .pipe(lambda w: w / w.sum()) - acwi_weights.loc[cur, REGIONS].astype(float)).abs().sum() / 2
    for prev, cur in zip(dates[:-1], dates[1:])
}, name="drift error")
print(f"Drift error over one quarter: mean {drift_error.mean():.2%}, max {drift_error.max():.2%} ({drift_error.idxmax().date()})")

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.stackplot(dates, acwi_weights[REGIONS].T.astype(float), labels=REGIONS)
ax.set_title("MSCI ACWI region weights (iShares MSCI ACWI ETF holdings, SEC filings)")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_ylim(0, 1)
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
fig.tight_layout()
fig.savefig(OUTPUT / "fig_acwi_weights.png", dpi=200)
plt.show()"""),
    md("""## 3. Backtest
Three portfolios are compared, each starting at 1:
- **Regional Momentum:** the strategy.
- **Passive replica:** the same six ETFs at ACWI weights, without tilt. It shows how well the building
  blocks replicate the index and isolates the contribution of the momentum signal.
- **MSCI ACWI (ACWI ETF):** the benchmark.

At each rebalancing date the signal and the target weights use the closing prices of that day, and the
trades are executed at that close. Between rebalancing dates weights drift with prices."""),
    code("""res = run_all(prices, acwi_weights, rf, PARAMS)
navs, returns, table = res["navs"], res["returns"], res["table"]
print(f"Sample of monthly returns: {returns.index[0]:%b %Y} to {returns.index[-1]:%b %Y} ({len(returns)} months)")
table.to_csv(OUTPUT / "performance_table.csv")
pct = ["Average return p.a.", "Geometric return p.a.", "Volatility p.a.", "Alpha p.a.",
       "Tracking error p.a.", "Max drawdown", "Turnover p.a.", "Trading costs p.a."]
display(table.T.style.format("{:.2%}", subset=pd.IndexSlice[pct, :]).format("{:.2f}", subset=pd.IndexSlice[[i for i in table.columns if i not in pct], :]))"""),
    md("""**Calculation assumptions.** Monthly returns from month-end values. Average return = arithmetic mean
x 12; volatility = standard deviation x sqrt(12). Sharpe ratio = annualised mean excess return over the
T-bill rate / annualised volatility of excess returns. Alpha and beta from the OLS regression
(r - rf) = alpha + beta (r_ACWI - rf) + e on monthly excess returns; alpha is annualised x 12, beta is not
annualised. Information ratio = annualised mean active return (r - r_ACWI) / tracking error (annualised
standard deviation of active returns). Trading costs are deducted from the portfolio value at each
rebalancing (the initial purchase is not charged)."""),
    code("""fig, ax = plt.subplots(figsize=(10, 5))
for col, style in zip(navs.columns, ["-", "--", "-"]):
    ax.plot(navs.index, navs[col], style, label=col, lw=1.6 if col == "Regional Momentum" else 1.2)
ax.axhline(1, color="grey", lw=0.6)
ax.set_title("Cumulative performance (total return, USD, start = 1)")
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT / "fig_cumulative_performance.png", dpi=200)
plt.show()

active = (navs["Regional Momentum"] / navs["MSCI ACWI (ACWI ETF)"])
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(active.index, active, color="black", lw=1)
ax.axhline(1, color="grey", lw=0.6)
ax.set_title("Relative performance: Regional Momentum / MSCI ACWI")
fig.tight_layout()
fig.savefig(OUTPUT / "fig_relative_performance.png", dpi=200)
plt.show()"""),
    md("### Active weights over time (strategy minus passive replica)"),
    code("""active_w = res["strategy"].weights - res["passive"].weights
fig, ax = plt.subplots(figsize=(10, 4))
for r in REGIONS:
    ax.plot(active_w.index, active_w[r], label=r, lw=1)
ax.axhline(0, color="grey", lw=0.6)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_title("Active weights vs. ACWI region weights")
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
fig.tight_layout()
fig.savefig(OUTPUT / "fig_active_weights.png", dpi=200)
plt.show()"""),
    md("### Rebalancing log and an example of the allocation rule (latest rebalancing date)"),
    code("""trades = res["strategy"].trades
trades.to_csv(OUTPUT / "rebalancing_log.csv")
last = trades.iloc[-1]
example = pd.DataFrame({
    "ACWI weight": [last[f"base {r}"] for r in REGIONS],
    f"{PARAMS.lookback_months}m return": [last[f"signal {r}"] for r in REGIONS],
    "Target weight": [last[f"target {r}"] for r in REGIONS],
}, index=REGIONS)
example["Active weight"] = example["Target weight"] - example["ACWI weight"]
example["Rank"] = example[f"{PARAMS.lookback_months}m return"].rank(ascending=False).astype(int)
print(f"Rebalancing on {trades.index[-1].date()}, ACWI weights from holdings as of {last['anchor'].date()}, turnover {last['turnover']:.1%}")
example.to_csv(OUTPUT / "example_rebalancing.csv")
display(example.sort_values("Rank").style.format("{:.2%}", subset=["ACWI weight", f"{PARAMS.lookback_months}m return", "Target weight", "Active weight"]))
display(trades[["anchor", "turnover", "cost"]].style.format({"turnover": "{:.1%}", "cost": "{:.3%}"}))"""),
    md("### Calendar-year returns"),
    code("""annual = navs.resample("YE").last().pct_change()
annual.iloc[0] = navs.resample("YE").last().iloc[0] / navs.iloc[0] - 1  # 2008: from 30 June
annual.index = annual.index.year
annual["Active (vs. ACWI)"] = annual["Regional Momentum"] - annual["MSCI ACWI (ACWI ETF)"]
annual.to_csv(OUTPUT / "annual_returns.csv")
display(annual.style.format("{:.2%}"))
print(f"Strategy beat the ACWI in {(annual['Active (vs. ACWI)'] > 0).sum()} of {len(annual)} calendar years")"""),
    md("""## 4. Sensitivity analysis
Each row changes one assumption of the base case and keeps everything else fixed."""),
    code("""grid = {
    "rebalance": ["Q", "S", "A"],
    "lookback_months": [6, 12],
    "tilt_pp": [0.0, 3.0, 6.0, 10.0],
    "cost_bps": [0.0, 15.0, 50.0],
}
sens = sensitivity(prices, acwi_weights, rf, PARAMS, grid)
sens.to_csv(OUTPUT / "sensitivity.csv")
display(sens.style.format({c: "{:.2%}" for c in ["Average return p.a.", "Volatility p.a.", "Alpha p.a.", "Turnover p.a."]})
                  .format({c: "{:.2f}" for c in ["Sharpe ratio", "Beta", "Information ratio"]}))"""),
    md("""## 5. Consistency with the Streamlit app
The app imports the same functions from `regional_momentum/core.py`. With its default settings it must
show the numbers below."""),
    code("""default_metrics = table.round(6).to_dict(orient="index")
(OUTPUT / "default_metrics.json").write_text(json.dumps({"params": PARAMS.__dict__, "metrics": default_metrics}, indent=2))
pd.DataFrame(default_metrics).T[["Average return p.a.", "Volatility p.a.", "Sharpe ratio", "Alpha p.a.", "Beta", "Information ratio"]]"""),
]


def module_cells() -> list:
    cells, title, lines = [], None, []
    for line in CORE.read_text(encoding="utf-8").splitlines():
        if line.startswith("# %% "):
            if title:
                cells += [md(f"### Code: {title}"), code("\n".join(lines).strip())]
            title, lines = line[5:].strip(), []
        else:
            lines.append(line)
    cells += [md(f"### Code: {title}"), code("\n".join(lines).strip())]
    return cells


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["colab"] = {"provenance": []}
    nb.cells = [
        md(INTRO),
        code(SETUP),
        md("# Strategy code\nThe same code is used by the Streamlit app (`regional_momentum/core.py` in the repository)."),
        *module_cells(),
        *ANALYSIS,
    ]
    OUT.parent.mkdir(exist_ok=True)
    nbf.write(nb, OUT)
    print(f"Written {OUT.relative_to(ROOT)} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
