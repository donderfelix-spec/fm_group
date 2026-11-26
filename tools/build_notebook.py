"""Builds notebook/regional_momentum.ipynb from regional_momentum/core.py plus the analysis cells below.

The strategy code lives in one file (core.py) and is used by both the notebook and the Streamlit app.
Each "# %% Title" section of core.py becomes a code cell; the comment lines directly below the title
become the text above that cell. After changing core.py or this file, run:

    python tools/build_notebook.py
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "regional_momentum" / "core.py"
OUT = ROOT / "notebook" / "regional_momentum.ipynb"

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

INTRO = """# Regional Momentum on the MSCI ACWI

**2270 Financial Modeling, Group Assignment**

- Moodle group: *TODO*
- Members: *TODO*
- Streamlit app: *TODO (URL)*
- GitHub repository: *TODO (URL)*

## Setup
Open this notebook in Google Colab and choose *Runtime > Run all*. No data files are needed: all data are
downloaded through APIs (Yahoo Finance via `yfinance`, FRED, SEC EDGAR). A full run takes a few minutes,
mostly for downloading the SEC holdings reports. Missing packages are installed in the first code cell.

## The strategy in short
The portfolio rebuilds the MSCI ACWI from six regional ETFs (SPY, IEV, EEM, EWJ, EWC, EPP). On the last
trading day of June and December it starts from the current ACWI region weights, ranks the regions by
their total return over the past 12 months and tilts the weights by +6/+4/+2/-2/-4/-6 percentage points.
No region may deviate by more than 50% from its ACWI weight; if a limit binds, the other side is scaled
down so that the weights still add up to 100%. Trading costs are 15 bp of the traded volume.

## Notebook structure
1. Strategy code (also used by the Streamlit app)
2. Parameters and data
3. Backtest and performance measures
4. Figures and tables for the prospectus
5. Sensitivity analysis
6. Check against the app

## Generative AI disclosure
*TODO: tools, purpose, how outputs were checked.*
"""

SETUP = """# install packages that are missing (Colab already has most of them)
import importlib.util
import subprocess
import sys

for package in ["yfinance", "statsmodels"]:
    if importlib.util.find_spec(package) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])"""

ANALYSIS = [
    md("""# 2. Parameters and data
The start and end dates are set explicitly. The start (30 June 2008) is the first rebalancing date on
which an ACWI holdings report was already public (holdings as of 30 April 2008, published 26 June 2008).
These are the base-case settings of the prospectus; the app uses the same defaults."""),
    code("""from pathlib import Path
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from IPython.display import display

OUTPUT = Path("output")      # tables and figures for the prospectus
OUTPUT.mkdir(exist_ok=True)
SEC_CACHE = Path("sec_cache")  # downloaded SEC reports, so a second run is fast

PARAMS = Params(
    start="2008-06-30",
    end="2026-10-31",
    rebalance="S",
    lookback_months=12,
    tilt_pp=6.0,
    max_rel_deviation=0.5,
    cost_bps=15.0,
)
PARAMS"""),
    md("""**Data sources**
- *ETF prices:* Yahoo Finance via `yfinance`, daily closing prices adjusted for splits and dividends
  (total return in USD). The running ETF costs (TER) are already deducted in these prices.
- *Risk-free rate:* 3-month US Treasury bill (FRED series DTB3), converted to a monthly return.
- *ACWI region weights:* quarterly holdings of the iShares MSCI ACWI ETF from SEC EDGAR (form N-Q until
  2018, NPORT-EX for April 2019, NPORT-P since October 2019). A report is only used once it was public
  (filing date before the rebalancing date)."""),
    code("""prices = load_prices(PARAMS.start, PARAMS.end)
rf = load_riskfree(PARAMS.start, PARAMS.end)
acwi_weights = load_acwi_region_weights(SEC_CACHE)

print("Prices:", prices.index[0].date(), "to", prices.index[-1].date(), "-", len(prices), "trading days")
print("Risk-free rate:", rf.index[0].date(), "to", rf.index[-1].date(), f"- average {rf.mean() * 12:.2%} p.a.")
print("ACWI reports:", len(acwi_weights), "reporting dates from", acwi_weights.index[0].date(), "to", acwi_weights.index[-1].date())
print("Forms:", acwi_weights["form"].value_counts().to_dict())

acwi_weights.to_csv(OUTPUT / "acwi_region_weights.csv")
acwi_weights[["filed", "form"] + REGIONS].iloc[::4].round(3)"""),
    md("""### ACWI region weights over time, and how good the roll-forward is
Between a reporting date and a rebalancing date we roll the weights forward with the ETF returns. To
check this, we roll every report forward to the next reporting date (6 months later until 2019, 3 months
since then) and compare with the weights that were actually reported. The error is the share of the
portfolio that would be in the wrong region. In the backtest the roll-forward covers only about 2 months."""),
    code("""errors = {}
dates = acwi_weights.index
for previous, current in zip(dates[:-1], dates[1:]):
    rolled, _ = acwi_weights_on(current, acwi_weights.loc[:previous], region_prices(prices))
    actual = acwi_weights.loc[current, REGIONS].astype(float)
    errors[current] = (rolled - actual).abs().sum() / 2
errors = pd.Series(errors)
print(f"Roll-forward error to the next report: average {errors.mean():.2%}, largest {errors.max():.2%} ({errors.idxmax().date()})")

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.stackplot(dates, acwi_weights[REGIONS].T.astype(float), labels=REGIONS)
ax.set_title("MSCI ACWI region weights (iShares MSCI ACWI ETF holdings, SEC reports)")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_ylim(0, 1)
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
fig.tight_layout()
fig.savefig(OUTPUT / "fig_acwi_weights.png", dpi=200)
plt.show()"""),
    md("""# 3. Backtest and performance measures
Three portfolios, each starting at 1:
- **Regional Momentum:** the strategy.
- **Passive replica:** the same six ETFs at ACWI weights, without tilt. It shows how well the six ETFs
  replicate the index, so the effect of the momentum signal can be measured on its own.
- **MSCI ACWI (ACWI ETF):** the benchmark."""),
    code("""result = run_all(prices, acwi_weights, rf, PARAMS)
navs = result["navs"]
returns = result["returns"]
table = result["table"]
table.to_csv(OUTPUT / "performance_table.csv")

print("Monthly returns from", returns.index[0].strftime("%b %Y"), "to", returns.index[-1].strftime("%b %Y"), f"({len(returns)} months)")

percent_columns = ["Average return p.a.", "Geometric return p.a.", "Volatility p.a.", "Alpha p.a.",
                   "Tracking error p.a.", "Max drawdown", "Turnover p.a.", "Trading costs p.a."]
pretty = table.copy()
for column in pretty.columns:
    number_format = "{:.2%}" if column in percent_columns else "{:.2f}"
    pretty[column] = table[column].map(lambda x: "" if pd.isna(x) else number_format.format(x))
pretty.T"""),
    md("""**Calculation assumptions**
- Monthly returns from month-end values (July 2008 to October 2026).
- Average return = arithmetic mean x 12; volatility = standard deviation x sqrt(12).
- Sharpe ratio = annualised average excess return over the T-bill rate / annualised volatility of the
  excess returns.
- Alpha and beta from the regression (r - rf) = alpha + beta x (r_ACWI - rf) + error on monthly excess
  returns. Alpha is multiplied by 12, beta is not annualised.
- Information ratio = annualised average active return (r - r_ACWI) / tracking error (annualised standard
  deviation of the active returns).
- Trading costs are deducted at each rebalancing; the initial purchase is not charged."""),
    md("# 4. Figures and tables for the prospectus"),
    code("""fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(navs.index, navs["Regional Momentum"], label="Regional Momentum", lw=1.6)
ax.plot(navs.index, navs["Passive replica"], "--", label="Passive replica", lw=1.2)
ax.plot(navs.index, navs["MSCI ACWI (ACWI ETF)"], label="MSCI ACWI (ACWI ETF)", lw=1.2)
ax.axhline(1, color="grey", lw=0.6)
ax.set_title("Cumulative performance (total return, USD, start = 1)")
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT / "fig_cumulative_performance.png", dpi=200)
plt.show()

relative = navs["Regional Momentum"] / navs["MSCI ACWI (ACWI ETF)"]
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(relative.index, relative, color="black", lw=1)
ax.axhline(1, color="grey", lw=0.6)
ax.set_title("Relative performance: Regional Momentum / MSCI ACWI")
fig.tight_layout()
fig.savefig(OUTPUT / "fig_relative_performance.png", dpi=200)
plt.show()"""),
    md("### Active weights over time (strategy minus passive replica)"),
    code("""active_weights = result["strategy"].weights - result["passive"].weights

fig, ax = plt.subplots(figsize=(10, 4))
for region in REGIONS:
    ax.plot(active_weights.index, active_weights[region], label=region, lw=1)
ax.axhline(0, color="grey", lw=0.6)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_title("Active weights vs. ACWI region weights")
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
fig.tight_layout()
fig.savefig(OUTPUT / "fig_active_weights.png", dpi=200)
plt.show()"""),
    md("### Example of the allocation rule: the latest rebalancing date"),
    code("""trades = result["strategy"].trades
trades.to_csv(OUTPUT / "rebalancing_log.csv")
last = trades.iloc[-1]

example = pd.DataFrame(index=REGIONS)
example["ACWI weight"] = [last[f"base {region}"] for region in REGIONS]
example["Momentum"] = [last[f"signal {region}"] for region in REGIONS]
example["Rank"] = example["Momentum"].rank(ascending=False).astype(int)
example["Target weight"] = [last[f"target {region}"] for region in REGIONS]
example["Active weight"] = example["Target weight"] - example["ACWI weight"]
example = example.sort_values("Rank")
example.to_csv(OUTPUT / "example_rebalancing.csv")

print("Rebalancing on", trades.index[-1].date(), "- ACWI weights from the report as of", last["anchor"].date(),
      f"- turnover {last['turnover']:.1%}")
example.round(4)"""),
    md("### Turnover and trading costs per rebalancing date"),
    code("""trades[["anchor", "turnover", "cost"]].round(5)"""),
    md("### Calendar-year returns"),
    code("""year_end_values = navs.resample("YE").last()
annual = year_end_values.pct_change()
annual.iloc[0] = year_end_values.iloc[0] / navs.iloc[0] - 1  # 2008 starts on 30 June
annual.index = annual.index.year
annual["Active (vs. ACWI)"] = annual["Regional Momentum"] - annual["MSCI ACWI (ACWI ETF)"]
annual.to_csv(OUTPUT / "annual_returns.csv")

years_ahead = (annual["Active (vs. ACWI)"] > 0).sum()
print(f"The strategy beat the ACWI in {years_ahead} of {len(annual)} calendar years")
annual.round(4)"""),
    md("""# 5. Sensitivity analysis
Each row changes one assumption of the base case and keeps everything else as it is."""),
    code("""grid = {
    "rebalance": ["Q", "S", "A"],
    "lookback_months": [6, 12],
    "tilt_pp": [0.0, 3.0, 6.0, 10.0],
    "cost_bps": [0.0, 15.0, 50.0],
}
sens = sensitivity(prices, acwi_weights, rf, PARAMS, grid)
sens.to_csv(OUTPUT / "sensitivity.csv")
sens.round(4)"""),
    md("""# 6. Check against the app
The Streamlit app imports the same functions from `regional_momentum/core.py`. With its default settings
it must show the numbers below."""),
    code("""main_numbers = table[["Average return p.a.", "Volatility p.a.", "Sharpe ratio", "Alpha p.a.", "Beta", "Information ratio"]]
(OUTPUT / "default_metrics.json").write_text(json.dumps(main_numbers.round(6).to_dict(orient="index"), indent=2))
main_numbers.round(4)"""),
]


def module_cells():
    """One markdown and one code cell per '# %% Title' section of core.py."""
    cells = []
    sections = CORE.read_text(encoding="utf-8").split("# %% ")[1:]
    for section in sections:
        lines = section.splitlines()
        title = lines[0].strip()
        description = []
        i = 1
        while i < len(lines) and lines[i].startswith("# "):
            description.append(lines[i][2:])
            i += 1
        cells.append(md(f"### Code: {title}\n" + " ".join(description)))
        cells.append(code("\n".join(lines[i:]).strip()))
    return cells


def main():
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["colab"] = {"provenance": []}
    nb.cells = [
        md(INTRO),
        code(SETUP),
        md("# 1. Strategy code\nThe same code is used by the Streamlit app (`regional_momentum/core.py` in the repository)."),
        *module_cells(),
        *ANALYSIS,
    ]
    OUT.parent.mkdir(exist_ok=True)
    nbf.write(nb, OUT)
    print(f"Written {OUT.relative_to(ROOT)} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
