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

- Moodle group: BT
- Members: Renke Löschen (82016), Matti Raether (81317), Ben Bormann (78448), Felix Donder (75246),
  Leopold Querfurth (78371), Clara Liehr (80350)
- Streamlit app: *TODO (not yet deployed)*
- GitHub repository: https://github.com/donderfelix-spec/fm_group

## Setup
Open this notebook in Google Colab and choose *Runtime > Run all*. No data files are needed: all data are
downloaded through APIs (Yahoo Finance via `yfinance`, FRED, SEC EDGAR). A full run takes a few minutes,
mostly for downloading the SEC holdings reports. Missing packages are installed in the first code cell.

## The strategy in short
The portfolio rebuilds the MSCI ACWI from six regional iShares UCITS ETFs (USA, Europe, Emerging Markets,
Japan, Canada, Pacific ex Japan), bought on Xetra in euros. On the last trading day of June and December it
starts from the current MSCI ACWI region weights, ranks the regions by their total return over the past 12
months and tilts the weights by +6/+4/+2/-2/-4/-6 percentage points. No region may deviate by more than 50%
from its MSCI ACWI weight; if a limit binds, the other side is scaled down so that the weights still add up
to 100%. Trading costs are 15 bp of the traded volume.

Our investors are Portuguese retail investors. They can only buy UCITS funds, and they measure returns in
euros, so the whole backtest uses UCITS ETFs and euro prices (no currency hedging).

## Notebook structure
1. Strategy code (also used by the Streamlit app)
2. Parameters, funds and data, including checks of the MSCI ACWI weights against iShares
3. Backtest and performance measures
4. Figures and tables for the prospectus
5. Risk class (SRI)
6. Sensitivity analysis
7. Check against the app

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
    md("""# 2. Parameters, funds and data
The start and end dates are set explicitly. The backtest starts on the first trading day of 2013. All six
regional UCITS ETFs have prices from 2010 on (needed for the first 12-month signal). The benchmark UCITS ETF
was launched in October 2011, but it was rarely traded on Xetra in 2012 (78 days without a price change,
prices up to 5% away from fair value), so we start once its prices are reliable. These are the base-case
settings of the prospectus; the app uses the same defaults."""),
    code("""from pathlib import Path
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from IPython.display import display

OUTPUT = Path("output")      # tables and figures for the prospectus
OUTPUT.mkdir(exist_ok=True)
SEC_CACHE = Path("sec_cache")  # downloaded SEC reports, so a second run is fast

PARAMS = Params(
    start="2013-01-01",
    end="2026-10-31",
    rebalance="S",
    lookback_months=12,
    tilt_pp=6.0,
    max_rel_deviation=0.5,
    cost_bps=15.0,
)
PARAMS"""),
    md("""### The funds
All funds are Irish-domiciled iShares UCITS ETFs with accumulating share classes (dividends are
reinvested inside the fund), registered for sale in Portugal. Facts from the iShares product pages
(26 November 2026); TERs should be re-checked in the key information documents before investing."""),
    code("""funds = FUND_FACTS.copy()
funds["TER"] = funds["TER"].map("{:.2%}".format)
funds.to_csv(OUTPUT / "fund_table.csv")
funds"""),
    md("""**Data sources**
- *ETF prices:* Yahoo Finance via `yfinance`, daily closing prices of the Xetra listings in EUR, for the six
  regional ETFs and the benchmark (iShares MSCI ACWI UCITS ETF). The share classes are accumulating, so the
  prices are total-return prices; the TER is already deducted. Because all prices are in EUR, no currency
  conversion is needed.
- *Risk-free rate:* 3-month Euribor, monthly average (European Central Bank data portal), converted to a
  monthly return.
- *MSCI ACWI region weights:* quarterly holdings of the US-listed iShares MSCI ACWI ETF, which tracks the
  same index, from SEC EDGAR (form N-Q until 2018, NPORT-EX for April 2019, NPORT-P since October 2019).
  A report is only used once it was public (filing date before the rebalancing date)."""),
    code("""prices = load_prices(PARAMS.start, PARAMS.end)
rf = load_riskfree(PARAMS.start, PARAMS.end)
acwi_weights = load_acwi_region_weights(SEC_CACHE)

print("Prices:", prices.index[0].date(), "to", prices.index[-1].date(), "-", len(prices), "Xetra trading days")
print("Risk-free rate:", rf.index[0].date(), "to", rf.index[-1].date(), f"- average {rf.mean() * 12:.2%} p.a.")
print("MSCI ACWI reports:", len(acwi_weights), "reporting dates from", acwi_weights.index[0].date(), "to", acwi_weights.index[-1].date())
print("Forms:", acwi_weights["form"].value_counts().to_dict())

acwi_weights.to_csv(OUTPUT / "acwi_region_weights.csv")
acwi_weights[["filed", "form"] + REGIONS].iloc[::4].round(3)"""),
    md("""### MSCI ACWI region weights over time, and how good the roll-forward is
Between a reporting date and a rebalancing date we roll the weights forward with the ETF returns. To
check this, we roll every report forward to the next reporting date (6 months later until 2019, 3 months
since then) and compare with the weights that were actually reported. The error is the share of the
portfolio that would be in the wrong region. In the backtest the roll-forward covers only about 2 months."""),
    code("""errors = {}
dates = acwi_weights.index[acwi_weights.index >= prices.index[0]]  # reports within the price history
for previous, current in zip(dates[:-1], dates[1:]):
    rolled, _ = acwi_weights_on(current, acwi_weights.loc[:previous], region_prices(prices))
    actual = acwi_weights.loc[current, REGIONS].astype(float)
    errors[current] = (rolled - actual).abs().sum() / 2
errors = pd.Series(errors)
print(f"Roll-forward error to the next report: average {errors.mean():.2%}, largest {errors.max():.2%} ({errors.idxmax().date()})")

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.stackplot(acwi_weights.index, acwi_weights[REGIONS].T.astype(float), labels=REGIONS)
ax.set_title("MSCI ACWI region weights (iShares MSCI ACWI ETF holdings, SEC reports)")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_ylim(0, 1)
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
fig.tight_layout()
fig.savefig(OUTPUT / "fig_acwi_weights.png", dpi=200)
plt.show()"""),
    md("""### Spot checks against iShares
iShares publishes the full holdings list of the ACWI ETF on its website. Old versions of this list are kept
by the Internet Archive (web.archive.org). For four dates we compare the region weights in the iShares
list with our SEC weights, rolled forward to the same date. The 2017 list still classifies companies by
their legal domicile (e.g. Bermuda), not by MSCI country, so a larger difference is expected there."""),
    code("""import gzip
import io
import requests

ISHARES_ARCHIVE = [
    "https://web.archive.org/web/20170218043220id_/https://www.ishares.com/us/products/239600/ishares-msci-acwi-etf/1467271812596.ajax?fileType=csv&fileName=ACWI_holdings&dataType=fund",
    "https://web.archive.org/web/20200506013502id_/https://www.ishares.com/us/products/239600/ishares-msci-acwi-etf/1467271812596.ajax?fileType=csv&fileName=ACWI_holdings&dataType=fund",
    "https://web.archive.org/web/20230328222332id_/https://www.ishares.com/us/products/239600/ishares-msci-acwi-etf/1467271812596.ajax?fileType=csv&fileName=ACWI_holdings&dataType=fund",
    "https://web.archive.org/web/20261118090456id_/https://www.ishares.com/us/products/239600/ishares-msci-acwi-etf/latest-holdings.csv",
]


def ishares_region_weights(url):
    \"\"\"Date and region weights of an archived iShares holdings list.\"\"\"
    content = requests.get(url, timeout=120).content
    if content[:2] == b"\\x1f\\x8b":  # some archived files are stored gzip-compressed
        content = gzip.decompress(content)
    lines = content.decode("utf-8-sig").splitlines()

    as_of_line = [line for line in lines if line.startswith("Fund Holdings as of")][0]
    as_of = pd.to_datetime(as_of_line.split(",", 1)[1].strip().strip('"'))

    header = [i for i, line in enumerate(lines) if line.startswith("Ticker,")][0]
    holdings = pd.read_csv(io.StringIO("\\n".join(lines[header:])))
    holdings = holdings[holdings["Asset Class"] == "Equity"]
    country_column = "Location" if "Location" in holdings.columns else "Country"
    market_value = pd.to_numeric(holdings["Market Value"].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")

    totals = {}
    for country, value in zip(holdings[country_column], market_value):
        if pd.notna(value):
            region = region_of_country(str(country), as_of)
            totals[region] = totals.get(region, 0.0) + value
    weights = pd.Series(totals).reindex(REGIONS).fillna(0.0)
    return as_of, weights / weights.sum()


rp = region_prices(prices)
checks = []
for url in ISHARES_ARCHIVE:
    try:
        as_of, ishares = ishares_region_weights(url)
    except Exception as error:  # the archive is an outside service; one failed download should not stop the notebook
        print("Could not load", url[:60], "-", error)
        continue
    # latest SEC report before that date, rolled forward to the same date
    report_date = acwi_weights.index[acwi_weights.index <= as_of].max()
    sec = acwi_weights.loc[report_date, REGIONS].astype(float) * rp.asof(as_of) / rp.asof(report_date)
    sec = sec / sec.sum()
    for region in REGIONS:
        checks.append({"iShares list": as_of.date(), "SEC report": report_date.date(), "Region": region,
                       "iShares": ishares[region], "SEC (rolled forward)": sec[region],
                       "Difference": sec[region] - ishares[region]})

checks = pd.DataFrame(checks)
checks.to_csv(OUTPUT / "ishares_checks.csv", index=False)
summary = checks.groupby(["iShares list", "SEC report"])["Difference"].apply(lambda d: d.abs().sum() / 2)
print("Share of the portfolio in a different region than in the iShares list:")
print(summary.map("{:.2%}".format).to_string())
checks.pivot(index="Region", columns="iShares list", values="Difference").reindex(REGIONS).round(4)"""),
    md("""# 3. Backtest and performance measures
Three portfolios, each starting at 1:
- **Regional Momentum:** the strategy.
- **Passive replica:** the same six ETFs at MSCI ACWI weights, without tilt. It shows how well the six ETFs
  replicate the index, so the effect of the momentum signal can be measured on its own.
- **MSCI ACWI (iShares UCITS ETF):** the benchmark.

All values are in euros."""),
    code("""result = run_all(prices, acwi_weights, rf, PARAMS)
navs = result["navs"]
returns = result["returns"]
table = result["table"]
table.to_csv(OUTPUT / "performance_table.csv")

print("Monthly returns from", returns.index[0].strftime("%b %Y"), "to", returns.index[-1].strftime("%b %Y"), f"({len(returns)} months)")

percent_columns = ["Average return p.a.", "Geometric return p.a.", "Volatility p.a.", "Alpha p.a.",
                   "Tracking error p.a.", "Max drawdown", "Turnover p.a.", "Trading costs p.a."]


def pretty(table):
    \"\"\"Performance table with percentages, metrics as rows.\"\"\"
    shown = table.copy()
    for column in shown.columns:
        number_format = "{:.2%}" if column in percent_columns else "{:.2f}"
        shown[column] = table[column].map(lambda x: "" if pd.isna(x) else number_format.format(x))
    return shown.T


pretty(table)"""),
    md("""**Calculation assumptions**
- All values in EUR, without currency hedging. The momentum signal uses the euro prices; converting all
  regions with the same exchange rate would not change their ranking.
- Monthly returns from month-end values (January 2013 to October 2026).
- Average return = arithmetic mean x 12; volatility = standard deviation x sqrt(12).
- Sharpe ratio = annualised average excess return over the 3-month Euribor / annualised volatility of the
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
ax.plot(navs.index, navs[BENCHMARK_NAME], label=BENCHMARK_NAME, lw=1.2)
ax.axhline(1, color="grey", lw=0.6)
ax.set_title("Cumulative performance (total return in EUR, start = 1)")
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT / "fig_cumulative_performance.png", dpi=200)
plt.show()

relative = navs["Regional Momentum"] / navs[BENCHMARK_NAME]
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
ax.set_title("Active weights vs. MSCI ACWI region weights")
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
fig.tight_layout()
fig.savefig(OUTPUT / "fig_active_weights.png", dpi=200)
plt.show()"""),
    md("### Example of the allocation rule: the latest rebalancing date"),
    code("""trades = result["strategy"].trades
trades.to_csv(OUTPUT / "rebalancing_log.csv")
last = trades.iloc[-1]

example = pd.DataFrame(index=REGIONS)
example["MSCI ACWI weight"] = [last[f"base {region}"] for region in REGIONS]
example["Momentum"] = [last[f"signal {region}"] for region in REGIONS]
example["Rank"] = example["Momentum"].rank(ascending=False).astype(int)
example["Target weight"] = [last[f"target {region}"] for region in REGIONS]
example["Active weight"] = example["Target weight"] - example["MSCI ACWI weight"]
example = example.sort_values("Rank")
example.to_csv(OUTPUT / "example_rebalancing.csv")

print("Rebalancing on", trades.index[-1].date(), "- MSCI ACWI weights from the report as of", last["anchor"].date(),
      f"- turnover {last['turnover']:.1%}")
example.round(4)"""),
    md("### Turnover and trading costs per rebalancing date"),
    code("""trades[["anchor", "turnover", "cost"]].round(5)"""),
    md("""### Running costs (TER)
The TERs are already deducted in the fund prices, so they are part of all results above. For the
prospectus we show the weighted TER of the portfolio at the latest rebalancing date."""),
    code("""ter = FUND_FACTS["TER"]
costs = pd.DataFrame({
    "Weight": example["Target weight"],
    "TER": ter.reindex(example.index),
})
costs["Weighted TER"] = costs["Weight"] * costs["TER"]

passive_ter = sum(example["MSCI ACWI weight"][region] * ter[region] for region in REGIONS)
print(f"Weighted TER: Regional Momentum {costs['Weighted TER'].sum():.3%}, passive replica {passive_ter:.3%}, "
      f"benchmark fund {ter['Benchmark']:.2%}")
print(f"Trading costs from rebalancing: {table.loc['Regional Momentum', 'Trading costs p.a.']:.3%} p.a.")
costs.to_csv(OUTPUT / "weighted_ter.csv")
costs.round(5)"""),
    md("### Calendar-year returns (EUR)"),
    code("""year_end_values = navs.resample("YE").last()
annual = year_end_values.pct_change()
annual.iloc[0] = year_end_values.iloc[0] / navs.iloc[0] - 1  # first year from the start date
annual.index = annual.index.year
annual["Active (vs. MSCI ACWI)"] = annual["Regional Momentum"] - annual[BENCHMARK_NAME]
annual.to_csv(OUTPUT / "annual_returns.csv")

years_ahead = (annual["Active (vs. MSCI ACWI)"] > 0).sum()
print(f"The strategy beat the MSCI ACWI in {years_ahead} of {len(annual)} calendar years")
annual.round(4)"""),
    md("""# 5. Risk class (SRI)
The summary risk indicator of the EU PRIIPs rules, on a scale from 1 (lowest) to 7 (highest), as shown in
the key information document of every fund sold to retail investors in Portugal. It uses daily values in
EUR over the last 5 years of the sample and a recommended holding period of 7 years (our clients'
horizon). The class limits for the VaR-equivalent volatility are 0.5%, 5%, 12%, 20%, 30% and 80%."""),
    code("""risk = pd.DataFrame({name: risk_class(navs[name], PARAMS.end, holding_years=7) for name in navs.columns}).T
risk.to_csv(OUTPUT / "risk_class.csv")
risk["SRI (1-7)"] = risk["SRI (1-7)"].astype(int)
risk.round(4)"""),
    md("""# 6. Sensitivity analysis
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
    md("""# 7. Check against the app
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
