from pathlib import Path

import pandas as pd
import streamlit as st

from regional_momentum.core import (
    BENCHMARK_NAME,
    REGIONS,
    Params,
    load_acwi_region_weights,
    load_prices,
    load_riskfree,
    run_all,
)

ROOT = Path(__file__).parent
WEIGHTS_SNAPSHOT = ROOT / "data" / "acwi_region_weights.csv"
DEFAULT = Params()
FIRST_DATE = pd.Timestamp(DEFAULT.start)
# The app runs to the last completed month-end, so no partial month enters the monthly returns.
# The notebook keeps the fixed end date of the prospectus (Params.end).
LAST_DATE = pd.Timestamp.today().normalize() - pd.offsets.MonthEnd(1)

st.set_page_config(page_title="Regional Momentum on the MSCI ACWI", layout="wide")


@st.cache_data(ttl=24 * 3600, show_spinner="Loading prices and interest rates ...")
def market_data(end):
    # end is part of the cache key, so the data is reloaded once a new month is complete
    return load_prices(DEFAULT.start, end), load_riskfree(DEFAULT.start, end)


@st.cache_data(show_spinner="Loading MSCI ACWI region weights from SEC filings ...")
def acwi_weights():
    if WEIGHTS_SNAPSHOT.exists():
        return pd.read_csv(WEIGHTS_SNAPSHOT, index_col="period", parse_dates=["period", "filed"])
    return load_acwi_region_weights(ROOT / "sec_cache", verbose=False)


st.title("Regional Momentum on the MSCI ACWI")
st.caption(
    "Six regional iShares UCITS ETFs at current MSCI ACWI weights, tilted towards the regions with the strongest "
    f"past returns. All values in EUR. Default settings = base case of the prospectus and the notebook; "
    f"data up to {LAST_DATE:%d %b %Y}."
)

with st.sidebar:
    st.header("Assumptions")
    rebalance = st.radio(
        "Rebalancing", ["Q", "S", "A"], index=1,
        format_func={"Q": "Quarterly", "S": "Semi-annual (Jun/Dec)", "A": "Annual (Dec)"}.get,
    )
    lookback = st.select_slider("Momentum lookback (months)", options=[3, 6, 9, 12], value=DEFAULT.lookback_months)
    tilt = st.slider("Tilt of the top region (pp)", 0.0, 10.0, DEFAULT.tilt_pp, 0.5,
                     help="Ranks 1-6 get +1, +2/3, +1/3, -1/3, -2/3, -1 times this value.")
    cap = st.slider("Max. deviation from MSCI ACWI weight", 0.0, 1.0, DEFAULT.max_rel_deviation, 0.05,
                    format="%.2f", help="0.5 = weight stays between 0.5x and 1.5x the MSCI ACWI weight")
    cost = st.slider("Trading costs (bp of traded volume)", 0.0, 50.0, DEFAULT.cost_bps, 1.0)
    start, end = st.slider(
        "Sample", min_value=FIRST_DATE.date(), max_value=LAST_DATE.date(),
        value=(FIRST_DATE.date(), LAST_DATE.date()), format="MMM YYYY",
    )

params = Params(
    start=str(start), end=str(end), rebalance=rebalance, lookback_months=lookback,
    tilt_pp=tilt, max_rel_deviation=cap, cost_bps=cost,
)

prices, rf = market_data(str(LAST_DATE.date()))
weights = acwi_weights()
res = run_all(prices, weights, rf, params)
table, navs = res["table"], res["navs"]
strat, bench = table.loc["Regional Momentum"], table.loc[BENCHMARK_NAME]

cols = st.columns(6)
cols[0].metric("Average return p.a.", f"{strat['Average return p.a.']:.2%}",
               f"{strat['Average return p.a.'] - bench['Average return p.a.']:+.2%} vs. MSCI ACWI")
cols[1].metric("Volatility p.a.", f"{strat['Volatility p.a.']:.2%}",
               f"{strat['Volatility p.a.'] - bench['Volatility p.a.']:+.2%} vs. MSCI ACWI", delta_color="inverse")
cols[2].metric("Sharpe ratio", f"{strat['Sharpe ratio']:.2f}", f"{strat['Sharpe ratio'] - bench['Sharpe ratio']:+.2f} vs. MSCI ACWI")
cols[3].metric("Alpha p.a.", f"{strat['Alpha p.a.']:.2%}", f"t = {strat['Alpha t-stat']:.2f}", delta_color="off")
cols[4].metric("Beta", f"{strat['Beta']:.2f}")
cols[5].metric("Information ratio", f"{strat['Information ratio']:.2f}", f"TE {strat['Tracking error p.a.']:.2%}", delta_color="off")

st.subheader("Cumulative performance in EUR (start = 1)")
st.line_chart(navs, height=380)

left, right = st.columns(2)
with left:
    st.subheader("Performance measures")
    pct = ["Average return p.a.", "Geometric return p.a.", "Volatility p.a.", "Alpha p.a.",
           "Tracking error p.a.", "Max drawdown", "Turnover p.a.", "Trading costs p.a."]
    shown = table.T.copy().astype(object)
    for row in shown.index:
        shown.loc[row] = [("" if pd.isna(v) else f"{v:.2%}" if row in pct else f"{v:.2f}") for v in table[row]]
    st.dataframe(shown, width="stretch")
with right:
    st.subheader("Active weights vs. MSCI ACWI")
    st.line_chart(res["strategy"].weights - res["passive"].weights, height=330)

st.subheader("Latest rebalancing")
last = res["strategy"].trades.iloc[-1]
latest = pd.DataFrame({
    "MSCI ACWI weight": [last[f"base {r}"] for r in REGIONS],
    f"{lookback}m return": [last[f"signal {r}"] for r in REGIONS],
    "Target weight": [last[f"target {r}"] for r in REGIONS],
}, index=REGIONS)
latest["Active weight"] = latest["Target weight"] - latest["MSCI ACWI weight"]
st.caption(f"{res['strategy'].trades.index[-1]:%d %b %Y}, MSCI ACWI weights from holdings as of {last['anchor']:%d %b %Y}")
st.dataframe(latest.sort_values(f"{lookback}m return", ascending=False).style.format("{:.2%}"), width="stretch")

st.caption(
    "Data: Yahoo Finance (Xetra prices of the UCITS ETFs in EUR), ECB 3-month Euribor (risk-free rate), "
    "SEC EDGAR holdings of the US-listed iShares MSCI ACWI ETF (region weights). "
    "Past performance is no guarantee of future results."
)
