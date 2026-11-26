# Regional Momentum on the MSCI ACWI

Group assignment for 2270 Financial Modeling. The strategy rebuilds the MSCI ACWI from six regional ETFs
and tilts the weights towards the regions with the strongest 12-month returns.

- Streamlit app: *TODO (not yet deployed)*
- Moodle group: BT

## Strategy

| | |
|---|---|
| Building blocks | SPY (USA), IEV (Europe), EEM (Emerging Markets), EWJ (Japan), EWC (Canada), EPP (Pacific ex Japan) |
| Benchmark | iShares MSCI ACWI ETF (ACWI) |
| Base weights | MSCI ACWI region weights from the latest SEC holdings filing public before the rebalancing date, rolled forward with ETF returns |
| Signal | Total return over the past 12 months (USD) |
| Tilt | Ranks 1-6: +6 / +4 / +2 / -2 / -4 / -6 percentage points |
| Limits | Each weight stays within 0.5x-1.5x its MSCI ACWI weight; if a cap binds, the other side is scaled down pro rata |
| Rebalancing | Semi-annual, last trading day of June and December |
| Trading costs | 15 bp of traded volume |
| Sample | 30 June 2008 - 31 October 2026 |

## Repository

```
regional_momentum/core.py   strategy code: data, SEC parsing, allocation rule, backtest, metrics
notebook/                   Colab notebook (built from core.py + analysis cells)
streamlit_app.py            Streamlit app, imports regional_momentum/core.py
data/                       snapshot of MSCI ACWI region weights for the app
tools/build_notebook.py     rebuilds the notebook after changes to core.py
tools/update_weights.py     refreshes the snapshot in data/
```

The notebook and the app share the same code, so the app with default settings reproduces the numbers in
the notebook and the prospectus.

## Data sources

- **ETF prices:** Yahoo Finance via `yfinance` (daily, adjusted for dividends and splits, USD)
- **Risk-free rate:** FRED, 3-month Treasury bill (DTB3)
- **MSCI ACWI region weights:** SEC EDGAR filings of the iShares MSCI ACWI ETF (series S000021461): N-Q
  (2008-2018), NPORT-EX (April 2019), NPORT-P (since October 2019)

## Run the notebook

Open `notebook/regional_momentum.ipynb` in Google Colab and choose *Runtime > Run all*. All data are
downloaded through APIs; a full run takes a few minutes.

## Run the app locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy on Streamlit Community Cloud

Push the repository to GitHub, create a new app on share.streamlit.io and select `streamlit_app.py` as the
main file.

## After changing the strategy code

```bash
python tools/build_notebook.py
```

The SEC asks automated clients to identify themselves: set your own contact e-mail in `SEC_HEADERS` in
`regional_momentum/core.py`.
