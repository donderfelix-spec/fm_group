# Regional Momentum on the MSCI ACWI

Group assignment for 2270 Financial Modeling. The strategy rebuilds the MSCI ACWI from six regional iShares
UCITS ETFs and tilts the weights towards the regions with the strongest 12-month returns. It is designed for
Portuguese retail investors, so all funds are UCITS and all results are in EUR.

- Streamlit app: *TODO (not yet deployed)*
- Moodle group: BT
- Members: Renke Löschen (82016), Matti Raether (81317), Ben Bormann (78448), Felix Donder (75246),
  Leopold Querfurth (78371), Clara Liehr (80350)

## Strategy

| | |
|---|---|
| Building blocks | iShares UCITS ETFs (Acc), Xetra: SXR8 (S&P 500), EUNK (MSCI Europe), EUNM (MSCI EM), SXR5 (MSCI Japan), SXR2 (MSCI Canada), SXR1 (MSCI Pacific ex Japan) |
| Benchmark | iShares MSCI ACWI UCITS ETF (Acc), Xetra: IUSQ |
| Base weights | MSCI ACWI region weights from the latest SEC holdings filing of the US-listed iShares MSCI ACWI ETF public before the rebalancing date, rolled forward with ETF returns |
| Signal | Total return over the past 12 months (EUR) |
| Tilt | Ranks 1-6: +6 / +4 / +2 / -2 / -4 / -6 percentage points |
| Limits | Each weight stays within 0.5x-1.5x its MSCI ACWI weight; if a cap binds, the other side is scaled down pro rata |
| Rebalancing | Semi-annual, last trading day of June and December |
| Trading costs | 15 bp of traded volume |
| Sample | 2 January 2013 - 31 October 2026 |

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

- **ETF prices:** Yahoo Finance via `yfinance` (daily Xetra closing prices in EUR, accumulating share classes)
- **Risk-free rate:** ECB data portal, 3-month Euribor
- **Spot checks of the region weights:** archived iShares holdings lists from the Internet Archive
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
