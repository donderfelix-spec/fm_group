# %% Imports and constants
from __future__ import annotations

import hashlib
import html
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
import yfinance as yf

REGIONS = ["USA", "Europe", "Emerging Markets", "Japan", "Canada", "Pacific ex Japan"]
ETF = {
    "USA": "SPY",
    "Europe": "IEV",
    "Emerging Markets": "EEM",
    "Japan": "EWJ",
    "Canada": "EWC",
    "Pacific ex Japan": "EPP",
}
BENCHMARK = "ACWI"

# iShares MSCI ACWI ETF on SEC EDGAR. The SEC asks automated clients to identify themselves.
SEC_SERIES = "S000021461"
SEC_HEADERS = {"User-Agent": "FM-Group-Project research@example.com"}
# Quarterly holdings filings. Annual/semi-annual reports (N-CSR/N-CSRS) are filed in April and
# October, i.e. never before one of our rebalancing dates, so they are not needed.
SEC_FORMS = ["N-Q", "NPORT-EX", "NPORT-P"]


@dataclass(frozen=True)
class Params:
    start: str = "2008-06-30"
    end: str = "2026-10-31"
    rebalance: str = "S"  # "Q" quarterly, "S" semi-annual, "A" annual
    lookback_months: int = 12
    tilt_pp: float = 6.0  # active weight of the top-ranked region in percentage points
    max_rel_deviation: float = 0.5  # weight stays within (1 +/- x) * ACWI weight
    cost_bps: float = 15.0  # per unit of traded volume


# %% Market data: ETF prices (Yahoo Finance) and risk-free rate (FRED)
def load_prices(start: str, end: str, buffer_months: int = 13) -> pd.DataFrame:
    """Daily dividend- and split-adjusted closing prices in USD."""
    first = pd.Timestamp(start) - pd.DateOffset(months=buffer_months)
    tickers = [ETF[r] for r in REGIONS] + [BENCHMARK]
    raw = yf.download(
        tickers,
        start=first,
        end=pd.Timestamp(end) + pd.Timedelta(days=1),
        auto_adjust=True,
        progress=False,
    )
    px = raw["Close"][tickers]
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    return px.dropna(subset=[ETF[r] for r in REGIONS])


def load_riskfree(start: str, end: str) -> pd.Series:
    """Monthly risk-free return from the 3-month T-bill rate (FRED: DTB3).

    The rate observed at the end of month m-1 is used for month m.
    """
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3"
    df = pd.read_csv(url, index_col=0, parse_dates=True, na_values=".")
    rate = df.iloc[:, 0].astype(float).ffill() / 100
    monthly = rate.resample("ME").last()
    rf = ((1 + monthly) ** (1 / 12) - 1).shift(1)
    return rf.loc[start:end].rename("rf")


# %% ACWI region weights from SEC filings (N-Q, NPORT-EX, NPORT-P)
_EUROPE = {
    "UNITED KINGDOM", "SWITZERLAND", "FRANCE", "GERMANY", "NETHERLANDS", "SWEDEN", "DENMARK",
    "ITALY", "SPAIN", "FINLAND", "BELGIUM", "NORWAY", "IRELAND", "AUSTRIA", "PORTUGAL",
    "LUXEMBOURG",
}
_PACIFIC = {"AUSTRALIA", "HONG KONG", "SINGAPORE", "NEW ZEALAND", "MACAU"}
_EMERGING = {
    "ARGENTINA", "BRAZIL", "CHILE", "CHINA", "COLOMBIA", "CZECH REPUBLIC", "EGYPT", "HUNGARY",
    "INDIA", "INDONESIA", "KUWAIT", "MALAYSIA", "MEXICO", "PAKISTAN", "PERU", "PHILIPPINES",
    "POLAND", "QATAR", "RUSSIA", "SAUDI ARABIA", "SOUTH AFRICA", "SOUTH KOREA", "KOREA",
    "TAIWAN", "THAILAND", "TURKEY", "UNITED ARAB EMIRATES",
}
_ALL_COUNTRIES = {"UNITED STATES", "CANADA", "JAPAN", "ISRAEL", "GREECE"} | _EUROPE | _PACIFIC | _EMERGING
# MSCI reclassifications inside the sample: Israel EM -> DM (2010), Greece DM -> EM (2013).
_ISRAEL_DM_FROM = pd.Timestamp("2010-06-01")
_GREECE_EM_FROM = pd.Timestamp("2013-12-01")

_ISO_EUROPE = {"GB", "CH", "FR", "DE", "NL", "SE", "DK", "IT", "ES", "FI", "BE", "NO", "IE", "AT", "PT", "LU", "JE", "IM", "IL"}
_ISO_PACIFIC = {"AU", "HK", "SG", "NZ", "MO"}
# Offshore domiciles typical for US companies (e.g. Medtronic in Ireland, Chubb in Switzerland).
_US_DOMICILES = {"US", "IE", "CH", "NL", "GB", "BM", "LU", "LR", "CW", "PA", "JE", "IM", "IL", "PR", "BS"}
_EUROPE_CCY = {"EUR", "GBP", "GBX", "CHF", "SEK", "DKK", "NOK", "ILS"}


def _region_from_country(country: str, period: pd.Timestamp) -> str:
    c = country.upper()
    if c == "UNITED STATES":
        return "USA"
    if c == "CANADA":
        return "Canada"
    if c == "JAPAN":
        return "Japan"
    if c == "ISRAEL":
        return "Europe" if period >= _ISRAEL_DM_FROM else "Emerging Markets"
    if c == "GREECE":
        return "Emerging Markets" if period >= _GREECE_EM_FROM else "Europe"
    if c in _EUROPE:
        return "Europe"
    if c in _PACIFIC:
        return "Pacific ex Japan"
    return "Emerging Markets"


def _region_from_holding(domicile: str, currency: str, name: str, period: pd.Timestamp) -> str:
    """N-PORT reports the legal domicile, not the MSCI country; trading currency resolves most cases."""
    if "ISHARES" in name:  # country ETFs held instead of local shares, e.g. iShares MSCI India ETF
        return "Japan" if "JAPAN" in name else "Canada" if "CANADA" in name else "Emerging Markets"
    if domicile == "GR":
        return _region_from_country("GREECE", period)
    if currency == "USD":
        if domicile == "CA":
            return "Canada"
        if domicile == "JP":
            return "Japan"
        if domicile in _ISO_PACIFIC:
            return "Pacific ex Japan"
        return "USA" if domicile in _US_DOMICILES else "Emerging Markets"
    if currency == "CAD":
        return "Canada"
    if currency == "JPY":
        return "Japan"
    if currency in _EUROPE_CCY:
        return "Europe"
    if currency in {"AUD", "NZD", "SGD"}:
        return "Pacific ex Japan"
    if currency == "HKD":
        return "Pacific ex Japan" if domicile in {"HK", "MO"} else "Emerging Markets"
    return "Emerging Markets"


def _normalise(weights: dict, source: str) -> pd.Series:
    w = pd.Series(weights, dtype=float).reindex(REGIONS).fillna(0.0)
    total = w.sum()
    if not 95 <= total <= 101:
        raise ValueError(f"Country weights in {source} add up to {total:.2f}% - parser needs checking")
    return w / total


def _weights_from_html(doc: str, period: pd.Timestamp, source: str) -> pd.Series:
    """Schedule of investments grouped by country, e.g. 'JAPAN - 9.29%'."""
    text = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", doc)))
    heads = [
        m.start()
        for m in re.finditer(r"MSCI ACWI (?:INDEX FUND|Index Fund|ETF)\b", text)
        if re.search(r"Security Shares Value (?:COMMON STOCKS|Common Stocks)", text[m.start(): m.start() + 160])
        and "Hedged" not in text[max(0, m.start() - 40): m.start()]
    ]
    if not heads:
        raise ValueError(f"No ACWI schedule of investments found in {source}")
    end = re.search(r"TOTAL INVESTMENTS|Total Investments", text[heads[0]:])
    section = text[heads[0]: heads[0] + end.start()]
    names = "|".join(sorted(map(re.escape, _ALL_COUNTRIES), key=len, reverse=True))
    weights: dict[str, float] = {}
    for country, pct in re.findall(rf"(?<![A-Za-z])({names}) ?[—–] ?(\d+(?:\.\d+)?)%", section, re.I):
        region = _region_from_country(country, period)
        weights[region] = weights.get(region, 0.0) + float(pct)
    return _normalise(weights, source)


def _weights_from_nport(xml: str, period: pd.Timestamp, source: str) -> pd.Series:
    weights: dict[str, float] = {}
    for h in re.findall(r"<invstOrSec>(.*?)</invstOrSec>", xml, re.S):
        asset = re.search(r"<assetCat>(.*?)</assetCat>", h)
        if not asset or asset.group(1) not in ("EC", "EP"):
            continue
        pct = float(re.search(r"<pctVal>(.*?)</pctVal>", h).group(1))  # already in percent
        dom = re.search(r'invCountry="?([A-Z]{2})', h) or re.search(r"<invCountry>(.*?)</invCountry>", h)
        ccy = re.search(r'curCd="([A-Z]{3})"', h) or re.search(r"<curCd>(.*?)</curCd>", h)
        name = re.search(r"<name>(.*?)</name>", h).group(1).upper()
        region = _region_from_holding(dom.group(1) if dom else "", ccy.group(1) if ccy else "", name, period)
        weights[region] = weights.get(region, 0.0) + pct
    return _normalise(weights, source)


def _sec_get(url: str, cache_dir: Path | None) -> str:
    path = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / hashlib.sha1(url.encode()).hexdigest()
        if path.exists():
            return path.read_text(encoding="utf-8")
    time.sleep(0.15)  # SEC fair-access limit is 10 requests per second
    r = requests.get(url, headers=SEC_HEADERS, timeout=120)
    r.raise_for_status()
    text = r.content.decode("utf-8", "ignore")
    if path is not None:
        path.write_text(text, encoding="utf-8")
    return text


def list_sec_filings() -> pd.DataFrame:
    rows = []
    for form in SEC_FORMS:
        url = (
            "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
            f"&CIK={SEC_SERIES}&type={form}&dateb=&owner=include&count=100&output=atom"
        )
        # The list itself is never cached so that new filings are picked up; filings never change.
        for entry in re.findall(r"<entry>(.*?)</entry>", _sec_get(url, None), re.S):
            ftype = re.search(r"<filing-type>(.*?)</filing-type>", entry).group(1)
            if ftype != form:  # drops amendments (/A)
                continue
            rows.append({
                "form": ftype,
                "filed": pd.Timestamp(re.search(r"<filing-date>(.*?)</filing-date>", entry).group(1)),
                "index_url": re.search(r"<filing-href>(.*?)</filing-href>", entry).group(1),
            })
    return pd.DataFrame(rows).drop_duplicates("index_url").sort_values("filed").reset_index(drop=True)


def load_acwi_region_weights(cache_dir: Path | None = None, verbose: bool = True) -> pd.DataFrame:
    """ACWI region weights per reporting date ('period') and the date the filing became public ('filed')."""
    rows = []
    filings = list_sec_filings()
    if verbose:
        print(f"Downloading and parsing {len(filings)} SEC filings ...")
    for f in filings.itertuples():
        index_page = _sec_get(f.index_url, cache_dir)
        period = pd.Timestamp(re.search(r'Period of Report</div>\s*<div class="info">([\d-]+)<', index_page).group(1))
        folder = f.index_url.rsplit("/", 1)[0]
        if f.form == "NPORT-P":
            w = _weights_from_nport(_sec_get(f"{folder}/primary_doc.xml", cache_dir), period, f.index_url)
        else:
            docs = [
                d for d in re.findall(r'href="(/Archives/[^"]+\.htm)"', index_page)
                if "-index" not in d and not re.search(r"cert|ex99|ex-99|ethic", d, re.I)
            ]
            w = _weights_from_html(_sec_get("https://www.sec.gov" + docs[0], cache_dir), period, f.index_url)
        rows.append({"period": period, "filed": f.filed, "form": f.form, "url": f.index_url, **w.to_dict()})
    df = pd.DataFrame(rows).sort_values(["period", "filed"]).drop_duplicates("period", keep="first")
    return df.set_index("period")


# %% Allocation rule: base weights, momentum signal and tilt
def region_prices(prices: pd.DataFrame) -> pd.DataFrame:
    rp = prices[[ETF[r] for r in REGIONS]].copy()
    rp.columns = REGIONS
    return rp


def rebalance_dates(index: pd.DatetimeIndex, p: Params) -> list[pd.Timestamp]:
    months = {"Q": (3, 6, 9, 12), "S": (6, 12), "A": (12,)}[p.rebalance]
    idx = index[(index >= pd.Timestamp(p.start)) & (index <= pd.Timestamp(p.end))]
    month_ends = pd.Series(idx, index=idx).groupby([idx.year, idx.month]).max()
    first = idx[0]
    return [first] + [d for d in month_ends if d.month in months and d > first]


def base_weights_at(date: pd.Timestamp, sec_weights: pd.DataFrame, rprices: pd.DataFrame) -> tuple[pd.Series, pd.Timestamp]:
    """Latest ACWI weights public before `date`, drifted with region returns since the reporting date."""
    available = sec_weights[sec_weights["filed"] < date]
    if available.empty:
        raise ValueError(f"No ACWI holdings filing was public before {date.date()}")
    anchor = available.index.max()
    w = available.loc[anchor, REGIONS].astype(float)
    w = w * (rprices.asof(date) / rprices.asof(anchor))
    return w / w.sum(), anchor


def momentum_scores(date: pd.Timestamp, rprices: pd.DataFrame, months: int) -> pd.Series:
    return rprices.asof(date) / rprices.asof(date - pd.DateOffset(months=months)) - 1


def target_weights(base: pd.Series, scores: pd.Series, tilt_pp: float, max_rel_deviation: float) -> pd.Series:
    """Rank-based tilt (+3/+2/+1/-1/-2/-3 thirds of tilt_pp), capped relative to the ACWI weight.

    If caps bind, the larger side (over- or underweights) is scaled down pro rata so that
    active weights sum to zero and no cap is violated.
    """
    order = scores.sort_values(ascending=False).index
    k = len(order) // 2
    steps = np.r_[np.arange(k, 0, -1), -np.arange(1, k + 1)] / k
    active = pd.Series(steps * tilt_pp / 100, index=order).reindex(base.index)
    limit = max_rel_deviation * base
    active = active.clip(-limit, limit)
    over, under = active[active > 0].sum(), -active[active < 0].sum()
    if over > under:
        active[active > 0] *= under / over
    elif under > over:
        active[active < 0] *= over / under
    return base + active


# %% Backtest engine
@dataclass
class BacktestResult:
    nav: pd.Series  # daily, starts at 1
    weights: pd.DataFrame  # daily end-of-day weights
    trades: pd.DataFrame  # one row per rebalancing date


def run_backtest(prices: pd.DataFrame, sec_weights: pd.DataFrame, p: Params, tilt: bool = True) -> BacktestResult:
    """Signals and weights use the close of the rebalancing day; trades execute at that close."""
    rp = region_prices(prices)
    sim = rp.loc[p.start: p.end]
    rets = sim.pct_change().fillna(0.0).to_numpy()
    rdates = set(rebalance_dates(sim.index, p))
    w = None
    nav = 1.0
    navs, hist, trades = [], [], []
    for i, d in enumerate(sim.index):
        if w is not None:
            gross = w * (1 + rets[i])
            nav *= gross.sum()
            w = gross / gross.sum()
        if d in rdates:
            base, anchor = base_weights_at(d, sec_weights, rp)
            scores = momentum_scores(d, rp, p.lookback_months)
            target = target_weights(base, scores, p.tilt_pp if tilt else 0.0, p.max_rel_deviation)
            turnover = 0.0 if w is None else float(np.abs(target.to_numpy() - w).sum())
            cost = turnover * p.cost_bps / 1e4
            nav *= 1 - cost
            trades.append({
                "date": d, "anchor": anchor, "turnover": turnover, "cost": cost,
                **{f"base {r}": base[r] for r in REGIONS},
                **{f"signal {r}": scores[r] for r in REGIONS},
                **{f"target {r}": target[r] for r in REGIONS},
            })
            w = target.to_numpy().copy()
        navs.append(nav)
        hist.append(w.copy())
    return BacktestResult(
        nav=pd.Series(navs, index=sim.index, name="nav"),
        weights=pd.DataFrame(hist, index=sim.index, columns=REGIONS),
        trades=pd.DataFrame(trades).set_index("date"),
    )


def benchmark_nav(prices: pd.DataFrame, p: Params) -> pd.Series:
    b = prices[BENCHMARK].loc[p.start: p.end]
    return (b / b.iloc[0]).rename("nav")


# %% Performance measures
def monthly_returns(nav: pd.Series) -> pd.Series:
    return nav.resample("ME").last().pct_change().dropna()


def performance_table(returns: pd.DataFrame, benchmark: str, rf: pd.Series) -> pd.DataFrame:
    """Monthly returns; means x12, volatilities x sqrt(12); CAPM alpha x12; beta not annualised."""
    rf = rf.reindex(returns.index)
    rb = returns[benchmark]
    rows = {}
    for name, r in returns.items():
        ex, exb = r - rf, rb - rf
        fit = sm.OLS(ex.to_numpy(), sm.add_constant(exb.to_numpy())).fit()
        active = r - rb
        te = active.std() * np.sqrt(12)
        cum = (1 + r).cumprod()
        rows[name] = {
            "Average return p.a.": r.mean() * 12,
            "Geometric return p.a.": cum.iloc[-1] ** (12 / len(r)) - 1,
            "Volatility p.a.": r.std() * np.sqrt(12),
            "Sharpe ratio": ex.mean() * 12 / (ex.std() * np.sqrt(12)),
            "Alpha p.a.": fit.params[0] * 12,
            "Alpha t-stat": fit.tvalues[0] if name != benchmark else np.nan,
            "Beta": fit.params[1],
            "Tracking error p.a.": te,
            "Information ratio": active.mean() * 12 / te if te > 0 else np.nan,
            "Max drawdown": (cum / cum.cummax() - 1).min(),
        }
    return pd.DataFrame(rows).T


def run_all(prices: pd.DataFrame, sec_weights: pd.DataFrame, rf: pd.Series, p: Params) -> dict:
    strategy = run_backtest(prices, sec_weights, p, tilt=True)
    passive = run_backtest(prices, sec_weights, p, tilt=False)
    navs = pd.DataFrame({
        "Regional Momentum": strategy.nav,
        "Passive replica": passive.nav,
        "MSCI ACWI (ACWI ETF)": benchmark_nav(prices, p),
    }).dropna()
    returns = navs.apply(monthly_returns)
    table = performance_table(returns, "MSCI ACWI (ACWI ETF)", rf)
    years = (navs.index[-1] - navs.index[0]).days / 365.25
    table["Turnover p.a."] = [strategy.trades["turnover"].sum() / years, passive.trades["turnover"].sum() / years, 0.0]
    table["Trading costs p.a."] = [strategy.trades["cost"].sum() / years, passive.trades["cost"].sum() / years, 0.0]
    return {"strategy": strategy, "passive": passive, "navs": navs, "returns": returns, "table": table}


def sensitivity(prices: pd.DataFrame, sec_weights: pd.DataFrame, rf: pd.Series, base: Params, grid: dict) -> pd.DataFrame:
    """Re-run the strategy changing one parameter at a time."""
    rows = []
    for field, values in grid.items():
        for v in values:
            p = Params(**{**base.__dict__, field: v})
            t = run_all(prices, sec_weights, rf, p)["table"].loc["Regional Momentum"]
            rows.append({"parameter": field, "value": v, **t[[
                "Average return p.a.", "Volatility p.a.", "Sharpe ratio", "Alpha p.a.", "Beta",
                "Information ratio", "Turnover p.a.",
            ]].to_dict()})
    return pd.DataFrame(rows).set_index(["parameter", "value"])
