# %% Imports and settings
# The six regional UCITS ETFs that make up the portfolio, the benchmark, and the default assumptions
# of the strategy. All funds are Irish-domiciled iShares UCITS ETFs with accumulating share classes,
# registered for sale in Portugal. Changing a value in Params changes the backtest.
import hashlib
import html
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
import yfinance as yf

REGIONS = ["USA", "Europe", "Emerging Markets", "Japan", "Canada", "Pacific ex Japan"]

# Yahoo Finance tickers of the Xetra listings (prices in EUR)
ETF = {
    "USA": "SXR8.DE",
    "Europe": "EUNK.DE",
    "Emerging Markets": "EUNM.DE",
    "Japan": "SXR5.DE",
    "Canada": "SXR2.DE",
    "Pacific ex Japan": "SXR1.DE",
}
# Benchmark: iShares MSCI ACWI UCITS ETF (Acc), also on Xetra
BENCHMARK = "IUSQ.DE"
BENCHMARK_NAME = "MSCI ACWI (iShares UCITS ETF)"

# Fund facts from the iShares product pages (26 Nov 2026): name, ISIN, index, TER, fund launch
FUND_FACTS = pd.DataFrame(
    [
        ["USA", "iShares Core S&P 500 UCITS ETF (Acc)", "IE00B5BMR087", "S&P 500", 0.0007, "2010-05-18"],
        ["Europe", "iShares Core MSCI Europe UCITS ETF (Acc)", "IE00B4K48X80", "MSCI Europe", 0.0012, "2009-09-25"],
        ["Emerging Markets", "iShares MSCI EM UCITS ETF (Acc)", "IE00B4L5YC18", "MSCI Emerging Markets", 0.0018, "2009-09-25"],
        ["Japan", "iShares MSCI Japan UCITS ETF (Acc)", "IE00B53QDK08", "MSCI Japan", 0.0012, "2010-01-11"],
        ["Canada", "iShares MSCI Canada UCITS ETF (Acc)", "IE00B52SF786", "MSCI Canada", 0.0048, "2010-01-12"],
        ["Pacific ex Japan", "iShares Core MSCI Pacific ex-Japan UCITS ETF (Acc)", "IE00B52MJY50", "MSCI Pacific ex Japan", 0.0020, "2010-01-11"],
        ["Benchmark", "iShares MSCI ACWI UCITS ETF (Acc)", "IE00B6R52259", "MSCI ACWI", 0.0020, "2011-10-21"],
    ],
    columns=["Region", "Fund", "ISIN", "Index", "TER", "Launch"],
).set_index("Region")
FUND_FACTS["Ticker"] = [ETF.get(region, BENCHMARK) for region in FUND_FACTS.index]

# iShares MSCI ACWI ETF on SEC EDGAR. The SEC wants automated downloads to include a contact e-mail.
SEC_SERIES = "S000021461"
SEC_HEADERS = {"User-Agent": "FM-Group-Project 75246@novasbe.pt"}
SEC_FORMS = ["N-Q", "NPORT-EX", "NPORT-P"]


@dataclass
class Params:
    start: str = "2013-01-01"
    end: str = "2026-10-31"
    rebalance: str = "S"            # "Q" quarterly, "S" semi-annual (Jun/Dec), "A" annual (Dec)
    lookback_months: int = 12       # momentum = total return over this many months
    tilt_pp: float = 6.0            # active weight of the best region, in percentage points
    max_rel_deviation: float = 0.5  # 0.5 = each weight stays between 0.5x and 1.5x its MSCI ACWI weight
    cost_bps: float = 15.0          # trading costs in basis points of the traded volume


# %% Market data
# ETF prices come from Yahoo Finance (yfinance). All funds are traded on Xetra in EUR, so the prices already
# include the exchange-rate effect for a euro investor. The risk-free rate is the 3-month Euribor (European
# Central Bank).
def load_prices(start, end):
    """Daily Xetra closing prices in EUR of the six ETFs and the benchmark (accumulating, so total return)."""
    # 13 extra months, so that the first 12-month momentum signal can be calculated
    first_day = pd.Timestamp(start) - pd.DateOffset(months=13)
    last_day = pd.Timestamp(end) + pd.Timedelta(days=1)  # yfinance leaves out the end date itself
    tickers = list(ETF.values()) + [BENCHMARK]

    data = yf.download(tickers, start=first_day, end=last_day, auto_adjust=True, progress=False)
    prices = data["Close"][tickers]
    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None)
    return prices.dropna(subset=list(ETF.values()))


def load_riskfree(start, end):
    """Monthly risk-free return in EUR from the 3-month Euribor (ECB data portal, monthly average)."""
    url = ("https://data-api.ecb.europa.eu/service/data/FM/M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA"
           "?format=csvdata&startPeriod=2007-01")
    data = pd.read_csv(url)
    yearly_rate = pd.Series(data["OBS_VALUE"].to_numpy() / 100,
                            index=pd.PeriodIndex(data["TIME_PERIOD"], freq="M").to_timestamp(how="end").normalize())
    monthly_return = (1 + yearly_rate) ** (1 / 12) - 1

    # month m earns the average rate of month m-1 (known at the start of month m)
    rf = monthly_return.shift(1)
    return rf.loc[start:end].rename("rf")


# %% MSCI ACWI region weights from SEC filings
# The US-listed iShares MSCI ACWI ETF tracks the same index as our benchmark and reports its holdings to the
# SEC every quarter, so its reports give the MSCI ACWI region weights over time. We download these reports,
# add up the weights per country and group the countries into our six regions.
# Until 2019 the reports are HTML tables by country (forms N-Q and NPORT-EX), since then XML files with
# every single holding (form NPORT-P).
EUROPE = {
    "UNITED KINGDOM", "SWITZERLAND", "FRANCE", "GERMANY", "NETHERLANDS", "SWEDEN", "DENMARK",
    "ITALY", "SPAIN", "FINLAND", "BELGIUM", "NORWAY", "IRELAND", "AUSTRIA", "PORTUGAL", "LUXEMBOURG",
}
PACIFIC = {"AUSTRALIA", "HONG KONG", "SINGAPORE", "NEW ZEALAND", "MACAU"}
EMERGING = {
    "ARGENTINA", "BRAZIL", "CHILE", "CHINA", "COLOMBIA", "CZECH REPUBLIC", "EGYPT", "HUNGARY",
    "INDIA", "INDONESIA", "KUWAIT", "MALAYSIA", "MEXICO", "PAKISTAN", "PERU", "PHILIPPINES",
    "POLAND", "QATAR", "RUSSIA", "SAUDI ARABIA", "SOUTH AFRICA", "SOUTH KOREA", "KOREA",
    "TAIWAN", "THAILAND", "TURKEY", "UNITED ARAB EMIRATES",
}
ALL_COUNTRIES = {"UNITED STATES", "CANADA", "JAPAN", "ISRAEL", "GREECE"} | EUROPE | PACIFIC | EMERGING

# MSCI moved two countries during our sample period
ISRAEL_DEVELOPED_FROM = pd.Timestamp("2010-06-01")
GREECE_EMERGING_FROM = pd.Timestamp("2013-12-01")

# Two-letter country codes used in the NPORT-P files
EUROPE_CODES = {"GB", "CH", "FR", "DE", "NL", "SE", "DK", "IT", "ES", "FI", "BE", "NO", "IE", "AT", "PT", "LU", "JE", "IM", "IL"}
PACIFIC_CODES = {"AU", "HK", "SG", "NZ", "MO"}
# Countries where many US companies are registered (e.g. Medtronic in Ireland, Chubb in Switzerland)
US_COMPANY_CODES = {"US", "IE", "CH", "NL", "GB", "BM", "LU", "LR", "CW", "PA", "JE", "IM", "IL", "PR", "BS"}
EUROPE_CURRENCIES = {"EUR", "GBP", "GBX", "CHF", "SEK", "DKK", "NOK", "ILS"}


def region_of_country(country, report_date):
    """Region of a country name as printed in the HTML reports."""
    country = country.upper()
    if country == "UNITED STATES":
        return "USA"
    if country == "CANADA":
        return "Canada"
    if country == "JAPAN":
        return "Japan"
    if country == "ISRAEL":
        return "Europe" if report_date >= ISRAEL_DEVELOPED_FROM else "Emerging Markets"
    if country == "GREECE":
        return "Emerging Markets" if report_date >= GREECE_EMERGING_FROM else "Europe"
    if country in EUROPE:
        return "Europe"
    if country in PACIFIC:
        return "Pacific ex Japan"
    return "Emerging Markets"


def region_of_holding(country_code, currency, name, report_date):
    """Region of a single NPORT-P holding.

    NPORT-P gives the country where a company is registered, not the MSCI country. The trading
    currency fixes most cases: Medtronic is registered in Ireland but trades in USD (= USA),
    Alibaba is registered in the Cayman Islands and trades in HKD or USD (= Emerging Markets).
    """
    if "ISHARES" in name:  # the fund sometimes holds a country ETF, e.g. iShares MSCI India ETF
        if "JAPAN" in name:
            return "Japan"
        if "CANADA" in name:
            return "Canada"
        return "Emerging Markets"
    if country_code == "GR":
        return region_of_country("GREECE", report_date)

    if currency == "USD":
        if country_code == "CA":
            return "Canada"
        if country_code == "JP":
            return "Japan"
        if country_code in PACIFIC_CODES:
            return "Pacific ex Japan"
        if country_code in US_COMPANY_CODES:
            return "USA"
        return "Emerging Markets"  # e.g. Chinese, Indian or Brazilian shares traded in USD
    if currency == "CAD":
        return "Canada"
    if currency == "JPY":
        return "Japan"
    if currency in EUROPE_CURRENCIES:
        return "Europe"
    if currency in ("AUD", "NZD", "SGD"):
        return "Pacific ex Japan"
    if currency == "HKD":
        return "Pacific ex Japan" if country_code in ("HK", "MO") else "Emerging Markets"
    return "Emerging Markets"


def find(pattern, text):
    """First group of a regular expression, or an empty string if there is no match."""
    match = re.search(pattern, text, re.S)
    return match.group(1) if match else ""


def to_weights(totals):
    """Turn summed percentages per region into weights that add up to 1."""
    weights = pd.Series(totals, dtype=float).reindex(REGIONS).fillna(0.0)
    total = weights.sum()
    if total < 95 or total > 101:  # the rest is cash; anything else means the parsing went wrong
        raise ValueError(f"Country weights add up to {total:.1f}% instead of about 100%")
    return weights / total


def weights_from_html(document, report_date):
    """Region weights from an HTML report (N-Q or NPORT-EX)."""
    # remove the HTML tags and collapse the white space
    text = html.unescape(re.sub(r"<[^>]+>", " ", document))
    text = re.sub(r"\s+", " ", text)

    # One report covers many iShares funds. The MSCI ACWI table starts with the fund name, followed by the
    # column headers and "Common Stocks". The currency-hedged MSCI ACWI fund is a different fund.
    start = None
    for match in re.finditer(r"MSCI ACWI (INDEX FUND|Index Fund|ETF)\b", text):
        before = text[max(0, match.start() - 40): match.start()]
        after = text[match.start(): match.start() + 160]
        if "Hedged" not in before and re.search(r"Security Shares Value (COMMON STOCKS|Common Stocks)", after):
            start = match.start()
            break
    if start is None:
        raise ValueError("MSCI ACWI table not found in the report")
    end = start + re.search(r"TOTAL INVESTMENTS|Total Investments", text[start:]).start()
    table = text[start:end]

    # country lines look like "JAPAN — 9.29%" (older reports) or "Japan — 7.5%" (newer reports)
    country_names = "|".join(sorted(ALL_COUNTRIES, key=len, reverse=True))
    pattern = r"(?<![A-Za-z])(" + country_names + r") ?[—–] ?(\d+(?:\.\d+)?)%"

    totals = {}
    for country, percent in re.findall(pattern, table, re.I):
        region = region_of_country(country, report_date)
        totals[region] = totals.get(region, 0.0) + float(percent)
    return to_weights(totals)


def weights_from_nport(xml, report_date):
    """Region weights from an NPORT-P XML file, one <invstOrSec> block per holding."""
    totals = {}
    for holding in re.findall(r"<invstOrSec>(.*?)</invstOrSec>", xml, re.S):
        asset_type = find(r"<assetCat>(.*?)</assetCat>", holding)
        if asset_type not in ("EC", "EP"):  # only common and preferred shares, no cash or futures
            continue
        percent = float(find(r"<pctVal>(.*?)</pctVal>", holding))  # already in percent
        country_code = find(r'invCountry="?([A-Z]{2})', holding) or find(r"<invCountry>(.*?)</invCountry>", holding)
        currency = find(r'curCd="([A-Z]{3})"', holding) or find(r"<curCd>(.*?)</curCd>", holding)
        name = find(r"<name>(.*?)</name>", holding).upper()

        region = region_of_holding(country_code, currency, name, report_date)
        totals[region] = totals.get(region, 0.0) + percent
    return to_weights(totals)


def sec_download(url, cache_folder=None):
    """Download a page from SEC EDGAR; with a cache folder each file is only downloaded once."""
    cache_file = None
    if cache_folder is not None:
        Path(cache_folder).mkdir(parents=True, exist_ok=True)
        cache_file = Path(cache_folder) / hashlib.sha1(url.encode()).hexdigest()
        if cache_file.exists():
            return cache_file.read_text(encoding="utf-8")

    time.sleep(0.15)  # the SEC allows at most 10 requests per second
    response = requests.get(url, headers=SEC_HEADERS, timeout=120)
    response.raise_for_status()
    text = response.content.decode("utf-8", "ignore")

    if cache_file is not None:
        cache_file.write_text(text, encoding="utf-8")
    return text


def list_sec_filings():
    """All quarterly holdings filings of the ACWI ETF: form, filing date and link."""
    rows = []
    for form in SEC_FORMS:
        url = (
            "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
            f"&CIK={SEC_SERIES}&type={form}&dateb=&owner=include&count=100&output=atom"
        )
        feed = sec_download(url)  # never cached, so that new filings show up
        for entry in re.findall(r"<entry>(.*?)</entry>", feed, re.S):
            if find(r"<filing-type>(.*?)</filing-type>", entry) != form:  # skips amendments like N-Q/A
                continue
            rows.append({
                "form": form,
                "filed": pd.Timestamp(find(r"<filing-date>(.*?)</filing-date>", entry)),
                "index_url": find(r"<filing-href>(.*?)</filing-href>", entry),
            })
    filings = pd.DataFrame(rows).drop_duplicates("index_url")
    return filings.sort_values("filed").reset_index(drop=True)


def load_acwi_region_weights(cache_folder=None, verbose=True):
    """MSCI ACWI region weights for every reporting date, plus the date each report became public."""
    filings = list_sec_filings()
    if verbose:
        print(f"Downloading and parsing {len(filings)} SEC filings ...")

    rows = []
    for filing in filings.itertuples():
        index_page = sec_download(filing.index_url, cache_folder)
        report_date = pd.Timestamp(find(r'Period of Report</div>\s*<div class="info">([\d-]+)<', index_page))
        folder = filing.index_url.rsplit("/", 1)[0]

        if filing.form == "NPORT-P":
            xml = sec_download(folder + "/primary_doc.xml", cache_folder)
            weights = weights_from_nport(xml, report_date)
        else:
            # the main document is the first .htm link that is not an index page or a certification
            links = re.findall(r'href="(/Archives/[^"]+\.htm)"', index_page)
            links = [link for link in links if "-index" not in link and not re.search(r"cert|ex99|ex-99|ethic", link, re.I)]
            document = sec_download("https://www.sec.gov" + links[0], cache_folder)
            weights = weights_from_html(document, report_date)

        rows.append({"period": report_date, "filed": filing.filed, "form": filing.form,
                     "url": filing.index_url, **weights.to_dict()})

    table = pd.DataFrame(rows).sort_values(["period", "filed"])
    table = table.drop_duplicates("period", keep="first")  # one row per reporting date
    return table.set_index("period")


# %% Allocation rule
# On every rebalancing date: (1) take the current MSCI ACWI weights, (2) rank the regions by momentum,
# (3) tilt the weights towards the winners within the limits.
def region_prices(prices):
    """ETF prices with region names as column names."""
    result = prices[[ETF[region] for region in REGIONS]].copy()
    result.columns = REGIONS
    return result


def rebalancing_dates(trading_days, params):
    """First day of the sample, then the last trading day of each rebalancing month."""
    months = {"Q": [3, 6, 9, 12], "S": [6, 12], "A": [12]}[params.rebalance]
    days = trading_days[(trading_days >= pd.Timestamp(params.start)) & (trading_days <= pd.Timestamp(params.end))]

    dates = [days[0]]
    for i, day in enumerate(days):
        last_day_of_month = i == len(days) - 1 or days[i + 1].month != day.month
        if last_day_of_month and day.month in months and day > days[0]:
            dates.append(day)
    return dates


def acwi_weights_on(day, acwi_weights, prices):
    """MSCI ACWI weights on a given day, using only reports that were already public.

    The latest public report is usually about two months old, so its weights are rolled forward
    with the price changes of each region since the reporting date.
    """
    public = acwi_weights[acwi_weights["filed"] < day]
    if public.empty:
        raise ValueError(f"No MSCI ACWI report was public before {day.date()}")
    report_date = public.index.max()

    weights = public.loc[report_date, REGIONS].astype(float)
    price_change = prices.asof(day) / prices.asof(report_date)
    weights = weights * price_change
    return weights / weights.sum(), report_date


def momentum_on(day, prices, months):
    """Total return of each region over the past `months` months."""
    past_day = day - pd.DateOffset(months=months)
    return prices.asof(day) / prices.asof(past_day) - 1


def target_weights(acwi, momentum, tilt_pp, max_deviation):
    """MSCI ACWI weights plus active weights by momentum rank.

    With 6 regions and tilt_pp = 6 the active weights are +6, +4, +2, -2, -4, -6 percentage points.
    No region may move more than max_deviation times its MSCI ACWI weight. If that limit cuts some active
    weights, the other side is scaled down so that over- and underweights still cancel out.
    """
    ranking = momentum.sort_values(ascending=False).index  # best region first
    half = len(ranking) // 2
    steps = list(range(half, 0, -1)) + list(range(-1, -half - 1, -1))  # 3, 2, 1, -1, -2, -3

    active = pd.Series(0.0, index=acwi.index)
    for region, step in zip(ranking, steps):
        active[region] = step / half * tilt_pp / 100

    # limit relative to the MSCI ACWI weight
    limit = max_deviation * acwi
    active = active.clip(lower=-limit, upper=limit)

    # over- and underweights must add up to zero
    over = active[active > 0].sum()
    under = -active[active < 0].sum()
    if over > under:
        active[active > 0] *= under / over
    elif under > over:
        active[active < 0] *= over / under

    return acwi + active


# %% Backtest
# Day by day: prices move and the weights drift; on rebalancing dates the portfolio is set back to
# the target weights and trading costs are paid. Signals and trades both use the closing price of
# the rebalancing day, so every decision only uses information available at that time.
@dataclass
class BacktestResult:
    nav: pd.Series         # portfolio value, starts at 1
    weights: pd.DataFrame  # weights at the end of each day
    trades: pd.DataFrame   # one row per rebalancing date


def run_backtest(prices, acwi_weights, params, with_tilt=True):
    all_prices = region_prices(prices)
    sample = all_prices.loc[params.start:params.end]
    daily_returns = sample.pct_change().fillna(0.0).to_numpy()
    rebalancing_days = set(rebalancing_dates(sample.index, params))

    value = 1.0
    weights = None  # numpy array in the order of REGIONS
    values, weight_history, trades = [], [], []

    for i, day in enumerate(sample.index):
        # 1) prices move: the portfolio value changes and the weights drift
        if weights is not None:
            grown = weights * (1 + daily_returns[i])
            value = value * grown.sum()
            weights = grown / grown.sum()

        # 2) rebalancing: new target weights, trading costs on the traded volume
        if day in rebalancing_days:
            acwi, report_date = acwi_weights_on(day, acwi_weights, all_prices)
            momentum = momentum_on(day, all_prices, params.lookback_months)
            tilt = params.tilt_pp if with_tilt else 0.0
            target = target_weights(acwi, momentum, tilt, params.max_rel_deviation)

            if weights is None:
                turnover = 0.0  # the initial purchase is not charged
            else:
                turnover = float(np.abs(target.to_numpy() - weights).sum())
            cost = turnover * params.cost_bps / 10000
            value = value * (1 - cost)

            trade = {"date": day, "anchor": report_date, "turnover": turnover, "cost": cost}
            for region in REGIONS:
                trade[f"base {region}"] = acwi[region]
                trade[f"signal {region}"] = momentum[region]
                trade[f"target {region}"] = target[region]
            trades.append(trade)
            weights = target.to_numpy()

        values.append(value)
        weight_history.append(weights)

    return BacktestResult(
        nav=pd.Series(values, index=sample.index, name="nav"),
        weights=pd.DataFrame(weight_history, index=sample.index, columns=REGIONS),
        trades=pd.DataFrame(trades).set_index("date"),
    )


def benchmark_nav(prices, params):
    acwi = prices[BENCHMARK].loc[params.start:params.end]
    return (acwi / acwi.iloc[0]).rename("nav")


# %% Performance measures
# All measures use monthly returns. Averages are multiplied by 12, volatilities by the square root
# of 12. Alpha and beta come from a regression of the monthly excess returns on the excess returns of
# the MSCI ACWI; alpha is multiplied by 12, beta is not annualised. All values are in EUR.
def monthly_returns(nav):
    month_end_values = nav.resample("ME").last()
    # if the sample does not start on a month-end, the first month runs from the start day
    if nav.index[0] < month_end_values.index[0]:
        month_end_values = pd.concat([nav.iloc[:1], month_end_values])
    return month_end_values.pct_change().dropna()


def performance_table(returns, benchmark, rf):
    rf = rf.reindex(returns.index)
    benchmark_returns = returns[benchmark]
    rows = {}

    for name in returns.columns:
        r = returns[name]
        excess = r - rf
        benchmark_excess = benchmark_returns - rf

        # CAPM regression: excess = alpha + beta * benchmark_excess + error
        regression = sm.OLS(excess.to_numpy(), sm.add_constant(benchmark_excess.to_numpy())).fit()
        alpha_monthly, beta = regression.params
        alpha_t = regression.tvalues[0]

        active = r - benchmark_returns
        tracking_error = active.std() * np.sqrt(12)
        growth = (1 + r).cumprod()

        rows[name] = {
            "Average return p.a.": r.mean() * 12,
            "Geometric return p.a.": growth.iloc[-1] ** (12 / len(r)) - 1,
            "Volatility p.a.": r.std() * np.sqrt(12),
            "Sharpe ratio": excess.mean() * 12 / (excess.std() * np.sqrt(12)),
            "Alpha p.a.": alpha_monthly * 12,
            "Alpha t-stat": alpha_t if name != benchmark else np.nan,
            "Beta": beta,
            "Tracking error p.a.": tracking_error,
            "Information ratio": active.mean() * 12 / tracking_error if tracking_error > 0 else np.nan,
            "Max drawdown": (growth / growth.cummax() - 1).min(),
        }
    return pd.DataFrame(rows).T


def run_all(prices, acwi_weights, rf, params):
    """Strategy, passive replica (same ETFs, no tilt) and MSCI ACWI, with their performance table."""
    strategy = run_backtest(prices, acwi_weights, params, with_tilt=True)
    passive = run_backtest(prices, acwi_weights, params, with_tilt=False)

    navs = pd.DataFrame({
        "Regional Momentum": strategy.nav,
        "Passive replica": passive.nav,
        BENCHMARK_NAME: benchmark_nav(prices, params),
    }).dropna()
    returns = navs.apply(monthly_returns)
    table = performance_table(returns, BENCHMARK_NAME, rf)

    years = (navs.index[-1] - navs.index[0]).days / 365.25
    table["Turnover p.a."] = [strategy.trades["turnover"].sum() / years, passive.trades["turnover"].sum() / years, 0.0]
    table["Trading costs p.a."] = [strategy.trades["cost"].sum() / years, passive.trades["cost"].sum() / years, 0.0]
    return {"strategy": strategy, "passive": passive, "navs": navs, "returns": returns, "table": table}


def sensitivity(prices, acwi_weights, rf, base_params, grid):
    """Re-run the strategy, changing one assumption at a time."""
    columns = ["Average return p.a.", "Volatility p.a.", "Sharpe ratio", "Alpha p.a.", "Beta",
               "Information ratio", "Turnover p.a."]
    rows = []
    for setting, values in grid.items():
        for value in values:
            params = replace(base_params, **{setting: value})
            result = run_all(prices, acwi_weights, rf, params)["table"].loc["Regional Momentum"]
            rows.append({"parameter": setting, "value": value, **result[columns].to_dict()})
    return pd.DataFrame(rows).set_index(["parameter", "value"])


# %% Risk class (SRI)
# The summary risk indicator (SRI, scale 1 to 7) of the EU PRIIPs rules, which every key information
# document for retail investors in Portugal shows. For an equity fund without credit risk the SRI equals
# the market risk class. It comes from the 97.5% value-at-risk over the recommended holding period,
# estimated from 5 years of daily returns with the Cornish-Fisher formula (which adds skewness and fat
# tails to the normal distribution), and turned into a "VaR-equivalent volatility" (VEV).
VEV_CLASS_LIMITS = [0.005, 0.05, 0.12, 0.20, 0.30, 0.80]  # upper limits of classes 1 to 6


def risk_class(nav, end, holding_years=7, history_years=5):
    """SRI of a daily value series, using the 5 years of history up to `end`."""
    end = pd.Timestamp(end)
    values = nav.loc[end - pd.DateOffset(years=history_years):end]
    log_returns = np.log(values).diff().dropna()

    sigma = log_returns.std()
    skew = log_returns.skew()
    excess_kurtosis = log_returns.kurt()  # pandas already subtracts 3
    n = 256 * holding_years  # trading days in the holding period (PRIIPs uses 256 per year)

    var = (sigma * np.sqrt(n) * (-1.96 + 0.474 * skew / np.sqrt(n) - 0.0687 * excess_kurtosis / n
                                 + 0.146 * skew ** 2 / n)
           - 0.5 * sigma ** 2 * n)
    vev = (np.sqrt(3.842 - 2 * var) - 1.96) / np.sqrt(holding_years)

    sri = 1
    for limit in VEV_CLASS_LIMITS:
        if vev >= limit:
            sri += 1
    return {
        "Volatility p.a. (daily, 5y)": sigma * np.sqrt(256),
        "Skewness": skew,
        "Excess kurtosis": excess_kurtosis,
        "VaR 97.5% (holding period)": var,
        "VaR-equivalent volatility": vev,
        "SRI (1-7)": sri,
    }
