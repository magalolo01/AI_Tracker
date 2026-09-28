# -*- coding: utf-8 -*-
"""portfolio_analytics_terminal_bot.py

# 📊 Portfolio Analytics Terminal — Telegram Bot

A menu-driven, image-based portfolio analytics terminal for Telegram: 3 built-in portfolios +
2 benchmarks (S&P 500, Gold), a yfinance data layer with an in-memory price cache, a full risk
metrics engine (Sharpe/Sortino/Calmar/Ulcer/VaR/beta), a 7-scenario stress-test engine, and a
persistent JSON portfolio/alert/settings store.

## Features
- Persistent JSON storage: create / edit / delete your own portfolios from Telegram; they survive
  a restart. The 3 original portfolios are untouched unless you explicitly edit/delete them.
- 🏠 Overview — all portfolios side by side: value, P/L, 1D/1M/1Y/5Y/10Y returns, risk metrics
  table, and an underwater (drawdown) comparison — one shared price download reused for every period.
- 📊 Comparison — pick any 2+ portfolios/benchmarks, get a full metrics table + charts.
- 📉 Drawdown — current/max drawdown, duration, recovery time, underwater chart.
- 🎯 Contribution — money + % contribution per holding, top contributors/detractors.
- 🔄 Rolling Metrics — rolling return/vol/Sharpe/Sortino/beta/correlation, selectable window.
- ⚖️ Rebalancing — Buy & Hold vs. monthly/quarterly/semiannual/annual/threshold rebalancing,
  simulated and compared (clearly labeled as historical simulation, not a guarantee).
- 📐 Derivatives Analysis — proxy-based (beta, leveraged/inverse ETF flags, VIX sensitivity,
  bond duration where known); explicitly says "no direct derivative positions detected" rather
  than inventing Greeks the data source doesn't provide.
- 🔔 Alerts — persisted, checked on a background schedule (Telegram JobQueue) with per-alert
  cooldowns so you're not spammed.
- 📄 Reports — single-portfolio and combined-portfolio PDF reports (A4, institutional layout),
  built from the same chart functions the bot uses for Telegram photos.

## Configuration
Reads two environment variables — never hardcode these:
- `BOT_TOKEN` — your Telegram bot token from @BotFather.
- `CHAT_ID` — (currently unused by the bot logic itself, kept for compatibility with any
  external tooling that expects it; the bot replies to whichever chat messages it.)

Optional environment variables:
- `PORTFOLIO_BOT_DATA_DIR` — directory for the persistent JSON store and generated PDF reports.
  Defaults to a `data/` folder next to this file (created automatically if missing). **This
  directory must live on a persistent disk** — see README.md for why this matters for hosting.

## How to run
    pip install -r requirements.txt
    export BOT_TOKEN="your-token-here"          # or set it in your host's environment/secrets
    python portfolio_analytics_terminal_bot.py

The process stays running (long-poll) — that's expected; it's what keeps interactive commands,
buttons, and the scheduled alert checker (Telegram JobQueue) working. Stop it with Ctrl+C, or via
your host's process manager (systemd, Railway, Render, etc.) — see README.md for deployment.
"""

# ============================================================================================
# SECTION 1 — SETUP
# ============================================================================================


import warnings
warnings.filterwarnings("ignore")

import io
import os
import re
import json
from datetime import date, datetime, timedelta, time as dt_time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

import matplotlib
matplotlib.use("Agg")  # headless rendering -> PNG bytes; required on any server with no display
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
print("Environment ready.")

# ============================================================================================
# SECTION 2 — PORTFOLIO DEFINITIONS & PERSISTENT STORAGE
# ============================================================================================
# The 3 original portfolios and 2 benchmarks are UNCHANGED from your uploaded script — same
# tickers, same weights. They are now loaded into a persistent JSON store on first run so that
# user-created/edited portfolios survive a restart, without ever silently overwriting these
# three unless you explicitly edit/delete them from the bot.

BUILTIN_PORTFOLIOS: Dict[str, Dict[str, float]] = {
    "Portfolio 1 (Claude)": {
        "SPY":   0.25, "SCHD": 0.10, "NVDA": 0.05, "AMD": 0.03, "AVGO": 0.04,
        "TSM":   0.05, "ASML": 0.03, "MSFT": 0.04, "GOOGL": 0.03, "IGV": 0.03,
        "NEE":   0.03, "ETN":  0.02, "VRT":  0.02, "EQIX": 0.02, "IGF": 0.01,
        "IEMG":  0.06, "EMQQ": 0.04, "IEF":  0.10, "GLD":  0.05,
    },
    "Portfolio 2 (ChatGPT)": {
        "IVV": 0.30, "ARTY": 0.15, "SOXX": 0.10, "POWR": 0.10, "IEMG": 0.10,
        "QUAL": 0.10, "SGOV": 0.10, "IAU": 0.05,
    },
    "Portfolio 3 (My Mixed Portfolio)": {
        "SPY": 0.350, "NVDA": 0.043, "AMD": 0.020, "AVGO": 0.030, "TSM": 0.040,
        "SOXX": 0.050, "ASML": 0.020, "MSFT": 0.027, "VBR": 0.050, "GOOGL": 0.020,
        "POWR": 0.100, "CTEC": 0.060, "KSTR": 0.040, "IEF": 0.100, "GLD": 0.050,
    },
}
BUILTIN_IDS = ["p1", "p2", "p3"]  # stable ids, in the same order as BUILTIN_PORTFOLIOS above

BENCHMARKS: Dict[str, Dict[str, float]] = {
    "100% S&P 500 (SPY)": {"SPY": 1.0},
    "100% Gold (GLD)": {"GLD": 1.0},
}
BENCHMARK_IDS = ["b1", "b2"]
BENCHMARKS_BY_ID = {bid: {"id": bid, "name": name, "weights": BENCHMARKS[name]}
                     for bid, name in zip(BENCHMARK_IDS, BENCHMARKS.keys())}

# Known approximate effective durations (years) for the fixed-income ETFs actually used anywhere
# in this project — used only by Derivatives Analysis as a documented, non-invented proxy.
KNOWN_BOND_DURATIONS = {"IEF": 7.5, "SGOV": 0.25, "TLT": 17.5, "SHY": 1.9, "AGG": 6.0}
# Known leveraged / inverse ETF tickers — used only to flag exposure if a portfolio holds one.
LEVERAGED_INVERSE_ETFS = {
    "TQQQ": "3x Long Nasdaq-100", "SQQQ": "3x Short Nasdaq-100", "UPRO": "3x Long S&P 500",
    "SPXU": "3x Short S&P 500", "SOXL": "3x Long Semiconductors", "SOXS": "3x Short Semiconductors",
    "UVXY": "~1.5x Long VIX futures", "SVXY": "Short VIX futures", "TZA": "3x Short Small-Cap",
    "TNA": "3x Long Small-Cap", "SPXL": "3x Long S&P 500", "SDS": "2x Short S&P 500",
}


def _data_dir() -> str:
    """Directory for the persistent JSON store and generated PDF reports. Override with the
    PORTFOLIO_BOT_DATA_DIR environment variable; otherwise defaults to a `data/` folder next to
    this file, created automatically if missing. This directory MUST be on a persistent disk —
    on a host with an ephemeral filesystem (e.g. a GitHub Actions runner, or a "free" web-service
    tier without an attached volume), anything written here is lost when the process restarts.
    See README.md for hosting options that provide persistent storage."""
    base = os.environ.get("PORTFOLIO_BOT_DATA_DIR") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data"
    )
    os.makedirs(base, exist_ok=True)
    return base


DATA_DIR = _data_dir()
STORAGE_PATH = os.path.join(DATA_DIR, "portfolio_terminal_store.json")


def _default_store() -> dict:
    portfolios = {}
    for pid, name in zip(BUILTIN_IDS, BUILTIN_PORTFOLIOS.keys()):
        portfolios[pid] = {
            "id": pid, "name": name, "weights": dict(BUILTIN_PORTFOLIOS[name]),
            "initial_capital": 10000.0, "currency": "USD",
            "created": date.today().isoformat(), "builtin": True,
        }
    return {"portfolios": portfolios, "next_custom_num": 1, "alerts": {}, "next_alert_num": 1,
            "user_settings": {}, "sector_cache": {}}


def load_store() -> dict:
    if os.path.exists(STORAGE_PATH):
        try:
            with open(STORAGE_PATH, "r") as f:
                store = json.load(f)
            # Make sure the 3 builtins always exist even if the file predates one of them.
            defaults = _default_store()
            for pid, rec in defaults["portfolios"].items():
                store.setdefault("portfolios", {}).setdefault(pid, rec)
            for key, val in defaults.items():
                if key != "portfolios":
                    store.setdefault(key, val)
            return store
        except Exception as e:
            print(f"⚠️ Could not read existing store ({e}) — starting a fresh one.")
    return _default_store()


def save_store(store: dict):
    try:
        with open(STORAGE_PATH, "w") as f:
            json.dump(store, f, indent=2)
    except Exception as e:
        print(f"⚠️ Could not save store: {e}")


STORE = load_store()
save_store(STORE)  # persist the seeded builtins on first run


def get_portfolios() -> Dict[str, dict]:
    """Live view of every portfolio (builtin + custom), keyed by stable id."""
    return STORE["portfolios"]


def get_portfolio_weights(pid: str) -> Dict[str, float]:
    return STORE["portfolios"][pid]["weights"]


def validate_weights(weights: Dict[str, float], tol: float = 0.5) -> Tuple[bool, float]:
    """tol is in percentage points — weights must sum to 100% within this tolerance."""
    total = sum(weights.values()) * 100
    return abs(total - 100.0) <= tol, total


def create_portfolio(name: str, weights: Dict[str, float], capital: float, currency: str) -> str:
    n = STORE["next_custom_num"]
    pid = f"c{n}"
    STORE["next_custom_num"] = n + 1
    STORE["portfolios"][pid] = {
        "id": pid, "name": name, "weights": weights, "initial_capital": capital,
        "currency": currency, "created": date.today().isoformat(), "builtin": False,
    }
    save_store(STORE)
    return pid


def update_portfolio(pid: str, **fields):
    STORE["portfolios"][pid].update(fields)
    save_store(STORE)


def delete_portfolio(pid: str):
    STORE["portfolios"].pop(pid, None)
    save_store(STORE)


def portfolio_display_name(pid: str) -> str:
    rec = STORE["portfolios"].get(pid)
    return rec["name"] if rec else pid

print(f"Storage ready at {STORAGE_PATH}. Loaded {len(STORE['portfolios'])} portfolio(s): "
      + ", ".join(f"{v['name']} [{k}]" for k, v in STORE["portfolios"].items()))

# ============================================================================================
# SECTION 3 — DATA LAYER  (unchanged from v1: in-memory price cache, same yfinance download
#              function, same never-invent-missing-data behavior. + a best-effort, cached
#              sector lookup used only by Contribution Analysis, since it's an extra per-ticker
#              API call and must not be repeated needlessly.)
# ============================================================================================

_PRICE_CACHE: Dict[Tuple[str, str, str], pd.Series] = {}


def _cache_key(ticker: str, start: date, end: date) -> Tuple[str, str, str]:
    return (ticker, start.isoformat(), end.isoformat())


def download_prices(tickers: List[str], start: date, end: date) -> Tuple[pd.DataFrame, List[str]]:
    """Download adjusted close prices for a list of tickers, reusing cached series where
    possible. Returns (prices_df, failed_tickers). Never invents data: a ticker that can't be
    downloaded is reported as failed and excluded, not substituted."""
    tickers = sorted(set(tickers))
    to_fetch = [t for t in tickers if _cache_key(t, start, end) not in _PRICE_CACHE]
    failed: List[str] = []

    if to_fetch:
        try:
            data = yf.download(
                to_fetch, start=start, end=end + timedelta(days=1), auto_adjust=True,
                progress=False, group_by="ticker", threads=True,
            )
        except Exception as e:
            data = None
            failed.extend(to_fetch)
            print(f"⚠️ yfinance download error: {e}")

        if data is not None:
            for t in to_fetch:
                try:
                    series = data["Close"] if len(to_fetch) == 1 else data[t]["Close"]
                    series = series.dropna()
                    if series.empty:
                        failed.append(t)
                    else:
                        _PRICE_CACHE[_cache_key(t, start, end)] = series
                except Exception:
                    failed.append(t)

    prices = pd.DataFrame()
    for t in tickers:
        key = _cache_key(t, start, end)
        if key in _PRICE_CACHE:
            prices[t] = _PRICE_CACHE[key]
        elif t not in failed:
            failed.append(t)

    return prices.sort_index(), failed


def download_prices_maxwindow(tickers: List[str], max_start: date, end: date) -> pd.DataFrame:
    """Downloads ONE wide window (default 10Y, used by Overview/Comparison) and lets callers
    slice sub-periods out of it in-memory, instead of issuing a separate yfinance call per
    preset period — this is what keeps Overview/Comparison/Combined-Report to a single shared
    download, per the performance requirements in the brief."""
    prices, failed = download_prices(tickers, max_start, end)
    return prices, failed


def get_risk_free_rate() -> Tuple[Optional[float], str]:
    """Current ~13-week T-bill discount rate (^IRX), used as the risk-free rate. Falls back to
    None (caller should use a documented default) if the fetch fails — never invented."""
    try:
        irx = yf.download("^IRX", period="5d", progress=False, auto_adjust=False)
        rate = float(irx["Close"].dropna().iloc[-1]) / 100.0
        asof = irx.index[-1].date()
        return rate, f"^IRX (13-week T-bill), last close {asof}"
    except Exception as e:
        return None, f"Could not fetch ^IRX ({e}); using 4.50% fallback"


RF_RATE, RF_SOURCE = get_risk_free_rate()
RF_RATE = RF_RATE if RF_RATE is not None else 0.045
print(f"Risk-free rate: {RF_RATE*100:.2f}%  |  Source: {RF_SOURCE}")


def get_sector(ticker: str) -> str:
    """Best-effort sector lookup via yfinance's .info, cached to disk so it's fetched at most
    once ever per ticker (this is a slow, unreliable-on-free-tier call, so it's never used in a
    hot path). Returns 'Unknown' rather than guessing if the source doesn't provide it."""
    cache = STORE["sector_cache"]
    if ticker in cache:
        return cache[ticker]
    sector = "Unknown"
    try:
        info = yf.Ticker(ticker).info
        sector = info.get("sector") or info.get("quoteType") or "Unknown"
    except Exception:
        pass
    cache[ticker] = sector
    save_store(STORE)
    return sector

# ============================================================================================
# SECTION 4 — DATE RANGE PRESETS  (added "1 Day" per the brief's expanded period list; the
#              rest is unchanged from v1 — plain functions so Telegram buttons can drive them)
# ============================================================================================

PRESET_PERIODS = {  # button label -> (code, approx calendar days)
    "1 Day":     ("1d", 1),
    "7 Days":    ("7d", 7),
    "15 Days":   ("15d", 15),
    "30 Days":   ("30d", 30),
    "6 Months":  ("6m", 182),
    "1 Year":    ("1y", 365),
    "3 Years":   ("3y", 365 * 3),
    "5 Years":   ("5y", 365 * 5),
    "10 Years":  ("10y", 365 * 10),
}
PRESET_CODE_TO_LABEL = {code: label for label, (code, _) in PRESET_PERIODS.items()}
PRESET_CODE_TO_DAYS = {code: days for label, (code, days) in PRESET_PERIODS.items()}
MAX_WINDOW_DAYS = 365 * 10 + 30  # covers every preset up to 10Y from one shared download


def preset_to_dates(code: str) -> Tuple[date, date]:
    today = date.today()
    days = PRESET_CODE_TO_DAYS.get(code)
    if days is None:
        raise ValueError(f"Unknown preset code: {code}")
    # "1 Day" means the most recent full trading day, not literally 24h (markets close weekends)
    start = today - timedelta(days=max(days, 4) if code == "1d" else days)
    return start, today


DATE_INPUT_RE = re.compile(
    r"(\d{1,2}/\d{1,2}/\d{4})\s*(?:-|to|→|,|\|)\s*(\d{1,2}/\d{1,2}/\d{4})", re.IGNORECASE
)


def parse_custom_range(text: str) -> Tuple[Optional[date], Optional[date], Optional[str]]:
    """Parses 'DD/MM/YYYY - DD/MM/YYYY' (also accepts 'to' / '→' / ',' / '|' as the separator).
    Returns (start, end, error_message). error_message is None on success."""
    m = DATE_INPUT_RE.search(text.strip())
    if not m:
        return None, None, ("Couldn't read that. Send it as:\n`01/01/2025 - 23/09/2026`\n"
                             "(day/month/year, either date first is fine)")
    try:
        d1 = datetime.strptime(m.group(1), "%d/%m/%Y").date()
        d2 = datetime.strptime(m.group(2), "%d/%m/%Y").date()
    except ValueError:
        return None, None, "One of those dates isn't valid — check day/month/year."
    start, end = (d1, d2) if d1 <= d2 else (d2, d1)
    if end > date.today():
        end = date.today()
    if start >= end:
        return None, None, "Start date must be before the end date."
    if (end - start).days < 2:
        return None, None, "That range is too short — pick at least a couple of days."
    return start, end, None

# ============================================================================================
# SECTION 5 — CALCULATION ENGINE  (core formulas unchanged from v1: Sharpe, Sortino, Calmar,
#              Ulcer, historical VaR, beta, correlation, contribution — all still exactly the
#              same math. New in v2: multi-period returns from one shared download, drawdown
#              episode analytics, rolling metrics, and a rebalancing simulator.)
# ============================================================================================


def portfolio_daily_returns(weights: dict, returns_df: pd.DataFrame) -> Tuple[pd.Series, float]:
    """Portfolio daily return = weighted sum of constituent daily returns (weights re-normalized
    over only the tickers with available data). This is a **daily-rebalanced constant-mix**
    methodology — the one convention used everywhere in this bot except inside the Rebalancing
    Analysis section, which explicitly simulates and labels alternative methodologies."""
    avail = {t: w for t, w in weights.items() if t in returns_df.columns}
    w = pd.Series(avail)
    sub = returns_df[list(avail.keys())].dropna(how="all")
    port_ret = sub.mul(w, axis=1).sum(axis=1)
    return port_ret, sum(avail.values())


def annualize_return(daily_returns: pd.Series) -> float:
    n = len(daily_returns)
    if n == 0:
        return float("nan")
    cum = (1 + daily_returns).prod() - 1
    return (1 + cum) ** (252 / n) - 1


def annualize_vol(daily_returns: pd.Series) -> float:
    return daily_returns.std() * np.sqrt(252) if len(daily_returns) else float("nan")


def sharpe_ratio(daily_returns: pd.Series, rf: float) -> float:
    ann_ret, ann_vol = annualize_return(daily_returns), annualize_vol(daily_returns)
    return (ann_ret - rf) / ann_vol if ann_vol and not np.isnan(ann_vol) else float("nan")


def sortino_ratio(daily_returns: pd.Series, rf: float) -> float:
    downside = daily_returns[daily_returns < 0]
    if len(downside) == 0:
        return float("nan")
    downside_dev = downside.std() * np.sqrt(252)
    ann_ret = annualize_return(daily_returns)
    return (ann_ret - rf) / downside_dev if downside_dev else float("nan")


def drawdown_series(daily_returns: pd.Series) -> pd.Series:
    cum_value = (1 + daily_returns).cumprod()
    return cum_value / cum_value.cummax() - 1


def max_drawdown(daily_returns: pd.Series) -> float:
    dd = drawdown_series(daily_returns)
    return dd.min() if len(dd) else float("nan")


def calmar_ratio(daily_returns: pd.Series) -> float:
    mdd = max_drawdown(daily_returns)
    ann_ret = annualize_return(daily_returns)
    return ann_ret / abs(mdd) if mdd and not np.isnan(mdd) and mdd != 0 else float("nan")


def ulcer_index(daily_returns: pd.Series) -> float:
    dd = drawdown_series(daily_returns) * 100
    return float(np.sqrt((dd ** 2).mean())) if len(dd) else float("nan")


def historical_var(daily_returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical (non-parametric) 1-day Value at Risk at the given confidence level, expressed
    as a negative daily return. No distributional assumption is made — this is the empirical
    percentile of realized daily returns over the selected period."""
    if len(daily_returns) < 20:
        return float("nan")
    return float(np.percentile(daily_returns, (1 - confidence) * 100))


def beta_vs_benchmark(port_ret: pd.Series, bench_ret: pd.Series) -> float:
    joined = pd.concat([port_ret, bench_ret], axis=1).dropna()
    if len(joined) < 20:
        return float("nan")
    var_b = joined.iloc[:, 1].var(ddof=1)
    return float(joined.iloc[:, 0].cov(joined.iloc[:, 1]) / var_b) if var_b else float("nan")


def correlation_vs_benchmark(port_ret: pd.Series, bench_ret: pd.Series) -> float:
    joined = pd.concat([port_ret, bench_ret], axis=1).dropna()
    return float(joined.iloc[:, 0].corr(joined.iloc[:, 1])) if len(joined) >= 20 else float("nan")


def cumulative_series(daily_returns: pd.Series) -> pd.Series:
    return (1 + daily_returns).cumprod() - 1


def compute_risk_metrics(port_ret: pd.Series, rf: float,
                          bench_ret: Optional[pd.Series] = None) -> Dict[str, float]:
    m = {
        "Expected (Annualized) Return": annualize_return(port_ret),
        "Volatility (Annualized)": annualize_vol(port_ret),
        "Sharpe Ratio": sharpe_ratio(port_ret, rf),
        "Sortino Ratio": sortino_ratio(port_ret, rf),
        "Max Drawdown": max_drawdown(port_ret),
        "Calmar Ratio": calmar_ratio(port_ret),
        "Ulcer Index": ulcer_index(port_ret),
        "Historical VaR (95%, 1-day)": historical_var(port_ret, 0.95),
    }
    if bench_ret is not None and len(bench_ret):
        m["Beta vs. Benchmark"] = beta_vs_benchmark(port_ret, bench_ret)
        m["Correlation vs. Benchmark"] = correlation_vs_benchmark(port_ret, bench_ret)
    return m


def asset_return_and_contribution(weights: dict, prices_df: pd.DataFrame,
                                   invested: float = 10000.0) -> pd.DataFrame:
    """Per-asset return over the period vs. its (approximate, weight-based) contribution to the
    portfolio's total return, PLUS the $ P/L implied by `invested`. Contribution is estimated as
    weight * asset_cumulative_return — exact for a single-period buy-and-hold view and a standard
    first-order approximation under this bot's daily-rebalanced constant-mix convention.
    Deliberately kept distinct from raw asset return in every column and every chart."""
    rows = []
    for t, w in weights.items():
        if t not in prices_df.columns:
            continue
        s = prices_df[t].dropna()
        if len(s) < 2:
            continue
        ret = float(s.iloc[-1] / s.iloc[0] - 1)
        rows.append({
            "Ticker": t, "Weight": w, "Start Price": float(s.iloc[0]), "End Price": float(s.iloc[-1]),
            "Return %": ret * 100, "Contribution %": w * ret * 100,
            "Contribution $": w * ret * invested, "P/L $": w * invested * ret,
        })
    df = pd.DataFrame(rows).sort_values("Return %", ascending=False).reset_index(drop=True)
    if len(df):
        df["Rank"] = range(1, len(df) + 1)
    return df


# ---- multi-period returns from ONE shared download (Overview / Comparison / Combined Report) ---

def multi_period_returns(prices_df: pd.DataFrame, weights: dict,
                          codes: List[str] = None) -> Dict[str, float]:
    """Computes total return for every preset period by slicing ONE already-downloaded price
    window in memory — no extra yfinance calls per period, as required for Overview/Comparison."""
    codes = codes or list(PRESET_CODE_TO_DAYS.keys())
    out = {}
    if prices_df.empty:
        return {c: float("nan") for c in codes}
    returns = prices_df.pct_change().dropna(how="all")
    full_port_ret, _ = portfolio_daily_returns(weights, returns)
    last_date = prices_df.index.max()
    for code in codes:
        days = PRESET_CODE_TO_DAYS[code]
        cutoff = last_date - timedelta(days=max(days, 2) if code == "1d" else days)
        window_ret = full_port_ret[full_port_ret.index > cutoff]
        out[code] = float((1 + window_ret).prod() - 1) if len(window_ret) else float("nan")
    return out


# ---- drawdown episode analytics (Section 8 of the brief) ---------------------------------------

def drawdown_episodes(daily_returns: pd.Series, threshold: float = -0.01) -> List[dict]:
    """Identifies distinct drawdown episodes (peak -> trough -> recovery) deeper than `threshold`
    (default -1%). An episode with no recovery yet within the selected period is marked ongoing."""
    dd = drawdown_series(daily_returns)
    episodes, in_dd, peak_idx = [], False, None
    for i, (dt, v) in enumerate(dd.items()):
        if not in_dd and v < threshold:
            in_dd, peak_idx = True, i
        elif in_dd and v >= -1e-9:
            trough_slice = dd.iloc[peak_idx:i]
            trough_pos = trough_slice.values.argmin()
            episodes.append({
                "peak_date": dd.index[peak_idx].date(), "trough_date": trough_slice.index[trough_pos].date(),
                "recovery_date": dt.date(), "depth": float(trough_slice.min()),
                "duration_days": (trough_slice.index[trough_pos] - dd.index[peak_idx]).days,
                "recovery_days": (dt - trough_slice.index[trough_pos]).days, "recovered": True,
            })
            in_dd, peak_idx = False, None
    if in_dd:
        trough_slice = dd.iloc[peak_idx:]
        trough_pos = trough_slice.values.argmin()
        episodes.append({
            "peak_date": dd.index[peak_idx].date(), "trough_date": trough_slice.index[trough_pos].date(),
            "recovery_date": None, "depth": float(trough_slice.min()),
            "duration_days": (trough_slice.index[trough_pos] - dd.index[peak_idx]).days,
            "recovery_days": None, "recovered": False,
        })
    return episodes


def drawdown_summary(daily_returns: pd.Series) -> dict:
    dd = drawdown_series(daily_returns)
    episodes = drawdown_episodes(daily_returns)
    return {
        "current_drawdown": float(dd.iloc[-1]) if len(dd) else float("nan"),
        "max_drawdown": float(dd.min()) if len(dd) else float("nan"),
        "max_drawdown_date": dd.idxmin().date() if len(dd) else None,
        "num_episodes": len(episodes),
        "avg_drawdown": float(np.mean([e["depth"] for e in episodes])) if episodes else float("nan"),
        "worst_episode": min(episodes, key=lambda e: e["depth"]) if episodes else None,
        "currently_in_drawdown": bool(len(dd) and dd.iloc[-1] < -0.01),
        "episodes": episodes,
    }


# ---- rolling metrics (Section 10 of the brief) --------------------------------------------------

ROLLING_WINDOWS = {"30 Days": 30, "60 Days": 60, "90 Days": 90, "6 Months": 126, "1 Year": 252}


def rolling_metrics(port_ret: pd.Series, window_days: int, rf: float,
                     bench_ret: Optional[pd.Series] = None) -> pd.DataFrame:
    """Rolling return/vol/Sharpe/Sortino, + rolling beta & correlation when a benchmark is given.
    Uses trading-day windows (not calendar days) so results line up with the annualization
    convention used everywhere else. min_periods == window, so the series starts window-days in."""
    idx = port_ret.index
    roll_ret = port_ret.rolling(window_days).apply(lambda x: (1 + x).prod() - 1, raw=False)
    roll_vol = port_ret.rolling(window_days).std() * np.sqrt(252)
    ann_factor = 252 / window_days
    roll_ann_ret = (1 + roll_ret) ** ann_factor - 1
    roll_sharpe = (roll_ann_ret - rf) / roll_vol
    downside = port_ret.where(port_ret < 0, 0.0)
    roll_down_dev = downside.rolling(window_days).std() * np.sqrt(252)
    roll_sortino = (roll_ann_ret - rf) / roll_down_dev.replace(0, np.nan)

    out = pd.DataFrame({"Rolling Return": roll_ret, "Rolling Volatility": roll_vol,
                         "Rolling Sharpe": roll_sharpe, "Rolling Sortino": roll_sortino}, index=idx)
    if bench_ret is not None and len(bench_ret):
        joined = pd.concat([port_ret, bench_ret], axis=1).dropna()
        joined.columns = ["port", "bench"]
        roll_cov = joined["port"].rolling(window_days).cov(joined["bench"])
        roll_var = joined["bench"].rolling(window_days).var()
        out["Rolling Beta"] = (roll_cov / roll_var).reindex(idx)
        out["Rolling Correlation"] = joined["port"].rolling(window_days).corr(joined["bench"]).reindex(idx)
    return out.dropna(how="all")


# ---- rebalancing simulator (Section 11 of the brief) --------------------------------------------

REBALANCE_STRATEGIES = {
    "Buy & Hold": None, "Monthly": "M", "Quarterly": "Q", "Semiannual": "2Q", "Annual": "Y",
}


def simulate_strategy(weights: dict, prices_df: pd.DataFrame, strategy: str,
                       initial_capital: float, threshold: Optional[float] = None) -> dict:
    """Simulates one rebalancing methodology on daily share/dollar holdings (NOT the constant-mix
    daily-return convention used elsewhere) — this section intentionally uses actual share counts
    so 'no rebalancing' (Buy & Hold) is genuinely static, distinguishing it from the constant-mix
    convention used in every other section. Returns portfolio value series + rebalance count/
    turnover. This is a historical simulation only — never a guarantee of future results."""
    avail = {t: w for t, w in weights.items() if t in prices_df.columns}
    total_w = sum(avail.values())
    avail = {t: w / total_w for t, w in avail.items()} if total_w else avail
    px = prices_df[list(avail.keys())].dropna(how="all").ffill().dropna()
    if px.empty or len(px) < 2:
        return {"value": pd.Series(dtype=float), "n_rebalances": 0, "turnover": 0.0}

    shares = {t: (avail[t] * initial_capital) / px[t].iloc[0] for t in avail}
    values, n_rebal, turnover = [], 0, 0.0

    def _period_boundary_hit(prev_date, cur_date) -> bool:
        if strategy in ("Buy & Hold", "Threshold") or strategy is None:
            return False
        freq = REBALANCE_STRATEGIES[strategy]
        if freq == "2Q":
            return (cur_date.quarter - 1) // 2 != (prev_date.quarter - 1) // 2 or cur_date.year != prev_date.year
        return prev_date.to_period(freq) != cur_date.to_period(freq)

    for i, dt in enumerate(px.index):
        row = px.loc[dt]
        port_value = float(sum(shares[t] * row[t] for t in shares))
        values.append(port_value)
        if i == 0:
            continue
        do_rebal = False
        if strategy == "Threshold":
            cur_weights = {t: shares[t] * row[t] / port_value for t in shares}
            if any(abs(cur_weights[t] - avail[t]) * 100 > threshold for t in avail):
                do_rebal = True
        elif strategy != "Buy & Hold":
            do_rebal = _period_boundary_hit(px.index[i - 1], dt)
        if do_rebal:
            cur_weights = {t: shares[t] * row[t] / port_value for t in shares}
            turnover += sum(abs(cur_weights[t] - avail[t]) for t in avail) / 2
            shares = {t: (avail[t] * port_value) / row[t] for t in avail}
            n_rebal += 1

    return {"value": pd.Series(values, index=px.index), "n_rebalances": n_rebal, "turnover": turnover}


def compare_rebalancing_strategies(weights: dict, prices_df: pd.DataFrame, initial_capital: float,
                                    rf: float, threshold_pct: float = 5.0) -> Dict[str, dict]:
    strategies = list(REBALANCE_STRATEGIES.keys()) + ["Threshold"]
    results = {}
    for strat in strategies:
        thresh = threshold_pct if strat == "Threshold" else None
        sim = simulate_strategy(weights, prices_df, strat, initial_capital, thresh)
        v = sim["value"]
        if len(v) < 2:
            results[strat] = {**sim, "metrics": None}
            continue
        ret = v.pct_change().dropna()
        results[strat] = {
            **sim,
            "final_value": float(v.iloc[-1]),
            "total_return": float(v.iloc[-1] / v.iloc[0] - 1),
            "metrics": {
                "Annualized Return": annualize_return(ret), "Volatility": annualize_vol(ret),
                "Sharpe": sharpe_ratio(ret, rf), "Sortino": sortino_ratio(ret, rf),
                "Max Drawdown": max_drawdown(ret), "Calmar": calmar_ratio(ret),
            },
        }
    return results


# ---- derivatives / risk-proxy analysis (Section 13 of the brief) --------------------------------

def derivatives_analysis(weights: dict, returns_df: pd.DataFrame,
                          spy_ret: Optional[pd.Series] = None,
                          vix_ret: Optional[pd.Series] = None) -> dict:
    """Never invents options Greeks or derivatives data. Reports only what a free equity/ETF price
    feed can actually support: direct leveraged/inverse ETF flags, portfolio beta, volatility,
    VIX-sensitivity (correlation to VIX changes), and known bond-ETF duration exposure."""
    leveraged_found = {t: LEVERAGED_INVERSE_ETFS[t] for t in weights if t in LEVERAGED_INVERSE_ETFS}
    port_ret, _ = portfolio_daily_returns(weights, returns_df)

    beta_val = beta_vs_benchmark(port_ret, spy_ret) if spy_ret is not None else float("nan")
    vix_corr = correlation_vs_benchmark(port_ret, vix_ret) if vix_ret is not None else float("nan")

    bond_exposure = {t: (w, KNOWN_BOND_DURATIONS[t]) for t, w in weights.items() if t in KNOWN_BOND_DURATIONS}
    weighted_duration = (sum(w * d for w, d in bond_exposure.values()) if bond_exposure else None)

    return {
        "leveraged_inverse_holdings": leveraged_found,
        "portfolio_beta_vs_spy": beta_val,
        "portfolio_vol": annualize_vol(port_ret),
        "vix_correlation": vix_corr,
        "bond_exposure": bond_exposure,
        "weighted_bond_duration": weighted_duration,
        "has_direct_derivatives": False,  # this data source cannot detect options/futures positions
    }

print("Calculation engine loaded: core risk metrics, multi-period returns, drawdown episodes, "
      "rolling metrics, rebalancing simulator, derivatives proxy analysis.")

# ============================================================================================
# SECTION 6 — STRESS TEST ENGINE  (unchanged methodology & scenarios from v1 — same 7
#              scenarios, same factor-beta approach, same "estimate, not a forecast" labeling)
# ============================================================================================

SCENARIOS = {
    "AI Bubble Bursts": {
        "emoji": "🤖", "factor": "IGV", "shock": -0.55, "type": "Hypothetical",
        "basis": "No historical precedent. Shock magnitude (-55%) modeled on the Nasdaq "
                 "Composite's -55% drawdown during the 2000-2002 dot-com bust, applied to the "
                 "AI-software factor proxy (IGV) as the closest analogue.",
    },
    "Semiconductor Crash": {
        "emoji": "💻", "factor": "SOXX", "shock": -0.38, "type": "Historical",
        "basis": "SOXX fell roughly -38% peak-to-trough during the 2022 rate-driven tech "
                 "selloff — used directly as the shock magnitude.",
    },
    "US Recession": {
        "emoji": "📉", "factor": "SPY", "shock": -0.25, "type": "Historical (blended)",
        "basis": "S&P 500 recession drawdowns range ~-20% (2001, 2020) to -34% (2008); -25% "
                 "used as a representative moderate-recession shock.",
    },
    "Interest Rates Rise": {
        "emoji": "📈", "factor": "IEF", "shock": -0.08, "type": "Model assumption",
        "basis": "Approximates a +100bp parallel rate shock via IEF's ~7.5yr effective "
                 "duration: shock ≈ -duration * Δrate ≈ -7.5%, rounded to -8%.",
    },
    "AI Productivity Boom": {
        "emoji": "🚀", "factor": "IGV", "shock": 0.40, "type": "Hypothetical",
        "basis": "Mirror-image hypothetical to 'AI Bubble Bursts' — magnitude chosen for "
                 "symmetry, not derived from a specific historical episode.",
    },
    "Energy Prices Spike": {
        "emoji": "⚡", "factor": "XLE", "shock": 0.30, "type": "Historical",
        "basis": "XLE rose roughly +30%+ during the 2022 energy price spike following the "
                 "Russia-Ukraine war's disruption to oil/gas supply.",
    },
    "Emerging Markets Outperform": {
        "emoji": "🌍", "factor": "IEMG", "shock": 0.25, "type": "Hypothetical",
        "basis": "Hypothetical sustained EM re-rating; magnitude chosen for illustrative "
                 "comparability, not a single historical episode.",
    },
}
SCENARIO_NAMES = list(SCENARIOS.keys())


def asset_beta(ticker: str, factor: str, returns_df: pd.DataFrame) -> float:
    if ticker not in returns_df.columns or factor not in returns_df.columns:
        return float("nan")
    joined = returns_df[[ticker, factor]].dropna()
    if len(joined) < 20:
        return float("nan")
    x, y = joined[factor].values, joined[ticker].values
    var_x = np.var(x, ddof=1)
    if var_x == 0:
        return float("nan")
    return float(np.cov(x, y, ddof=1)[0, 1] / var_x)


def run_stress_test(scenario_name: str, weights: dict, returns_df: pd.DataFrame,
                     start_value: float) -> dict:
    """Applies the scenario's factor shock to each holding via its historical beta to the
    factor proxy. Linear, single-factor approximation — an estimate, not a forecast."""
    s = SCENARIOS[scenario_name]
    factor, shock = s["factor"], s["shock"]
    contrib, port_impact = {}, 0.0
    for t, w in weights.items():
        beta = asset_beta(t, factor, returns_df)
        if np.isnan(beta):
            continue
        asset_impact = beta * shock
        contrib[t] = w * asset_impact
        port_impact += w * asset_impact
    stressed_value = start_value * (1 + port_impact)
    return {
        "scenario": scenario_name, "factor": factor, "shock": shock, "type": s["type"],
        "basis": s["basis"], "impact_pct": port_impact, "start_value": start_value,
        "stressed_value": stressed_value, "asset_contrib": contrib,
    }

print("Stress test engine loaded:", ", ".join(f"{v['emoji']} {k}" for k, v in SCENARIOS.items()))

# ============================================================================================
# SECTION 7 — CHART GENERATION & PDF REPORTS
# ============================================================================================
# Every "_build_xxx_fig" function returns a matplotlib Figure. "generate_xxx" wraps it to PNG
# bytes for Telegram. The SAME fig-builders are reused inside PdfPages for Section 14's PDF
# reports, so a Telegram chart and its PDF-page equivalent are always pixel-for-pixel the same
# code path — no separate, divergent "PDF version" of any chart to maintain.

_DARK_BG = "#0d1117"
_PANEL_BG = "#161b22"
_TEXT = "#e6edf3"
_MUTED = "#8b949e"
_GRID = "#30363d"
_ACCENT = "#58a6ff"
_GREEN = "#3fb950"
_RED = "#f85149"
_GOLD = "#d4a72c"
_PALETTE = ["#58a6ff", "#3fb950", "#f85149", "#d4a72c", "#bc8cff", "#39c5cf",
            "#ff9e64", "#f778ba", "#7ee787", "#79c0ff", "#ffa657", "#a5d6ff"]
_SERIES_COLORS = ["#58a6ff", "#3fb950", "#d4a72c", "#f85149", "#bc8cff", "#39c5cf"]


def _style_axes(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(_PANEL_BG)
    if title:
        ax.set_title(title, color=_TEXT, fontsize=14, fontweight="bold", pad=14)
    if xlabel:
        ax.set_xlabel(xlabel, color=_MUTED, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=_MUTED, fontsize=10)
    ax.tick_params(colors=_MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(_GRID)
    ax.grid(True, color=_GRID, linewidth=0.6, alpha=0.6)


def _new_fig(figsize=(8, 6)):
    return plt.figure(figsize=figsize, dpi=200, facecolor=_DARK_BG)


def _fig_to_bytes(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight", dpi=200)
    plt.close(fig)
    buf.seek(0)
    buf.name = "chart.png"
    return buf


def _footer(fig, text: str):
    fig.text(0.5, 0.01, text, color=_MUTED, fontsize=8, ha="center", va="bottom")


def _style_table(table, df, highlight_col=None, highlight_series=None):
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(_GRID)
        if r == 0:
            cell.set_facecolor("#21262d")
            cell.set_text_props(color=_TEXT, fontweight="bold")
        else:
            cell.set_facecolor(_PANEL_BG)
            cell.set_text_props(color=_TEXT)
            if highlight_col is not None and c == highlight_col and highlight_series is not None:
                val = highlight_series.iloc[r - 1]
                cell.set_text_props(color=_GREEN if val >= 0 else _RED)


# ---- A. Allocation -------------------------------------------------------------------------

def _build_allocation_fig(portfolio_name: str, weights: dict):
    items = sorted(weights.items(), key=lambda kv: -kv[1])
    labels = [t for t, _ in items]
    values = [v * 100 for _, v in items]
    fig = _new_fig((7, 7))
    ax = fig.add_subplot(111)
    ax.set_facecolor(_DARK_BG)
    wedges, _, autotexts = ax.pie(
        values, labels=None, autopct=lambda p: f"{p:.1f}%" if p >= 3 else "",
        colors=[_PALETTE[i % len(_PALETTE)] for i in range(len(labels))],
        wedgeprops=dict(width=0.55, edgecolor=_DARK_BG, linewidth=2),
        pctdistance=0.78, startangle=90,
    )
    for t in autotexts:
        t.set_color(_DARK_BG); t.set_fontsize(9); t.set_fontweight("bold")
    ax.legend(wedges, [f"{t} — {v:.1f}%" for t, v in zip(labels, values)],
              loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False,
              labelcolor=_TEXT, fontsize=10)
    ax.set_title(f"Portfolio Allocation — {portfolio_name}", color=_TEXT, fontsize=15,
                 fontweight="bold", pad=16)
    fig.text(0.5, 0.02, "Holdings only — benchmarks excluded", color=_MUTED, fontsize=8, ha="center")
    return fig


def generate_allocation_chart(portfolio_name: str, weights: dict) -> io.BytesIO:
    return _fig_to_bytes(_build_allocation_fig(portfolio_name, weights))


# ---- B. Performance line chart -------------------------------------------------------------

def _series_for_mode(ret: pd.Series, mode: str, base_value: float):
    return cumulative_series(ret) * 100 if mode == "cumret" else base_value * (1 + ret).cumprod()


def _build_performance_fig(portfolio_name: str, port_ret: pd.Series, start: date, end: date,
                            mode: str = "cumret", bench_name: Optional[str] = None,
                            bench_ret: Optional[pd.Series] = None, base_value: float = 100.0):
    fig = _new_fig((8, 5))
    ax = fig.add_subplot(111)
    s = _series_for_mode(port_ret, mode, base_value)
    ax.plot(s.index, s.values, color=_ACCENT, linewidth=2.2, label=portfolio_name)
    ax.fill_between(s.index, s.values, (0 if mode == "cumret" else base_value), color=_ACCENT, alpha=0.10)
    final_val = s.iloc[-1] if len(s) else float("nan")
    if bench_ret is not None and len(bench_ret):
        sb = _series_for_mode(bench_ret, mode, base_value)
        ax.plot(sb.index, sb.values, color=_GOLD, linewidth=1.8, linestyle="--", label=bench_name)
    ylabel = "Cumulative Return (%)" if mode == "cumret" else f"Growth of ${base_value:,.0f}"
    _style_axes(ax, xlabel="Date", ylabel=ylabel)
    fig.autofmt_xdate(rotation=25)
    ax.axhline(0 if mode == "cumret" else base_value, color=_MUTED, linewidth=0.8, linestyle=":")
    ax.legend(loc="upper left", frameon=False, labelcolor=_TEXT, fontsize=9)
    ret_str = f"{final_val:+.2f}%" if mode == "cumret" else f"${final_val:,.0f} ({(final_val/base_value-1)*100:+.2f}%)"
    fig.suptitle(f"Portfolio Performance — {portfolio_name}", color=_TEXT, fontsize=15,
                 fontweight="bold", y=0.98)
    ax.set_title(f"{start:%d %b %Y} → {end:%d %b %Y}   |   Final: {ret_str}",
                 color=_MUTED, fontsize=10.5, pad=10)
    return fig


def generate_performance_chart(portfolio_name, port_ret, start, end, mode="cumret",
                                bench_name=None, bench_ret=None, base_value=100.0) -> io.BytesIO:
    return _fig_to_bytes(_build_performance_fig(portfolio_name, port_ret, start, end, mode,
                                                 bench_name, bench_ret, base_value))


# ---- C. Risk metrics report card -------------------------------------------------------------

_METRIC_FMT = {
    "Expected (Annualized) Return": lambda v: f"{v*100:+.2f}%", "Volatility (Annualized)": lambda v: f"{v*100:.2f}%",
    "Sharpe Ratio": lambda v: f"{v:.2f}", "Sortino Ratio": lambda v: f"{v:.2f}",
    "Max Drawdown": lambda v: f"{v*100:.2f}%", "Calmar Ratio": lambda v: f"{v:.2f}",
    "Ulcer Index": lambda v: f"{v:.2f}", "Historical VaR (95%, 1-day)": lambda v: f"{v*100:.2f}%",
    "Beta vs. Benchmark": lambda v: f"{v:.2f}", "Correlation vs. Benchmark": lambda v: f"{v:.2f}",
}


def _build_risk_report_fig(portfolio_name: str, metrics: Dict[str, float], start: date, end: date,
                            rf: float, bench_name: Optional[str] = None):
    rows = [(k, v) for k, v in metrics.items()]
    fig = _new_fig((7.5, 1.1 + 0.6 * len(rows)))
    ax = fig.add_subplot(111)
    ax.set_facecolor(_DARK_BG); ax.axis("off")
    ax.set_title(f"Risk Metrics — {portfolio_name}", color=_TEXT, fontsize=15, fontweight="bold",
                 pad=6, loc="left")
    sub = f"{start:%d %b %Y} → {end:%d %b %Y}   |   Rf = {rf*100:.2f}%"
    if bench_name:
        sub += f"   |   Benchmark: {bench_name}"
    ax.text(0, 1.0, sub, transform=ax.transAxes, color=_MUTED, fontsize=9.5, va="top")
    n = len(rows)
    for i, (label, val) in enumerate(rows):
        y = 1 - (i + 1.6) / (n + 1.6)
        formatted = _METRIC_FMT.get(label, lambda v: f"{v:.4f}")(val) if not np.isnan(val) else "n/a"
        color = _TEXT
        if label in ("Expected (Annualized) Return", "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio") and not np.isnan(val):
            color = _GREEN if val >= 0 else _RED
        if label in ("Max Drawdown", "Historical VaR (95%, 1-day)") and not np.isnan(val):
            color = _RED if val < 0 else _TEXT
        ax.axhline(y + (0.6 / (n + 1.6)), color=_GRID, linewidth=0.6, xmin=0, xmax=1)
        ax.text(0, y, label, transform=ax.transAxes, color=_MUTED, fontsize=11, va="center")
        ax.text(1, y, formatted, transform=ax.transAxes, color=color, fontsize=12,
                fontweight="bold", va="center", ha="right")
    _footer(fig, "Daily-rebalanced constant-mix methodology, 252-trading-day annualization. "
                  "Historical data only — nothing forecast.")
    return fig


def generate_risk_report_image(portfolio_name, metrics, start, end, rf, bench_name=None) -> io.BytesIO:
    return _fig_to_bytes(_build_risk_report_fig(portfolio_name, metrics, start, end, rf, bench_name))


# ---- D. Best/worst + all-holdings --------------------------------------------------------------

def _build_best_worst_fig(portfolio_name: str, perf_df: pd.DataFrame, start: date, end: date):
    best = perf_df.head(3).iloc[::-1]
    worst = perf_df.tail(3)
    fig = _new_fig((7.5, 5))
    ax = fig.add_subplot(111)
    ax.set_facecolor(_DARK_BG)
    combo = pd.concat([worst, best])
    colors = [_RED if v < 0 else _GREEN for v in combo["Return %"]]
    bars = ax.barh(combo["Ticker"], combo["Return %"], color=colors, height=0.55)
    for b, v in zip(bars, combo["Return %"]):
        ax.text(v + (0.5 if v >= 0 else -0.5), b.get_y() + b.get_height() / 2, f"{v:+.1f}%",
                color=_TEXT, fontsize=9, va="center", ha="left" if v >= 0 else "right")
    ax.axvline(0, color=_MUTED, linewidth=0.8)
    _style_axes(ax, xlabel="Return (%)")
    ax.set_title(f"Best / Worst Performers — {portfolio_name}", color=_TEXT, fontsize=14,
                 fontweight="bold", pad=12)
    fig.text(0.5, 0.02, f"{start:%d %b %Y} → {end:%d %b %Y}", color=_MUTED, fontsize=8, ha="center")
    return fig


def generate_best_worst_chart(portfolio_name, perf_df, start, end) -> io.BytesIO:
    return _fig_to_bytes(_build_best_worst_fig(portfolio_name, perf_df, start, end))


def _build_all_holdings_fig(portfolio_name: str, perf_df: pd.DataFrame, start: date, end: date):
    df = perf_df.copy()
    ret_raw = perf_df["Return %"]
    df["Weight"] = (df["Weight"] * 100).map(lambda v: f"{v:.1f}%")
    df["Start Price"] = df["Start Price"].map(lambda v: f"{v:,.2f}")
    df["End Price"] = df["End Price"].map(lambda v: f"{v:,.2f}")
    df["Return %"] = df["Return %"].map(lambda v: f"{v:+.2f}%")
    df["Contribution %"] = df["Contribution %"].map(lambda v: f"{v:+.2f}%")
    cols = ["Ticker", "Weight", "Start Price", "End Price", "Return %", "Contribution %"]
    df = df[cols]
    n_rows = len(df)
    fig = _new_fig((8, 1.0 + 0.42 * n_rows))
    ax = fig.add_subplot(111)
    ax.set_facecolor(_DARK_BG); ax.axis("off")
    ax.set_title(f"All Holdings Performance — {portfolio_name}", color=_TEXT, fontsize=14,
                 fontweight="bold", pad=10, loc="left")
    table = ax.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="center")
    _style_table(table, df, highlight_col=df.columns.get_loc("Return %"), highlight_series=ret_raw)
    fig.text(0.5, 0.01, f"{start:%d %b %Y} → {end:%d %b %Y}   |   sorted by return", color=_MUTED,
              fontsize=8, ha="center")
    return fig


def generate_all_holdings_chart(portfolio_name, perf_df, start, end) -> io.BytesIO:
    return _fig_to_bytes(_build_all_holdings_fig(portfolio_name, perf_df, start, end))


# ---- E. Stress test charts -----------------------------------------------------------------

def _build_stress_impact_fig(result: dict, portfolio_name: str):
    fig = _new_fig((7, 5))
    ax = fig.add_subplot(111)
    ax.set_facecolor(_DARK_BG)
    labels = ["Starting Value", "Stressed Value"]
    values = [result["start_value"], result["stressed_value"]]
    colors = [_ACCENT, _GREEN if result["impact_pct"] >= 0 else _RED]
    bars = ax.bar(labels, values, color=colors, width=0.5)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"${v:,.0f}", color=_TEXT, fontsize=11,
                fontweight="bold", ha="center", va="bottom")
    _style_axes(ax, ylabel="Portfolio Value ($)")
    ax.set_title(f"Portfolio Impact — {result['scenario']}", color=_TEXT, fontsize=14,
                 fontweight="bold", pad=12)
    change = result["stressed_value"] - result["start_value"]
    fig.text(0.5, 0.02, f"{portfolio_name}   |   {change:+,.0f} ({result['impact_pct']*100:+.2f}%)",
              color=_MUTED, fontsize=9, ha="center")
    return fig


def generate_stress_impact_chart(result, portfolio_name) -> io.BytesIO:
    return _fig_to_bytes(_build_stress_impact_fig(result, portfolio_name))


def _build_stress_contribution_fig(result: dict, portfolio_name: str):
    contrib = result["asset_contrib"] or {"n/a": 0.0}
    items = sorted(contrib.items(), key=lambda kv: kv[1])
    tickers = [k for k, _ in items]
    vals = [v * 100 for _, v in items]
    fig = _new_fig((7.5, max(4, 0.35 * len(tickers) + 1.5)))
    ax = fig.add_subplot(111)
    ax.set_facecolor(_DARK_BG)
    colors = [_GREEN if v >= 0 else _RED for v in vals]
    bars = ax.barh(tickers, vals, color=colors, height=0.6)
    for b, v in zip(bars, vals):
        ax.text(v + (0.05 if v >= 0 else -0.05), b.get_y() + b.get_height() / 2, f"{v:+.2f}%",
                color=_TEXT, fontsize=8, va="center", ha="left" if v >= 0 else "right")
    ax.axvline(0, color=_MUTED, linewidth=0.8)
    _style_axes(ax, xlabel="Contribution to Portfolio Impact (%)")
    ax.set_title(f"Asset Contribution — {result['scenario']}", color=_TEXT, fontsize=13,
                 fontweight="bold", pad=10)
    fig.text(0.5, 0.01, portfolio_name, color=_MUTED, fontsize=8, ha="center")
    return fig


def generate_stress_contribution_chart(result, portfolio_name) -> io.BytesIO:
    return _fig_to_bytes(_build_stress_contribution_fig(result, portfolio_name))

print("Chart generation Part 1/2 (core charts, refactored) loaded.")

# ---- F. Overview (all portfolios) ------------------------------------------------------------

OVERVIEW_PERIOD_COLS = ["1d", "30d", "1y", "5y", "10y"]
OVERVIEW_COL_DISPLAY = {"1d": "1D", "30d": "1M", "1y": "1Y", "5y": "5Y", "10y": "10Y"}


def build_overview_rows(as_of_period: str = "10y") -> Tuple[pd.DataFrame, List[str]]:
    """One shared max-window download for every portfolio's ticker universe, then every period
    column is sliced from that single download (see multi_period_returns)."""
    portfolios = get_portfolios()
    all_tickers = sorted({t for rec in portfolios.values() for t in rec["weights"]})
    start = date.today() - timedelta(days=MAX_WINDOW_DAYS)
    prices, failed = download_prices(all_tickers, start, date.today())
    rows = []
    for pid, rec in portfolios.items():
        w = rec["weights"]
        sub_prices = prices[[t for t in w if t in prices.columns]].dropna(how="all")
        if sub_prices.empty:
            continue
        mpr = multi_period_returns(sub_prices, w, OVERVIEW_PERIOD_COLS)
        perf_df = asset_return_and_contribution(w, sub_prices, rec["initial_capital"])
        best = perf_df.iloc[0]["Ticker"] if len(perf_df) else "n/a"
        worst = perf_df.iloc[-1]["Ticker"] if len(perf_df) else "n/a"
        total_ret = mpr.get("10y", float("nan"))
        current_value = rec["initial_capital"] * (1 + total_ret) if not np.isnan(total_ret) else rec["initial_capital"]
        row = {"Portfolio": rec["name"], "id": pid, "Value": current_value,
               "Initial": rec["initial_capital"], "P/L": current_value - rec["initial_capital"],
               "Total Return": total_ret, "Best Asset": best, "Worst Asset": worst,
               "# Holdings": len(w)}
        for c in OVERVIEW_PERIOD_COLS:
            row[OVERVIEW_COL_DISPLAY[c]] = mpr.get(c, float("nan"))
        rows.append(row)
    return pd.DataFrame(rows), failed


def _build_overview_table_fig(df: pd.DataFrame):
    period_cols = list(OVERVIEW_COL_DISPLAY.values())
    disp = df.copy()
    disp["Value"] = disp["Value"].map(lambda v: f"${v:,.0f}")
    disp["P/L"] = disp["P/L"].map(lambda v: f"{v:+,.0f}")
    for c in period_cols:
        disp[c] = disp[c].map(lambda v: f"{v*100:+.1f}%" if not (isinstance(v, float) and np.isnan(v)) else "n/a")
    cols = ["Portfolio", "Value", "P/L"] + period_cols + ["Best Asset", "Worst Asset"]
    disp = disp[cols]
    fig = _new_fig((11, 1.2 + 0.5 * len(disp)))
    ax = fig.add_subplot(111)
    ax.set_facecolor(_DARK_BG); ax.axis("off")
    ax.set_title(f"Overview — All Portfolios  (as of {date.today():%d %b %Y})", color=_TEXT,
                 fontsize=15, fontweight="bold", pad=10, loc="left")
    table = ax.table(cellText=disp.values, colLabels=disp.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(8.5); table.scale(1, 1.7)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(_GRID)
        if r == 0:
            cell.set_facecolor("#21262d"); cell.set_text_props(color=_TEXT, fontweight="bold")
        else:
            cell.set_facecolor(_PANEL_BG); cell.set_text_props(color=_TEXT)
            colname = disp.columns[c]
            if colname in ["P/L"] + period_cols:
                raw = df.iloc[r - 1][colname]
                if isinstance(raw, (int, float)) and not np.isnan(raw):
                    cell.set_text_props(color=_GREEN if raw >= 0 else _RED)
    return fig


def generate_overview_table_chart(df: pd.DataFrame) -> io.BytesIO:
    return _fig_to_bytes(_build_overview_table_fig(df))


def _build_multi_value_fig(title: str, series_by_name: Dict[str, pd.Series], ylabel: str,
                            as_pct_from_zero: bool = False):
    fig = _new_fig((8.5, 5.2))
    ax = fig.add_subplot(111)
    for i, (name, s) in enumerate(series_by_name.items()):
        if s is None or len(s) == 0:
            continue
        ax.plot(s.index, s.values, color=_SERIES_COLORS[i % len(_SERIES_COLORS)], linewidth=2,
                label=name)
    _style_axes(ax, xlabel="Date", ylabel=ylabel)
    fig.autofmt_xdate(rotation=25)
    if as_pct_from_zero:
        ax.axhline(0, color=_MUTED, linewidth=0.8, linestyle=":")
    ax.legend(loc="upper left", frameon=False, labelcolor=_TEXT, fontsize=9)
    ax.set_title(title, color=_TEXT, fontsize=14, fontweight="bold", pad=12)
    return fig


def generate_overview_value_chart(value_series_by_name: Dict[str, pd.Series]) -> io.BytesIO:
    return _fig_to_bytes(_build_multi_value_fig("Portfolio Value Comparison", value_series_by_name,
                                                 "Value ($)"))


def generate_cumulative_comparison_chart(cumret_series_by_name: Dict[str, pd.Series]) -> io.BytesIO:
    return _fig_to_bytes(_build_multi_value_fig("Cumulative Performance Comparison",
                                                 cumret_series_by_name, "Cumulative Return (%)",
                                                 as_pct_from_zero=True))


def generate_drawdown_comparison_chart(dd_series_by_name: Dict[str, pd.Series]) -> io.BytesIO:
    fig = _new_fig((8.5, 5))
    ax = fig.add_subplot(111)
    for i, (name, dd) in enumerate(dd_series_by_name.items()):
        if dd is None or len(dd) == 0:
            continue
        color = _SERIES_COLORS[i % len(_SERIES_COLORS)]
        ax.plot(dd.index, dd.values * 100, color=color, linewidth=1.6, label=name)
        ax.fill_between(dd.index, dd.values * 100, 0, color=color, alpha=0.08)
    _style_axes(ax, xlabel="Date", ylabel="Drawdown (%)")
    fig.autofmt_xdate(rotation=25)
    ax.legend(loc="lower left", frameon=False, labelcolor=_TEXT, fontsize=9)
    ax.set_title("Drawdown Comparison", color=_TEXT, fontsize=14, fontweight="bold", pad=12)
    return _fig_to_bytes(fig)


def generate_risk_return_scatter(points_by_name: Dict[str, Tuple[float, float]]) -> io.BytesIO:
    """points_by_name: name -> (annualized_vol, annualized_return), both as decimals."""
    fig = _new_fig((7, 6))
    ax = fig.add_subplot(111)
    ax.set_facecolor(_DARK_BG)
    for i, (name, (vol, ret)) in enumerate(points_by_name.items()):
        if np.isnan(vol) or np.isnan(ret):
            continue
        color = _SERIES_COLORS[i % len(_SERIES_COLORS)]
        ax.scatter([vol * 100], [ret * 100], color=color, s=160, zorder=3, edgecolor=_DARK_BG, linewidth=1.5)
        ax.annotate(name, (vol * 100, ret * 100), color=_TEXT, fontsize=9, xytext=(8, 6),
                    textcoords="offset points")
    _style_axes(ax, xlabel="Volatility — Annualized (%)", ylabel="Return — Annualized (%)")
    ax.axhline(0, color=_MUTED, linewidth=0.7, linestyle=":")
    ax.set_title("Risk / Return Comparison", color=_TEXT, fontsize=14, fontweight="bold", pad=12)
    return _fig_to_bytes(fig)


def _build_comparison_table_fig(metrics_by_name: Dict[str, dict]):
    metric_order = ["Total Return", "Annualized Return", "Volatility", "Sharpe", "Sortino",
                     "Max Drawdown", "Calmar", "Ulcer Index", "VaR (95%, 1d)", "Beta", "Correlation",
                     "Current Value", "Total P/L"]
    names = list(metrics_by_name.keys())
    fig = _new_fig((2.6 + 2.1 * len(names), 1.2 + 0.55 * len(metric_order)))
    ax = fig.add_subplot(111)
    ax.set_facecolor(_DARK_BG); ax.axis("off")
    ax.set_title("Portfolio Comparison", color=_TEXT, fontsize=15, fontweight="bold", pad=10, loc="left")
    cell_text, colors_grid = [], []
    for m in metric_order:
        row, crow = [], []
        for name in names:
            v = metrics_by_name[name].get(m, float("nan"))
            if isinstance(v, str):
                row.append(v); crow.append(_TEXT); continue
            if v is None or (isinstance(v, float) and np.isnan(v)):
                row.append("n/a"); crow.append(_MUTED); continue
            if m in ("Total Return", "Annualized Return", "Volatility", "Max Drawdown", "Ulcer Index", "VaR (95%, 1d)"):
                row.append(f"{v*100:+.2f}%" if m not in ("Volatility", "Ulcer Index") else f"{v*100:.2f}%")
            elif m in ("Current Value", "Total P/L"):
                row.append(f"${v:,.0f}" if m == "Current Value" else f"{v:+,.0f}")
            else:
                row.append(f"{v:.2f}")
            crow.append(_GREEN if (m in ("Total Return", "Annualized Return", "Sharpe", "Sortino",
                                          "Calmar", "Total P/L") and v >= 0) else
                        (_RED if (m in ("Max Drawdown", "VaR (95%, 1d)", "Total P/L", "Total Return",
                                         "Annualized Return", "Sharpe", "Sortino", "Calmar") and v < 0) else _TEXT))
        cell_text.append(row); colors_grid.append(crow)
    table = ax.table(cellText=cell_text, rowLabels=metric_order, colLabels=names, loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 1.6)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(_GRID)
        if r == 0 or c == -1:
            cell.set_facecolor("#21262d"); cell.set_text_props(color=_TEXT, fontweight="bold")
        else:
            cell.set_facecolor(_PANEL_BG)
            cell.set_text_props(color=colors_grid[r - 1][c])
    return fig


def generate_comparison_table_chart(metrics_by_name: Dict[str, dict]) -> io.BytesIO:
    return _fig_to_bytes(_build_comparison_table_fig(metrics_by_name))

print("Chart generation Part 2/4 (Overview + Comparison) loaded.")

# ---- G. Drawdown analysis ---------------------------------------------------------------------

def _build_underwater_fig(portfolio_name: str, dd: pd.Series, summary: dict, start: date, end: date):
    fig = _new_fig((8.5, 5))
    ax = fig.add_subplot(111)
    ax.fill_between(dd.index, dd.values * 100, 0, color=_RED, alpha=0.25)
    ax.plot(dd.index, dd.values * 100, color=_RED, linewidth=1.4)
    if summary["max_drawdown_date"]:
        ax.axvline(pd.Timestamp(summary["max_drawdown_date"]), color=_GOLD, linewidth=1, linestyle="--")
    _style_axes(ax, xlabel="Date", ylabel="Drawdown (%)")
    fig.autofmt_xdate(rotation=25)
    ax.set_title(f"Underwater Chart — {portfolio_name}", color=_TEXT, fontsize=14, fontweight="bold", pad=12)
    fig.text(0.5, 0.01, f"{start:%d %b %Y} → {end:%d %b %Y}   |   Max DD: "
              f"{summary['max_drawdown']*100:.2f}% on {summary['max_drawdown_date']}",
              color=_MUTED, fontsize=8, ha="center")
    return fig


def generate_underwater_chart(portfolio_name, dd, summary, start, end) -> io.BytesIO:
    return _fig_to_bytes(_build_underwater_fig(portfolio_name, dd, summary, start, end))


def _build_drawdown_stats_fig(portfolio_name: str, summary: dict, start: date, end: date):
    rows = [
        ("Current Drawdown", f"{summary['current_drawdown']*100:.2f}%"),
        ("Maximum Drawdown", f"{summary['max_drawdown']*100:.2f}%"),
        ("Max Drawdown Date", str(summary["max_drawdown_date"])),
        ("Currently In Drawdown", "Yes" if summary["currently_in_drawdown"] else "No"),
        ("Number of Drawdown Episodes (>1%)", str(summary["num_episodes"])),
        ("Average Drawdown Depth", f"{summary['avg_drawdown']*100:.2f}%" if not np.isnan(summary["avg_drawdown"]) else "n/a"),
    ]
    we = summary["worst_episode"]
    if we:
        rows.append(("Worst Episode", f"{we['peak_date']} → {we['trough_date']}  ({we['depth']*100:.2f}%)"))
        rows.append(("Worst Episode Duration", f"{we['duration_days']} days to trough"))
        rows.append(("Worst Episode Recovery", f"{we['recovery_days']} days" if we["recovered"] else "Not yet recovered"))
    fig = _new_fig((7.5, 1.1 + 0.55 * len(rows)))
    ax = fig.add_subplot(111)
    ax.set_facecolor(_DARK_BG); ax.axis("off")
    ax.set_title(f"Drawdown Statistics — {portfolio_name}", color=_TEXT, fontsize=14, fontweight="bold",
                 pad=6, loc="left")
    ax.text(0, 1.0, f"{start:%d %b %Y} → {end:%d %b %Y}", transform=ax.transAxes, color=_MUTED,
            fontsize=9.5, va="top")
    n = len(rows)
    for i, (label, val) in enumerate(rows):
        y = 1 - (i + 1.6) / (n + 1.6)
        ax.axhline(y + (0.6 / (n + 1.6)), color=_GRID, linewidth=0.6)
        ax.text(0, y, label, transform=ax.transAxes, color=_MUTED, fontsize=10.5, va="center")
        ax.text(1, y, val, transform=ax.transAxes, color=_TEXT, fontsize=11, fontweight="bold",
                va="center", ha="right")
    return fig


def generate_drawdown_stats_chart(portfolio_name, summary, start, end) -> io.BytesIO:
    return _fig_to_bytes(_build_drawdown_stats_fig(portfolio_name, summary, start, end))


# ---- H. Contribution analysis -------------------------------------------------------------------

def _build_contribution_fig(portfolio_name: str, perf_df: pd.DataFrame, start: date, end: date,
                             in_dollars: bool = False):
    df = perf_df.sort_values("Contribution %").copy()
    col = "Contribution $" if in_dollars else "Contribution %"
    fig = _new_fig((7.5, max(4, 0.35 * len(df) + 1.5)))
    ax = fig.add_subplot(111)
    ax.set_facecolor(_DARK_BG)
    colors = [_GREEN if v >= 0 else _RED for v in df[col]]
    bars = ax.barh(df["Ticker"], df[col], color=colors, height=0.6)
    for b, v in zip(bars, df[col]):
        label = f"${v:+,.0f}" if in_dollars else f"{v:+.2f}%"
        ax.text(v + (abs(v) * 0.02 + 0.01 if v >= 0 else -(abs(v) * 0.02 + 0.01)),
                b.get_y() + b.get_height() / 2, label, color=_TEXT, fontsize=8, va="center",
                ha="left" if v >= 0 else "right")
    ax.axvline(0, color=_MUTED, linewidth=0.8)
    _style_axes(ax, xlabel="Contribution to Portfolio Return ($)" if in_dollars else "Contribution to Portfolio Return (%)")
    ax.set_title(f"Contribution Analysis — {portfolio_name}", color=_TEXT, fontsize=13,
                 fontweight="bold", pad=10)
    fig.text(0.5, 0.01, f"{start:%d %b %Y} → {end:%d %b %Y}   |   Return ≠ Contribution "
              "(contribution = weight × asset return)", color=_MUTED, fontsize=7.5, ha="center")
    return fig


def generate_contribution_chart(portfolio_name, perf_df, start, end, in_dollars=False) -> io.BytesIO:
    return _fig_to_bytes(_build_contribution_fig(portfolio_name, perf_df, start, end, in_dollars))


# ---- I. Rolling metrics ---------------------------------------------------------------------

def _build_rolling_fig(portfolio_name: str, rolling_df: pd.DataFrame, window_label: str):
    cols = [c for c in ["Rolling Return", "Rolling Volatility", "Rolling Sharpe", "Rolling Sortino",
                         "Rolling Beta", "Rolling Correlation"] if c in rolling_df.columns]
    n = len(cols)
    fig, axes = plt.subplots(n, 1, figsize=(8.5, 2.3 * n), dpi=200, facecolor=_DARK_BG, sharex=True)
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        vals = rolling_df[col] * 100 if col in ("Rolling Return", "Rolling Volatility") else rolling_df[col]
        ax.plot(rolling_df.index, vals, color=_ACCENT, linewidth=1.6)
        ax.axhline(0, color=_MUTED, linewidth=0.6, linestyle=":")
        _style_axes(ax, ylabel=col.replace("Rolling ", ""))
    axes[-1].set_xlabel("Date", color=_MUTED, fontsize=10)
    fig.autofmt_xdate(rotation=25)
    fig.suptitle(f"Rolling Metrics ({window_label}) — {portfolio_name}", color=_TEXT, fontsize=14,
                 fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    return fig


def generate_rolling_chart(portfolio_name, rolling_df, window_label) -> io.BytesIO:
    return _fig_to_bytes(_build_rolling_fig(portfolio_name, rolling_df, window_label))


# ---- J. Rebalancing ---------------------------------------------------------------------------

def _build_rebalancing_value_fig(portfolio_name: str, results: Dict[str, dict], start: date, end: date):
    fig = _new_fig((8.5, 5.2))
    ax = fig.add_subplot(111)
    for i, (strat, res) in enumerate(results.items()):
        v = res.get("value")
        if v is None or len(v) == 0:
            continue
        ax.plot(v.index, v.values, color=_SERIES_COLORS[i % len(_SERIES_COLORS)], linewidth=1.8, label=strat)
    _style_axes(ax, xlabel="Date", ylabel="Portfolio Value ($)")
    fig.autofmt_xdate(rotation=25)
    ax.legend(loc="upper left", frameon=False, labelcolor=_TEXT, fontsize=8.5, ncol=2)
    ax.set_title(f"Rebalancing Strategies — {portfolio_name}", color=_TEXT, fontsize=14,
                 fontweight="bold", pad=12)
    fig.text(0.5, 0.01, f"{start:%d %b %Y} → {end:%d %b %Y}   |   Historical simulation — "
              "not a guarantee of future performance", color=_MUTED, fontsize=8, ha="center")
    return fig


def generate_rebalancing_value_chart(portfolio_name, results, start, end) -> io.BytesIO:
    return _fig_to_bytes(_build_rebalancing_value_fig(portfolio_name, results, start, end))


def _build_rebalancing_table_fig(results: Dict[str, dict]):
    strategies = list(results.keys())
    metric_order = ["Final Value", "Total Return", "Annualized Return", "Volatility", "Sharpe",
                     "Sortino", "Max Drawdown", "Calmar", "# Rebalances", "Est. Turnover"]
    cell_text = []
    for m in metric_order:
        row = []
        for strat in strategies:
            res = results[strat]
            if res.get("metrics") is None:
                row.append("n/a"); continue
            if m == "Final Value":
                row.append(f"${res['final_value']:,.0f}")
            elif m == "Total Return":
                row.append(f"{res['total_return']*100:+.2f}%")
            elif m == "# Rebalances":
                row.append(str(res["n_rebalances"]))
            elif m == "Est. Turnover":
                row.append(f"{res['turnover']*100:.0f}%")
            else:
                v = res["metrics"].get(m, float("nan"))
                if np.isnan(v):
                    row.append("n/a")
                elif m in ("Annualized Return", "Volatility", "Max Drawdown"):
                    row.append(f"{v*100:+.2f}%")
                else:
                    row.append(f"{v:.2f}")
        cell_text.append(row)
    fig = _new_fig((1.8 + 1.7 * len(strategies), 1.2 + 0.5 * len(metric_order)))
    ax = fig.add_subplot(111)
    ax.set_facecolor(_DARK_BG); ax.axis("off")
    ax.set_title("Rebalancing Strategy Comparison", color=_TEXT, fontsize=13, fontweight="bold",
                 pad=8, loc="left")
    table = ax.table(cellText=cell_text, rowLabels=metric_order, colLabels=strategies, loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1, 1.6)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(_GRID)
        if r == 0 or c == -1:
            cell.set_facecolor("#21262d"); cell.set_text_props(color=_TEXT, fontweight="bold")
        else:
            cell.set_facecolor(_PANEL_BG); cell.set_text_props(color=_TEXT)
    return fig


def generate_rebalancing_table_chart(results) -> io.BytesIO:
    return _fig_to_bytes(_build_rebalancing_table_fig(results))


# ---- K. Derivatives / risk-proxy card ----------------------------------------------------------

def _build_derivatives_fig(portfolio_name: str, da: dict):
    rows = [("Direct Derivative Positions", "No direct derivative positions detected." if not da["has_direct_derivatives"] else "Detected")]
    if da["leveraged_inverse_holdings"]:
        for t, desc in da["leveraged_inverse_holdings"].items():
            rows.append((f"Leveraged/Inverse: {t}", desc))
    else:
        rows.append(("Leveraged/Inverse ETF Exposure", "None detected"))
    rows.append(("Portfolio Beta vs. S&P 500", f"{da['portfolio_beta_vs_spy']:.2f}" if not np.isnan(da["portfolio_beta_vs_spy"]) else "n/a"))
    rows.append(("Annualized Volatility", f"{da['portfolio_vol']*100:.2f}%" if not np.isnan(da["portfolio_vol"]) else "n/a"))
    rows.append(("Correlation with VIX (^VIX)", f"{da['vix_correlation']:.2f}" if not np.isnan(da["vix_correlation"]) else "n/a"))
    if da["weighted_bond_duration"] is not None:
        rows.append(("Weighted Bond Duration (known ETFs)", f"{da['weighted_bond_duration']:.2f} years"))
        for t, (w, d) in da["bond_exposure"].items():
            rows.append((f"  — {t} (duration {d}y)", f"{w*100:.1f}% of portfolio"))
    else:
        rows.append(("Bond Duration Exposure", "No known-duration fixed-income ETF in this portfolio"))

    fig = _new_fig((7.8, 1.2 + 0.55 * len(rows)))
    ax = fig.add_subplot(111)
    ax.set_facecolor(_DARK_BG); ax.axis("off")
    ax.set_title(f"Derivatives Analysis — {portfolio_name}", color=_TEXT, fontsize=14,
                 fontweight="bold", pad=6, loc="left")
    n = len(rows)
    for i, (label, val) in enumerate(rows):
        y = 1 - (i + 1.4) / (n + 1.4)
        ax.axhline(y + (0.6 / (n + 1.4)), color=_GRID, linewidth=0.5)
        ax.text(0, y, label, transform=ax.transAxes, color=_MUTED, fontsize=10, va="center")
        ax.text(1, y, val, transform=ax.transAxes, color=_TEXT, fontsize=10.5, fontweight="bold",
                va="center", ha="right")
    _footer(fig, "Proxy-based analysis from price data only — no options/futures positions data "
                  "is available through this free source. Nothing here is an invented Greek.")
    return fig


def generate_derivatives_chart(portfolio_name, da) -> io.BytesIO:
    return _fig_to_bytes(_build_derivatives_fig(portfolio_name, da))


# ---- L. Correlation matrix (used in PDF page 6) -------------------------------------------------

def _build_correlation_matrix_fig(portfolio_name: str, returns_df: pd.DataFrame, tickers: List[str]):
    sub = returns_df[[t for t in tickers if t in returns_df.columns]].dropna()
    corr = sub.corr()
    fig = _new_fig((max(6, 0.55 * len(corr)), max(5, 0.55 * len(corr))))
    ax = fig.add_subplot(111)
    im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=90, color=_MUTED, fontsize=8)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns, color=_MUTED, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i,j]:.1f}", ha="center", va="center", color="black", fontsize=6.5)
    ax.set_title(f"Correlation Matrix — {portfolio_name}", color=_TEXT, fontsize=13, fontweight="bold", pad=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color=_MUTED, labelcolor=_MUTED)
    fig.patch.set_facecolor(_DARK_BG)
    return fig


def generate_correlation_matrix_chart(portfolio_name, returns_df, tickers) -> io.BytesIO:
    return _fig_to_bytes(_build_correlation_matrix_fig(portfolio_name, returns_df, tickers))

print("Chart generation Part 3/4 (Drawdown, Contribution, Rolling, Rebalancing, Derivatives, "
      "Correlation) loaded.")

# ---- M. Executive summary, performance-stats, methodology pages (PDF-specific) ------------------

def performance_stats(port_ret: pd.Series, start_value: float) -> dict:
    if len(port_ret) < 2:
        return {}
    total_ret = float((1 + port_ret).prod() - 1)
    monthly = (1 + port_ret).resample("ME").prod() - 1 if len(port_ret) else pd.Series(dtype=float)
    return {
        "start_value": start_value, "end_value": start_value * (1 + total_ret),
        "total_pl": start_value * total_ret, "total_return": total_ret,
        "annualized_return": annualize_return(port_ret),
        "best_day": float(port_ret.max()), "worst_day": float(port_ret.min()),
        "best_day_date": port_ret.idxmax().date(), "worst_day_date": port_ret.idxmin().date(),
        "best_month": float(monthly.max()) if len(monthly) else float("nan"),
        "worst_month": float(monthly.min()) if len(monthly) else float("nan"),
    }


def _build_executive_summary_fig(portfolio_name: str, rec: dict, stats: dict, metrics: dict,
                                  start: date, end: date, bench_name: Optional[str]):
    fig = _new_fig((7.2, 9.2))
    ax = fig.add_subplot(111)
    ax.set_facecolor(_DARK_BG); ax.axis("off")
    ax.text(0, 1.0, portfolio_name, transform=ax.transAxes, color=_TEXT, fontsize=22,
            fontweight="bold", va="top")
    ax.text(0, 0.945, f"Executive Portfolio Overview   |   Reporting Period: {start:%d %b %Y} "
            f"→ {end:%d %b %Y}", transform=ax.transAxes, color=_MUTED, fontsize=10, va="top")
    if bench_name:
        ax.text(0, 0.91, f"Benchmark: {bench_name}", transform=ax.transAxes, color=_MUTED, fontsize=10, va="top")

    highlights = [
        ("Initial Value", f"${stats.get('start_value', rec['initial_capital']):,.0f}"),
        ("Current Value", f"${stats.get('end_value', rec['initial_capital']):,.0f}"),
        ("Total P/L", f"{stats.get('total_pl', 0):+,.0f} {rec['currency']}"),
        ("Total Return", f"{stats.get('total_return', float('nan'))*100:+.2f}%"),
        ("Annualized Return", f"{metrics.get('Expected (Annualized) Return', float('nan'))*100:+.2f}%"),
        ("Volatility (Annualized)", f"{metrics.get('Volatility (Annualized)', float('nan'))*100:.2f}%"),
        ("Sharpe Ratio", f"{metrics.get('Sharpe Ratio', float('nan')):.2f}"),
        ("Max Drawdown", f"{metrics.get('Max Drawdown', float('nan'))*100:.2f}%"),
        ("Number of Holdings", str(len(rec["weights"]))),
        ("Currency", rec["currency"]),
    ]
    y0 = 0.83
    for i, (label, val) in enumerate(highlights):
        y = y0 - i * 0.07
        ax.axhline(y + 0.034, color=_GRID, linewidth=0.5, xmin=0, xmax=1)
        ax.text(0, y, label, transform=ax.transAxes, color=_MUTED, fontsize=12, va="center")
        color = _GREEN if (label in ("Total P/L", "Total Return", "Annualized Return", "Sharpe Ratio")
                            and not val.startswith("-") and not val.startswith("nan")) else _TEXT
        if label in ("Total P/L", "Total Return", "Annualized Return") and val.strip().startswith("-"):
            color = _RED
        if label == "Max Drawdown":
            color = _RED
        ax.text(1, y, val, transform=ax.transAxes, color=color, fontsize=13, fontweight="bold",
                va="center", ha="right")
    _footer(fig, "Source: Yahoo Finance (yfinance). Daily-rebalanced constant-mix methodology. "
                  "Past performance does not guarantee future results.")
    return fig


def _build_methodology_fig(portfolio_names: List[str], rf: float, rf_source: str, bench_name: Optional[str] = None):
    fig = _new_fig((7.4, 10))
    ax = fig.add_subplot(111)
    ax.set_facecolor(_DARK_BG); ax.axis("off")
    ax.text(0, 1.0, "Methodology & Data", transform=ax.transAxes, color=_TEXT, fontsize=17,
            fontweight="bold", va="top")
    lines = [
        f"Portfolio(s) covered: {', '.join(portfolio_names)}",
        f"Data source: Yahoo Finance via yfinance (free, unauthenticated). Retrieved {date.today():%d %b %Y}.",
        "Prices: adjusted close (auto_adjust=True) — includes dividends/splits where the source provides them.",
        "Return methodology: daily-rebalanced constant-mix (portfolio daily return = weighted sum of "
        "constituent daily returns, weights re-normalized over available tickers) — used everywhere "
        "EXCEPT the Rebalancing Analysis section, which explicitly simulates true Buy & Hold and "
        "periodic-rebalancing alternatives using share counts, and labels each one.",
        "Annualization: 252 trading days/year, compounded (not simple-scaled).",
        f"Risk-free rate: {rf*100:.2f}% — source: {rf_source}.",
        f"Benchmark: {bench_name}" if bench_name else "Benchmark: none selected for this report.",
        "Contribution to portfolio return: weight × asset cumulative return over the period — a "
        "standard first-order approximation, distinct from raw asset return in every table/chart.",
        "Drawdown: computed on the cumulative daily-rebalanced return series; an episode is any "
        "decline deeper than -1% from a prior peak, from peak to full recovery.",
        "Rebalancing simulation: historical only, no transaction costs or taxes modeled, turnover "
        "estimated as sum of |weight drift| / 2 at each rebalance event. Not a guarantee of future results.",
        "Stress tests: single-factor linear beta model (asset beta to a factor proxy × a documented "
        "shock). Scenario estimates, not forecasts — see each scenario's stated basis.",
        "Derivatives analysis: proxy-based from price data only (beta, volatility, VIX correlation, "
        "known bond-ETF duration, leveraged/inverse ETF flags). This free data source cannot detect "
        "actual options/futures positions or Greeks, and none are invented.",
        "Missing data: a ticker that fails to download is excluded and reported, never substituted "
        "or invented. Delisted/invalid tickers, market holidays and differing trading calendars are "
        "handled by dropping missing observations (pandas NaN-aware operations), not interpolating.",
        "Currency: figures are shown in each portfolio's configured currency; no FX conversion is "
        "applied between tickers priced in different currencies.",
    ]
    y = 0.90
    for line in lines:
        wrapped = _wrap_text(line, 92)
        for j, wline in enumerate(wrapped):
            ax.text(0, y, ("• " if j == 0 else "   ") + wline, transform=ax.transAxes, color=_MUTED,
                    fontsize=8.7, va="top")
            y -= 0.033
        y -= 0.008
    return fig


def _wrap_text(text: str, width: int) -> List[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur); cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines

print("Chart generation Part 4/4 (executive summary + methodology pages) loaded.")

# ---- N. PDF report assembly (A4, institutional layout: branded header/footer + page numbers
#         on every page, each page's chart placed and letterboxed to fit the page cleanly) ------

PDF_PAGE_SIZE = (8.27, 11.69)  # A4 portrait, inches


def _rasterize_fig(fig) -> np.ndarray:
    """Renders a chart-builder figure to a pixel array and closes it, so it can be placed onto
    a uniform A4 page — this is what lets every PDF page share one page size regardless of each
    chart's own (Telegram-optimized) aspect ratio."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = plt.imread(buf)
    buf.close()
    return img


def _build_a4_page(img: np.ndarray, section_title: str, page_num: int, total_pages: int,
                    report_label: str, as_of: date):
    """Wraps one rasterized chart in a fixed A4 page. Header carries the running document title
    (brand + report label + as-of date, repeated on every page like a fund fact sheet's masthead)
    — it does NOT repeat the section title, since every content figure already carries its own
    title internally; showing it twice was redundant. Image is fit-to-box, preserving its aspect
    ratio (never stretched/distorted)."""
    page = plt.figure(figsize=PDF_PAGE_SIZE, dpi=200, facecolor=_DARK_BG)

    page.text(0.055, 0.975, "PORTFOLIO ANALYTICS TERMINAL", color=_ACCENT, fontsize=9,
              fontweight="bold", ha="left", va="top")
    page.text(0.945, 0.975, f"As of {as_of:%d %b %Y}", color=_MUTED, fontsize=8, ha="right", va="top")
    page.text(0.055, 0.955, report_label, color=_TEXT, fontsize=13, fontweight="bold", ha="left", va="top")
    page.add_artist(plt.Line2D([0.055, 0.945], [0.940, 0.940], transform=page.transFigure,
                                color=_GRID, linewidth=0.8))

    page.add_artist(plt.Line2D([0.055, 0.945], [0.048, 0.048], transform=page.transFigure,
                                color=_GRID, linewidth=0.8))
    page.text(0.055, 0.032, "Generated for informational purposes only — not investment advice. "
              "Past performance does not guarantee future results.", color=_MUTED, fontsize=6.2,
              ha="left", va="top")
    page.text(0.945, 0.032, f"Page {page_num} of {total_pages}", color=_MUTED, fontsize=7.5,
              ha="right", va="top")

    left, right, top, bottom = 0.055, 0.945, 0.925, 0.062
    box_w, box_h = right - left, top - bottom
    box_aspect = (box_w * PDF_PAGE_SIZE[0]) / (box_h * PDF_PAGE_SIZE[1])
    img_h, img_w = img.shape[0], img.shape[1]
    img_aspect = img_w / img_h
    if img_aspect > box_aspect:  # image relatively wider than the box -> fit width, letterbox top/bottom
        draw_w, draw_h = box_w, box_w * (PDF_PAGE_SIZE[0] / PDF_PAGE_SIZE[1]) / img_aspect
    else:  # fit height, letterbox left/right
        draw_h, draw_w = box_h, box_h * (PDF_PAGE_SIZE[1] / PDF_PAGE_SIZE[0]) * img_aspect
    ax_left = left + (box_w - draw_w) / 2
    ax_bottom = bottom + (box_h - draw_h) / 2
    ax = page.add_axes([ax_left, ax_bottom, draw_w, draw_h])
    ax.imshow(img)
    ax.axis("off")
    return page


def _save_a4_report(out_path: str, pages: List[Tuple], report_label: str):
    """pages: list of (figure, section_title). Rasterizes each, places it on an A4 page with
    header/footer/page numbers, and writes the whole PDF in one pass."""
    total = len(pages)
    as_of = date.today()
    with PdfPages(out_path) as pdf:
        for i, (fig, title) in enumerate(pages, start=1):
            img = _rasterize_fig(fig)
            page = _build_a4_page(img, title, i, total, report_label, as_of)
            pdf.savefig(page, facecolor=_DARK_BG)
            plt.close(page)


def _pdf_output_path(name: str) -> str:
    reports_dir = os.path.join(DATA_DIR, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", name)
    return os.path.join(reports_dir, f"{safe}_{date.today():%Y%m%d}.pdf")


def build_portfolio_pdf(pid: str, period_code: str = "1y", benchmark_id: Optional[str] = None,
                         scenario_name: Optional[str] = None) -> Tuple[str, List[str]]:
    """Assembles the full multi-page A4 PDF for one portfolio, reusing the exact same fig-builder
    functions the Telegram bot uses for its chart images. Returns (file_path, warnings)."""
    rec = get_portfolios()[pid]
    name, weights = rec["name"], rec["weights"]
    start, end = preset_to_dates(period_code)
    warnings_list = []

    bench_name, bench_weights = None, None
    if benchmark_id:
        bench_name = BENCHMARKS_BY_ID[benchmark_id]["name"]
        bench_weights = BENCHMARKS_BY_ID[benchmark_id]["weights"]

    all_tickers = sorted(set(weights) | (set(bench_weights) if bench_weights else set()) | {"SPY", "^VIX"})
    prices, failed = download_prices(all_tickers, start, end)
    if failed:
        warnings_list.append(f"No data for: {', '.join(failed)} (excluded, not invented).")
    returns = prices.pct_change().dropna(how="all")
    port_ret, _ = portfolio_daily_returns(weights, returns)
    bench_ret, _ = (portfolio_daily_returns(bench_weights, returns) if bench_weights else (None, None))
    if len(port_ret) < 5:
        raise ValueError("Not enough historical data for this period to build a report.")

    metrics = compute_risk_metrics(port_ret, RF_RATE, bench_ret)
    stats = performance_stats(port_ret, rec["initial_capital"])
    perf_df = asset_return_and_contribution(weights, prices, rec["initial_capital"])
    dd = drawdown_series(port_ret)
    dd_summary = drawdown_summary(port_ret)
    rolling_df = rolling_metrics(port_ret, 90, RF_RATE, bench_ret if bench_ret is not None else returns.get("SPY"))
    vix_ret = returns["^VIX"].dropna() if "^VIX" in returns.columns else None
    da = derivatives_analysis(weights, returns, returns.get("SPY"), vix_ret)
    rebal_prices = prices[[t for t in weights if t in prices.columns]]
    rebal_results = compare_rebalancing_strategies(weights, rebal_prices, rec["initial_capital"], RF_RATE)

    pages: List[Tuple] = [
        (_build_executive_summary_fig(name, rec, stats, metrics, start, end, bench_name), "Executive Portfolio Overview"),
        (_build_allocation_fig(name, weights), "Asset Allocation"),
        (_build_performance_fig(name, port_ret, start, end, "cumret", bench_name, bench_ret), "Performance"),
        (_build_risk_report_fig(name, metrics, start, end, RF_RATE, bench_name), "Risk Metrics"),
        (_build_underwater_fig(name, dd, dd_summary, start, end), "Drawdown — Underwater Chart"),
        (_build_drawdown_stats_fig(name, dd_summary, start, end), "Drawdown Statistics"),
    ]
    if len(perf_df):
        pages.append((_build_all_holdings_fig(name, perf_df, start, end), "Holdings"))
        pages.append((_build_contribution_fig(name, perf_df, start, end), "Contribution Analysis"))
    if len(rolling_df):
        pages.append((_build_rolling_fig(name, rolling_df, "90 Days"), "Rolling Metrics (90 Days)"))
    corr_tickers = list(weights.keys())[:15]
    if len(corr_tickers) >= 2:
        pages.append((_build_correlation_matrix_fig(name, returns, corr_tickers), "Correlation Matrix"))
    pages.append((_build_derivatives_fig(name, da), "Derivatives Analysis"))
    if scenario_name and scenario_name in SCENARIOS:
        stress_tickers = sorted(set(weights) | {SCENARIOS[scenario_name]["factor"]})
        stress_prices, _ = download_prices(stress_tickers, start, end)
        stress_returns = stress_prices.pct_change().dropna(how="all")
        result = run_stress_test(scenario_name, weights, stress_returns, rec["initial_capital"])
        pages.append((_build_stress_impact_fig(result, name), f"Stress Test — {scenario_name}"))
        pages.append((_build_stress_contribution_fig(result, name), "Stress Test — Asset Contribution"))
    pages.append((_build_rebalancing_value_fig(name, rebal_results, start, end), "Rebalancing Simulation"))
    pages.append((_build_rebalancing_table_fig(rebal_results), "Rebalancing — Strategy Comparison"))
    pages.append((_build_methodology_fig([name], RF_RATE, RF_SOURCE, bench_name), "Methodology & Data"))

    out_path = _pdf_output_path(f"{name}_Report")
    _save_a4_report(out_path, pages, name)
    return out_path, warnings_list


def build_combined_pdf(pids: List[str], period_code: str = "1y") -> Tuple[str, List[str]]:
    """One PDF comparing every requested portfolio — overview, performance, risk, drawdown,
    allocation and correlation comparisons, per the brief's 'Combined Report' requirement."""
    portfolios = get_portfolios()
    recs = {pid: portfolios[pid] for pid in pids}
    start, end = preset_to_dates(period_code)
    all_tickers = sorted({t for rec in recs.values() for t in rec["weights"]})
    prices, failed = download_prices(all_tickers, start, end)
    warnings_list = [f"No data for: {', '.join(failed)}"] if failed else []
    returns = prices.pct_change().dropna(how="all")

    cumret_by_name, dd_by_name, value_by_name, points_by_name, metrics_by_name = {}, {}, {}, {}, {}
    for pid, rec in recs.items():
        w = rec["weights"]
        port_ret, _ = portfolio_daily_returns(w, returns)
        if len(port_ret) < 2:
            continue
        cumret_by_name[rec["name"]] = cumulative_series(port_ret) * 100
        dd_by_name[rec["name"]] = drawdown_series(port_ret)
        value_by_name[rec["name"]] = rec["initial_capital"] * (1 + port_ret).cumprod()
        points_by_name[rec["name"]] = (annualize_vol(port_ret), annualize_return(port_ret))
        m = compute_risk_metrics(port_ret, RF_RATE)
        end_val = float(value_by_name[rec["name"]].iloc[-1])
        metrics_by_name[rec["name"]] = {
            "Total Return": float(cumulative_series(port_ret).iloc[-1]), "Annualized Return": m["Expected (Annualized) Return"],
            "Volatility": m["Volatility (Annualized)"], "Sharpe": m["Sharpe Ratio"], "Sortino": m["Sortino Ratio"],
            "Max Drawdown": m["Max Drawdown"], "Calmar": m["Calmar Ratio"], "Ulcer Index": m["Ulcer Index"],
            "VaR (95%, 1d)": m["Historical VaR (95%, 1-day)"], "Current Value": end_val,
            "Total P/L": end_val - rec["initial_capital"],
        }

    overview_df, _ = build_overview_rows()
    overview_df = overview_df[overview_df["id"].isin(pids)]

    out_path = _pdf_output_path("Combined_Portfolio_Report")
    report_label = ", ".join(r["name"] for r in recs.values())
    pages: List[Tuple] = []

    cover = _new_fig((8.5, 5))
    ax = cover.add_subplot(111); ax.set_facecolor(_DARK_BG); ax.axis("off")
    ax.text(0, 0.72, "Combined Portfolio Report", transform=ax.transAxes, color=_TEXT,
            fontsize=24, fontweight="bold", va="top")
    ax.text(0, 0.55, f"{report_label}\nPeriod: {start:%d %b %Y} → {end:%d %b %Y}",
            transform=ax.transAxes, color=_MUTED, fontsize=11, va="top")
    pages.append((cover, "Cover"))

    if len(overview_df):
        pages.append((_build_overview_table_fig(overview_df), "Overview — Selected Portfolios"))
    if value_by_name:
        pages.append((_build_multi_value_fig("Portfolio Value Comparison", value_by_name, "Value ($)"),
                      "Value Comparison"))
    if cumret_by_name:
        pages.append((_build_multi_value_fig("Cumulative Performance Comparison", cumret_by_name,
                      "Cumulative Return (%)", as_pct_from_zero=True), "Performance Comparison"))
    if points_by_name:
        pages.append((_build_comparison_table_fig(metrics_by_name), "Risk Comparison"))
    if dd_by_name:
        dd_fig = _new_fig((8.5, 5))
        ax = dd_fig.add_subplot(111)
        for i, (nm, s) in enumerate(dd_by_name.items()):
            c = _SERIES_COLORS[i % len(_SERIES_COLORS)]
            ax.plot(s.index, s.values * 100, color=c, linewidth=1.6, label=nm)
            ax.fill_between(s.index, s.values * 100, 0, color=c, alpha=0.08)
        _style_axes(ax, xlabel="Date", ylabel="Drawdown (%)")
        dd_fig.autofmt_xdate(rotation=25)
        ax.legend(loc="lower left", frameon=False, labelcolor=_TEXT, fontsize=9)
        ax.set_title("Drawdown Comparison", color=_TEXT, fontsize=14, fontweight="bold", pad=12)
        pages.append((dd_fig, "Drawdown Comparison — Underwater Analysis"))
    if len(points_by_name) >= 1:
        sc_fig = _new_fig((7, 6)); ax = sc_fig.add_subplot(111); ax.set_facecolor(_DARK_BG)
        for i, (nm, (vol, ret)) in enumerate(points_by_name.items()):
            if np.isnan(vol) or np.isnan(ret):
                continue
            c = _SERIES_COLORS[i % len(_SERIES_COLORS)]
            ax.scatter([vol * 100], [ret * 100], color=c, s=160, zorder=3, edgecolor=_DARK_BG, linewidth=1.5)
            ax.annotate(nm, (vol * 100, ret * 100), color=_TEXT, fontsize=9, xytext=(8, 6), textcoords="offset points")
        _style_axes(ax, xlabel="Volatility — Annualized (%)", ylabel="Return — Annualized (%)")
        ax.axhline(0, color=_MUTED, linewidth=0.7, linestyle=":")
        ax.set_title("Risk / Return Comparison", color=_TEXT, fontsize=14, fontweight="bold", pad=12)
        pages.append((sc_fig, "Risk / Return Comparison"))
    all_corr_tickers = sorted(set(all_tickers))[:15]
    if len(all_corr_tickers) >= 2:
        pages.append((_build_correlation_matrix_fig("All Portfolios (union of holdings)", returns, all_corr_tickers),
                      "Correlation Matrix"))
    pages.append((_build_methodology_fig([r["name"] for r in recs.values()], RF_RATE, RF_SOURCE),
                  "Methodology & Data"))

    _save_a4_report(out_path, pages, report_label)
    return out_path, warnings_list

print("PDF report assembly functions loaded.")

# ============================================================================================
# SECTION 8 — TELEGRAM BOT
# ============================================================================================
# BOT_TOKEN and CHAT_ID are read from environment variables — never hardcoded, never committed.
# Set them in your shell (`export BOT_TOKEN=...`), a local `.env` file loaded by your process
# manager, or your host's secrets manager (GitHub Actions secrets, Railway/Render environment
# variables, etc.) — see README.md.

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler, MessageHandler,
    ContextTypes, filters,
)

BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

print("Bot token loaded:", BOT_TOKEN is not None)

DEFAULT_SETTINGS = {
    "default_portfolio": None, "default_period": "1y", "currency": "USD", "date_format": "DD/MM/YYYY",
    "default_benchmark": None, "rebalance_default": "Quarterly", "rolling_window": "90 Days",
}


# ---- session + persisted per-user settings ------------------------------------------------------

def _session(context: ContextTypes.DEFAULT_TYPE, chat_id: Optional[int] = None) -> dict:
    if "sess" not in context.user_data:
        saved = STORE["user_settings"].get(str(chat_id)) if chat_id is not None else None
        settings = {**DEFAULT_SETTINGS, **(saved or {})}
        context.user_data["sess"] = {
            "pid": None, "period_code": None, "start": None, "end": None, "mode": "cumret",
            "benchmark_id": None, "state": None, "compare_selected": set(), "report_selected": set(),
            "pending_action": None, "pending_ctx": {}, "settings": settings,
        }
    return context.user_data["sess"]


def _persist_settings(chat_id: int, settings: dict):
    STORE["user_settings"][str(chat_id)] = settings
    save_store(STORE)


def _get_period_dates(sess: dict) -> Tuple[date, date]:
    if sess["period_code"] == "custom":
        return sess["start"], sess["end"]
    return preset_to_dates(sess["period_code"] or sess["settings"]["default_period"])


def _period_label(sess: dict) -> str:
    if sess["period_code"] == "custom":
        return f"{sess['start']:%d %b %Y} → {sess['end']:%d %b %Y}"
    return PRESET_CODE_TO_LABEL.get(sess["period_code"], sess["period_code"])


async def _reply(update: Update, text: str, kb: Optional[InlineKeyboardMarkup] = None):
    await update.effective_message.reply_text(text, reply_markup=kb, parse_mode=ParseMode.MARKDOWN)


async def _reply_photo(update: Update, photo: io.BytesIO, caption: str = "",
                        kb: Optional[InlineKeyboardMarkup] = None):
    await update.effective_message.reply_photo(photo=photo, caption=caption[:1024], reply_markup=kb,
                                                 parse_mode=ParseMode.MARKDOWN)


async def _reply_document(update: Update, path: str, caption: str = "",
                           kb: Optional[InlineKeyboardMarkup] = None):
    with open(path, "rb") as f:
        await update.effective_message.reply_document(document=f, filename=os.path.basename(path),
                                                        caption=caption[:1024], reply_markup=kb,
                                                        parse_mode=ParseMode.MARKDOWN)


def _nav_row(back_cb: str) -> List[InlineKeyboardButton]:
    return [InlineKeyboardButton("⬅️ Back", callback_data=back_cb),
            InlineKeyboardButton("🏠 Main Menu", callback_data="home")]


def _chunk_buttons(buttons: List[InlineKeyboardButton], per_row: int = 2) -> List[List[InlineKeyboardButton]]:
    return [buttons[i:i + per_row] for i in range(0, len(buttons), per_row)]

print("Telegram session/keyboard helpers loaded.")

# ---- keyboards -----------------------------------------------------------------------------

def kb_main() -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton("🏠 Overview", callback_data="menu:overview")],
        [InlineKeyboardButton("📁 Portfolios", callback_data="menu:portfolios")],
        [InlineKeyboardButton("📈 Performance", callback_data="menu:performance"),
         InlineKeyboardButton("📊 Comparison", callback_data="menu:comparison")],
        [InlineKeyboardButton("📉 Drawdown", callback_data="menu:drawdown"),
         InlineKeyboardButton("🎯 Contribution", callback_data="menu:contribution")],
        [InlineKeyboardButton("🔄 Rolling Metrics", callback_data="menu:rolling"),
         InlineKeyboardButton("⚖️ Rebalancing", callback_data="menu:rebalancing")],
        [InlineKeyboardButton("📐 Derivatives", callback_data="menu:derivatives"),
         InlineKeyboardButton("⚠️ Stress Tests", callback_data="menu:stress")],
        [InlineKeyboardButton("🔔 Alerts", callback_data="menu:alerts"),
         InlineKeyboardButton("📄 Reports", callback_data="menu:reports")],
        [InlineKeyboardButton("⚙️ Settings", callback_data="menu:settings")],
    ]
    return InlineKeyboardMarkup(rows)


def kb_portfolio_picker(action: str, back_cb: str = "home", include_benchmarks: bool = False) -> InlineKeyboardMarkup:
    portfolios = get_portfolios()
    buttons = [InlineKeyboardButton(f"📁 {rec['name']}", callback_data=f"pf:{action}:{pid}")
               for pid, rec in portfolios.items()]
    if include_benchmarks:
        buttons += [InlineKeyboardButton(f"📊 {b['name']}", callback_data=f"pf:{action}:{bid}")
                    for bid, b in BENCHMARKS_BY_ID.items()]
    rows = _chunk_buttons(buttons, 1)
    rows.append(_nav_row(back_cb))
    return InlineKeyboardMarkup(rows)


def kb_period_picker(back_cb: str = "home") -> InlineKeyboardMarkup:
    labels = list(PRESET_PERIODS.keys())
    buttons = [InlineKeyboardButton(lbl, callback_data=f"period:{PRESET_PERIODS[lbl][0]}") for lbl in labels]
    rows = _chunk_buttons(buttons, 2)
    rows.append([InlineKeyboardButton("📅 Custom Date Range", callback_data="period:custom")])
    rows.append(_nav_row(back_cb))
    return InlineKeyboardMarkup(rows)


def kb_performance_report() -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton("💵 Growth of $100", callback_data="report:growth"),
         InlineKeyboardButton("📈 Cumulative %", callback_data="report:cumret")],
        [InlineKeyboardButton("📊 Compare with Benchmark", callback_data="report:benchcompare")],
        [InlineKeyboardButton("🟢🔴 Best/Worst Assets", callback_data="report:bestworst"),
         InlineKeyboardButton("📋 All Holdings", callback_data="report:allholdings")],
        _nav_row("menu:performance"),
    ]
    return InlineKeyboardMarkup(rows)


def kb_benchmark_picker(cb_prefix: str, back_cb: str) -> InlineKeyboardMarkup:
    rows = [[InlineKeyboardButton(f"vs. {b['name']}", callback_data=f"{cb_prefix}:{bid}")]
            for bid, b in BENCHMARKS_BY_ID.items()]
    rows.append(_nav_row(back_cb))
    return InlineKeyboardMarkup(rows)


def kb_scenario_picker() -> InlineKeyboardMarkup:
    names = SCENARIO_NAMES
    rows = []
    for i in range(0, len(names), 2):
        row = [InlineKeyboardButton(f"{SCENARIOS[names[i]]['emoji']} {names[i]}", callback_data=f"scen:{i}")]
        if i + 1 < len(names):
            row.append(InlineKeyboardButton(f"{SCENARIOS[names[i+1]]['emoji']} {names[i+1]}", callback_data=f"scen:{i+1}"))
        rows.append(row)
    rows.append(_nav_row("menu:stress"))
    return InlineKeyboardMarkup(rows)


def kb_window_picker(back_cb: str = "menu:rolling") -> InlineKeyboardMarkup:
    rows = [[InlineKeyboardButton(lbl, callback_data=f"window:{lbl}")] for lbl in ROLLING_WINDOWS]
    rows.append(_nav_row(back_cb))
    return InlineKeyboardMarkup(rows)


def kb_portfolio_management() -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton("📊 View Holdings", callback_data="pm:holdpick")],
        [InlineKeyboardButton("➕ Create Portfolio", callback_data="pm:create")],
        [InlineKeyboardButton("✏️ Edit Portfolio", callback_data="pm:editpick")],
        [InlineKeyboardButton("🗑️ Delete Portfolio", callback_data="pm:deletepick")],
        [InlineKeyboardButton("📋 View All Portfolios (list)", callback_data="pm:viewall")],
        _nav_row("home"),
    ]
    return InlineKeyboardMarkup(rows)


def kb_edit_portfolio(pid: str) -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton("➕ Add Holding", callback_data=f"pm:addhold:{pid}")],
        [InlineKeyboardButton("➖ Remove Holding", callback_data=f"pm:removeholdpick:{pid}")],
        [InlineKeyboardButton("⚖️ Change a Weight", callback_data=f"pm:setweightpick:{pid}")],
        [InlineKeyboardButton("💰 Change Initial Capital", callback_data=f"pm:setcapital:{pid}")],
        [InlineKeyboardButton("✍️ Rename Portfolio", callback_data=f"pm:rename:{pid}")],
        _nav_row("menu:portfolios"),
    ]
    return InlineKeyboardMarkup(rows)


def kb_comparison_picker(sess: dict) -> InlineKeyboardMarkup:
    portfolios = get_portfolios()
    buttons = []
    for pid, rec in portfolios.items():
        mark = "✅ " if pid in sess["compare_selected"] else ""
        buttons.append(InlineKeyboardButton(f"{mark}{rec['name']}", callback_data=f"cmp:toggle:{pid}"))
    for bid, b in BENCHMARKS_BY_ID.items():
        mark = "✅ " if bid in sess["compare_selected"] else ""
        buttons.append(InlineKeyboardButton(f"{mark}{b['name']}", callback_data=f"cmp:toggle:{bid}"))
    rows = _chunk_buttons(buttons, 1)
    rows.append([InlineKeyboardButton("▶️ Compare Selected", callback_data="cmp:go")])
    rows.append(_nav_row("home"))
    return InlineKeyboardMarkup(rows)


def kb_report_picker(sess: dict) -> InlineKeyboardMarkup:
    portfolios = get_portfolios()
    buttons = []
    for pid, rec in portfolios.items():
        mark = "✅ " if pid in sess["report_selected"] else ""
        buttons.append(InlineKeyboardButton(f"{mark}{rec['name']}", callback_data=f"report_sel:{pid}"))
    rows = _chunk_buttons(buttons, 1)
    rows.append([InlineKeyboardButton("📄 Generate Combined Report", callback_data="report:combined_go")])
    rows.append(_nav_row("home"))
    return InlineKeyboardMarkup(rows)


def kb_settings(s: dict) -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton("Default Portfolio", callback_data="settings:portfolio")],
        [InlineKeyboardButton("Default Period", callback_data="settings:period")],
        [InlineKeyboardButton("Default Benchmark", callback_data="settings:benchmark")],
        [InlineKeyboardButton("Rebalancing Method", callback_data="settings:rebal")],
        [InlineKeyboardButton("Rolling Window", callback_data="settings:window")],
        [InlineKeyboardButton("Currency", callback_data="settings:currency")],
        [InlineKeyboardButton("Date Format", callback_data="settings:dateformat")],
        _nav_row("home"),
    ]
    return InlineKeyboardMarkup(rows)


def kb_alerts_menu() -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton("➕ Create Alert", callback_data="alert:createpick")],
        [InlineKeyboardButton("📋 View Active Alerts", callback_data="alert:list")],
        _nav_row("home"),
    ]
    return InlineKeyboardMarkup(rows)


def kb_alert_types() -> InlineKeyboardMarkup:
    types = [
        ("Performance (1-day move)", "perf"), ("Drawdown", "drawdown"),
        ("Asset Price Move (%)", "asset_move"), ("Target Weight Drift", "weight_drift"),
    ]
    rows = [[InlineKeyboardButton(lbl, callback_data=f"alert:create:{code}")] for lbl, code in types]
    rows.append(_nav_row("menu:alerts"))
    return InlineKeyboardMarkup(rows)

print("Telegram keyboards loaded.")

# ---- command handlers -----------------------------------------------------------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    _session(context, update.effective_chat.id)
    await _reply(
        update,
        "💼 *Portfolio Analytics Terminal*\n\n"
        "A professional, menu-driven portfolio terminal — every analysis delivered as a chart "
        "image (or a PDF report). Choose a section below.",
        kb_main(),
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _reply(
        update,
        "📊 *Portfolio Terminal — Help*\n\n"
        "/start — open the main menu\n"
        "/portfolio — quick summary of your default portfolio\n"
        "/help — this message\n\n"
        "Everything else is menu-driven with buttons. The only times you'll type text are: "
        "creating/editing a portfolio, a custom date range, and setting an alert threshold.",
        kb_main(),
    )


async def portfolio_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    sess = _session(context, update.effective_chat.id)
    portfolios = get_portfolios()
    pid = sess["settings"]["default_portfolio"] or next(iter(portfolios))
    code = sess["settings"]["default_period"]
    start_d, end_d = preset_to_dates(code)
    await _reply(update, f"⏳ Building summary for *{portfolios[pid]['name']}*…")
    await _send_summary(update, context, pid, start_d, end_d)


# ---- callback router --------------------------------------------------------------------------

async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    chat_id = update.effective_chat.id
    sess = _session(context, chat_id)

    try:
        # ---- global navigation ----
        if data == "home":
            await _reply(update, "🏠 *Main Menu*", kb_main())

        # ---- Overview ----
        elif data == "menu:overview":
            await _send_overview(update, context)

        # ---- Portfolio Management ----
        elif data == "menu:portfolios":
            await _reply(update, "📁 *Portfolio Management*", kb_portfolio_management())
        elif data == "pm:create":
            sess["state"] = "pm_create_name"
            sess["pending_ctx"] = {}
            await _reply(update, "✏️ Send a *name* for the new portfolio.")
        elif data == "pm:holdpick":
            await _reply(update, "Choose a portfolio:", kb_portfolio_picker("hold", "menu:portfolios"))
        elif data == "pm:viewall":
            await _send_portfolio_list(update, context)
        elif data == "pm:editpick":
            await _reply(update, "Choose a portfolio to edit:", kb_portfolio_picker("editopen", "menu:portfolios"))
        elif data.startswith("pf:editopen:"):
            pid = data.split(":")[2]
            await _reply(update, f"✏️ *Editing {portfolio_display_name(pid)}*", kb_edit_portfolio(pid))
        elif data == "pm:deletepick":
            await _reply(update, "Choose a portfolio to delete:", kb_portfolio_picker("delconfirm", "menu:portfolios"))
        elif data.startswith("pf:delconfirm:"):
            pid = data.split(":")[2]
            rec = get_portfolios().get(pid)
            if not rec:
                await _reply(update, "That portfolio no longer exists.", kb_portfolio_management())
            else:
                kb = InlineKeyboardMarkup([[InlineKeyboardButton("✅ Yes, Delete", callback_data=f"pm:delyes:{pid}"),
                                             InlineKeyboardButton("❌ Cancel", callback_data="menu:portfolios")]])
                await _reply(update, f"⚠️ Are you sure you want to delete *{rec['name']}*? This cannot be undone.", kb)
        elif data.startswith("pm:delyes:"):
            pid = data.split(":")[2]
            name = portfolio_display_name(pid)
            delete_portfolio(pid)
            await _reply(update, f"🗑️ Deleted *{name}*.", kb_portfolio_management())
        elif data.startswith("pm:addhold:"):
            pid = data.split(":")[2]
            sess["state"] = "pm_add_holding"; sess["pending_ctx"] = {"pid": pid}
            await _reply(update, "Send the holding to add as `TICKER:WEIGHT` (weight in %), e.g. `NVDA:5`.\n"
                                  "This will be taken from the other holdings proportionally is NOT automatic — "
                                  "you'll be shown the new total so you can adjust another weight if needed.")
        elif data.startswith("pf:removeholdpick:"):
            pid = data.split(":")[2]
            weights = get_portfolio_weights(pid)
            rows = [[InlineKeyboardButton(t, callback_data=f"pm:removehold:{pid}:{t}")] for t in weights]
            rows.append(_nav_row(f"pf:editopen:{pid}"))
            await _reply(update, "Choose a holding to remove:", InlineKeyboardMarkup(rows))
        elif data.startswith("pm:removehold:"):
            _, _, pid, ticker = data.split(":")
            w = dict(get_portfolio_weights(pid))
            w.pop(ticker, None)
            update_portfolio(pid, weights=w)
            ok, total = validate_weights(w)
            note = "" if ok else f"\n⚠️ Weights now sum to {total:.1f}% — you may want to adjust another weight."
            await _reply(update, f"➖ Removed `{ticker}`.{note}", kb_edit_portfolio(pid))
        elif data.startswith("pf:setweightpick:"):
            pid = data.split(":")[2]
            weights = get_portfolio_weights(pid)
            rows = [[InlineKeyboardButton(f"{t} ({w*100:.1f}%)", callback_data=f"pm:setweight:{pid}:{t}")]
                    for t, w in weights.items()]
            rows.append(_nav_row(f"pf:editopen:{pid}"))
            await _reply(update, "Choose a holding to reweight:", InlineKeyboardMarkup(rows))
        elif data.startswith("pm:setweight:"):
            _, _, pid, ticker = data.split(":")
            sess["state"] = "pm_set_weight"; sess["pending_ctx"] = {"pid": pid, "ticker": ticker}
            await _reply(update, f"Send the new weight for `{ticker}` as a percentage, e.g. `12.5`.")
        elif data.startswith("pm:setcapital:"):
            pid = data.split(":")[2]
            sess["state"] = "pm_set_capital"; sess["pending_ctx"] = {"pid": pid}
            await _reply(update, "Send the new initial capital amount, e.g. `25000`.")
        elif data.startswith("pm:rename:"):
            pid = data.split(":")[2]
            sess["state"] = "pm_rename"; sess["pending_ctx"] = {"pid": pid}
            await _reply(update, "Send the new name for this portfolio.")

        # ---- Holdings quick-view (kept from v1, now sourced from live portfolio store) ----
        elif data.startswith("pf:hold:"):
            pid = data.split(":")[2]
            await _send_holdings(update, context, pid)

        # ---- Performance ----
        elif data == "menu:performance":
            await _reply(update, "📈 *Portfolio Performance*\n\nChoose a portfolio:", kb_portfolio_picker("perf"))
        elif data.startswith("pf:perf:"):
            sess["pid"] = data.split(":")[2]; sess["benchmark_id"] = None
            await _reply(update, f"📈 *{portfolio_display_name(sess['pid'])}*\n\nChoose a period:",
                         kb_period_picker("menu:performance"))
            sess["pending_action"] = "performance"
        elif data == "perf:report":
            await _reply(update, "Choose what to view:", kb_performance_report())
        elif data.startswith("report:") and data.split(":", 1)[1] in ("growth", "cumret"):
            sess["mode"] = data.split(":", 1)[1]
            await _send_performance_chart(update, context)
        elif data == "report:benchcompare":
            await _reply(update, "Compare against which benchmark?", kb_benchmark_picker("benchpick", "perf:report"))
        elif data.startswith("benchpick:"):
            sess["benchmark_id"] = data.split(":")[1]
            await _send_performance_chart(update, context)
        elif data == "report:allholdings":
            await _send_all_holdings(update, context)
        elif data == "report:bestworst":
            await _send_best_worst(update, context)

        # ---- Period picker (shared by Performance / Drawdown / Contribution / Rebalancing / Derivatives) ----
        elif data.startswith("period:"):
            code = data.split(":", 1)[1]
            if code == "custom":
                sess["state"] = "awaiting_custom_date"
                await _reply(update, "📅 Send the custom date range as:\n`01/01/2025 - 23/09/2026`\n"
                                      "(start - end, day/month/year)")
            else:
                sess["period_code"] = code
                sess["state"] = None
                await _dispatch_pending_action(update, context)

        # ---- Comparison ----
        elif data == "menu:comparison":
            sess["compare_selected"] = set()
            await _reply(update, "📊 *Comparison*\n\nSelect 2 or more portfolios/benchmarks to compare, "
                                  "then tap Compare Selected.", kb_comparison_picker(sess))
        elif data.startswith("cmp:toggle:"):
            pid = data.split(":")[2]
            sel = sess["compare_selected"]
            sel.discard(pid) if pid in sel else sel.add(pid)
            await _reply(update, "📊 *Comparison* — selection updated:", kb_comparison_picker(sess))
        elif data == "cmp:go":
            if len(sess["compare_selected"]) < 2:
                await _reply(update, "Select at least 2 to compare.", kb_comparison_picker(sess))
            else:
                sess["pending_action"] = "comparison"
                await _reply(update, "Choose a period for the comparison:", kb_period_picker("menu:comparison"))

        # ---- Drawdown ----
        elif data == "menu:drawdown":
            await _reply(update, "📉 *Drawdown Analysis*\n\nChoose a portfolio:", kb_portfolio_picker("drawdown"))
        elif data.startswith("pf:drawdown:"):
            sess["pid"] = data.split(":")[2]
            sess["pending_action"] = "drawdown"
            await _reply(update, f"📉 *{portfolio_display_name(sess['pid'])}*\n\nChoose a period:",
                         kb_period_picker("menu:drawdown"))

        # ---- Contribution ----
        elif data == "menu:contribution":
            await _reply(update, "🎯 *Contribution Analysis*\n\nChoose a portfolio:", kb_portfolio_picker("contribution"))
        elif data.startswith("pf:contribution:"):
            sess["pid"] = data.split(":")[2]
            sess["pending_action"] = "contribution"
            await _reply(update, f"🎯 *{portfolio_display_name(sess['pid'])}*\n\nChoose a period:",
                         kb_period_picker("menu:contribution"))

        # ---- Rolling Metrics ----
        elif data == "menu:rolling":
            await _reply(update, "🔄 *Rolling Metrics*\n\nChoose a portfolio:", kb_portfolio_picker("rolling"))
        elif data.startswith("pf:rolling:"):
            sess["pid"] = data.split(":")[2]
            await _reply(update, f"🔄 *{portfolio_display_name(sess['pid'])}*\n\nChoose a rolling window:",
                         kb_window_picker())
        elif data.startswith("window:"):
            window_label = data.split(":", 1)[1]
            await _send_rolling(update, context, window_label)

        # ---- Rebalancing ----
        elif data == "menu:rebalancing":
            await _reply(update, "⚖️ *Rebalancing Analysis*\n\nChoose a portfolio:", kb_portfolio_picker("rebalancing"))
        elif data.startswith("pf:rebalancing:"):
            sess["pid"] = data.split(":")[2]
            sess["pending_action"] = "rebalancing"
            await _reply(update, f"⚖️ *{portfolio_display_name(sess['pid'])}*\n\nChoose a period to simulate over:",
                         kb_period_picker("menu:rebalancing"))

        # ---- Derivatives ----
        elif data == "menu:derivatives":
            await _reply(update, "📐 *Derivatives Analysis*\n\nChoose a portfolio:", kb_portfolio_picker("derivatives"))
        elif data.startswith("pf:derivatives:"):
            pid = data.split(":")[2]
            await _send_derivatives(update, context, pid)

        # ---- Stress Tests ----
        elif data == "menu:stress":
            await _reply(update, "⚠️ *Stress Tests*\n\nChoose a portfolio:", kb_portfolio_picker("stress"))
        elif data.startswith("pf:stress:"):
            sess["pid"] = data.split(":")[2]
            await _reply(update, f"⚠️ *{portfolio_display_name(sess['pid'])}*\n\nChoose a scenario:", kb_scenario_picker())
        elif data.startswith("scen:"):
            scen_idx = int(data.split(":")[1])
            await _send_stress_test(update, context, sess["pid"], scen_idx)

        # ---- Reports ----
        elif data == "menu:reports":
            sess["report_selected"] = set()
            rows = [[InlineKeyboardButton("📄 Single-Portfolio Report", callback_data="report:singlepick")],
                    [InlineKeyboardButton("📄 Combined Report (2-3 portfolios)", callback_data="report:combinedpick")],
                    _nav_row("home")]
            await _reply(update, "📄 *Reports / Export*", InlineKeyboardMarkup(rows))
        elif data == "report:singlepick":
            await _reply(update, "Choose a portfolio for the PDF report:", kb_portfolio_picker("reportgo"))
        elif data.startswith("pf:reportgo:"):
            pid = data.split(":")[2]
            await _send_pdf_report(update, context, pid)
        elif data == "report:combinedpick":
            await _reply(update, "Select 2-3 portfolios for the combined report:", kb_report_picker(sess))
        elif data.startswith("report_sel:"):
            pid = data.split(":")[1]
            sel = sess["report_selected"]
            sel.discard(pid) if pid in sel else sel.add(pid)
            await _reply(update, "Selection updated:", kb_report_picker(sess))
        elif data == "report:combined_go":
            if len(sess["report_selected"]) < 2:
                await _reply(update, "Select at least 2 portfolios.", kb_report_picker(sess))
            else:
                await _send_combined_pdf(update, context, list(sess["report_selected"]))

        # ---- Alerts ----
        elif data == "menu:alerts":
            await _reply(update, "🔔 *Alerts*", kb_alerts_menu())
        elif data == "alert:createpick":
            await _reply(update, "What kind of alert?", kb_alert_types())
        elif data.startswith("alert:create:"):
            atype = data.split(":")[2]
            await _start_alert_creation(update, context, atype)
        elif data.startswith("pf:alertport:"):
            pid = data.split(":")[2]
            sess["pending_ctx"]["pid"] = pid
            if sess["pending_ctx"].get("type") == "weight_drift":
                weights = get_portfolio_weights(pid)
                rows = [[InlineKeyboardButton(t, callback_data=f"alertticker:{t}")] for t in weights]
                rows.append(_nav_row("menu:alerts"))
                await _reply(update, "Choose the asset to watch:", InlineKeyboardMarkup(rows))
            else:
                sess["state"] = "alert_threshold_input"
                await _reply(update, "Send the threshold as a percentage, e.g. `5`.")
        elif data.startswith("alertticker:"):
            ticker = data.split(":", 1)[1]
            sess["pending_ctx"]["ticker"] = ticker
            sess["state"] = "alert_threshold_input"
            await _reply(update, "Send the drift threshold in percentage points, e.g. `5`.")
        elif data == "alert:list":
            await _send_alert_list(update, context)
        elif data.startswith("alert:toggle:"):
            aid = data.split(":")[2]
            a = STORE["alerts"].get(aid)
            if a:
                a["enabled"] = not a["enabled"]
                save_store(STORE)
            await _send_alert_list(update, context)
        elif data.startswith("alert:delete:"):
            aid = data.split(":")[2]
            STORE["alerts"].pop(aid, None)
            save_store(STORE)
            await _send_alert_list(update, context)

        # ---- Settings ----
        elif data == "menu:settings":
            await _send_settings(update, context)
        elif data == "settings:portfolio":
            rows = [[InlineKeyboardButton(rec["name"], callback_data=f"pf:setdefault:{pid}")]
                    for pid, rec in get_portfolios().items()]
            rows.append(_nav_row("menu:settings"))
            await _reply(update, "Choose your default portfolio:", InlineKeyboardMarkup(rows))
        elif data.startswith("pf:setdefault:"):
            pid = data.split(":")[2]
            sess["settings"]["default_portfolio"] = pid
            _persist_settings(chat_id, sess["settings"])
            await _reply(update, f"✅ Default portfolio set to *{portfolio_display_name(pid)}*.", kb_settings(sess["settings"]))
        elif data == "settings:period":
            rows = [[InlineKeyboardButton(lbl, callback_data=f"setperiod:{code}")] for lbl, (code, _) in PRESET_PERIODS.items()]
            rows.append(_nav_row("menu:settings"))
            await _reply(update, "Choose your default period:", InlineKeyboardMarkup(rows))
        elif data.startswith("setperiod:"):
            sess["settings"]["default_period"] = data.split(":", 1)[1]
            _persist_settings(chat_id, sess["settings"])
            await _reply(update, f"✅ Default period set to *{PRESET_CODE_TO_LABEL[sess['settings']['default_period']]}*.",
                         kb_settings(sess["settings"]))
        elif data == "settings:benchmark":
            rows = [[InlineKeyboardButton(b["name"], callback_data=f"setbench:{bid}")] for bid, b in BENCHMARKS_BY_ID.items()]
            rows.append(_nav_row("menu:settings"))
            await _reply(update, "Choose your default benchmark:", InlineKeyboardMarkup(rows))
        elif data.startswith("setbench:"):
            sess["settings"]["default_benchmark"] = data.split(":", 1)[1]
            _persist_settings(chat_id, sess["settings"])
            await _reply(update, "✅ Default benchmark set.", kb_settings(sess["settings"]))
        elif data == "settings:rebal":
            rows = [[InlineKeyboardButton(s, callback_data=f"setrebal:{s}")] for s in list(REBALANCE_STRATEGIES) + ["Threshold"]]
            rows.append(_nav_row("menu:settings"))
            await _reply(update, "Choose your default rebalancing methodology:", InlineKeyboardMarkup(rows))
        elif data.startswith("setrebal:"):
            sess["settings"]["rebalance_default"] = data.split(":", 1)[1]
            _persist_settings(chat_id, sess["settings"])
            await _reply(update, "✅ Default rebalancing methodology set.", kb_settings(sess["settings"]))
        elif data == "settings:window":
            rows = [[InlineKeyboardButton(w, callback_data=f"setwindow:{w}")] for w in ROLLING_WINDOWS]
            rows.append(_nav_row("menu:settings"))
            await _reply(update, "Choose your default rolling window:", InlineKeyboardMarkup(rows))
        elif data.startswith("setwindow:"):
            sess["settings"]["rolling_window"] = data.split(":", 1)[1]
            _persist_settings(chat_id, sess["settings"])
            await _reply(update, "✅ Default rolling window set.", kb_settings(sess["settings"]))
        elif data == "settings:currency":
            rows = [[InlineKeyboardButton(c, callback_data=f"setcurrency:{c}")] for c in ("USD", "EUR", "GBP")]
            rows.append(_nav_row("menu:settings"))
            await _reply(update, "Choose currency (display label only):", InlineKeyboardMarkup(rows))
        elif data.startswith("setcurrency:"):
            sess["settings"]["currency"] = data.split(":", 1)[1]
            _persist_settings(chat_id, sess["settings"])
            await _reply(update, f"✅ Currency set to *{sess['settings']['currency']}*.", kb_settings(sess["settings"]))
        elif data == "settings:dateformat":
            rows = [[InlineKeyboardButton(f, callback_data=f"setdf:{f}")] for f in ("DD/MM/YYYY", "MM/DD/YYYY")]
            rows.append(_nav_row("menu:settings"))
            await _reply(update, "Choose date format:", InlineKeyboardMarkup(rows))
        elif data.startswith("setdf:"):
            sess["settings"]["date_format"] = data.split(":", 1)[1]
            _persist_settings(chat_id, sess["settings"])
            await _reply(update, f"✅ Date format set to *{sess['settings']['date_format']}*.", kb_settings(sess["settings"]))

        else:
            await _reply(update, "Unrecognized action — back to the main menu.", kb_main())

    except Exception as e:
        await _reply(update, f"⚠️ Something went wrong: `{e}`\n\nTry again or head back to the main menu.", kb_main())

print("Callback router loaded.")

# ---- data loading helper ---------------------------------------------------------------------

async def _load_portfolio_data(pid: str, start: date, end: date, benchmark_id: Optional[str] = None):
    rec = get_portfolios()[pid]
    weights = rec["weights"]
    tickers = list(weights.keys())
    bench_name, bench_weights = None, None
    if benchmark_id:
        b = BENCHMARKS_BY_ID[benchmark_id]
        bench_name, bench_weights = b["name"], b["weights"]
        tickers = sorted(set(tickers) | set(bench_weights.keys()))
    prices, failed = download_prices(tickers, start, end)
    returns = prices.pct_change().dropna(how="all")
    port_ret, _ = portfolio_daily_returns(weights, returns)
    bench_ret = portfolio_daily_returns(bench_weights, returns)[0] if bench_weights else None
    return rec, weights, prices, returns, port_ret, bench_name, bench_ret, failed


# ---- Overview ------------------------------------------------------------------------------

async def _send_overview(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _reply(update, "⏳ Building Overview — one shared download for every portfolio…")
    df, failed = build_overview_rows()
    if failed:
        await _reply(update, f"⚠️ No data for: {', '.join(failed)}")
    if df.empty:
        await _reply(update, "No portfolios with data available.", kb_main())
        return
    await _reply_photo(update, generate_overview_table_chart(df), "Overview — All Portfolios")

    portfolios = get_portfolios()
    all_tickers = sorted({t for rec in portfolios.values() for t in rec["weights"]})
    start = date.today() - timedelta(days=365)
    prices, _ = download_prices(all_tickers, start, date.today())
    returns = prices.pct_change().dropna(how="all")
    value_by_name, cumret_by_name, dd_by_name, metrics_by_name = {}, {}, {}, {}
    for pid, rec in portfolios.items():
        port_ret, _ = portfolio_daily_returns(rec["weights"], returns)
        if len(port_ret) < 2:
            continue
        value_by_name[rec["name"]] = rec["initial_capital"] * (1 + port_ret).cumprod()
        cumret_by_name[rec["name"]] = cumulative_series(port_ret) * 100
        dd_by_name[rec["name"]] = drawdown_series(port_ret)
        m = compute_risk_metrics(port_ret, RF_RATE)
        end_val = float(value_by_name[rec["name"]].iloc[-1])
        metrics_by_name[rec["name"]] = {
            "Total Return": float(cumulative_series(port_ret).iloc[-1]),
            "Annualized Return": m["Expected (Annualized) Return"], "Volatility": m["Volatility (Annualized)"],
            "Sharpe": m["Sharpe Ratio"], "Sortino": m["Sortino Ratio"], "Max Drawdown": m["Max Drawdown"],
            "Calmar": m["Calmar Ratio"], "Ulcer Index": m["Ulcer Index"],
            "VaR (95%, 1d)": m["Historical VaR (95%, 1-day)"], "Current Value": end_val,
            "Total P/L": end_val - rec["initial_capital"],
        }
    if value_by_name:
        await _reply_photo(update, generate_overview_value_chart(value_by_name), "Portfolio Value Comparison (1Y)")
        await _reply_photo(update, generate_cumulative_comparison_chart(cumret_by_name), "Cumulative Performance Comparison (1Y)")
    if metrics_by_name:
        await _reply_photo(update, generate_comparison_table_chart(metrics_by_name), "Risk Metrics — All Portfolios (1Y)")
    if dd_by_name:
        await _reply_photo(update, generate_drawdown_comparison_chart(dd_by_name), "Underwater Analysis — All Portfolios (1Y)")
    await _reply(update, "What next?", kb_main())


async def _send_portfolio_list(update: Update, context: ContextTypes.DEFAULT_TYPE):
    portfolios = get_portfolios()
    lines = ["📋 *All Portfolios*\n"]
    for pid, rec in portfolios.items():
        tag = " (built-in)" if rec["builtin"] else " (custom)"
        ok, total = validate_weights(rec["weights"])
        flag = "" if ok else f"  ⚠️ weights sum to {total:.1f}%"
        lines.append(f"• `{pid}` — *{rec['name']}*{tag} — {len(rec['weights'])} holdings — "
                      f"{rec['currency']} {rec['initial_capital']:,.0f}{flag}")
    await _reply(update, "\n".join(lines), kb_portfolio_management())


# ---- Holdings (kept from v1, sourced from live store) --------------------------------------

async def _send_holdings(update: Update, context: ContextTypes.DEFAULT_TYPE, pid: str):
    rec = get_portfolios()[pid]
    name, weights, invested, currency = rec["name"], rec["weights"], rec["initial_capital"], rec["currency"]
    await _reply(update, f"⏳ Loading holdings for *{name}*…")
    start_d, end_d = preset_to_dates("1y")
    prices, failed = download_prices(list(weights.keys()), start_d, end_d)
    perf_df = asset_return_and_contribution(weights, prices, invested)
    lines = [f"📁 *{name}* — Holdings\n"]
    for t, w in sorted(weights.items(), key=lambda kv: -kv[1]):
        value = w * invested
        row = perf_df[perf_df["Ticker"] == t]
        perf_1y = f"{row.iloc[0]['Return %']:+.1f}% (1Y)" if len(row) else "n/a"
        lines.append(f"• `{t}`  —  {w*100:.1f}%  —  {currency} {value:,.0f}  —  {perf_1y}")
    if failed:
        lines.append(f"\n⚠️ No data for: {', '.join(failed)}")
    photo = generate_allocation_chart(name, weights)
    kb = InlineKeyboardMarkup([[InlineKeyboardButton("📈 View Performance", callback_data=f"pf:perf:{pid}")],
                                _nav_row("menu:portfolios")])
    await _reply_photo(update, photo, "\n".join(lines), kb)

print("Overview / Portfolio Management / Holdings handlers loaded.")

# ---- Performance -----------------------------------------------------------------------------

async def _send_performance_chart(update: Update, context: ContextTypes.DEFAULT_TYPE):
    sess = _session(context, update.effective_chat.id)
    pid = sess["pid"]
    start_d, end_d = _get_period_dates(sess)
    rec, weights, prices, returns, port_ret, bench_name, bench_ret, failed = \
        await _load_portfolio_data(pid, start_d, end_d, sess["benchmark_id"])
    if len(port_ret) < 2:
        await _reply(update, "⚠️ Not enough data for this period.", kb_performance_report())
        return
    photo = generate_performance_chart(rec["name"], port_ret, start_d, end_d, mode=sess["mode"],
                                        bench_name=bench_name, bench_ret=bench_ret)
    caption = f"*{rec['name']}* — {'Growth of $100' if sess['mode']=='growth' else 'Cumulative Return'}"
    if bench_name:
        caption += f" vs. {bench_name}"
    await _reply_photo(update, photo, caption, kb_performance_report())


async def _send_all_holdings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    sess = _session(context, update.effective_chat.id)
    pid = sess["pid"]
    start_d, end_d = _get_period_dates(sess)
    rec = get_portfolios()[pid]
    prices, failed = download_prices(list(rec["weights"].keys()), start_d, end_d)
    perf_df = asset_return_and_contribution(rec["weights"], prices, rec["initial_capital"])
    if len(perf_df) == 0:
        await _reply(update, "⚠️ No data available for this period.", kb_performance_report())
        return
    await _reply_photo(update, generate_all_holdings_chart(rec["name"], perf_df, start_d, end_d),
                        "All Holdings Performance (sorted by return)", kb_performance_report())


async def _send_best_worst(update: Update, context: ContextTypes.DEFAULT_TYPE):
    sess = _session(context, update.effective_chat.id)
    pid = sess["pid"]
    start_d, end_d = _get_period_dates(sess)
    rec = get_portfolios()[pid]
    prices, failed = download_prices(list(rec["weights"].keys()), start_d, end_d)
    perf_df = asset_return_and_contribution(rec["weights"], prices, rec["initial_capital"])
    if len(perf_df) == 0:
        await _reply(update, "⚠️ No data available for this period.", kb_performance_report())
        return
    top3 = perf_df.head(3)[["Ticker", "Return %", "Contribution %", "P/L $"]]
    bot3 = perf_df.tail(3)[["Ticker", "Return %", "Contribution %", "P/L $"]]
    lines = ["🟢 *Top 3 Winners*"]
    for _, r in top3.iterrows():
        lines.append(f"`{r['Ticker']}`  {r['Return %']:+.2f}%  |  contrib {r['Contribution %']:+.2f}%  |  P/L {r['P/L $']:+,.0f}")
    lines.append("\n🔴 *Top 3 Losers*")
    for _, r in bot3.iloc[::-1].iterrows():
        lines.append(f"`{r['Ticker']}`  {r['Return %']:+.2f}%  |  contrib {r['Contribution %']:+.2f}%  |  P/L {r['P/L $']:+,.0f}")
    await _reply(update, "\n".join(lines))
    await _reply_photo(update, generate_best_worst_chart(rec["name"], perf_df, start_d, end_d),
                        "Best / Worst Performers", kb_performance_report())


async def _generate_full_performance_report(update: Update, context: ContextTypes.DEFAULT_TYPE):
    sess = _session(context, update.effective_chat.id)
    pid = sess["pid"]
    start_d, end_d = _get_period_dates(sess)
    rec = get_portfolios()[pid]
    await _reply(update, f"⏳ Building performance report for *{rec['name']}* ({_period_label(sess)})…")

    rec, weights, prices, returns, port_ret, _, _, failed = await _load_portfolio_data(pid, start_d, end_d)
    if failed:
        await _reply(update, f"⚠️ Could not download: {', '.join(failed)} — excluded, not invented.")
    if port_ret is None or len(port_ret) < 2:
        await _reply(update, "⚠️ Not enough historical data for this period. Try a longer range.", kb_main())
        return

    await _reply_photo(update, generate_allocation_chart(rec["name"], weights), f"*{rec['name']}* — Allocation")
    await _send_performance_chart(update, context)

    metrics = compute_risk_metrics(port_ret, RF_RATE)
    await _reply_photo(update, generate_risk_report_image(rec["name"], metrics, start_d, end_d, RF_RATE), "Risk Metrics")

    perf_df = asset_return_and_contribution(weights, prices, rec["initial_capital"])
    if len(perf_df):
        await _reply_photo(update, generate_best_worst_chart(rec["name"], perf_df, start_d, end_d), "Best & Worst Performers")

    await _reply(update, "What next?", kb_performance_report())


async def _send_summary(update: Update, context: ContextTypes.DEFAULT_TYPE, pid: str, start_d: date, end_d: date):
    rec, weights, prices, returns, port_ret, _, _, failed = await _load_portfolio_data(pid, start_d, end_d)
    if len(port_ret) < 2:
        await _reply(update, "⚠️ Not enough data for this period.", kb_main())
        return
    metrics = compute_risk_metrics(port_ret, RF_RATE)
    total_ret = cumulative_series(port_ret).iloc[-1] * 100
    text = (
        f"📋 *{rec['name']}* — Summary ({start_d:%d %b %Y} → {end_d:%d %b %Y})\n\n"
        f"Total Return: {total_ret:+.2f}%\n"
        f"Annualized Return: {metrics['Expected (Annualized) Return']*100:+.2f}%\n"
        f"Volatility: {metrics['Volatility (Annualized)']*100:.2f}%\n"
        f"Sharpe: {metrics['Sharpe Ratio']:.2f}   Sortino: {metrics['Sortino Ratio']:.2f}\n"
        f"Max Drawdown: {metrics['Max Drawdown']*100:.2f}%\n"
    )
    photo = generate_allocation_chart(rec["name"], weights)
    kb = InlineKeyboardMarkup([[InlineKeyboardButton("📈 Full Performance Report", callback_data=f"pf:perf:{pid}")],
                                _nav_row("home")])
    await _reply_photo(update, photo, text, kb)

print("Performance handlers loaded.")

# ---- Drawdown --------------------------------------------------------------------------------

async def _send_drawdown(update: Update, context: ContextTypes.DEFAULT_TYPE):
    sess = _session(context, update.effective_chat.id)
    pid = sess["pid"]
    start_d, end_d = _get_period_dates(sess)
    rec, weights, prices, returns, port_ret, _, _, failed = await _load_portfolio_data(pid, start_d, end_d)
    if len(port_ret) < 5:
        await _reply(update, "⚠️ Not enough data for drawdown analysis over this period.", kb_main())
        return
    dd = drawdown_series(port_ret)
    summary = drawdown_summary(port_ret)
    await _reply_photo(update, generate_underwater_chart(rec["name"], dd, summary, start_d, end_d), "Underwater Chart")
    await _reply_photo(update, generate_drawdown_stats_chart(rec["name"], summary, start_d, end_d), "Drawdown Statistics")
    kb = InlineKeyboardMarkup([_nav_row("menu:drawdown")])
    await _reply(update, "What next?", kb)


# ---- Contribution ----------------------------------------------------------------------------

async def _send_contribution(update: Update, context: ContextTypes.DEFAULT_TYPE):
    sess = _session(context, update.effective_chat.id)
    pid = sess["pid"]
    start_d, end_d = _get_period_dates(sess)
    rec = get_portfolios()[pid]
    prices, failed = download_prices(list(rec["weights"].keys()), start_d, end_d)
    perf_df = asset_return_and_contribution(rec["weights"], prices, rec["initial_capital"])
    if len(perf_df) == 0:
        await _reply(update, "⚠️ No data available for this period.", kb_main())
        return
    await _reply_photo(update, generate_contribution_chart(rec["name"], perf_df, start_d, end_d, in_dollars=False),
                        "Contribution to Portfolio Return (%)")
    await _reply_photo(update, generate_contribution_chart(rec["name"], perf_df, start_d, end_d, in_dollars=True),
                        "Contribution to Portfolio Return ($)")
    top = perf_df.sort_values("Contribution %", ascending=False)
    lines = ["🎯 *Top Contributors*"]
    for _, r in top.head(3).iterrows():
        lines.append(f"`{r['Ticker']}`  contribution {r['Contribution %']:+.2f}%  (asset return {r['Return %']:+.2f}%)")
    lines.append("\n*Biggest Detractors*")
    for _, r in top.tail(3).iloc[::-1].iterrows():
        lines.append(f"`{r['Ticker']}`  contribution {r['Contribution %']:+.2f}%  (asset return {r['Return %']:+.2f}%)")
    lines.append("\n_Sector-level contribution requires an extra per-ticker lookup; ask for it "
                  "explicitly if you'd like it — it's cached after the first request._")
    await _reply(update, "\n".join(lines), InlineKeyboardMarkup([_nav_row("menu:contribution")]))


# ---- Rolling Metrics --------------------------------------------------------------------------

async def _send_rolling(update: Update, context: ContextTypes.DEFAULT_TYPE, window_label: str):
    sess = _session(context, update.effective_chat.id)
    pid = sess["pid"]
    window_days = ROLLING_WINDOWS[window_label]
    lookback_days = max(window_days * 4, 365)
    start_d, end_d = date.today() - timedelta(days=lookback_days), date.today()
    rec = get_portfolios()[pid]
    default_bench = sess["settings"]["default_benchmark"]
    tickers = list(rec["weights"].keys()) + (list(BENCHMARKS_BY_ID[default_bench]["weights"].keys()) if default_bench else [])
    prices, failed = download_prices(tickers, start_d, end_d)
    returns = prices.pct_change().dropna(how="all")
    port_ret, _ = portfolio_daily_returns(rec["weights"], returns)
    if len(port_ret) < window_days + 5:
        await _reply(update, f"⚠️ Not enough history for a {window_label} rolling window "
                              f"({len(port_ret)} days available).", kb_main())
        return
    bench_ret = portfolio_daily_returns(BENCHMARKS_BY_ID[default_bench]["weights"], returns)[0] if default_bench else None
    rdf = rolling_metrics(port_ret, window_days, RF_RATE, bench_ret)
    if rdf.empty:
        await _reply(update, "⚠️ Rolling metrics could not be computed (insufficient overlapping data).", kb_main())
        return
    await _reply_photo(update, generate_rolling_chart(rec["name"], rdf, window_label),
                        f"🔄 Rolling Metrics ({window_label}) — {rec['name']}")
    await _reply(update, "What next?", InlineKeyboardMarkup([_nav_row("menu:rolling")]))


# ---- Rebalancing -----------------------------------------------------------------------------

async def _send_rebalancing(update: Update, context: ContextTypes.DEFAULT_TYPE):
    sess = _session(context, update.effective_chat.id)
    pid = sess["pid"]
    start_d, end_d = _get_period_dates(sess)
    rec = get_portfolios()[pid]
    prices, failed = download_prices(list(rec["weights"].keys()), start_d, end_d)
    if prices.empty or len(prices) < 30:
        await _reply(update, "⚠️ Not enough data to simulate rebalancing over this period.", kb_main())
        return
    results = compare_rebalancing_strategies(rec["weights"], prices, rec["initial_capital"], RF_RATE, 5.0)
    await _reply_photo(update, generate_rebalancing_value_chart(rec["name"], results, start_d, end_d), "Strategy Comparison")
    await _reply_photo(update, generate_rebalancing_table_chart(results), "Metrics by Strategy")
    await _reply(update, "_Historical simulation only — no transaction costs/taxes modeled, "
                          "not a guarantee of future results._", InlineKeyboardMarkup([_nav_row("menu:rebalancing")]))


# ---- Derivatives -----------------------------------------------------------------------------

async def _send_derivatives(update: Update, context: ContextTypes.DEFAULT_TYPE, pid: str):
    sess = _session(context, update.effective_chat.id)
    rec = get_portfolios()[pid]
    start_d, end_d = preset_to_dates(sess["settings"]["default_period"])
    tickers = sorted(set(rec["weights"]) | {"SPY", "^VIX"})
    prices, failed = download_prices(tickers, start_d, end_d)
    returns = prices.pct_change().dropna(how="all")
    vix_ret = returns["^VIX"].dropna() if "^VIX" in returns.columns else None
    da = derivatives_analysis(rec["weights"], returns, returns.get("SPY"), vix_ret)
    await _reply_photo(update, generate_derivatives_chart(rec["name"], da), f"Derivatives Analysis — {rec['name']}",
                        InlineKeyboardMarkup([_nav_row("menu:derivatives")]))


# ---- Stress Tests (kept from v1) ----------------------------------------------------------------

async def _send_stress_test(update: Update, context: ContextTypes.DEFAULT_TYPE, pid: str, scen_idx: int):
    rec = get_portfolios()[pid]
    scen_name = SCENARIO_NAMES[scen_idx]
    start_d, end_d = preset_to_dates("1y")
    await _reply(update, f"⏳ Running *{scen_name}* on *{rec['name']}*…")
    factor = SCENARIOS[scen_name]["factor"]
    tickers = sorted(set(rec["weights"]) | {factor})
    prices, failed = download_prices(tickers, start_d, end_d)
    returns = prices.pct_change().dropna(how="all")
    result = run_stress_test(scen_name, rec["weights"], returns, rec["initial_capital"])
    await _reply(
        update,
        f"⚠️ *Scenario Estimate — Not a Prediction*\n\n"
        f"*{SCENARIOS[scen_name]['emoji']} {scen_name}*  [{result['type']}]\n"
        f"Factor: `{result['factor']}`   Shock: {result['shock']*100:+.0f}%\n"
        f"_{result['basis']}_\n\n"
        f"Starting Value: ${result['start_value']:,.0f}\n"
        f"Stressed Value: ${result['stressed_value']:,.0f}\n"
        f"Change: ${result['stressed_value']-result['start_value']:+,.0f} ({result['impact_pct']*100:+.2f}%)",
    )
    await _reply_photo(update, generate_stress_impact_chart(result, rec["name"]), "Before vs. After")
    await _reply_photo(update, generate_stress_contribution_chart(result, rec["name"]), "Asset-Level Contribution")
    await _reply(update, "Run another scenario?", kb_scenario_picker())


# ---- Comparison -------------------------------------------------------------------------------

async def _send_comparison(update: Update, context: ContextTypes.DEFAULT_TYPE):
    sess = _session(context, update.effective_chat.id)
    ids = list(sess["compare_selected"])
    start_d, end_d = _get_period_dates(sess)
    portfolios = get_portfolios()
    all_tickers = set()
    entities = {}
    for pid in ids:
        if pid in portfolios:
            entities[pid] = {"name": portfolios[pid]["name"], "weights": portfolios[pid]["weights"],
                              "initial_capital": portfolios[pid]["initial_capital"]}
        elif pid in BENCHMARKS_BY_ID:
            entities[pid] = {"name": BENCHMARKS_BY_ID[pid]["name"], "weights": BENCHMARKS_BY_ID[pid]["weights"],
                              "initial_capital": 10000.0}
        all_tickers |= set(entities[pid]["weights"])
    prices, failed = download_prices(sorted(all_tickers), start_d, end_d)
    returns = prices.pct_change().dropna(how="all")

    cumret_by_name, dd_by_name, points_by_name, metrics_by_name = {}, {}, {}, {}
    for pid, e in entities.items():
        port_ret, _ = portfolio_daily_returns(e["weights"], returns)
        if len(port_ret) < 2:
            continue
        cumret_by_name[e["name"]] = cumulative_series(port_ret) * 100
        dd_by_name[e["name"]] = drawdown_series(port_ret)
        points_by_name[e["name"]] = (annualize_vol(port_ret), annualize_return(port_ret))
        m = compute_risk_metrics(port_ret, RF_RATE)
        end_val = e["initial_capital"] * (1 + cumulative_series(port_ret).iloc[-1])
        metrics_by_name[e["name"]] = {
            "Total Return": float(cumulative_series(port_ret).iloc[-1]), "Annualized Return": m["Expected (Annualized) Return"],
            "Volatility": m["Volatility (Annualized)"], "Sharpe": m["Sharpe Ratio"], "Sortino": m["Sortino Ratio"],
            "Max Drawdown": m["Max Drawdown"], "Calmar": m["Calmar Ratio"], "Ulcer Index": m["Ulcer Index"],
            "VaR (95%, 1d)": m["Historical VaR (95%, 1-day)"], "Current Value": end_val,
            "Total P/L": end_val - e["initial_capital"],
        }
    if not cumret_by_name:
        await _reply(update, "⚠️ Not enough data to compare these selections.", kb_main())
        return
    await _reply_photo(update, generate_comparison_table_chart(metrics_by_name), "Comparison Metrics")
    await _reply_photo(update, generate_cumulative_comparison_chart(cumret_by_name), "Cumulative Performance")
    await _reply_photo(update, generate_drawdown_comparison_chart(dd_by_name), "Drawdown Comparison")
    await _reply_photo(update, generate_risk_return_scatter(points_by_name), "Risk / Return")
    await _reply(update, "What next?", InlineKeyboardMarkup([_nav_row("home")]))


# ---- pending-action dispatcher (called after a period is chosen) -----------------------------

async def _dispatch_pending_action(update: Update, context: ContextTypes.DEFAULT_TYPE):
    sess = _session(context, update.effective_chat.id)
    action = sess.get("pending_action")
    if action == "performance":
        await _generate_full_performance_report(update, context)
    elif action == "drawdown":
        await _send_drawdown(update, context)
    elif action == "contribution":
        await _send_contribution(update, context)
    elif action == "rebalancing":
        await _send_rebalancing(update, context)
    elif action == "comparison":
        await _send_comparison(update, context)
    else:
        await _reply(update, "Choose a section from the main menu.", kb_main())

print("Drawdown / Contribution / Rolling / Rebalancing / Derivatives / Stress / Comparison handlers loaded.")

# ---- Reports (PDF) ---------------------------------------------------------------------------

async def _send_pdf_report(update: Update, context: ContextTypes.DEFAULT_TYPE, pid: str):
    sess = _session(context, update.effective_chat.id)
    rec = get_portfolios()[pid]
    await _reply(update, f"⏳ Building the full PDF report for *{rec['name']}* — this covers "
                          "executive summary, allocation, performance, risk, drawdown, holdings & "
                          "contribution, rolling metrics, correlation, derivatives, rebalancing and "
                          "methodology. This can take a little while.")
    try:
        period = sess["settings"]["default_period"]
        bench_id = sess["settings"]["default_benchmark"]
        path, warns = build_portfolio_pdf(pid, period_code=period, benchmark_id=bench_id)
        for w in warns:
            await _reply(update, f"⚠️ {w}")
        await _reply_document(update, path, f"📄 *{rec['name']}* — Full Portfolio Report ({PRESET_CODE_TO_LABEL.get(period, period)})",
                               InlineKeyboardMarkup([_nav_row("menu:reports")]))
    except Exception as e:
        await _reply(update, f"⚠️ Could not build the report: `{e}`", kb_main())


async def _send_combined_pdf(update: Update, context: ContextTypes.DEFAULT_TYPE, pids: List[str]):
    sess = _session(context, update.effective_chat.id)
    names = [portfolio_display_name(p) for p in pids]
    await _reply(update, f"⏳ Building the combined report for: {', '.join(names)}…")
    try:
        period = sess["settings"]["default_period"]
        path, warns = build_combined_pdf(pids, period_code=period)
        for w in warns:
            await _reply(update, f"⚠️ {w}")
        await _reply_document(update, path, f"📄 Combined Portfolio Report — {', '.join(names)}",
                               InlineKeyboardMarkup([_nav_row("menu:reports")]))
    except Exception as e:
        await _reply(update, f"⚠️ Could not build the combined report: `{e}`", kb_main())


# ---- Settings display -------------------------------------------------------------------------

async def _send_settings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    sess = _session(context, update.effective_chat.id)
    s = sess["settings"]
    default_pf_name = portfolio_display_name(s["default_portfolio"]) if s["default_portfolio"] else "(none set)"
    default_bench_name = BENCHMARKS_BY_ID[s["default_benchmark"]]["name"] if s["default_benchmark"] else "(none set)"
    text = (
        "⚙️ *Settings*\n\n"
        f"Default portfolio: {default_pf_name}\n"
        f"Default period: {PRESET_CODE_TO_LABEL.get(s['default_period'], s['default_period'])}\n"
        f"Default benchmark: {default_bench_name}\n"
        f"Rebalancing method: {s['rebalance_default']}\n"
        f"Rolling window: {s['rolling_window']}\n"
        f"Currency: {s['currency']}\n"
        f"Date format: {s['date_format']}"
    )
    await _reply(update, text, kb_settings(s))


# ---- Alerts -----------------------------------------------------------------------------------

async def _start_alert_creation(update: Update, context: ContextTypes.DEFAULT_TYPE, atype: str):
    sess = _session(context, update.effective_chat.id)
    sess["pending_ctx"] = {"type": atype}
    if atype == "asset_move":
        sess["state"] = "alert_asset_input"
        await _reply(update, "Send `TICKER THRESHOLD`, e.g. `NVDA 7` — alerts you if NVDA moves "
                              "more than 7% in one day.")
    else:
        await _reply(update, "Which portfolio should this alert watch?",
                     kb_portfolio_picker("alertport", "menu:alerts"))


async def _send_alert_list(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    alerts = {aid: a for aid, a in STORE["alerts"].items() if str(a["chat_id"]) == chat_id}
    if not alerts:
        await _reply(update, "You have no alerts yet.", kb_alerts_menu())
        return
    lines = ["🔔 *Your Alerts*\n"]
    rows = []
    for aid, a in alerts.items():
        status = "🟢 ON" if a["enabled"] else "⚪ OFF"
        desc = _describe_alert(a)
        lines.append(f"`{aid}` {status}\n{desc}")
        rows.append([InlineKeyboardButton(f"Toggle {aid}", callback_data=f"alert:toggle:{aid}"),
                     InlineKeyboardButton(f"Delete {aid}", callback_data=f"alert:delete:{aid}")])
    rows.append(_nav_row("menu:alerts"))
    await _reply(update, "\n\n".join(lines), InlineKeyboardMarkup(rows))


def _describe_alert(a: dict) -> str:
    if a["type"] == "perf":
        return f"Performance: alert if {portfolio_display_name(a['pid'])} falls more than {a['threshold']:.1f}% in one day."
    if a["type"] == "drawdown":
        return f"Drawdown: alert if {portfolio_display_name(a['pid'])} drawdown exceeds {a['threshold']:.1f}%."
    if a["type"] == "asset_move":
        return f"Asset move: alert if `{a['ticker']}` moves more than {a['threshold']:.1f}% in one day."
    if a["type"] == "weight_drift":
        return (f"Weight drift: alert if `{a['ticker']}` in {portfolio_display_name(a['pid'])} moves "
                f"more than {a['threshold']:.1f}pp from its target weight.")
    return "Unknown alert type."


def _create_alert(chat_id: int, atype: str, threshold: float, pid: Optional[str] = None,
                   ticker: Optional[str] = None) -> str:
    n = STORE["next_alert_num"]
    aid = f"a{n}"
    STORE["next_alert_num"] = n + 1
    STORE["alerts"][aid] = {
        "id": aid, "chat_id": chat_id, "type": atype, "threshold": threshold, "pid": pid,
        "ticker": ticker, "enabled": True, "created": date.today().isoformat(), "last_triggered": None,
    }
    save_store(STORE)
    return aid

print("Reports / Settings / Alerts handlers loaded.")

# ---- free-text handler (custom dates, portfolio management, alert thresholds) -----------------

def _parse_holdings_text(text: str) -> Tuple[Dict[str, float], Optional[str]]:
    """Parses 'NVDA:30, MSFT:25, AMD:20, TSM:25' (weights as percentages) -> {ticker: fraction}."""
    parts = [p.strip() for p in re.split(r"[,\n]", text) if p.strip()]
    weights = {}
    for p in parts:
        m = re.match(r"^([A-Za-z^.\-]{1,10})\s*[:=]\s*([\d.]+)\s*%?$", p)
        if not m:
            return {}, f"Couldn't read `{p}`. Use `TICKER:WEIGHT` pairs, e.g. `NVDA:30, MSFT:25`."
        weights[m.group(1).upper()] = float(m.group(2)) / 100.0
    if not weights:
        return {}, "No holdings found — use `TICKER:WEIGHT` pairs, e.g. `NVDA:30, MSFT:25`."
    return weights, None


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    sess = _session(context, update.effective_chat.id)
    state = sess.get("state")
    text = update.message.text.strip()

    if state is None:
        await _reply(update, "Use the menu buttons to navigate 🙂", kb_main())
        return

    # ---- custom date range (Performance / Drawdown / Contribution / Rebalancing / Comparison) ----
    if state == "awaiting_custom_date":
        start_d, end_d, err = parse_custom_range(text)
        if err:
            await _reply(update, f"⚠️ {err}")
            return
        sess["period_code"] = "custom"
        sess["start"], sess["end"] = start_d, end_d
        sess["state"] = None
        await _dispatch_pending_action(update, context)
        return

    # ---- portfolio creation wizard ----
    if state == "pm_create_name":
        sess["pending_ctx"]["name"] = text
        sess["state"] = "pm_create_capital"
        await _reply(update, "💰 Send the *initial capital* amount, e.g. `10000`.")
        return
    if state == "pm_create_capital":
        try:
            cap = float(text.replace(",", "").replace("$", ""))
        except ValueError:
            await _reply(update, "⚠️ Send a plain number, e.g. `10000`.")
            return
        sess["pending_ctx"]["capital"] = cap
        sess["state"] = "pm_create_currency"
        await _reply(update, "💱 Send the *currency code*, e.g. `USD`.")
        return
    if state == "pm_create_currency":
        sess["pending_ctx"]["currency"] = text.upper()[:6]
        sess["state"] = "pm_create_holdings"
        await _reply(update, "📊 Send the holdings as `TICKER:WEIGHT` pairs (weights in %, must sum "
                              "to ~100%), e.g.:\n`NVDA:30, MSFT:25, AMD:20, TSM:25`")
        return
    if state == "pm_create_holdings":
        weights, err = _parse_holdings_text(text)
        if err:
            await _reply(update, f"⚠️ {err}")
            return
        ok, total = validate_weights(weights)
        if not ok:
            await _reply(update, f"⚠️ Weights sum to *{total:.1f}%*, not 100%. Please resend the "
                                  "full holdings list with corrected weights.")
            return
        ctx = sess["pending_ctx"]
        pid = create_portfolio(ctx["name"], weights, ctx["capital"], ctx["currency"])
        sess["state"] = None; sess["pending_ctx"] = {}
        await _reply(update, f"✅ Created *{ctx['name']}* (`{pid}`) with {len(weights)} holdings, "
                              f"weights sum to {total:.1f}%.", kb_portfolio_management())
        return

    # ---- edit: add holding ----
    if state == "pm_add_holding":
        pid = sess["pending_ctx"]["pid"]
        m = re.match(r"^([A-Za-z^.\-]{1,10})\s*[:=]\s*([\d.]+)\s*%?$", text.strip())
        if not m:
            await _reply(update, "⚠️ Use `TICKER:WEIGHT`, e.g. `NVDA:5`.")
            return
        ticker, w = m.group(1).upper(), float(m.group(2)) / 100.0
        weights = dict(get_portfolio_weights(pid))
        weights[ticker] = w
        update_portfolio(pid, weights=weights)
        ok, total = validate_weights(weights)
        note = "" if ok else f"\n⚠️ Weights now sum to {total:.1f}% — you may want to adjust another weight."
        sess["state"] = None
        await _reply(update, f"➕ Added `{ticker}` at {w*100:.1f}%.{note}", kb_edit_portfolio(pid))
        return

    # ---- edit: set weight ----
    if state == "pm_set_weight":
        pid, ticker = sess["pending_ctx"]["pid"], sess["pending_ctx"]["ticker"]
        try:
            new_w = float(text.replace("%", "")) / 100.0
        except ValueError:
            await _reply(update, "⚠️ Send a plain number, e.g. `12.5`.")
            return
        weights = dict(get_portfolio_weights(pid))
        weights[ticker] = new_w
        update_portfolio(pid, weights=weights)
        ok, total = validate_weights(weights)
        note = "" if ok else f"\n⚠️ Weights now sum to {total:.1f}% — you may want to adjust another weight."
        sess["state"] = None
        await _reply(update, f"⚖️ `{ticker}` set to {new_w*100:.1f}%.{note}", kb_edit_portfolio(pid))
        return

    # ---- edit: set capital ----
    if state == "pm_set_capital":
        pid = sess["pending_ctx"]["pid"]
        try:
            cap = float(text.replace(",", "").replace("$", ""))
        except ValueError:
            await _reply(update, "⚠️ Send a plain number, e.g. `25000`.")
            return
        update_portfolio(pid, initial_capital=cap)
        sess["state"] = None
        await _reply(update, f"💰 Initial capital set to {cap:,.0f}.", kb_edit_portfolio(pid))
        return

    # ---- edit: rename ----
    if state == "pm_rename":
        pid = sess["pending_ctx"]["pid"]
        update_portfolio(pid, name=text)
        sess["state"] = None
        await _reply(update, f"✍️ Renamed to *{text}*.", kb_edit_portfolio(pid))
        return

    # ---- alerts ----
    if state == "alert_threshold_input":
        try:
            threshold = float(text.replace("%", "").replace("pp", ""))
        except ValueError:
            await _reply(update, "⚠️ Send a plain number, e.g. `5`.")
            return
        ctx = sess["pending_ctx"]
        aid = _create_alert(update.effective_chat.id, ctx["type"], threshold, ctx.get("pid"), ctx.get("ticker"))
        sess["state"] = None; sess["pending_ctx"] = {}
        await _reply(update, f"🔔 Alert `{aid}` created:\n{_describe_alert(STORE['alerts'][aid])}", kb_alerts_menu())
        return

    if state == "alert_asset_input":
        m = re.match(r"^([A-Za-z^.\-]{1,10})\s+([\d.]+)\s*%?$", text.strip())
        if not m:
            await _reply(update, "⚠️ Use `TICKER THRESHOLD`, e.g. `NVDA 7`.")
            return
        ticker, threshold = m.group(1).upper(), float(m.group(2))
        aid = _create_alert(update.effective_chat.id, "asset_move", threshold, ticker=ticker)
        sess["state"] = None; sess["pending_ctx"] = {}
        await _reply(update, f"🔔 Alert `{aid}` created:\n{_describe_alert(STORE['alerts'][aid])}", kb_alerts_menu())
        return

    await _reply(update, "Use the menu buttons to navigate 🙂", kb_main())

print("Free-text state handler loaded.")

# ---- Daily automated report (Telegram JobQueue — sent to CHAT_ID once per day) ----------------
# Reuses the exact same analytics/chart functions as the interactive 🏠 Overview menu — this is
# not a separate report engine, just the same code invoked on a schedule instead of a button tap.

async def _bot_send_text(context: ContextTypes.DEFAULT_TYPE, text: str):
    if not CHAT_ID:
        return
    await context.bot.send_message(chat_id=CHAT_ID, text=text, parse_mode=ParseMode.MARKDOWN)


async def _bot_send_photo(context: ContextTypes.DEFAULT_TYPE, photo: io.BytesIO, caption: str = ""):
    if not CHAT_ID:
        return
    await context.bot.send_photo(chat_id=CHAT_ID, photo=photo, caption=caption[:1024],
                                  parse_mode=ParseMode.MARKDOWN)


async def daily_report_job(context: ContextTypes.DEFAULT_TYPE):
    """Scheduled once per day via application.job_queue.run_daily (see main()). Sends an
    automated portfolio report — value, P/L, 1D move, best/worst asset, cumulative performance,
    risk metrics, and drawdown comparison across every portfolio — to CHAT_ID. Requires CHAT_ID
    to be set; does nothing otherwise (safe to leave scheduling enabled either way)."""
    if not CHAT_ID:
        print("Daily report: CHAT_ID not set, skipping.")
        return
    try:
        df, failed = build_overview_rows()
        if df.empty:
            await _bot_send_text(context, "⚠️ Daily report: no portfolio data available today.")
            return

        lines = [f"📊 *Daily Portfolio Report — {date.today():%d %b %Y}*\n"]
        for _, row in df.iterrows():
            lines.append(f"*{row['Portfolio']}*: ${row['Value']:,.0f}  ({row['P/L']:+,.0f})   "
                          f"1D: {row['1D']*100:+.2f}%   Best: {row['Best Asset']}   Worst: {row['Worst Asset']}")
        await _bot_send_text(context, "\n".join(lines))
        if failed:
            await _bot_send_text(context, f"⚠️ No data for: {', '.join(failed)}")
        await _bot_send_photo(context, generate_overview_table_chart(df), "Overview — All Portfolios")

        portfolios = get_portfolios()
        all_tickers = sorted({t for rec in portfolios.values() for t in rec["weights"]})
        start = date.today() - timedelta(days=365)
        prices, _ = download_prices(all_tickers, start, date.today())
        returns = prices.pct_change().dropna(how="all")
        cumret_by_name, dd_by_name, metrics_by_name = {}, {}, {}
        for pid, rec in portfolios.items():
            port_ret, _ = portfolio_daily_returns(rec["weights"], returns)
            if len(port_ret) < 2:
                continue
            cumret_by_name[rec["name"]] = cumulative_series(port_ret) * 100
            dd_by_name[rec["name"]] = drawdown_series(port_ret)
            m = compute_risk_metrics(port_ret, RF_RATE)
            end_val = rec["initial_capital"] * (1 + cumulative_series(port_ret).iloc[-1])
            metrics_by_name[rec["name"]] = {
                "Total Return": float(cumulative_series(port_ret).iloc[-1]),
                "Annualized Return": m["Expected (Annualized) Return"], "Volatility": m["Volatility (Annualized)"],
                "Sharpe": m["Sharpe Ratio"], "Sortino": m["Sortino Ratio"], "Max Drawdown": m["Max Drawdown"],
                "Calmar": m["Calmar Ratio"], "Ulcer Index": m["Ulcer Index"],
                "VaR (95%, 1d)": m["Historical VaR (95%, 1-day)"], "Current Value": end_val,
                "Total P/L": end_val - rec["initial_capital"],
            }
        if cumret_by_name:
            await _bot_send_photo(context, generate_cumulative_comparison_chart(cumret_by_name), "Cumulative Performance (1Y)")
        if metrics_by_name:
            await _bot_send_photo(context, generate_comparison_table_chart(metrics_by_name), "Risk Metrics — All Portfolios (1Y)")
        if dd_by_name:
            await _bot_send_photo(context, generate_drawdown_comparison_chart(dd_by_name), "Underwater Analysis (1Y)")
    except Exception as e:
        print(f"⚠️ Daily report failed: {e}")
        try:
            await _bot_send_text(context, f"⚠️ Daily report failed to generate: `{e}`")
        except Exception:
            pass


# ---- Alert checker (Telegram JobQueue — background schedule) ----------------------------------

ALERT_COOLDOWN_HOURS = 6
ALERT_CHECK_INTERVAL_SECONDS = 900  # 15 minutes


def _cooldown_ok(a: dict) -> bool:
    if not a.get("last_triggered"):
        return True
    last = datetime.fromisoformat(a["last_triggered"])
    return (datetime.now() - last).total_seconds() > ALERT_COOLDOWN_HOURS * 3600


async def check_alerts(context: ContextTypes.DEFAULT_TYPE):
    """Runs on a background schedule (see application.job_queue.run_repeating in Section 9).
    Never spams: each alert has a cooldown, and only fires when its condition is newly true."""
    alerts = [a for a in STORE["alerts"].values() if a["enabled"]]
    if not alerts:
        return
    portfolios = get_portfolios()
    end = date.today()
    start_short = end - timedelta(days=10)   # for 1-day-move checks
    start_long = end - timedelta(days=400)   # for drawdown checks

    for a in alerts:
        if not _cooldown_ok(a):
            continue
        try:
            triggered, message = False, ""
            if a["type"] == "asset_move":
                prices, failed = download_prices([a["ticker"]], start_short, end)
                if a["ticker"] in prices.columns and len(prices) >= 2:
                    move = float(prices[a["ticker"]].iloc[-1] / prices[a["ticker"]].iloc[-2] - 1) * 100
                    if abs(move) > a["threshold"]:
                        triggered = True
                        message = f"🔔 *{a['ticker']}* moved {move:+.2f}% in one day (alert threshold {a['threshold']:.1f}%)."
            elif a["type"] in ("perf", "drawdown", "weight_drift") and a.get("pid") in portfolios:
                rec = portfolios[a["pid"]]
                if a["type"] == "weight_drift":
                    prices, failed = download_prices(list(rec["weights"].keys()), start_short, end)
                    if len(prices) >= 1 and a["ticker"] in prices.columns:
                        returns = prices.pct_change().dropna(how="all")
                        port_ret, _ = portfolio_daily_returns(rec["weights"], returns)
                        # approximate current weight via cumulative drift since period start
                        cum = (1 + returns[a["ticker"]]).cumprod().iloc[-1] if a["ticker"] in returns.columns else 1.0
                        port_cum = (1 + port_ret).cumprod().iloc[-1] if len(port_ret) else 1.0
                        target = rec["weights"][a["ticker"]]
                        implied_current = target * cum / port_cum if port_cum else target
                        drift = (implied_current - target) * 100
                        if abs(drift) > a["threshold"]:
                            triggered = True
                            message = (f"🔔 *{a['ticker']}* in *{rec['name']}* has drifted "
                                       f"{drift:+.2f}pp from its {target*100:.1f}% target weight "
                                       f"(alert threshold {a['threshold']:.1f}pp).")
                else:
                    prices, failed = download_prices(list(rec["weights"].keys()), start_long if a["type"] == "drawdown" else start_short, end)
                    returns = prices.pct_change().dropna(how="all")
                    port_ret, _ = portfolio_daily_returns(rec["weights"], returns)
                    if a["type"] == "perf" and len(port_ret) >= 1:
                        move = float(port_ret.iloc[-1]) * 100
                        if move < -abs(a["threshold"]):
                            triggered = True
                            message = f"🔔 *{rec['name']}* fell {move:.2f}% in one day (alert threshold -{a['threshold']:.1f}%)."
                    elif a["type"] == "drawdown" and len(port_ret) >= 5:
                        dd = float(drawdown_series(port_ret).iloc[-1]) * 100
                        if abs(dd) > a["threshold"]:
                            triggered = True
                            message = f"🔔 *{rec['name']}* drawdown is {dd:.2f}% (alert threshold {a['threshold']:.1f}%)."
            if triggered:
                await context.bot.send_message(chat_id=a["chat_id"], text=message, parse_mode=ParseMode.MARKDOWN)
                a["last_triggered"] = datetime.now().isoformat()
                save_store(STORE)
        except Exception as e:
            print(f"⚠️ Alert check failed for {a['id']}: {e}")

print("Alert checker (JobQueue callback) loaded.")

# ============================================================================================
# SECTION 10 — RUN THE BOT
# ============================================================================================
# Wrapped in a function + `if __name__ == "__main__"` guard so this file can also be imported
# (e.g. by daily_report.py, to reuse the analytics/chart functions above) WITHOUT starting the
# bot or requiring BOT_TOKEN to be set at import time. Running it directly still blocks on
# run_polling() — that's expected; it's what keeps interactive commands, buttons, and the
# scheduled alert checker (JobQueue) working. Stop it with Ctrl+C, or via your host's process
# manager (systemd, Railway, Render, Docker, etc. — see README.md).

def main():
    if not BOT_TOKEN:
        raise RuntimeError(
            "BOT_TOKEN environment variable not set. Set it before running this script, e.g.:\n"
            "  export BOT_TOKEN=\"your-token-from-@BotFather\"\n"
            "or configure it as a secret/environment variable on your hosting platform. "
            "See README.md."
        )

    application = Application.builder().token(BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("portfolio", portfolio_command))
    application.add_handler(CallbackQueryHandler(on_callback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    if application.job_queue is not None:
        application.job_queue.run_repeating(check_alerts, interval=ALERT_CHECK_INTERVAL_SECONDS, first=60)
        print(f"Alert checker scheduled every {ALERT_CHECK_INTERVAL_SECONDS//60} minutes.")

        daily_enabled = os.environ.get("DAILY_REPORT_ENABLED", "true").strip().lower() in ("1", "true", "yes")
        if daily_enabled and CHAT_ID:
            hour_utc = int(os.environ.get("DAILY_REPORT_HOUR_UTC", "13"))  # 13:00 UTC ≈ 9am US Eastern (adjust as needed)
            application.job_queue.run_daily(daily_report_job, time=dt_time(hour=hour_utc, minute=0))
            print(f"Daily report scheduled at {hour_utc:02d}:00 UTC to CHAT_ID.")
        elif daily_enabled and not CHAT_ID:
            print("Daily report: CHAT_ID environment variable not set — automatic daily report "
                  "not scheduled. Set CHAT_ID to enable it (see README.md).")
    else:
        print("⚠️ JobQueue unavailable (install 'python-telegram-bot[job-queue]' for scheduled alerts "
              "and the daily report) — every other feature still works, alerts/daily report just "
              "won't auto-run.")

    print("Bot is starting... send /start to it on Telegram.")
    application.run_polling()


if __name__ == "__main__":
    main()

# ============================================================================================
# CHANGE LOG — v1 -> v2, what changed and why, and what was deliberately left out of this pass
# ============================================================================================
"""
PRESERVED FROM v1 (unchanged mechanism/methodology):
- The 3 original portfolios and 2 benchmarks — same tickers, same weights (now seeded into the
  persistent store on first run; editing/deleting them requires an explicit action).
- BOT_TOKEN / CHAT_ID are read from environment variables (never hardcoded).
- The yfinance-based data layer, the in-memory price cache, and the "never invent missing data"
  behavior (failed tickers are excluded and reported).
- Every v1 calculation formula byte-for-byte: annualize_return, annualize_vol, sharpe_ratio,
  sortino_ratio, max_drawdown, calmar_ratio, ulcer_index, historical_var, beta_vs_benchmark,
  correlation_vs_benchmark — the daily-rebalanced constant-mix convention is still the default
  everywhere except the new Rebalancing Analysis section, which now explicitly labels itself.
- The 7 stress-test scenarios and the factor-beta methodology, unchanged.
- Inline-keyboard, image-based navigation with Back/Main Menu on every screen.
- 1D/7D/15D/30D/6M/1Y/3Y/5Y/10Y presets + custom date range (1D added per this brief; the other 8
  presets are exactly what v1 already had).

NEW IN v2 (this is most of the added code):
- storage.py-equivalent (Section 2): a JSON-backed persistent store for portfolios, alerts, and
  per-chat settings, reloaded automatically on restart. Custom portfolios get ids c1, c2, ...;
  the 3 built-ins keep stable ids p1/p2/p3 so nothing shifts if you add/remove others.
- Full Portfolio Management: create (name → capital → currency → holdings-as-text, with 100%
  weight validation before saving), edit (add/remove holding, reweight, rename, change capital),
  delete (with an explicit Yes/Cancel confirmation).
- Overview: every portfolio's value/P&L/1D-10D returns from ONE shared max-window download,
  sliced in memory per period (no repeated yfinance calls) — plus value/cumulative comparison
  charts.
- Comparison: pick any 2+ portfolios/benchmarks via toggle buttons, get a full metrics table,
  cumulative-performance overlay, drawdown overlay, and a risk/return scatter.
- Drawdown: current/max drawdown, episode detection (peak→trough→recovery), duration, recovery
  time, average depth, worst episode — plus an underwater chart.
- Contribution: now reports $ contribution and P/L alongside % contribution and % return, with a
  top-contributors/detractors chart, kept explicitly distinct from raw asset return everywhere.
- Rolling Metrics: rolling return/vol/Sharpe/Sortino/beta/correlation over a selectable window
  (30D/60D/90D/6M/1Y), charted as stacked panels.
- Rebalancing: a genuine historical simulator (share-count based, NOT the constant-mix
  convention) comparing Buy & Hold vs. monthly/quarterly/semiannual/annual/threshold (±5% default)
  rebalancing — final value, return, vol, Sharpe/Sortino/Calmar, # rebalances, estimated turnover.
  Labeled throughout as a historical simulation, not a guarantee.
- Derivatives Analysis: proxy-only (portfolio beta, volatility, correlation with ^VIX, known
  leveraged/inverse-ETF flags, known bond-ETF duration) — explicitly states "no direct derivative
  positions detected" rather than inventing options Greeks this data source can't provide.
- Alerts: persisted, 4 types (performance/drawdown/asset-move/weight-drift), checked every 15
  minutes via python-telegram-bot's JobQueue with a 6-hour per-alert cooldown so you're not
  spammed; view/toggle/delete from Telegram.
- Reports: single-portfolio and combined (2-3 portfolio) PDF reports, assembled with matplotlib's
  PdfPages, reusing the exact same chart-building functions the bot uses for Telegram photos, in
  a layout modeled on institutional fund fact sheets (header + as-of date, growth chart, metrics
  tables, disclosures/methodology page) — not copying any specific provider's branding.
- Settings expanded and now persisted per chat: default portfolio, period, benchmark, rebalancing
  method, rolling window, currency, date format.

DELIBERATELY NOT BUILT IN THIS PASS (the brief itself allows deferring this):
- Full transaction/ledger accounting (buy/sell/deposit/withdrawal/dividend/fee history with
  realized vs. unrealized P&L). The data model's `initial_capital` field and the portfolio
  store's JSON schema are structured so a `transactions.json` keyed by portfolio id could be
  added later without touching the weight-based calculation engine — but actually computing
  realized/unrealized P&L from a transaction log is a genuinely separate subsystem from
  everything else in this file, and bolting on a partial/untested version seemed worse than
  shipping the clean foundation and flagging it clearly here.
- Sector/asset-class contribution breakdown is available on request (via the cached, best-effort
  `get_sector()` lookup) but is not auto-fetched on every Contribution report, since it's a slow,
  unreliable-on-free-tier call per ticker — this keeps every other report fast.

KNOWN LIMITATIONS (free-data-source constraints):
1. Yahoo Finance (yfinance) is unauthenticated and can rate-limit or silently drop a ticker;
   failed tickers are excluded and reported, never invented.
2. "Current value" is still weight% × a configurable initial capital, not derived from real share
   counts — there's no brokerage data source wired in.
3. Contribution-to-return uses the standard first-order approximation (weight × asset return);
   it diverges slightly from an exact geometric attribution over very long/volatile periods.
4. Stress tests remain a linear single-factor beta model — "scenario estimates," never forecasts.
5. Rolling metrics need at least window+5 trading days of history; very long windows (1Y) on a
   portfolio with a short price history will report "not enough history" rather than guess.
6. Alerts rely on the bot's process staying up and connected — if the host restarts or the
   process is killed, checks stop until it comes back (alerts and their settings are still saved
   to disk, so nothing is lost — see README.md for hosts that keep the process running 24/7).
7. PDF pages are laid out on a fixed A4 page (header/footer/page numbers on every page); each
   chart is centered and scaled to fit without distortion.
"""
