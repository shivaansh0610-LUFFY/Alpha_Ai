"""
GBM (Geometric Brownian Motion) model for Bitcoin hourly price prediction.
Uses Student-t distribution for fat tails and rolling EWMA volatility clustering.
"""

import numpy as np
import pandas as pd
import requests
from scipy import stats
from typing import Tuple, Optional, List, Dict
from datetime import datetime, timezone


BINANCE_URL  = "https://data-api.binance.vision/api/v3/klines"
FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=1"
SYMBOL       = "BTCUSDT"
INTERVAL     = "1h"
N_SIMS       = 10_000
CONFIDENCE   = 0.95


# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_btc_candles(n_bars: int = 750, end_time_ms: Optional[int] = None) -> pd.DataFrame:
    """Fetch BTCUSDT 1-hour OHLCV bars from Binance public API (no key needed)."""
    # Fetch an extra bar because the last one might be incomplete (currently open)
    params = {"symbol": SYMBOL, "interval": INTERVAL, "limit": min(n_bars + 1, 1000)}
    if end_time_ms is not None:
        params["endTime"] = end_time_ms

    resp = requests.get(BINANCE_URL, params=params, timeout=15)
    resp.raise_for_status()

    df = pd.DataFrame(resp.json(), columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "n_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df["open_time"]  = pd.to_datetime(df["open_time"],  unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    # The prompt explicitly asks for "the very latest closed bar"
    # Binance includes the current incomplete/open candle, so we drop it.
    now = pd.Timestamp.now(timezone.utc)
    df = df[df["close_time"] < now].copy()

    # Ensure we return exactly n_bars
    return df.sort_values("open_time").reset_index(drop=True).tail(n_bars)


def fetch_fear_greed() -> dict:
    """
    Fetch the Bitcoin Fear & Greed Index from alternative.me (free, no key).
    Returns dict with 'value' (0-100) and 'classification' (e.g. 'Fear').
    """
    try:
        resp = requests.get(FEAR_GREED_URL, timeout=8)
        resp.raise_for_status()
        data = resp.json()["data"][0]
        return {
            "value": int(data["value"]),
            "classification": data["value_classification"],
            "timestamp": data["timestamp"],
        }
    except Exception:
        return {"value": 50, "classification": "Neutral", "timestamp": ""}


# ── Core math ─────────────────────────────────────────────────────────────────

def compute_log_returns(closes: np.ndarray) -> np.ndarray:
    """Log return at step i = log(price[i] / price[i-1])."""
    return np.diff(np.log(closes))


def fit_student_t(returns: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit a Student-t distribution to log returns.
    Returns (degrees_of_freedom, location, scale).
    BTC typically gives df≈3–5 (normal would be df=∞).
    Lower df = heavier tails = bigger surprise moves are more likely.
    """
    return stats.t.fit(returns)


def rolling_volatility(returns: np.ndarray, window: int = 10) -> float:
    """
    EWMA (Exponentially Weighted Moving Average) volatility.
    Recent bars get more weight than old bars.
    This captures 'volatility clustering': calm periods stay calm,
    wild periods stay wild — at least for a while.
    """
    if len(returns) < 2:
        return float(np.std(returns)) if len(returns) else 0.001
    recent  = returns[-window:] if len(returns) >= window else returns
    weights = np.exp(np.linspace(-1.0, 0.0, len(recent)))
    weights /= weights.sum()
    mean_w  = np.sum(weights * recent)
    var_w   = np.sum(weights * (recent - mean_w) ** 2)
    return float(np.sqrt(var_w))


# ── Prediction functions ──────────────────────────────────────────────────────

def predict_next_bar(
    closes:     np.ndarray,
    vol_window: int = 10,
    n_sims:     int = N_SIMS,
    confidence: float = CONFIDENCE,
    use_normal: bool = False,
    rng:        Optional[np.random.Generator] = None,
) -> Tuple[float, float, float]:
    """
    Core GBM prediction: simulate 10,000 possible next-hour prices,
    then read off the confidence interval.

    use_normal=True  → uses normal (Gaussian) distribution  — baseline
    use_normal=False → uses Student-t distribution          — our model
    """
    if rng is None:
        rng = np.random.default_rng()

    returns = compute_log_returns(closes)
    if len(returns) < 5:
        raise ValueError("Need at least 6 close prices.")

    fit_returns = returns[-500:] if len(returns) > 500 else returns
    df_t, loc_t, scale_t = fit_student_t(fit_returns)

    recent_vol = rolling_volatility(returns, window=vol_window)
    scale_use  = recent_vol if recent_vol > 1e-8 else scale_t
    drift      = float(np.mean(returns[-vol_window:]))

    if use_normal:
        shocks = rng.normal(loc=0, scale=scale_use, size=n_sims)
    else:
        shocks = stats.t.rvs(df=df_t, loc=0, scale=scale_use,
                             size=n_sims, random_state=rng)

    sim_prices = closes[-1] * np.exp(drift + shocks)
    alpha      = 1 - confidence
    lower      = float(np.quantile(sim_prices, alpha / 2))
    upper      = float(np.quantile(sim_prices, 1 - alpha / 2))

    return float(closes[-1]), lower, upper


def predict_multiple_bands(
    closes:            np.ndarray,
    confidence_levels: List[float] = [0.80, 0.90, 0.95],
    vol_window:        int = 10,
    n_sims:            int = N_SIMS,
    rng:               Optional[np.random.Generator] = None,
    vol_multiplier:    float = 1.0,
) -> Tuple[float, Dict[float, Tuple[float, float]], np.ndarray]:
    """
    Predict multiple confidence bands in a single simulation pass.
    Returns (current_price, {0.80: (lo, hi), 0.90: (lo, hi), 0.95: (lo, hi)})

    All bands come from the SAME 10,000 simulated paths — we just read
    different quantiles from the same distribution.
    """
    if rng is None:
        rng = np.random.default_rng()

    returns     = compute_log_returns(closes)
    fit_returns = returns[-500:] if len(returns) > 500 else returns
    df_t, _, _  = fit_student_t(fit_returns)

    recent_vol = rolling_volatility(returns, window=vol_window) * vol_multiplier
    drift      = float(np.mean(returns[-vol_window:]))

    shocks     = stats.t.rvs(df=df_t, loc=0, scale=recent_vol,
                              size=n_sims, random_state=rng)
    sim_prices = closes[-1] * np.exp(drift + shocks)

    bands = {}
    for conf in confidence_levels:
        alpha       = 1 - conf
        bands[conf] = (
            float(np.quantile(sim_prices, alpha / 2)),
            float(np.quantile(sim_prices, 1 - alpha / 2)),
        )

    return float(closes[-1]), bands, sim_prices


def get_volatility_regime(returns: np.ndarray, window: int = 10) -> dict:
    """
    Classify current market as CALM / MODERATE / VOLATILE
    by comparing recent vol to long-term vol percentile.
    """
    if len(returns) < 50:
        return {"label": "MODERATE", "color": "#f39c12", "emoji": "〰️", "percentile": 50.0}

    current_vol  = rolling_volatility(returns, window=window)
    all_vols     = [
        rolling_volatility(returns[:i], window=window)
        for i in range(window + 1, len(returns), 5)
    ]
    if not all_vols:
        pct = 50.0
    else:
        pct = float(stats.percentileofscore(all_vols, current_vol))

    if pct < 33:
        return {"label": "CALM",     "color": "#00c896", "emoji": "😴", "percentile": pct}
    elif pct < 67:
        return {"label": "MODERATE", "color": "#f39c12", "emoji": "👀", "percentile": pct}
    else:
        return {"label": "VOLATILE", "color": "#ff4b6e", "emoji": "🔥", "percentile": pct}


# ── Scoring ───────────────────────────────────────────────────────────────────

def winkler_score(lower: float, upper: float, actual: float,
                  confidence: float = CONFIDENCE) -> float:
    """
    Winkler interval score (lower = better forecaster).
    Inside range  → score = just the width (reward for being tight)
    Outside range → score = width + big penalty proportional to how far off
    """
    alpha = 1 - confidence
    width = upper - lower
    if lower <= actual <= upper:
        return width
    elif actual < lower:
        return width + (2.0 / alpha) * (lower - actual)
    else:
        return width + (2.0 / alpha) * (actual - upper)


def evaluate(predictions: list) -> dict:
    """
    Compute coverage, mean width, and mean Winkler score for a list of predictions.
    Each prediction dict needs: lower, upper, actual, confidence (optional).
    """
    inside   = []
    widths   = []
    winklers = []

    for p in predictions:
        lo, hi, actual = p["lower"], p["upper"], p["actual"]
        conf = p.get("confidence", CONFIDENCE)
        inside.append(1 if lo <= actual <= hi else 0)
        widths.append(hi - lo)
        winklers.append(winkler_score(lo, hi, actual, conf))

    return {
        "coverage_95":     float(np.mean(inside)),
        "mean_width":      float(np.mean(widths)),
        "mean_winkler_95": float(np.mean(winklers)),
        "n_predictions":   len(predictions),
    }
