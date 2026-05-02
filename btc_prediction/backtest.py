"""
Part A — 30-day backtest of BTC hourly GBM prediction model.

Fetches ~720 BTCUSDT 1-hour bars (30 days), then for each bar:
  - Uses ONLY data up to bar i to predict bar i+1
  - Records lower, upper, actual close, whether it was covered
  - Computes coverage, average width, and Winkler score

Saves results to backtest_results.jsonl (one JSON object per line).
"""

import json
import time
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from model import fetch_btc_candles, predict_next_bar, evaluate, CONFIDENCE

# Minimum bars of history needed before we start predicting.
# This ensures a stable volatility and Student-t fit.
MIN_HISTORY = 100

OUTPUT_FILE = Path(__file__).parent / "backtest_results.jsonl"


def run_backtest(n_total_bars: int = 850, vol_window: int = 20, n_sims: int = 10_000):
    """
    Rolling-window backtest.

    We fetch n_total_bars. The first MIN_HISTORY bars serve as warm-up
    history for the first prediction. We then make predictions for bars
    MIN_HISTORY .. n_total_bars-2 (each predicting the bar after it).

    NO PEEKING: at prediction step i, closes[:i+1] is used. The actual
    at bar i+1 is only revealed AFTER the prediction is locked in.
    """
    print(f"Fetching {n_total_bars} BTCUSDT 1h bars from Binance …")
    df = fetch_btc_candles(n_bars=n_total_bars)
    closes = df["close"].values
    times = df["open_time"].values  # numpy datetime64

    print(f"  Got {len(df)} bars: {df['open_time'].iloc[0]} → {df['open_time'].iloc[-1]}")

    predictions = []
    rng = np.random.default_rng(seed=42)

    n = len(closes)
    predict_indices = range(MIN_HISTORY, n - 1)
    print(f"Running backtest on {len(predict_indices)} prediction steps …")

    for i in tqdm(predict_indices, desc="Backtesting"):
        # --- NO PEEK: only closes[0 .. i] available ---
        history = closes[: i + 1]
        bar_time = pd.Timestamp(times[i]).isoformat()
        next_bar_time = pd.Timestamp(times[i + 1]).isoformat()

        try:
            current_price, lower, upper = predict_next_bar(
                history, vol_window=vol_window, n_sims=n_sims, rng=rng
            )
        except Exception as e:
            print(f"  Warning: prediction failed at bar {i}: {e}", file=sys.stderr)
            continue

        actual = float(closes[i + 1])
        covered = int(lower <= actual <= upper)

        record = {
            "bar_time": bar_time,
            "next_bar_time": next_bar_time,
            "current_price": float(current_price),
            "lower": float(lower),
            "upper": float(upper),
            "actual": actual,
            "covered": covered,
            "width": float(upper - lower),
            "confidence": CONFIDENCE,
        }
        predictions.append(record)

    print(f"\nSaving {len(predictions)} predictions to {OUTPUT_FILE} …")
    with open(OUTPUT_FILE, "w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")

    metrics = evaluate(predictions)
    print("\n" + "=" * 50)
    print("  BACKTEST RESULTS")
    print("=" * 50)
    print(f"  Predictions        : {metrics['n_predictions']}")
    print(f"  Coverage (target≈0.95): {metrics['coverage_95']:.4f}")
    print(f"  Mean width ($)     : ${metrics['mean_width']:,.2f}")
    print(f"  Mean Winkler score : {metrics['mean_winkler_95']:,.2f}")
    print("=" * 50)
    print(f"\nResults saved to: {OUTPUT_FILE}")

    return metrics


def load_backtest_results(path: Path = OUTPUT_FILE) -> list:
    """Load previously saved backtest results from JSONL file."""
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


if __name__ == "__main__":
    metrics = run_backtest(n_total_bars=850, vol_window=10, n_sims=10_000)
