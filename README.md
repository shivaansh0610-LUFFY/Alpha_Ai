<div align="center">
  <h1>Alpha_Ai × Polaris Challenge</h1>
  <p><strong>Advanced Probabilistic Forecasting for Bitcoin Hourly Prices</strong></p>

  <p>
    <a href="#-project-overview">Overview</a> •
    <a href="#-the-math--methodology">Methodology</a> •
    <a href="#-evaluation--metrics">Metrics</a> •
    <a href="#-standout-features">Features</a> •
    <a href="#-how-to-run">Installation</a>
  </p>
</div>

---

## 🚀 Project Overview

Developed for the **Alpha AI × Polaris Challenge**, this project is a robust, live-updating dashboard that predicts the price range of Bitcoin for the upcoming hour. 

Instead of generating a simple "up or down" point prediction (which is notoriously unreliable in crypto), this system models uncertainty. It outputs **confidence intervals** (e.g., 80%, 90%, 95%) that dynamically adapt to real-time market chaos. 

**Live Demo:** *[Insert Streamlit Cloud URL Here]*

## 🧠 The Math & Methodology

Financial assets like Bitcoin exhibit properties that break standard predictive models. To capture reality accurately, this project relies on a tailored **Geometric Brownian Motion (GBM)** implementation:

1. **Student-t Distribution for Fat Tails:** 
   Standard models use a Normal (Gaussian) distribution, which drastically underestimates the frequency of massive price spikes/crashes in crypto. By fitting historical log returns to a **Student-t distribution**, our model correctly anticipates these "black swan" events, keeping coverage safe.
2. **Exponentially Weighted Moving Average (EWMA) Volatility:** 
   Volatility clusters—calm periods stay calm, and wild periods stay wild. Standard rolling windows are too slow to react. EWMA applies heavier weighting to the most recent hours, allowing the model's prediction bands to instantly expand when the market breaks out.
3. **Monte Carlo Simulations:** 
   The model runs 10,000 parallel universe simulations of the next hour's price action. By finding the 2.5th and 97.5th percentiles of those 10,000 simulated paths, it defines the 95% Confidence Interval.

## 📊 Evaluation & Metrics

The system was rigorously backtested on 750 historical 1-hour candles to ensure the predictive bounds are both accurate and as tight as possible. The model enforces a strict "no-peeking" rule, evaluating entirely out-of-sample.

* **Target Coverage:** 95%
* **Actual Backtest Coverage:** **98.00%**
* **Mean Winkler Score (95% CI):** **2,055**

*(The Winkler Score heavily penalizes bounds that fail to capture the true price while rewarding narrow, highly confident bounds. A score of ~2055 on a ~$60K+ asset represents roughly a tight 3% prediction window that rarely breaks).*

## ✨ Standout Features

- **Real-Time Data Pipeline:** Streams live, free binance OHLCV data without requiring API keys.
- **Monte Carlo Probability Density Chart:** Physically plots the distribution of the 10,000 simulated paths so users can visualize the heavy tails and the model's confidence.
- **Interactive Stress Test:** A sidebar slider allows users to artificially inflate or deflate the volatility multiplier to watch how the model adapts to simulated market shocks in real-time.
- **Regime Analysis:** Breaks down the backtest score by "CALM" vs "VOLATILE" environments to definitively prove that the EWMA volatility clustering works.
- **Live Fear & Greed Integration:** Pulls current market sentiment directly into the UI to contextualize the price action.

## 💻 How to Run (Local Setup)

Want to run the backtest and the dashboard on your own machine?

1. **Clone the repository:**
   ```bash
   git clone https://github.com/shivaansh0610-LUFFY/Alpha_Ai.git
   cd Alpha_Ai/btc_prediction
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Backtest:**
   *(Run this once to generate `backtest_results.jsonl` and calculate the Winkler score)*
   ```bash
   python backtest.py
   ```

4. **Launch the Live App:**
   ```bash
   streamlit run dashboard.py
   ```
   *The dashboard will open automatically in your browser at `http://localhost:8501`.*

---

<div align="center">
  <i>Built with Python, Pandas, SciPy, and Streamlit.</i>
</div>
