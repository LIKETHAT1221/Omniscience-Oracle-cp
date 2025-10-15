# Omniscience Enhanced TA Stack (Streamlit Edition)

This project is a **numeric, machine-learning-ready sports odds analysis engine** implementing a robust technical analysis (TA) stack with explicit weighting, EV/Kelly calculations, trend projection, strict parsing, and advanced debug tracing. It fully preserves the logic and UX of the original Omniscience TA JavaScript UI, now in Python/Streamlit for interactive analytics and future ML/AI expansion.

---

## Features

- **Strict Odds Feed Parsing:** Paste your odds blocks (header + 5-line blocks) for robust, error-checked data ingestion.
- **Full TA Stack:** Numeric signals for AMA, SMA, EMA, EMA/SMA crossover, RSI, MACD, BB, ATR, Z-score, ROC, Steam, Greeks, Fibonacci (with polarity/fib swing debug).
- **Explicit Weights:** All signals get normalized, explicit weights for a composite bias/confidence score.
- **Recency Decay:** Recent ticks weigh more; stale ticks less.
- **EV & Kelly Sizing:** Model probability, market probability, explicit EV, and safe Kelly fraction (with debug).
- **Trend Projection:** Weighted projection and reversal zones using TA and Fib extensions.
- **Debug/Narrative:** All internals (signals, weights, bias, swing debug, projections) are logged and visible.
- **Streamlit UI:** Paste raw feed, click Analyze, see JSON output and debug, ready for ML export.

---

## Quickstart

1. **Install requirements**
    ```bash
    pip install -r requirements.txt
    ```

2. **Run the Streamlit app**
    ```bash
    streamlit run omniscience_ta.py
    ```

3. **Paste your odds feed** (first line header, then repeated 5-line blocks)

4. **Click Analyze**  
   - Get TA signals, projections, EV/Kelly, and full JSON/debug output.

---

## Odds Feed Format

- First line: header (required)
- Then, repeated 5-line blocks for each tick:
  1. `MM/DD h:mmAM/PM [team?] spread`
  2. `spread vig`
  3. `total (e.g. o154/u154.5)`
  4. `total vig`
  5. `awayML homeML`

---

## ML Integration

The output includes a `normalized_signals` vector (all numeric), ready for use in any ML pipeline (e.g., sklearn, xgboost, pytorch).  
You can easily extend the app to train supervised models on historical labeled outcomes.

---

## Requirements

- Python 3.8+
- streamlit
- numpy
- pandas
- scikit-learn

---

## License

MIT

---
