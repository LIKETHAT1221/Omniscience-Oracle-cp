import streamlit as st
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta

# === TA FUNCTIONS ===

def SMA(series, period):
    return pd.Series(series).rolling(period).mean().values

def EMA(series, period):
    return pd.Series(series).ewm(span=period, adjust=False).mean().values

def RSI(series, period=14):
    series = pd.Series(series)
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-12)
    return 100 - (100 / (1 + rs))

def MACD(series, fast=12, slow=26, signal=9):
    ema_fast = EMA(series, fast)
    ema_slow = EMA(series, slow)
    macd = ema_fast - ema_slow
    signal_line = EMA(macd, signal)
    hist = macd - signal_line
    return macd, signal_line, hist

def BollingerBands(series, period=20, mult=2):
    sma = SMA(series, period)
    std = pd.Series(series).rolling(period).std().values
    upper = sma + mult * std
    lower = sma - mult * std
    return upper, lower

def ATR(series, period=14):
    series = pd.Series(series)
    tr = series.diff().abs()
    return tr.rolling(window=period).mean().values

def zscore(series, period=10):
    ser = pd.Series(series)
    return ((ser - ser.rolling(period).mean()) / (ser.rolling(period).std() + 1e-12)).values

def adaptiveMA(series, fast=2, slow=10, efficiency_lookback=8):
    n = len(series)
    ama = [None]*n
    for i in range(slow+efficiency_lookback-1, n):
        change = abs(series[i] - series[i-efficiency_lookback])
        volatility = np.sum(np.abs(np.diff(series[i-efficiency_lookback:i+1])))
        ER = 0 if volatility == 0 else change / (volatility + 1e-12)
        # Volatility-normalized
        ER = np.clip(ER / (np.std(series[max(0,i-50):i+1])+1e-5), 0, 1)
        fastSC = 2/(fast+1)
        slowSC = 2/(slow+1)
        SC = (ER * (fastSC-slowSC) + slowSC)**2
        ama[i] = series[i-1] if ama[i-1] is None else ama[i-1] + SC*(series[i]-ama[i-1])
    return np.array(ama)

def ROC(series, period=2):
    arr = np.array(series)
    out = np.full_like(arr, np.nan, dtype=np.float64)
    for i in range(period, len(arr)):
        if arr[i-period] != 0:
            out[i] = 100*(arr[i]-arr[i-period])/abs(arr[i-period])
    return out

def steam_moves(series, threshold=2):
    return np.array([1 if abs(series[i]-series[i-1])>=threshold else 0 for i in range(1,len(series))]+[0])

def fib_levels(series, min_lookback=13):
    swing_start = series[-min_lookback]
    swing_end = series[-1]
    swing_diff = swing_end - swing_start
    polarity = np.sign(swing_diff)
    high, low = (swing_end, swing_start) if polarity>0 else (swing_start, swing_end)
    retracements = [high - abs(swing_diff)*r for r in [0.236,0.382,0.5,0.618,0.786]]
    extensions = [high + abs(swing_diff)*e for e in [0.236,0.382,0.5,0.618,1.0]]
    return {
        "swing_start": swing_start,
        "swing_end": swing_end,
        "swing_diff": swing_diff,
        "polarity": polarity,
        "retracements": retracements,
        "extensions": extensions,
        "debug": {
            "high": high, "low": low
        }
    }

def greek_delta(series):
    series = np.array(series)
    if len(series)<3: return 0
    delta = series[-1]-series[-2]
    prev_delta = series[-2]-series[-3]
    gamma = delta-prev_delta
    return delta, gamma

# === EV & KELLY ===

def implied_probability(odds):
    # American odds
    if odds > 0:
        return 100/(odds+100)
    else:
        return abs(odds)/(abs(odds)+100)

def decimal_odds(odds):
    return 1 + (odds/100) if odds > 0 else 1 + (100/abs(odds))

def calculate_no_vig_prob(odds1, odds2):
    imp1 = implied_probability(odds1)
    imp2 = implied_probability(odds2)
    total = imp1 + imp2
    return imp1/total, imp2/total

def expected_value(prob, odds):
    dec = decimal_odds(odds)
    return prob*(dec-1) - (1-prob)

def kelly_fraction(prob, odds):
    dec = decimal_odds(odds)
    b = dec-1
    q = 1-prob
    f_star = (b*prob-q)/b if b!=0 else 0
    return max(0, min(f_star, 1)) # [0,1]

# === RECENCY WEIGHTING ===

def recency_weights(timestamps, decay_minutes=60):
    now = max(timestamps)
    deltas = np.array([(now-t).total_seconds()/60 for t in timestamps])
    weights = np.exp(-deltas/decay_minutes)
    weights = weights / (weights.sum() + 1e-12)
    return weights

# === DATA PARSING ===

def parse_blocks(raw):
    lines = [l.strip() for l in raw.split('\n') if l.strip()]
    blocks = []
    i = 1 if lines and lines[0].lower().startswith('time') else 0
    while i+4 < len(lines):
        L1, L2, L3, L4, L5 = lines[i:i+5]
        tks = L1.split()
        try:
            ts = datetime.strptime(f"{tks[0]} {tks[1]}", "%m/%d %I:%M%p")
        except Exception: ts = None
        team = tks[2] if len(tks) > 3 and not tks[2].replace('.','').isdigit() else None
        spread = float(tks[3]) if team else float(tks[2])
        spread = -spread
        spread_vig = float(L2.split()[0])
        total_side = L3[0].lower() if L3[0].lower() in 'ou' else None
        total = float(L3[1:]) if total_side else float(L3)
        total_vig = float(L4.split()[0])
        ml_nums = [float(x) for x in L5.replace('even','100').split() if x.replace('.','',1).replace('-','',1).isdigit()]
        ml_away, ml_home = ml_nums[0], ml_nums[1] if len(ml_nums)>1 else (None,None)
        blocks.append(dict(
            time=ts, team=team, spread=spread, spread_vig=spread_vig, total=total, total_side=total_side,
            total_vig=total_vig, ml_away=ml_away, ml_home=ml_home
        ))
        i += 5
    return blocks

# === MAIN STREAMLIT APP ===

st.set_page_config(page_title="Omniscience TA Stack", layout="wide")
st.title("Omniscience TA Stack — Enhanced Numeric Engine")

st.markdown("**Paste your odds feed. Each block: 5 lines (time, spread, etc).**")
raw = st.text_area("Odds Feed", height=280)
analyze_btn = st.button("Analyze")

if analyze_btn and raw.strip():
    blocks = parse_blocks(raw)
    df = pd.DataFrame(blocks)
    df = df.sort_values("time").reset_index(drop=True)
    # Recency weighting
    rec_weights = recency_weights(df["time"].tolist())
    # All TA indicators (numeric)
    spread = df["spread"].values
    total = df["total"].values
    ml = df["ml_away"].values
    # --- Indicators ---
    sma = SMA(spread, 10)
    ema = EMA(spread, 5)
    rsi = RSI(spread, 7)
    macd, macd_sig, _ = MACD(spread)
    bb_upper, bb_lower = BollingerBands(spread)
    atr = ATR(spread)
    zs = zscore(spread)
    ama = adaptiveMA(spread, 2, 8, 6)
    roc = ROC(spread, 2)
    steam = steam_moves(spread)
    fib = fib_levels(spread)
    delta, gamma = greek_delta(spread)
    # Numeric signals (latest tick)
    idx = len(spread)-1
    signals = dict(
        AMA = (ama[idx] - spread[idx-1])/(np.std(spread[max(0,idx-20):idx+1])+1e-6) if ama[idx] is not None else 0,
        SMA = (sma[idx]-spread[idx-1])/(np.std(spread[max(0,idx-20):idx+1])+1e-6) if not np.isnan(sma[idx]) else 0,
        EMA = (ema[idx]-spread[idx-1])/(np.std(spread[max(0,idx-20):idx+1])+1e-6) if not np.isnan(ema[idx]) else 0,
        EMA_SMA_X = 1 if ema[idx] > sma[idx] else -1 if ema[idx] < sma[idx] else 0,
        RSI = (rsi[idx]-50)/50 if not np.isnan(rsi[idx]) else 0,
        MACD = (macd[idx]-macd_sig[idx])/(np.std(macd[max(0,idx-20):idx+1])+1e-6) if not np.isnan(macd[idx]) and not np.isnan(macd_sig[idx]) else 0,
        FIB = (spread[idx]-fib['retracements'][2])/(abs(fib['swing_diff'])+1e-6),
        STEAM = steam[idx] if idx<len(steam) else 0,
        ZSCORE = zs[idx] if not np.isnan(zs[idx]) else 0,
    )
    # Apply weights
    WEIGHTS = dict(AMA=0.28, SMA=0.12, EMA=0.15, EMA_SMA_X=0.10, RSI=0.10, MACD=0.10, FIB=0.10, STEAM=0.03, ZSCORE=0.02)
    WEIGHTS = {k:v/sum(WEIGHTS.values()) for k,v in WEIGHTS.items()}
    weighted_bias = sum(signals[k]*WEIGHTS[k] for k in WEIGHTS)
    # Confidence (abs(weighted_bias))
    confidence = min(1.0, abs(weighted_bias))
    # Model probability (from bias)
    prob = 0.5 + 0.4*weighted_bias if weighted_bias>0 else 0.5 + 0.4*weighted_bias
    prob = np.clip(prob, 0.05, 0.95)
    # Market prob
    last_block = blocks[-1]
    market_prob, _ = calculate_no_vig_prob(last_block["spread_vig"], -110)
    # EV/Kelly
    ev = expected_value(prob, last_block["spread_vig"])
    kf = kelly_fraction(prob, last_block["spread_vig"])
    stake = 1000 * kf
    # Projection
    fib_ext = fib['extensions']
    continuation_target = fib_ext[2] if weighted_bias>0 else fib_ext[0]
    reversal_zone = fib['retracements'][0 if weighted_bias<0 else -1]
    # Output
    out = dict(
        normalized_signals=signals,
        weights=WEIGHTS,
        weighted_bias_score=weighted_bias,
        confidence=float(confidence),
        model_prob=float(prob),
        market_prob=float(market_prob),
        expected_value=float(ev),
        kelly_fraction=float(kf),
        recommended_stake=float(stake),
        continuation_target=float(continuation_target),
        reversal_zone=float(reversal_zone),
        swing_debug=dict(
            swing_start=float(fib['swing_start']),
            swing_end=float(fib['swing_end']),
            swing_diff=float(fib['swing_diff']),
            polarity=int(fib['polarity'])
        ),
        fib_levels=fib,
        ama_series=[str(a) if a is not None else "null" for a in ama.tolist()],
        recency_weights=rec_weights.tolist(),
        debug=dict(
            signals=signals,
            weights=WEIGHTS,
            weighted_bias=weighted_bias,
            prob=prob,
            market_prob=market_prob,
            ev=ev,
            kelly_fraction=kf,
            stake=stake,
            projection=dict(continuation_target=continuation_target, reversal_zone=reversal_zone),
        ),
        narrative=dict(
            top_play="Favorite" if weighted_bias>0 else "Underdog",
            ev="{:.2f}%".format(ev*100),
            confidence="{:.1f}%".format(confidence*100),
            stake="${:.2f}".format(stake)
        )
    )
    st.json(out)
    st.write("### Debug Trace")
    st.write(json.dumps(out['debug'], indent=2))
