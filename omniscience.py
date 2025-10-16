import streamlit as st
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta

# ---------- SERIALIZATION UTIL ----------
def safe_json(obj):
    if isinstance(obj, (np.integer, np.int_)): return int(obj)
    if isinstance(obj, (np.floating, np.float_)): return float(obj)
    if isinstance(obj, (np.ndarray,)): return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime)): return str(obj)
    if isinstance(obj, dict): return {k: safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list): return [safe_json(v) for v in obj]
    return obj

# ---------- TA FUNCTIONS ----------
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
    if len(series)<3: return 0, 0
    delta = series[-1]-series[-2]
    prev_delta = series[-2]-series[-3]
    gamma = delta-prev_delta
    return delta, gamma

# ---------- EV & KELLY ----------
def implied_probability(odds):
    if odds == 0: return 0.5
    if odds > 0: return 100/(odds+100)
    else: return abs(odds)/(abs(odds)+100)

def decimal_odds(odds):
    return 1 + (odds/100) if odds > 0 else 1 + (100/abs(odds))

def calculate_no_vig_prob(odds1, odds2):
    imp1 = implied_probability(odds1)
    imp2 = implied_probability(odds2)
    total = imp1 + imp2
    return imp1/total if total else 0.5, imp2/total if total else 0.5

def expected_value(prob, odds):
    dec = decimal_odds(odds)
    return prob*(dec-1) - (1-prob)

def kelly_fraction(prob, odds):
    dec = decimal_odds(odds)
    b = dec-1
    q = 1-prob
    f_star = (b*prob-q)/b if b!=0 else 0
    return max(0, min(f_star, 1)) # [0,1]

# ---------- RECENCY WEIGHTING ----------
def recency_weights(timestamps, decay_minutes=60):
    now = max(timestamps)
    deltas = np.array([(now-t).total_seconds()/60 for t in timestamps])
    weights = np.exp(-deltas/decay_minutes)
    weights = weights / (weights.sum() + 1e-12)
    return weights

# ---------- STRICT PARSER WITH "even" LOGIC FIX ----------
def parse_blocks(raw):
    lines = [l.strip() for l in raw.split('\n') if l.strip()]
    blocks = []
    errors = []
    i = 1 if lines and lines[0].lower().startswith('time') else 0
    while i+4 < len(lines):
        try:
            L1, L2, L3, L4, L5 = lines[i:i+5]
            tks = L1.split()
            ts = datetime.strptime(f"{tks[0]} {tks[1]}", "%m/%d %I:%M%p")
            team = tks[2] if len(tks) > 3 and not tks[2].replace('.','').isdigit() else None
            spread = float(tks[3]) if team else float(tks[2])
            spread = -spread
            spread_vig = float(L2.split()[0])
            total_side = L3[0].lower() if L3[0].lower() in 'ou' else None
            total = float(L3[1:]) if total_side else float(L3)
            total_vig = float(L4.split()[0])
            # Treat "even" as +100 everywhere
            l5mod = L5.replace("even","+100").replace("EVEN","+100")
            ml_tokens = l5mod.replace(",", " ").split()
            ml_nums = []
            for tok in ml_tokens:
                try:
                    ml_nums.append(float(tok))
                except:
                    pass
            ml_away, ml_home = (ml_nums+[None,None])[:2]
            blocks.append(dict(
                time=ts, team=team, spread=spread, spread_vig=spread_vig, total=total, total_side=total_side,
                total_vig=total_vig, ml_away=ml_away, ml_home=ml_home
            ))
        except Exception as e:
            errors.append({"block": lines[i:i+5], "error": str(e)})
        i += 5
    return blocks, errors

# ---------- VOTING AND NARRATIVE ----------
def ta_signals(spread, idx, extra={}):
    sma = SMA(spread, 10)
    ema = EMA(spread, 5)
    rsi = RSI(spread, 7)
    macd, macd_sig, _ = MACD(spread)
    ama = adaptiveMA(spread, 2, 8, 6)
    zs = zscore(spread)
    steam = steam_moves(spread)
    fib = fib_levels(spread)
    delta, gamma = greek_delta(spread)
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
        DELTA = delta/(np.std(spread[max(0,idx-20):idx+1])+1e-6) if idx>=2 else 0,
        GAMMA = gamma/(np.std(spread[max(0,idx-20):idx+1])+1e-6) if idx>=3 else 0
    )
    votes = {k: (1 if v>0.08 else -1 if v<-0.08 else 0) for k,v in signals.items()}
    up = sum(1 for v in votes.values() if v==1)
    down = sum(1 for v in votes.values() if v==-1)
    neutral = sum(1 for v in votes.values() if v==0)
    direction = "Favorite" if up>down else "Underdog" if down>up else "No Clear Edge"
    return signals, votes, direction, up, down, neutral, fib

def weighted_bias(signals):
    WEIGHTS = dict(AMA=0.28, SMA=0.12, EMA=0.15, EMA_SMA_X=0.10, RSI=0.10, MACD=0.10, FIB=0.10, STEAM=0.03, ZSCORE=0.02)
    WEIGHTS = {k:v/sum(WEIGHTS.values()) for k,v in WEIGHTS.items()}
    score = sum(signals[k]*WEIGHTS[k] for k in WEIGHTS)
    return score, WEIGHTS

def build_narrative(direction, votes, up, down, neutral, rec, ev, stake, kelly, confidence):
    rec_icon = "🔥" if abs(ev)>0.04 and confidence>0.7 else "⚠️" if abs(ev)>0.02 else "ℹ️"
    if direction == "Favorite":
        dtext = "The technicals and market bias favor the favorite. Consider a play on the favorite spread."
    elif direction == "Underdog":
        dtext = "Underdog signals gaining strength. Upset or plus-points may be live—consider underdog spread/ML."
    else:
        dtext = "No strong TA consensus. Wait for edge or pass."
    return f"""
{rec_icon} **{direction} — Recommendation**
- Up votes: {up} | Down votes: {down} | Neutral: {neutral}
- Weighted Confidence: {confidence*100:.1f}%
- Expected Value: {ev*100:.2f}%
- Recommended Stake: ${stake:.2f} ({kelly*100:.1f}% Kelly)
- **Narrative:** {dtext}
"""

# ---------- UI ----------
st.set_page_config(layout="wide")
st.markdown("<h1 style='color:#7ee3d0'>Omniscience — Enhanced TA Engine (EV + Kelly + Backtesting)</h1>", unsafe_allow_html=True)
col1, col2 = st.columns([1.2,2])

with col1:
    st.subheader("Paste Odds Feed")
    odds_feed = st.text_area("First line header, then repeated 5-line blocks. Each block:\n1. time [team?] spread\n2. spread vig\n3. total (e.g. o154/u154.5)\n4. total vig\n5. awayML homeML", height=320)
    st.markdown("**Controls:**")
    c1, c2, c3 = st.columns(3)
    analyze = c1.button("Analyze")
    backtest = c2.button("Backtest")
    clear = c3.button("Clear")
    bankroll = st.selectbox("Bankroll", ["$1,000", "$5,000", "$10,000", "Custom..."], index=0)
    custom = None
    if bankroll=="Custom...":
        custom = st.number_input("Enter custom bankroll", min_value=100.0, value=1000.0, step=50.0)
    br_val = float(bankroll.replace("$","").replace(",","")) if bankroll!="Custom..." else (custom if custom else 1000)
    debug = st.checkbox("Show Debug Output", value=False)
    st.markdown("Malformed blocks are reported in the analysis.")
    # Parsed preview
    if odds_feed.strip():
        blocks, errors = parse_blocks(odds_feed)
        # Only display fully valid blocks
        required_fields = {"time", "spread", "spread_vig", "total", "total_vig", "ml_away", "ml_home"}
        valid_blocks = [b for b in blocks if all(f in b and b[f] is not None for f in required_fields)]
        if valid_blocks:
            try:
                dfprev = pd.DataFrame(valid_blocks)
                st.markdown("**Parsed Preview (chronological)**")
                st.dataframe(dfprev, hide_index=True)
            except Exception as e:
                st.warning(f"Could not display preview table: {e}")
        else:
            st.warning("No valid blocks parsed.")
    else:
        blocks, errors = [], []

with col2:
    if clear:
        st.experimental_rerun()
    if analyze and blocks:
        df = pd.DataFrame([b for b in blocks if all(k in b and b[k] is not None for k in ("time","spread","spread_vig","total","total_vig","ml_away","ml_home"))])
        df = df.sort_values("time").reset_index(drop=True)
        rec_weights = recency_weights(df["time"].tolist())
        spread = df["spread"].values
        idx = len(spread)-1
        signals, votes, direction, up, down, neutral, fib = ta_signals(spread, idx)
        w_bias, WEIGHTS = weighted_bias(signals)
        confidence = min(1.0, abs(w_bias))
        prob = 0.5 + 0.4*w_bias if w_bias>0 else 0.5 + 0.4*w_bias
        prob = np.clip(prob, 0.05, 0.95)
        last_block = df.iloc[-1].to_dict()
        market_prob, _ = calculate_no_vig_prob(last_block["spread_vig"], -110)
        ev = expected_value(prob, last_block["spread_vig"])
        kf = kelly_fraction(prob, last_block["spread_vig"])
        stake = br_val * kf
        fib_ext = fib['extensions']
        continuation_target = fib_ext[2] if w_bias>0 else fib_ext[0]
        reversal_zone = fib['retracements'][0 if w_bias<0 else -1]
        # Voting window
        st.markdown(f"### Top Play & Recommendation")
        st.markdown(build_narrative(direction, votes, up, down, neutral, rec_weights, ev, stake, kf, confidence))
        st.markdown("### TA Indicator Table (numeric signals, weights, votes)")
        ta_table = pd.DataFrame([
            {
                "Signal": k,
                "Value": float(signals[k]),
                "Weight": float(WEIGHTS.get(k, 0)),
                "Vote": int(votes[k])
            }
            for k in signals
        ])
        st.dataframe(ta_table, hide_index=True)
        st.markdown(f"**Trend Projection:** Continuation target: `{continuation_target:.2f}` | Reversal zone: `{reversal_zone:.2f}`")
        st.markdown(f"**EV:** `{ev*100:.2f}%` | **Kelly Fraction:** `{kf*100:.1f}%` | **Recommended Stake:** `${stake:.2f}` (Bankroll ${br_val:.2f})")
        if debug:
            st.markdown("### Debug Output")
            debugdict = dict(
                signals=safe_json(signals),
                votes=safe_json(votes),
                weights=safe_json(WEIGHTS),
                weighted_bias=w_bias,
                confidence=confidence,
                model_prob=prob,
                market_prob=market_prob,
                expected_value=ev,
                kelly_fraction=kf,
                stake=stake,
                projection=dict(continuation_target=continuation_target, reversal_zone=reversal_zone),
                swing_debug={k: float(fib[k]) if isinstance(fib[k], (int,float)) else fib[k] for k in ['swing_start','swing_end','swing_diff','polarity']}
            )
            st.code(json.dumps(safe_json(debugdict), indent=2))
        st.markdown("### Exportable JSON Output")
        out = dict(
            normalized_signals=safe_json(signals),
            votes=safe_json(votes),
            weights=safe_json(WEIGHTS),
            weighted_bias_score=w_bias,
            confidence=float(confidence),
            model_prob=float(prob),
            market_prob=float(market_prob),
            expected_value=float(ev),
            kelly_fraction=float(kf),
            recommended_stake=float(stake),
            continuation_target=float(continuation_target),
            reversal_zone=float(reversal_zone),
            swing_debug={k: float(fib[k]) if isinstance(fib[k], (int,float)) else fib[k] for k in ['swing_start','swing_end','swing_diff','polarity']},
            fib_levels=safe_json(fib),
            recency_weights=safe_json(rec_weights),
            top_play=direction,
            narrative=build_narrative(direction, votes, up, down, neutral, rec_weights, ev, stake, kf, confidence)
        )
        st.code(json.dumps(safe_json(out), indent=2))
        if errors:
            st.markdown("### Parser Warnings (malformed blocks):")
            st.code(json.dumps(errors[:5], indent=2))
    elif backtest and blocks:
        df = pd.DataFrame([b for b in blocks if all(k in b and b[k] is not None for k in ("time","spread","spread_vig","total","total_vig","ml_away","ml_home"))])
        df = df.sort_values("time").reset_index(drop=True)
        br = br_val
        br_hist = [br]
        bet_hist = []
        for i in range(2, len(df)-1):
            spread = df["spread"].values[:i+1]
            idx = len(spread)-1
            signals, votes, direction, up, down, neutral, fib = ta_signals(spread, idx)
            w_bias, WEIGHTS = weighted_bias(signals)
            prob = 0.5 + 0.4*w_bias if w_bias>0 else 0.5 + 0.4*w_bias
            prob = np.clip(prob, 0.05, 0.95)
            odds = df.loc[i,"spread_vig"]
            kf = kelly_fraction(prob, odds)
            stake = br * kf
            outcome = (df.loc[i+1,"spread"] < df.loc[i,"spread"]) if direction=="Favorite" else (df.loc[i+1,"spread"] > df.loc[i,"spread"])
            if outcome:
                br += stake * 0.91
                res = "Win"
            else:
                br -= stake
                res = "Loss"
            br_hist.append(br)
            bet_hist.append(dict(time=str(df.loc[i,"time"]), direction=direction, outcome=res, bankroll=br))
        st.markdown("### Backtest Results")
        st.markdown(f"- Starting bankroll: ${br_val:.2f}\n- Ending: ${br:.2f}\n- Profit: ${br-br_val:.2f}\n- # Bets: {len(bet_hist)}")
        st.line_chart(br_hist)
        st.dataframe(pd.DataFrame(bet_hist), hide_index=True)
    elif not blocks and (analyze or backtest):
        st.warning("No valid blocks parsed. Check your feed formatting.")
