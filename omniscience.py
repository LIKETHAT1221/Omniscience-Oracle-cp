import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ----- MARKETS -----
MARKETS = ['spread', 'total', 'ml_home', 'ml_away']
MARKET_LABELS = {'spread': 'Spread', 'total': 'Total', 'ml_home': 'Home ML', 'ml_away': 'Away ML'}

# ----- BASE WEIGHTS (anchors: AMA, EV) -----
BASE_WEIGHTS = dict(
    ADAPTIVEMA=0.27, EV=0.25, MOM=0.10, VOL=0.09, EMA=0.05, SMA=0.05,
    MACD=0.05, RSI=0.04, BB=0.03, ZSCORE=0.03, ROC=0.01, FIB=0.01, STEAM=0.02, BREAKOUT=0.01
)

# ----- SAFE JSON -----
def safe_json(obj):
    if isinstance(obj, (np.integer, np.int_)): return int(obj)
    if isinstance(obj, (np.floating, np.float_)): return float(obj)
    if isinstance(obj, (np.ndarray,)): return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime)): return str(obj)
    if isinstance(obj, dict): return {k: safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list): return [safe_json(v) for v in obj]
    return obj

# ----- RECENCY WEIGHTS -----
def recency_weights(times, half_life_hrs):
    now = max(times)
    deltas = np.array([(now-t).total_seconds()/3600 for t in times])
    decay = np.log(2)/half_life_hrs
    weights = np.exp(-decay * deltas)
    weights = weights / (weights.sum() + 1e-12)
    return weights

# ----- PARSER -----
def parse_blocks(raw):
    lines = [l.strip() for l in raw.split('\n') if l.strip()]
    blocks = []
    i = 1 if lines and lines[0].lower().startswith('time') else 0
    while i+4 < len(lines):
        try:
            L1, L2, L3, L4, L5 = lines[i:i+5]
            tks = L1.split()
            ts = datetime.strptime(f"{tks[0]} {tks[1]}", "%m/%d %I:%M%p")
            spread = -float(tks[-1])
            spread_vig = float(L2.split()[0])
            total = float(L3[1:]) if L3[0].lower() in 'ou' else float(L3)
            total_vig = float(L4.split()[0])
            l5mod = L5.replace("even","+100").replace("EVEN","+100")
            ml_tokens = l5mod.replace(",", " ").split()
            ml_away, ml_home = (np.nan, np.nan)
            if len(ml_tokens) >= 2:
                ml_away, ml_home = float(ml_tokens[0]), float(ml_tokens[1])
            blocks.append(dict(
                time=ts, spread=spread, spread_vig=spread_vig, total=total, total_vig=total_vig,
                ml_away=ml_away, ml_home=ml_home
            ))
        except Exception: pass
        i += 5
    return blocks

# ----- INDICATORS -----
def EMA(series, n): return pd.Series(series).ewm(span=n, adjust=False).mean().values
def SMA(series, n): return pd.Series(series).rolling(n).mean().values
def MACD(series, fast=3, slow=5, signal=2):
    ema_fast = EMA(series, fast)
    ema_slow = EMA(series, slow)
    macd = ema_fast - ema_slow
    macd_signal = pd.Series(macd).ewm(span=signal,adjust=False).mean().values
    return macd, macd_signal
def RSI(series, n=3):
    series = pd.Series(series)
    delta = series.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    roll_up = up.rolling(n).mean()
    roll_down = down.rolling(n).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))
def BB(series, n=3, mult=2):
    sma = SMA(series, n)
    std = pd.Series(series).rolling(n).std().values
    upper = sma + mult*std
    lower = sma - mult*std
    return upper, lower
def zscore(series, n=3):
    ser = pd.Series(series)
    return ((ser - ser.rolling(n).mean()) / (ser.rolling(n).std() + 1e-12)).values
def ROC(series, n=2):
    arr = np.array(series)
    out = np.full_like(arr, np.nan, dtype=np.float64)
    for i in range(n, len(arr)):
        out[i] = 100*(arr[i]-arr[i-n])/abs(arr[i-n])
    return out
def adaptiveMA(series, fast=2, slow=4, efficiency_lookback=2):
    n = len(series)
    ama = [np.nan]*n
    for i in range(slow+efficiency_lookback-1, n):
        change = abs(series[i] - series[i-efficiency_lookback])
        volatility = np.sum(np.abs(np.diff(series[i-efficiency_lookback:i+1])))
        ER = 0 if volatility == 0 else change / (volatility + 1e-12)
        ER = np.clip(ER / (np.std(series[max(0,i-10):i+1])+1e-5), 0, 1)
        fastSC = 2/(fast+1)
        slowSC = 2/(slow+1)
        SC = (ER * (fastSC-slowSC) + slowSC)**2
        ama[i] = series[i-1] if np.isnan(ama[i-1]) else ama[i-1] + SC*(series[i]-ama[i-1])
    return np.array(ama)
def MOM(series, n=2):
    arr = np.array(series)
    out = np.full_like(arr, np.nan, dtype=np.float64)
    for i in range(n, len(arr)):
        out[i] = arr[i] - arr[i-n]
    return out
def VOL(series, n=3):
    return pd.Series(series).rolling(n).std().values

# ----- EV & KELLY -----
def implied_probability(odds):
    if odds == 0: return 0.5
    if odds > 0: return 100/(odds+100)
    else: return abs(odds)/(abs(odds)+100)
def decimal_odds(odds):
    return 1 + (odds/100) if odds > 0 else 1 + (100/abs(odds))
def expected_value(prob, odds):
    dec = decimal_odds(odds)
    return prob*(dec-1) - (1-prob)
def kelly_fraction(prob, odds):
    dec = decimal_odds(odds)
    b = dec-1
    q = 1-prob
    f_star = (b*prob-q)/b if b!=0 else 0
    return max(0, min(f_star, 1))

# ----- PROJECTION -----
def fib_levels(series, window=3):
    if len(series) < window: return None
    swing_start = series[-window]
    swing_end = series[-1]
    swing_diff = swing_end - swing_start
    polarity = np.sign(swing_diff)
    high, low = (swing_end, swing_start) if polarity>0 else (swing_start, swing_end)
    retracements = [high - abs(swing_diff)*r for r in [0.236,0.382,0.5,0.618,0.786]]
    extensions = [high + abs(swing_diff)*e for e in [0.236,0.382,0.5,0.618,1.0]]
    return dict(
        swing_start=swing_start, swing_end=swing_end, swing_diff=swing_diff, polarity=polarity,
        retracements=retracements, extensions=extensions, debug={"high":high,"low":low}
    )
def project_line(series, fib, score):
    if fib is None: return None, None, "Not enough data for projection"
    if score > 0:
        proj = fib['extensions'][2]
        retrace = fib['retracements'][1]
        narrative = f"Projected continuation {proj:.2f}, retrace risk {retrace:.2f}."
    elif score < 0:
        proj = fib['extensions'][0]
        retrace = fib['retracements'][3]
        narrative = f"Projected continuation {proj:.2f}, retrace risk {retrace:.2f}."
    else:
        proj = fib['retracements'][2]
        retrace = fib['retracements'][2]
        narrative = "No strong trend; likely range-bound."
    return proj, retrace, narrative

# ----- FULL POWER ANALYSIS -----
def analyze_market(df, market, recency_half, ama_fast, ama_slow, ama_eff):
    col = market
    odds_col = 'spread_vig' if market == 'spread' else 'total_vig' if market == 'total' else market
    # Use only available (non-NaN) data for each market
    s = df[col].dropna().values
    odds = df[odds_col].dropna().values
    times = df['time'].values[-len(s):] if len(df['time']) >= len(s) else df['time'].values
    if len(s) < 2: return None, "No data"
    rec_weights = recency_weights(times, recency_half)
    out = []
    for i in range(len(s)):
        L = min(4, i+1)
        ama = adaptiveMA(s[:i+1], ama_fast, ama_slow, ama_eff)[-1] if i+1 >= ama_slow+ama_eff else np.nan
        ema = EMA(s[:i+1], L)[-1]
        sma = SMA(s[:i+1], L)[-1]
        macd, macd_signal = MACD(s[:i+1], fast=2, slow=L, signal=2)
        macd_val = macd[-1] - macd_signal[-1]
        rsi = RSI(s[:i+1], L)
        rsi_val = (rsi.iloc[-1]-50)/50 if not np.isnan(rsi.iloc[-1]) else 0
        bb_upper, bb_lower = BB(s[:i+1], L)
        zs = zscore(s[:i+1], L)
        roc = ROC(s[:i+1], 2)
        steam = int(abs(s[i]-np.mean(s[max(0,i-L):i+1])) > 2*np.std(s[max(0,i-L):i+1]))
        mom = MOM(s[:i+1], 2)[-1] if i+1>=2 else 0
        vol = VOL(s[:i+1], 2)[-1] if i+1>=2 else 0
        fib = fib_levels(s[:i+1], L)
        proj, retrace, narrative = project_line(s[:i+1], fib, ama if not np.isnan(ama) else 0)
        # Score (anchors: AMA and EV)
        ev = expected_value(0.5, odds[i]) if i<len(odds) else np.nan
        kelly = kelly_fraction(0.5, odds[i]) if i<len(odds) else np.nan
        score = 0.27*(ama if not np.isnan(ama) else 0) + 0.25*(ev if not np.isnan(ev) else 0)
        score += 0.10*(mom if not np.isnan(mom) else 0)
        score += 0.09*(vol if not np.isnan(vol) else 0)
        score += 0.05*(ema-s[i]) + 0.05*(sma-s[i])
        score += 0.05*macd_val + 0.04*rsi_val + 0.03*((s[i]-bb_upper[-1])/(np.std(s[max(0,i-L):i+1])+1e-6))
        score += 0.03*(zs[-1] if not np.isnan(zs[-1]) else 0) + 0.01*(roc[-1]/100 if not np.isnan(roc[-1]) else 0)
        score += 0.01*((s[i]-np.median(s[max(0,i-L):i+1]))/(np.std(s[max(0,i-L):i+1])+1e-6) if np.std(s[max(0,i-L):i+1])>0 else 0)
        score += 0.02*steam
        rec = "Bet Up" if score > 0.12 else "Bet Down" if score < -0.12 else "Hold"
        out.append(dict(
            time=times[i], line=s[i], AMA=ama, EMA=ema, SMA=sma, MACD=macd_val, RSI=rsi_val,
            MOM=mom, VOL=vol, EV=ev, Kelly=kelly, Score=score, Rec=rec,
            Project=proj, Retrace=retrace, Narrative=narrative
        ))
    df_out = pd.DataFrame(out)
    return df_out, "OK"

# ----- UI -----
st.set_page_config(layout="wide")
with st.sidebar:
    st.header("God Machine Controls")
    ama_fast = st.slider("AMA Fast", 1, 8, 2)
    ama_slow = st.slider("AMA Slow", 2, 8, 4)
    ama_eff = st.slider("AMA Eff", 1, 4, 2)
    recency_half = st.slider("Recency Half-life (hrs)", 6, 48, 18)
    st.markdown("**Backtest and Clear**")
    backtest = st.button("Backtest")
    clear = st.button("Clear")

if 'odds_feed' not in st.session_state: st.session_state['odds_feed'] = ''
if 'analysis_triggered' not in st.session_state: st.session_state['analysis_triggered'] = False

def clear_all():
    st.session_state['odds_feed'] = ''
    st.session_state['analysis_triggered'] = False

st.subheader("Paste Odds Feed")
st.text_area("Paste odds", key='odds_feed', height=320)
c1, c2 = st.columns(2)
if c1.button("Analyze"):
    st.session_state['analysis_triggered'] = True
if clear:
    clear_all()
    st.experimental_rerun()

blocks = parse_blocks(st.session_state['odds_feed'])
if blocks:
    df = pd.DataFrame(blocks)
    st.dataframe(df.tail(10))
else:
    st.info("Paste odds feed above for analysis.")

if st.session_state['analysis_triggered'] and blocks:
    df = pd.DataFrame(blocks)
    df = df.sort_values("time").reset_index(drop=True)
    for market in MARKETS:
        label = MARKET_LABELS[market]
        df_out, status = analyze_market(df, market, recency_half, ama_fast, ama_slow, ama_eff)
        if df_out is None or status != "OK" or df_out.empty:
            st.warning(f"No available data for {label} market.")
            continue
        st.markdown(f"### {label} Analysis")
        st.dataframe(df_out.tail(10))
        st.line_chart(df_out.set_index('time')[['line','AMA','EMA','SMA','Score']].tail(20))
        last = df_out.iloc[-1]
        st.markdown(
            f"**{label} Rec:** {last['Rec']} | "
            f"Score: {last['Score']:.3f} | "
            f"AMA: {last['AMA']:.3f} | "
            f"EV: {last['EV']:.2%} | "
            f"Kelly: {last['Kelly']:.2%} | "
            f"{last['Narrative']}"
        )

if backtest and blocks:
    df = pd.DataFrame(blocks)
    df = df.sort_values("time").reset_index(drop=True)
    df_out, status = analyze_market(df, "spread", recency_half, ama_fast, ama_slow, ama_eff)
    if df_out is None or status != "OK" or df_out.empty:
        st.warning("No available data for spread market for backtest.")
    else:
        st.header("Backtest Results (Spread, Kelly staking)")
        bankroll = 1000
        br_hist = [bankroll]
        for i in range(len(df_out)):
            if df_out['Rec'].iloc[i] != "Hold" and df_out['EV'].iloc[i]>0:
                stake = bankroll * df_out['Kelly'].iloc[i]
                win = np.random.choice([1,0])
                if win: bankroll += stake*0.91
                else: bankroll -= stake
            br_hist.append(bankroll)
        st.line_chart(br_hist)
        st.write(f"Final bankroll: {bankroll:.2f}")
