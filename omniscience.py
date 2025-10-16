import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ---------------- Parsing --------------------
def safe_json(obj):
    if isinstance(obj, (np.integer, np.int_)): return int(obj)
    if isinstance(obj, (np.floating, np.float_)): return float(obj)
    if isinstance(obj, (np.ndarray,)): return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime)): return str(obj)
    if isinstance(obj, dict): return {k: safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list): return [safe_json(v) for v in obj]
    return obj

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
        except Exception:
            pass  # skip bad block
        i += 5
    return blocks

# ---------------- Indicators ------------------
def EMA(series, n):
    try: return pd.Series(series).ewm(span=max(1,n), adjust=False).mean().values
    except: return np.full_like(series, np.nan, dtype=np.float64)

def SMA(series, n):
    try: return pd.Series(series).rolling(max(1,n)).mean().values
    except: return np.full_like(series, np.nan, dtype=np.float64)

def MACD(series, fast=3, slow=5, signal=2):
    try:
        ema_fast = EMA(series, fast)
        ema_slow = EMA(series, slow)
        macd = ema_fast - ema_slow
        macd_signal = pd.Series(macd).ewm(span=max(1,signal), adjust=False).mean().values
        return macd, macd_signal
    except: return np.full_like(series, 0), np.full_like(series, 0)

def RSI(series, n=3):
    try:
        series = pd.Series(series)
        delta = series.diff()
        up, down = delta.clip(lower=0), -delta.clip(upper=0)
        roll_up = up.rolling(max(1,n)).mean()
        roll_down = down.rolling(max(1,n)).mean()
        rs = roll_up / (roll_down + 1e-12)
        return (100 - (100 / (1 + rs))).values
    except: return np.full_like(series, 0, dtype=np.float64)

def BB(series, n=3, mult=2):
    sma = SMA(series, n)
    std = pd.Series(series).rolling(max(1,n)).std().values
    upper = sma + mult*std
    lower = sma - mult*std
    return upper, lower

def zscore(series, n=3):
    try:
        ser = pd.Series(series)
        return ((ser - ser.rolling(max(1,n)).mean()) / (ser.rolling(max(1,n)).std() + 1e-12)).values
    except: return np.full_like(series, 0, dtype=np.float64)

def ROC(series, n=2):
    arr = np.array(series)
    out = np.full_like(arr, np.nan, dtype=np.float64)
    try:
        for i in range(n, len(arr)):
            if arr[i-n] != 0:
                out[i] = 100*(arr[i]-arr[i-n])/abs(arr[i-n])
        return out
    except: return np.full_like(series, 0, dtype=np.float64)

def adaptiveMA(series, fast=2, slow=4, efficiency_lookback=2):
    n = len(series)
    ama = [np.nan]*n
    try:
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
    except: return np.full_like(series, 0, dtype=np.float64)

def MOM(series, n=2):
    arr = np.array(series)
    out = np.full_like(arr, np.nan, dtype=np.float64)
    try:
        for i in range(n, len(arr)):
            out[i] = arr[i] - arr[i-n]
        return out
    except: return np.full_like(series, 0, dtype=np.float64)

def VOL(series, n=3):
    try: return pd.Series(series).rolling(max(1,n)).std().values
    except: return np.full_like(series, 0, dtype=np.float64)

# ----- EV/Kelly/Projection -----
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
        retracements=retracements, extensions=extensions
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

def regime_adaptive_weights(vols, base_weights):
    weights = base_weights.copy()
    if vols is not None and len(vols) >= 5 and np.nanmedian(vols) > 0:
        if np.nanmean(vols[-5:]) > 2*np.nanmedian(vols):
            weights['VOL'] *= 1.6
            weights['ZSCORE'] *= 1.4
            weights['ADAPTIVEMA'] *= 0.8
    return weights

# ------------- Market Analysis --------------
def analyze_market_god(df, market, recency_half, ama_fast, ama_slow, ama_eff, base_weights):
    col = market
    odds_col = 'spread_vig' if market == 'spread' else 'total_vig' if market == 'total' else market
    if col not in df: return None, "No data"
    s = df[col].dropna().values
    odds = df[odds_col].dropna().values if odds_col in df else np.full_like(s, 100)
    times = df['time'].values[-len(s):] if len(df['time']) >= len(s) else df['time'].values
    if len(s) < 2: return None, "No data"
    rec_weights = np.ones_like(s, dtype=np.float64)  # can use recency_weights(times, recency_half)
    out = []
    vols = []
    for i in range(len(s)):
        L = min(4, i+1)
        try: ama = adaptiveMA(s[:i+1], ama_fast, ama_slow, ama_eff)[-1]
        except: ama = 0
        try: ema = EMA(s[:i+1], L)[-1]
        except: ema = 0
        try: sma = SMA(s[:i+1], L)[-1]
        except: sma = 0
        try:
            macd, macd_signal = MACD(s[:i+1], fast=2, slow=L, signal=2)
            macd_val = macd[-1] - macd_signal[-1]
        except: macd_val = 0
        try:
            rsi = RSI(s[:i+1], L)
            rsi_val = (rsi[-1]-50)/50 if not np.isnan(rsi[-1]) else 0
        except: rsi_val = 0
        try:
            bb_upper, bb_lower = BB(s[:i+1], L)
        except: bb_upper, bb_lower = [sma],[sma]
        try:
            zs = zscore(s[:i+1], L)
            z_val = zs[-1] if not np.isnan(zs[-1]) else 0
        except: z_val = 0
        try:
            roc = ROC(s[:i+1], 2)
            roc_val = roc[-1]/100 if not np.isnan(roc[-1]) else 0
        except: roc_val = 0
        try:
            steam = int(abs(s[i]-np.mean(s[max(0,i-L):i+1])) > 2*np.std(s[max(0,i-L):i+1]))
        except: steam = 0
        try:
            mom = MOM(s[:i+1], 2)[-1] if i+1>=2 else 0
        except: mom = 0
        try:
            vol = VOL(s[:i+1], 2)[-1] if i+1>=2 else 0
        except: vol = 0
        vols.append(vol)
        try:
            fib = fib_levels(s[:i+1], L)
            proj, retrace, narrative = project_line(s[:i+1], fib, ama if not np.isnan(ama) else 0)
        except:
            proj, retrace, narrative = (None, None, "Projection unavailable")
        try:
            ev = expected_value(0.5, odds[i]) if i<len(odds) else 0
        except: ev = 0
        try:
            kelly = kelly_fraction(0.5, odds[i]) if i<len(odds) else 0
        except: kelly = 0
        out.append(dict(
            time=times[i], line=s[i], AMA=ama, EMA=ema, SMA=sma, MACD=macd_val, RSI=rsi_val,
            MOM=mom, VOL=vol, EV=ev, Kelly=kelly, ZSCORE=z_val, ROC=roc_val, STEAM=steam, Score=0,
            Project=proj, Retrace=retrace, Narrative=narrative
        ))
    weights = regime_adaptive_weights(vols, base_weights)
    for row in out:
        score = (weights['ADAPTIVEMA']*(row['AMA'] if not np.isnan(row['AMA']) else 0)
                + weights['EV']*(row['EV'] if not np.isnan(row['EV']) else 0)
                + weights['MOM']*(row['MOM'] if not np.isnan(row['MOM']) else 0)
                + weights['VOL']*(row['VOL'] if not np.isnan(row['VOL']) else 0)
                + weights['EMA']*(row['EMA']-row['line'])
                + weights['SMA']*(row['SMA']-row['line'])
                + weights['MACD']*row['MACD']
                + weights['RSI']*row['RSI']
                + weights['BB']*((row['line']-row['EMA'])/((abs(row['VOL'])+1e-6)))
                + weights['ZSCORE']*row['ZSCORE']
                + weights['ROC']*row['ROC']
                + weights['STEAM']*row['STEAM'])
        row['Score'] = score
    df_out = pd.DataFrame(out)
    return df_out, "OK"

# ------------- UI --------------
MARKETS = ['spread', 'total', 'ml_home', 'ml_away']
MARKET_LABELS = {'spread': 'Spread', 'total': 'Total', 'ml_home': 'Home ML', 'ml_away': 'Away ML'}
BASE_WEIGHTS = dict(
    ADAPTIVEMA=0.27, EV=0.25, MOM=0.10, VOL=0.09, EMA=0.05, SMA=0.05,
    MACD=0.05, RSI=0.04, BB=0.03, ZSCORE=0.03, ROC=0.01, FIB=0.01, STEAM=0.02, BREAKOUT=0.01
)

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
    tabs = st.tabs([MARKET_LABELS[m] for m in MARKETS])
    for idx, market in enumerate(MARKETS):
        label = MARKET_LABELS[market]
        with tabs[idx]:
            df_out, status = analyze_market_god(df, market, recency_half, ama_fast, ama_slow, ama_eff, BASE_WEIGHTS)
            if df_out is None or status != "OK" or df_out.empty:
                st.warning(f"No available data for {label} market.")
                continue
            last = df_out.iloc[-1]
            color = "#19ad19" if last['Score'] > 0.12 else "#e05959" if last['Score'] < -0.12 else "#888"
            st.markdown(
                f"<div style='background-color:{color};padding:10px;border-radius:6px'>"
                f"<b>{label} Rec:</b> {last['Rec']} &nbsp;&nbsp;|&nbsp;"
                f"<b>Score:</b> {last['Score']:.3f} &nbsp;|&nbsp;"
                f"<b>AMA:</b> {last['AMA']:.3f} &nbsp;|&nbsp;"
                f"<b>EV:</b> {last['EV']:.2%} &nbsp;|&nbsp;"
                f"<b>Kelly:</b> {last['Kelly']:.2%}<br>"
                f"{last['Narrative']}</div>", unsafe_allow_html=True)
            st.metric("Model Score", value=last["Score"], delta=None)
            st.metric("Kelly", value=f"{last['Kelly']:.2%}")
            st.metric("EV", value=f"{last['EV']:.2%}")
            st.line_chart(df_out.set_index('time')[['line','AMA','EMA','SMA','Score']].tail(20))
            st.line_chart(df_out.set_index('time')[['MOM','VOL','RSI','MACD','ZSCORE']].tail(20))
            st.dataframe(df_out.tail(10))
            st.json(safe_json(df_out.tail(1).to_dict()))
if backtest and blocks:
    df = pd.DataFrame(blocks)
    df = df.sort_values("time").reset_index(drop=True)
    df_out, status = analyze_market_god(df, "spread", recency_half, ama_fast, ama_slow, ama_eff, BASE_WEIGHTS)
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
