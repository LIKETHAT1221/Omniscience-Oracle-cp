import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# --------- BASE WEIGHTS ---------
BASE_WEIGHTS = {
    'ADAPTIVEMA': 0.20, 'EV': 0.18, 'MOM': 0.10, 'VOL': 0.08, 'EMA': 0.06, 'SMA': 0.05,
    'MACD': 0.07, 'RSI': 0.06, 'BB': 0.05, 'ZSCORE': 0.05, 'ROC': 0.02, 'FIB': 0.02,
    'STEAM': 0.03, 'BREAKOUT': 0.03
}
MARKETS = ['spread', 'total', 'ml_home', 'ml_away']
MARKET_LABELS = {'spread': 'Spread', 'total': 'Total', 'ml_home': 'Home ML', 'ml_away': 'Away ML'}

# ----------- UTILS ---------------
def safe_json(obj):
    if isinstance(obj, (np.integer, np.int_)): return int(obj)
    if isinstance(obj, (np.floating, np.float_)): return float(obj)
    if isinstance(obj, (np.ndarray,)): return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime)): return str(obj)
    if isinstance(obj, dict): return {k: safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list): return [safe_json(v) for v in obj]
    return obj

# ----------- PARSING -------------
def parse_blocks(raw):
    lines = [l.strip() for l in raw.split('\n') if l.strip()]
    blocks, errors = [], []
    i = 1 if lines and lines[0].lower().startswith('time') else 0
    while i+4 < len(lines):
        try:
            L1, L2, L3, L4, L5 = lines[i:i+5]
            tks = L1.split()
            try:
                ts = datetime.strptime(f"{tks[0]} {tks[1]}", "%m/%d %I:%M%p")
            except Exception as e:
                errors.append({"block": lines[i:i+5], "error": str(e)})
                i += 5
                continue
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
        except Exception as e:
            errors.append({"block": lines[i:i+5], "error": str(e)})
        i += 5
    return blocks, errors

# ------ INDICATORS ------
def EMA(series, n):
    try:
        return pd.Series(series).ewm(span=max(1,n), adjust=False).mean().values
    except Exception as e:
        return np.full_like(series, np.nan)

def SMA(series, n):
    try:
        return pd.Series(series).rolling(max(1,n)).mean().values
    except Exception as e:
        return np.full_like(series, np.nan)

def MACD(series, fast=3, slow=5, signal=2):
    try:
        ema_fast = EMA(series, fast)
        ema_slow = EMA(series, slow)
        macd = ema_fast - ema_slow
        macd_signal = pd.Series(macd).ewm(span=max(1,signal), adjust=False).mean().values
        return macd, macd_signal
    except Exception as e:
        return np.full_like(series, 0), np.full_like(series, 0)

def RSI(series, n=3):
    try:
        series = pd.Series(series)
        delta = series.diff()
        up, down = delta.clip(lower=0), -delta.clip(upper=0)
        roll_up = up.rolling(max(1,n)).mean()
        roll_down = down.rolling(max(1,n)).mean()
        rs = roll_up / (roll_down + 1e-12)
        return (100 - (100 / (1 + rs))).values
    except Exception as e:
        return np.full_like(series, 0)

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
    except Exception as e:
        return np.full_like(series, 0)

def ROC(series, n=2):
    arr = np.array(series)
    out = np.full_like(arr, np.nan)
    try:
        for i in range(n, len(arr)):
            if arr[i-n] != 0:
                out[i] = 100*(arr[i]-arr[i-n])/abs(arr[i-n])
        return out
    except Exception as e:
        return np.full_like(series, 0)

def MOM(series, n=2):
    arr = np.array(series)
    out = np.full_like(arr, np.nan)
    try:
        for i in range(n, len(arr)):
            out[i] = arr[i] - arr[i-n]
        return out
    except Exception as e:
        return np.full_like(series, 0)

def VOL(series, n=3):
    try:
        return pd.Series(series).rolling(max(1,n)).std().values
    except Exception as e:
        return np.full_like(series, 0)

def adaptiveMA(price, fast=2, slow=8, efficiency_lookback=4):
    n = len(price)
    ama = [np.nan]*n
    for i in range(slow+efficiency_lookback-1, n):
        try:
            ER = abs(price[i] - price[i-efficiency_lookback])/sum(abs(np.diff(price[i-efficiency_lookback:i+1])))
            fastSC = 2/(fast+1)
            slowSC = 2/(slow+1)
            SC = (ER*(fastSC-slowSC) + slowSC)**2
            ama[i] = price[i-1] if np.isnan(ama[i-1]) else ama[i-1] + SC*(price[i]-ama[i-1])
        except Exception as e:
            ama[i] = np.nan
    return np.array(ama)

def crossover(arr1, arr2):
    prev = np.sign(np.array(arr1)-np.array(arr2))
    out = np.zeros_like(prev)
    for i in range(1,len(prev)):
        if prev[i-1]<0 and prev[i]>0: out[i]=1
        elif prev[i-1]>0 and prev[i]<0: out[i]=-1
    return out

# --- FIBONACCI ---
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

# --- RECENCY WEIGHTS ---
def recency_weights(times, half_life_hrs=18):
    deltas = [(max(times)-t).total_seconds()/3600 for t in times]
    decay = np.log(2)/half_life_hrs
    weights = np.exp(-decay*np.array(deltas))
    return weights / (weights.sum() + 1e-12)

# --- EV & KELLY ---
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

# --- PROJECTION ---
def project_line(series, fib_obj, score):
    if fib_obj is None: return None, None, "Not enough data for projection"
    retracements = fib_obj['retracements']
    extensions = fib_obj['extensions']
    if score > 0:
        proj = extensions[2]
        retrace = retracements[1]
        narrative = f"Projected continuation {proj:.2f}, retrace risk {retrace:.2f}."
    elif score < 0:
        proj = extensions[0]
        retrace = retracements[3]
        narrative = f"Projected continuation {proj:.2f}, retrace risk {retrace:.2f}."
    else:
        proj = retracements[2]
        retrace = retracements[2]
        narrative = "No strong trend; likely range-bound."
    return proj, retrace, narrative

# --- MAIN ANALYSIS ---
def analyze_market_god(df, market, recency_half, ama_fast, ama_slow, ama_eff, base_weights):
    col = market
    odds_col = 'spread_vig' if market == 'spread' else 'total_vig' if market == 'total' else market
    if col not in df: return None, "No data"
    s = df[col].dropna().values
    odds = df[odds_col].dropna().values if odds_col in df else np.full_like(s, 100)
    times = df['time'].values[-len(s):] if len(df['time']) >= len(s) else df['time'].values
    if len(s) < 2: return None, "No data"
    rec_weights = recency_weights(times, recency_half)
    out = []
    vols = []
    for i in range(len(s)):
        L = min(4, i+1)
        ama = adaptiveMA(s[:i+1], ama_fast, ama_slow, ama_eff)[-1] if i+1>=ama_slow+ama_eff else np.nan
        ema = EMA(s[:i+1], L)[-1]
        sma = SMA(s[:i+1], L)[-1]
        macd, macd_signal = MACD(s[:i+1], fast=2, slow=L, signal=2)
        macd_val = macd[-1] - macd_signal[-1]
        rsi = RSI(s[:i+1], L)
        rsi_val = (rsi[-1]-50)/50 if not np.isnan(rsi[-1]) else 0
        bb_upper, bb_lower = BB(s[:i+1], L)
        zs = zscore(s[:i+1], L)
        roc = ROC(s[:i+1], 2)
        steam = int(abs(s[i]-np.mean(s[max(0,i-L):i+1])) > 2*np.std(s[max(0,i-L):i+1]))
        mom = MOM(s[:i+1], 2)[-1] if i+1>=2 else 0
        vol = VOL(s[:i+1], 2)[-1] if i+1>=2 else 0
        vols.append(vol)
        fib_obj = fib_levels(s[:i+1], L)
        proj, retrace, narrative = project_line(s[:i+1], fib_obj, ama if not np.isnan(ama) else 0)
        ev = expected_value(0.5, odds[i]) if i<len(odds) else np.nan
        kelly = kelly_fraction(0.5, odds[i]) if i<len(odds) else np.nan
        # --- Crossover Detection Example ---
        cross_ama_ema = crossover([ama], [ema])[-1] if not np.isnan(ama) and not np.isnan(ema) else 0
        cross_ema_sma = crossover([ema], [sma])[-1] if not np.isnan(ema) and not np.isnan(sma) else 0
        # --- Signal Scoring ---
        signals = {
            'ADAPTIVEMA': ama if not np.isnan(ama) else 0,
            'EV': ev if not np.isnan(ev) else 0,
            'MOM': mom if not np.isnan(mom) else 0,
            'VOL': vol if not np.isnan(vol) else 0,
            'EMA': ema-s[i],
            'SMA': sma-s[i],
            'MACD': macd_val,
            'RSI': rsi_val,
            'BB': (s[i]-bb_upper[-1])/(np.std(s[max(0,i-L):i+1])+1e-6),
            'ZSCORE': zs[-1] if not np.isnan(zs[-1]) else 0,
            'ROC': roc[-1]/100 if not np.isnan(roc[-1]) else 0,
            'FIB': ((s[i]-np.median(s[max(0,i-L):i+1]))/(np.std(s[max(0,i-L):i+1])+1e-6)) if np.std(s[max(0,i-L):i+1])>0 else 0,
            'STEAM': steam,
            'BREAKOUT': int(s[i]>bb_upper[-1] and rsi_val<0.4 and zs[-1]>1.4 and steam)
        }
        score = sum(base_weights[k]*signals[k] for k in BASE_WEIGHTS)
        model_prob = np.clip(0.5 + 0.4*score, 0.05, 0.95)
        # --- Final Rec ---
        rec = "Bet Up" if score > 0.12 else "Bet Down" if score < -0.12 else "Hold"
        out.append(dict(
            time=times[i], line=s[i], AMA=ama, EMA=ema, SMA=sma, MACD=macd_val, RSI=rsi_val,
            MOM=mom, VOL=vol, EV=ev, Kelly=kelly, ZSCORE=signals['ZSCORE'], ROC=signals['ROC'],
            STEAM=steam, BREAKOUT=signals['BREAKOUT'], Score=score, ModelProb=model_prob, Rec=rec,
            Project=proj, Retrace=retrace, Narrative=narrative,
            cross_ama_ema=cross_ama_ema, cross_ema_sma=cross_ema_sma
        ))
    df_out = pd.DataFrame(out)
    return df_out, "OK"

# --------- STREAMLIT UI ----------
st.set_page_config(layout="wide")
with st.sidebar:
    st.header("God Machine Controls")
    ama_fast = st.slider("AMA Fast",1,8,2)
    ama_slow = st.slider("AMA Slow",4,16,8)
    ama_eff = st.slider("AMA Efficiency Lookback",2,8,4)
    recency_half = st.slider("Recency Half-life (hrs)",6,48,18)
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

blocks, errors = parse_blocks(st.session_state['odds_feed'])
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
            st.metric("Model Score", value=last["Score"])
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
