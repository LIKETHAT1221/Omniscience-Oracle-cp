import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime

# ---- Weighting ----
BASE_WEIGHTS = {
    'ADAPTIVEMA': 0.20, 'EV': 0.18, 'MOM': 0.10, 'VOL': 0.08, 'EMA': 0.06, 'SMA': 0.05,
    'MACD': 0.07, 'RSI': 0.06, 'BB': 0.05, 'ZSCORE': 0.05, 'ROC': 0.02, 'FIB': 0.02,
    'STEAM': 0.03, 'BREAKOUT': 0.03
}
LOOKBACKS = [3, 5, 8]
MARKETS = ['spread', 'total', 'ml_home', 'ml_away']
MARKET_LABELS = {'spread': 'Spread', 'total': 'Total', 'ml_home': 'Home ML', 'ml_away': 'Away ML'}

# ---- Utility ----
def safe_json(obj):
    if isinstance(obj, (np.integer, np.int_)): return int(obj)
    if isinstance(obj, (np.floating, np.float_)): return float(obj)
    if isinstance(obj, (np.ndarray,)): return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime)): return str(obj)
    if isinstance(obj, dict): return {k: safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list): return [safe_json(v) for v in obj]
    return obj

def normalize(x):
    if isinstance(x, (np.ndarray, list)): x = np.nan_to_num(x)
    try:
        mx = np.nanmax(np.abs(x))
        if mx == 0: return 0
        return float(x) / mx
    except:
        return float(x) if np.isfinite(x) else 0

# ---- Parsing ----
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
                if ts.year < 2000: ts = ts.replace(year=datetime.now().year)
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

# ---- TA Indicators ----
def EMA(series, n):
    series = pd.Series(series)
    return series.ewm(span=max(n,1), adjust=False).mean().values

def SMA(series, n):
    series = pd.Series(series)
    return series.rolling(max(n,1)).mean().values

def MACD(series, fast=3, slow=8, signal=2):
    fast_ema = EMA(series, fast)
    slow_ema = EMA(series, slow)
    macd = fast_ema - slow_ema
    macd_signal = pd.Series(macd).ewm(span=signal, adjust=False).mean().values
    return macd, macd_signal

def RSI(series, n=5):
    series = pd.Series(series)
    delta = series.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    roll_up = up.rolling(max(n,1)).mean()
    roll_down = down.rolling(max(n,1)).mean()
    rs = roll_up / (roll_down + 1e-12)
    return (100 - (100 / (1 + rs))).values

def BB(series, n=5, mult=2):
    sma = SMA(series, n)
    std = pd.Series(series).rolling(max(n,1)).std().values
    upper = sma + mult*std
    lower = sma - mult*std
    return upper, lower

def zscore(series, n=5):
    ser = pd.Series(series)
    mean = ser.rolling(max(n,1)).mean()
    std = ser.rolling(max(n,1)).std()
    return ((ser - mean) / (std + 1e-12)).values

def ROC(series, n=2):
    arr = np.array(series)
    out = np.full_like(arr, np.nan)
    for i in range(n, len(arr)):
        if arr[i-n] != 0:
            out[i] = 100*(arr[i]-arr[i-n])/abs(arr[i-n])
    return out

def MOM(series, n=2):
    arr = np.array(series)
    out = np.full_like(arr, np.nan)
    for i in range(n, len(arr)):
        out[i] = arr[i] - arr[i-n]
    return out

def VOL(series, n=5):
    return pd.Series(series).rolling(max(n,1)).std().values

def adaptiveMA(price, fast=2, slow=8, efficiency_lookback=4):
    price = np.array(price)
    n = len(price)
    ama = np.full(n, np.nan)
    for i in range(slow+efficiency_lookback-1, n):
        try:
            ER = abs(price[i] - price[i-efficiency_lookback]) / (np.sum(np.abs(np.diff(price[i-efficiency_lookback:i+1]))) + 1e-12)
            fastSC = 2/(fast+1)
            slowSC = 2/(slow+1)
            SC = (ER*(fastSC-slowSC) + slowSC)**2
            ama[i] = price[i-1] if np.isnan(ama[i-1]) else ama[i-1] + SC*(price[i]-ama[i-1])
        except Exception:
            ama[i] = np.nan
    return pd.Series(ama).fillna(method='ffill').fillna(method='bfill').values

def crossover(arr1, arr2):
    arr1 = np.nan_to_num(arr1)
    arr2 = np.nan_to_num(arr2)
    prev = np.sign(np.array(arr1)-np.array(arr2))
    out = np.zeros_like(prev)
    for i in range(1,len(prev)):
        if prev[i-1]<0 and prev[i]>0: out[i]=1
        elif prev[i-1]>0 and prev[i]<0: out[i]=-1
    return out

def fib_levels(series, window=5):
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

# ---- Recency Weighting ----
def recency_weights(times, half_life_hrs=18):
    times_converted = []
    for t in times:
        try:
            if isinstance(t, str):
                converted = pd.to_datetime(t)
            elif isinstance(t, (int, float)):
                converted = pd.to_datetime(t, unit='s')
            else:
                converted = pd.to_datetime(t)
            times_converted.append(converted)
        except Exception as e:
            print(f"Failed to convert {t} (type: {type(t)}): {e}")
            times_converted.append(pd.Timestamp.now())
    times = times_converted
    latest_time = max(times)
    deltas = []
    for t in times:
        try:
            delta = (latest_time - t).total_seconds() / 3600
            deltas.append(delta)
        except Exception as e:
            print(f"Error calculating delta for {t} (type: {type(t)}): {e}")
            deltas.append(0)
    weights = [2 ** (-delta / half_life_hrs) for delta in deltas]
    norm_weights = np.array(weights) / (np.sum(weights) + 1e-12)
    return norm_weights.tolist()

# ---- EV & Kelly ----
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

# ---- Projections ----
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

# ---- Main Market Analysis ----
def analyze_market_god(df, market, recency_half, ama_params):
    col = market
    odds_col = 'spread_vig' if market == 'spread' else 'total_vig' if market == 'total' else market
    if col not in df: return None, {"narrative": "No data"}
    s = df[col].dropna().values
    odds = df[odds_col].dropna().values if odds_col in df else np.full_like(s, 100)
    times = df['time'].values[-len(s):] if len(df['time']) >= len(s) else df['time'].values
    if len(s) < 2: return None, {"narrative": "No data"}
    rec_weights = recency_weights(times, recency_half)
    out = []
    for i in range(len(s)):
        indicator_vals = {}
        narrative_parts = []
        for L in LOOKBACKS:
            if i+1 < L: continue
            sub = s[:i+1]
            # --- All indicator values, scaled ---
            ama_full = adaptiveMA(sub, ama_params['fast'], ama_params['slow'], ama_params['eff'])
            ama = ama_full[-1]
            ema = EMA(sub, L)[-1]
            sma = SMA(sub, L)[-1]
            macd, macd_signal = MACD(sub, fast=3, slow=L, signal=2)
            macd_val = macd[-1] - macd_signal[-1]
            rsi = RSI(sub, L)[-1]
            bb_upper, bb_lower = BB(sub, L)
            bb_val = (sub[-1]-bb_upper[-1])/(np.std(sub)+1e-6)
            zs = zscore(sub, L)[-1]
            roc = ROC(sub, 2)[-1] if len(sub)>=2 else 0
            mom = MOM(sub, 2)[-1] if len(sub)>=2 else 0
            vol = VOL(sub, L)[-1]
            steam = int(abs(sub[-1]-np.mean(sub)) > 2*np.std(sub))
            fib_obj = fib_levels(sub, L)
            cross_ama_ema = crossover(ama_full, EMA(sub, L))[-1]
            cross_ema_sma = crossover(EMA(sub, L), SMA(sub, L))[-1]
            breakout = int(sub[-1]>bb_upper[-1] and (rsi-50)/50<0.4 and zs>1.4 and steam)
            # Save values
            indicator_vals[f"ADAPTIVEMA_{L}"] = normalize(ama)
            indicator_vals[f"EMA_{L}"] = normalize(ema)
            indicator_vals[f"SMA_{L}"] = normalize(sma)
            indicator_vals[f"MACD_{L}"] = normalize(macd_val)
            indicator_vals[f"RSI_{L}"] = normalize((rsi-50)/50)
            indicator_vals[f"BB_{L}"] = normalize(bb_val)
            indicator_vals[f"ZSCORE_{L}"] = normalize(zs)
            indicator_vals[f"ROC_{L}"] = normalize(roc)
            indicator_vals[f"MOM_{L}"] = normalize(mom)
            indicator_vals[f"VOL_{L}"] = normalize(vol)
            indicator_vals[f"STEAM_{L}"] = steam
            indicator_vals[f"BREAKOUT_{L}"] = breakout
            indicator_vals[f"cross_ama_ema_{L}"] = cross_ama_ema
            indicator_vals[f"cross_ema_sma_{L}"] = cross_ema_sma
            indicator_vals[f"FIB_{L}"] = normalize((sub[-1]-np.median(sub))/(np.std(sub)+1e-6) if np.std(sub)>0 else 0)
            # Narrative for this lookback
            narrative_parts.append(
                f"AMA({L}): {'+' if ama>0 else '-' if ama<0 else '0'}, EMA({L}): {ema:.2f}, SMA({L}): {sma:.2f}, "
                f"MACD({L}): {macd_val:.2f}, RSI({L}): {rsi:.2f}, BB({L}): {bb_val:.2f}, ZSCORE({L}): {zs:.2f}, "
                f"ROC({L}): {roc:.2f}, MOM({L}): {mom:.2f}, VOL({L}): {vol:.2f}, STEAM({L}): {steam}, "
                f"Breakout({L}): {breakout}, cross_ama_ema({L}): {cross_ama_ema}, cross_ema_sma({L}): {cross_ema_sma}."
            )
        # --- Scoring ---
        score = 0
        for k in BASE_WEIGHTS:
            signal_vals = [indicator_vals[f"{k}_{L}"] for L in LOOKBACKS if f"{k}_{L}" in indicator_vals]
            best = max(signal_vals, key=abs) if signal_vals else 0
            score += BASE_WEIGHTS[k] * best
        model_prob = np.clip(0.5 + 0.4*score, 0.05, 0.95)
        ev = expected_value(model_prob, odds[i]) if i<len(odds) else 0
        kelly = kelly_fraction(model_prob, odds[i]) if i<len(odds) else 0
        # --- Projection ---
        fib_obj = fib_levels(s[:i+1], min(LOOKBACKS))
        proj, retrace, proj_narr = project_line(s[:i+1], fib_obj, score)
        # --- Rec ---
        rec = "Bet Favorite" if score>0.15 else "Bet Underdog" if score<-0.15 else "Hold"
        # --- Full Narrative ---
        full_narr = (
            f"AMA: {'Positive' if indicator_vals.get('ADAPTIVEMA_3',0)>0 else 'Negative' if indicator_vals.get('ADAPTIVEMA_3',0)<0 else 'Neutral'}, "
            f"indicating {'trend alignment' if indicator_vals.get('ADAPTIVEMA_3',0)>0 else 'reversal risk' if indicator_vals.get('ADAPTIVEMA_3',0)<0 else 'no clear signal'}. "
            f"MACD: {indicator_vals.get('MACD_3',0):.2f}, {'trend continuation' if indicator_vals.get('MACD_3',0)>0 else 'reversal pressure' if indicator_vals.get('MACD_3',0)<0 else 'neutral'}. "
            f"RSI: {indicator_vals.get('RSI_3',0):.2f}, {'overbought' if indicator_vals.get('RSI_3',0)>0.8 else 'oversold' if indicator_vals.get('RSI_3',0)<-0.8 else 'neutral'}. "
            f"VOL: {indicator_vals.get('VOL_3',0):.2f}, indicating {'high volatility' if indicator_vals.get('VOL_3',0)>0.7 else 'normal'}. "
            f"MOM: {indicator_vals.get('MOM_3',0):.2f}, {'momentum supports selection' if indicator_vals.get('MOM_3',0)>0 else 'momentum weak'}. "
            f"Breakout: {indicator_vals.get('BREAKOUT_3',0)}, {'true breakout' if indicator_vals.get('BREAKOUT_3',0)>0 else 'no breakout'}. "
            f"Steam: {indicator_vals.get('STEAM_3',0)}, {'market move detected' if indicator_vals.get('STEAM_3',0)>0 else 'no steam'}. "
            f"ModelProb: {model_prob:.2f} → {rec}, Score: {score:.2f}. {proj_narr} "
            f"Crossovers: AMA/EMA: {indicator_vals.get('cross_ama_ema_3',0)}, EMA/SMA: {indicator_vals.get('cross_ema_sma_3',0)}. "
            f"\nIndicator details: {' | '.join(narrative_parts)}"
        )
        out.append(dict(
            time=times[i], line=s[i], **indicator_vals,
            Score=score, ModelProb=model_prob, EV=ev, Kelly=kelly, Rec=rec,
            Project=proj, Retrace=retrace, Narrative=full_narr
        ))
    df_out = pd.DataFrame(out)
    return df_out, {"narrative": out[-1]['Narrative'] if out else ""}

# ---- Streamlit UI ----
st.set_page_config(layout="wide")
with st.sidebar:
    st.header("God Machine Controls")
    ama_fast = st.slider("AMA Fast",1,8,2)
    ama_slow = st.slider("AMA Slow",4,16,8)
    ama_eff = st.slider("AMA Efficiency Lookback",2,8,4)
    recency_half = st.slider("Recency Half-life (hrs)",6,48,18)
    st.markdown("**Backtest and Clear**")
    clear = st.button("Clear")
    backtest = st.button("Backtest")

if 'odds_feed' not in st.session_state: st.session_state['odds_feed'] = ''
if 'analysis_triggered' not in st.session_state: st.session_state['analysis_triggered'] = False
if 'df' not in st.session_state: st.session_state['df'] = None

def clear_all():
    st.session_state.odds_feed = ""
    st.session_state.analysis_triggered = False
    st.session_state.df = None

st.subheader("Paste Odds Feed")
st.text_area("Paste odds", key='odds_feed', height=320)
c1, c2 = st.columns(2)
if c1.button("Analyze"):
    st.session_state.analysis_triggered = True
if clear:
    clear_all()
    st.experimental_rerun()

blocks, errors = parse_blocks(st.session_state.odds_feed)
if blocks:
    df = pd.DataFrame(blocks)
    st.session_state.df = df
    st.dataframe(df.tail(10))

if st.session_state.analysis_triggered and st.session_state.df is not None:
    df = st.session_state.df
    tabs = st.tabs([MARKET_LABELS[m] for m in MARKETS])
    ama_params = {'fast': ama_fast, 'slow': ama_slow, 'eff': ama_eff}
    recency_wt = recency_half
    for idx, market in enumerate(MARKETS):
        label = MARKET_LABELS[market]
        with tabs[idx]:
            df_out, meta = analyze_market_god(df, market, recency_wt, ama_params)
            if df_out is None or df_out.empty:
                st.warning(f"No available data for {label} market.")
                continue
            st.dataframe(df_out)
            st.markdown(f"**Narrative:**\n{meta['narrative']}")

if backtest and st.session_state.df is not None:
    df = st.session_state.df
    ama_params = {'fast': ama_fast, 'slow': ama_slow, 'eff': ama_eff}
    recency_wt = recency_half
    st.header("Backtest Results")
    for market in MARKETS:
        label = MARKET_LABELS[market]
        df_out, meta = analyze_market_god(df, market, recency_wt, ama_params)
        if df_out is None or df_out.empty:
            st.warning(f"No available data for {label} market.")
            continue
        st.markdown(f"### {label}")
        st.dataframe(df_out)
        st.markdown(f"**Narrative:**\n{meta['narrative']}")
