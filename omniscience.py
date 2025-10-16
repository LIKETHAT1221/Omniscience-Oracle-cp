import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# --- MARKETS AND BASE WEIGHTS ---
MARKETS = ['spread', 'total', 'ml_home', 'ml_away']
BASE_WEIGHTS = dict(
    ADAPTIVEMA=0.2, EV=0.18, MOM=0.1, VOL=0.08, EMA=0.06, SMA=0.05,
    MACD=0.07, RSI=0.06, BB=0.05, ZSCORE=0.05, ROC=0.02, FIB=0.02, STEAM=0.03, BREAKOUT=0.03
)

# --- SAFE JSON ---
def safe_json(obj):
    if isinstance(obj, (np.integer, np.int_)): return int(obj)
    if isinstance(obj, (np.floating, np.float_)): return float(obj)
    if isinstance(obj, (np.ndarray,)): return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime)): return str(obj)
    if isinstance(obj, dict): return {k: safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list): return [safe_json(v) for v in obj]
    return obj

# --- RECENCY WEIGHTS ---
def recency_weights(times, half_life_hrs=18):
    now = max(times)
    deltas = np.array([(now-t).total_seconds()/3600 for t in times])
    decay = np.log(2)/half_life_hrs
    weights = np.exp(-decay * deltas)
    weights = weights / (weights.sum() + 1e-12)
    return weights

# --- PARSER ---
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
            l5mod = L5.replace("even","+100").replace("EVEN","+100")
            ml_tokens = l5mod.replace(",", " ").split()
            ml_nums = []
            for tok in ml_tokens:
                try: ml_nums.append(float(tok))
                except: pass
            ml_away, ml_home = (ml_nums+[None,None])[:2]
            block = dict(
                time=ts, team=team, spread=spread, spread_vig=spread_vig, total=total, total_side=total_side,
                total_vig=total_vig, ml_away=ml_away, ml_home=ml_home)
            blocks.append(block)
        except Exception as e:
            errors.append({"block": lines[i:i+5], "error": str(e)})
        i += 5
    return blocks, errors

# --- INDICATORS (ALWAYS USE MAX AVAILABLE) ---
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
        if arr[i-n] != 0:
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
def crossover(arr1, arr2):
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)
    out = np.zeros_like(arr1)
    prev = arr1 - arr2
    prev = np.sign(prev)
    for i in range(1, len(arr1)):
        if prev[i-1] < 0 and prev[i] > 0: out[i] = 1
        elif prev[i-1] > 0 and prev[i] < 0: out[i] = -1
        else: out[i] = 0
    return out

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

def analyze_market_god(df, market_type, recency_wt, ama_params):
    try:
        col_map = {'spread':'spread','total':'total','ml_home':'ml_home','ml_away':'ml_away'}
        odds_map = {'spread':'spread_vig','total':'total_vig','ml_home':'ml_home','ml_away':'ml_away'}
        if market_type not in col_map: return None, {"error": "Unknown market_type"}
        series = df[col_map[market_type]].dropna().values
        odds = df[odds_map[market_type]].dropna().values
        N = len(series)
        if N < 2:
            return None, {"error": f"Not enough data: {N} ticks, need at least 2"}
        signals_hist = []
        for i in range(N):
            sigs = {}
            avail_Ls = [L for L in [2,3,4] if i >= L]
            for L in avail_Ls:
                ema = EMA(series[:i+1], L)[-1]
                sma = SMA(series[:i+1], L)[-1]
                macd, macd_signal = MACD(series[:i+1], fast=max(2,L//2), slow=L, signal=2)
                macd_val = macd[-1] - macd_signal[-1]
                rsi = RSI(series[:i+1], L)
                rsi_val = (rsi.iloc[-1]-50)/50 if not np.isnan(rsi.iloc[-1]) else 0
                bb_upper, bb_lower = BB(series[:i+1], L)
                zs = zscore(series[:i+1], L)
                roc = ROC(series[:i+1], 2)
                fib = fib_levels(series[:i+1], L)
                steam = int(abs(series[i]-np.mean(series[max(0,i-L):i+1])) > 2*np.std(series[max(0,i-L):i+1]))
                ama = adaptiveMA(series[:i+1], fast=ama_params['fast'], slow=L, efficiency_lookback=ama_params['eff'])[-1]
                mom = MOM(series[:i+1], 2)[-1] if i >= 2 else 0
                vol = VOL(series[:i+1], 2)[-1] if i >= 2 else 0
                cross_ama_ema = crossover([ama], [ema])[-1]
                cross_ema_sma = crossover([ema], [sma])[-1]
                sigs[f"ema_{L}"] = ema - series[i]
                sigs[f"sma_{L}"] = sma - series[i]
                sigs[f"macd_{L}"] = macd_val
                sigs[f"rsi_{L}"] = rsi_val
                sigs[f"bb_{L}"] = (series[i]-bb_upper[-1])/(np.std(series[max(0,i-L):i+1])+1e-6)
                sigs[f"zs_{L}"] = zs[-1] if not np.isnan(zs[-1]) else 0
                sigs[f"roc_{L}"] = roc[-1]/100 if not np.isnan(roc[-1]) else 0
                sigs[f"fib_{L}"] = ((series[i]-np.median(series[max(0,i-L):i+1]))/(np.std(series[max(0,i-L):i+1])+1e-6)) if np.std(series[max(0,i-L):i+1])>0 else 0
                sigs[f"steam_{L}"] = steam
                sigs[f"ama_{L}"] = ama - series[i] if not np.isnan(ama) else 0
                sigs[f"mom_{L}"] = mom if not np.isnan(mom) else 0
                sigs[f"vol_{L}"] = vol if not np.isnan(vol) else 0
                sigs[f"cross_ama_ema_{L}"] = cross_ama_ema
                sigs[f"cross_ema_sma_{L}"] = cross_ema_sma
                sigs[f"breakout_{L}"] = int(series[i] > bb_upper[-1] and rsi_val < 0.4 and zs[-1] > 1.4 and steam)
                if L == max(avail_Ls): fib_obj = fib
            for k in sigs:
                if isinstance(sigs[k], (float,int)): sigs[k] *= recency_wt[min(i,len(recency_wt)-1)]
            signals_hist.append(sigs.copy())
        weights = BASE_WEIGHTS.copy()
        scores = []
        evs, model_probs, kellys = [], [], []
        for i, sh in enumerate(signals_hist):
            score = 0
            for k0 in BASE_WEIGHTS:
                best = 0
                for L in [2,3,4]:
                    name = {
                        "EMA": f"ema_{L}", "SMA": f"sma_{L}", "MACD": f"macd_{L}", "RSI": f"rsi_{L}",
                        "BB": f"bb_{L}", "ZSCORE": f"zs_{L}", "ROC": f"roc_{L}", "FIB": f"fib_{L}",
                        "STEAM": f"steam_{L}", "ADAPTIVEMA": f"ama_{L}", "MOM": f"mom_{L}", "VOL": f"vol_{L}",
                        "BREAKOUT": f"breakout_{L}"
                    }[k0]
                    if name in sh and np.isfinite(sh[name]) and abs(sh[name]) > abs(best): best = sh[name]
                score += weights[k0]*best
            model_prob = 0.5 + 0.4*score if score>0 else 0.5 + 0.4*score
            model_prob = np.clip(model_prob, 0.05, 0.95)
            odd = odds[i] if i < len(odds) else 100
            ev = expected_value(model_prob, odd)
            kelly = kelly_fraction(model_prob, odd)
            scores.append(score)
            model_probs.append(model_prob)
            evs.append(ev)
            kellys.append(kelly)
            sh['score'] = score
            sh['ev'] = ev
            sh['model_prob'] = model_prob
            sh['kelly'] = kelly
        df_out = pd.DataFrame(signals_hist)
        df_out['time'] = df['time'].values[:N]
        df_out['line'] = series
        df_out['score'] = scores
        df_out['model_prob'] = model_probs
        df_out['ev'] = evs
        df_out['kelly'] = kellys
        last_obj = signals_hist[-1]
        final_score = last_obj['score']
        rec = "Bet Favorite" if final_score > 0.2 else "Bet Underdog" if final_score < -0.2 else "Hold"
        proj, retrace, narrative = project_line(series, fib_obj if 'fib_obj' in locals() else None, final_score)
        regime = "Normal"
        return df_out, dict(
            recommendation=rec, projected_line=proj, retrace_zone=retrace, projection_narrative=narrative,
            meta_regime=regime, weights=safe_json(weights), narrative=narrative, last_obj=last_obj
        )
    except Exception as e:
        return None, {"error": f"Exception in analysis: {e}"}

# --- STREAMLIT UI ---
st.set_page_config(layout="wide")
st.markdown("<h1 style='color:#d97706'>Omniscience Oracle Syndicate — The God Machine</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.header("God Machine Controls")
    ama_fast = st.slider("AMA Fast", 1, 8, 2)
    ama_slow = st.slider("AMA Slow", 2, 8, 4)
    ama_eff = st.slider("AMA Efficiency Lookback", 1, 5, 2)
    recency_half = st.slider("Recency Half-life (hrs)", 6, 48, 18)

st.subheader("Paste Odds Feed")
odds_feed = st.text_area("Header, then 5-line blocks for each tick. Each block: time [team?] spread, spread vig, total (e.g. o154/u154.5), total vig, awayML homeML", height=320)
c1, c2, c3 = st.columns(3)
analyze = c1.button("Analyze")
backtest = c2.button("Backtest")
clear = c3.button("Clear")

if clear:
    st.experimental_rerun()

blocks, errors = [], []
if odds_feed.strip():
    blocks, errors = parse_blocks(odds_feed)
    if blocks:
        dfprev = pd.DataFrame(blocks)
        st.markdown("**Parsed Preview (last 25, valid only)**")
        st.dataframe(dfprev.tail(25), hide_index=True)
    if errors:
        st.markdown("### Parser Warnings (malformed blocks):")
        st.code(safe_json(errors[:5]))

if analyze and blocks:
    df = pd.DataFrame(blocks)
    df = df.sort_values("time").reset_index(drop=True)
    cutoff = df.time.max() - timedelta(hours=48)
    df = df[df.time >= cutoff]
    rec_weights = recency_weights(df["time"].tolist(), half_life_hrs=recency_half)
    st.info(f"Parsed {len(df)} ticks. Minimum for each market: 2.")
    for mkt in MARKETS:
        mktlabel = {"spread":"Spread","total":"Total","ml_home":"Home ML","ml_away":"Away ML"}[mkt]
        result = analyze_market_god(df, mkt, rec_weights, {"fast":ama_fast,"slow":ama_slow,"eff":ama_eff})
        if result is None or (isinstance(result, tuple) and result[0] is None):
            msg = result[1]["error"] if (isinstance(result, tuple) and result[1] and "error" in result[1]) else "Unknown error"
            st.warning(f"Analysis skipped for {mktlabel}: {msg}.")
            continue
        df_out, meta = result
        st.success(f"🟢 Analysis SUCCESS for {mktlabel} ({len(df_out)} ticks analyzed)")
        st.dataframe(df_out.tail(10))
        st.markdown(f"**Meta:** {meta}")

if backtest and blocks:
    df = pd.DataFrame(blocks)
    df = df.sort_values("time").reset_index(drop=True)
    cutoff = df.time.max() - timedelta(hours=48)
    df = df[df.time >= cutoff]
    rec_weights = recency_weights(df["time"].tolist(), half_life_hrs=recency_half)
    st.header("Backtest Results (Spread Only, Kelly)")
    result = analyze_market_god(df, "spread", rec_weights, {"fast":ama_fast,"slow":ama_slow,"eff":ama_eff})
    if result is None or (isinstance(result, tuple) and result[0] is None):
        msg = result[1]["error"] if (isinstance(result, tuple) and result[1] and "error" in result[1]) else "Unknown error"
        st.warning(f"Backtest skipped: {msg}.")
    else:
        df_out, meta = result
        bankroll = 1000
        br_hist = [bankroll]
        for i in range(len(df_out)):
            if abs(df_out['score'].iloc[i]) > 0.2 and df_out['ev'].iloc[i] > 0:
                stake = bankroll * df_out['kelly'].iloc[i]
                outcome = np.random.choice([1,0])  # Simulate: 1=win, 0=loss
                if outcome:
                    bankroll += stake * 0.91
                else:
                    bankroll -= stake
            br_hist.append(bankroll)
        st.line_chart(br_hist)
        st.write(f"Final bankroll: ${bankroll:.2f}")
