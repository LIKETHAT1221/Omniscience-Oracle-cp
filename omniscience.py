import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ====== GOD MACHINE PARAMETERS ======
LOOKBACKS = [4, 8, 16]  # multi-horizon analysis
RECENCY_HALFLIFE_HRS = 18
MARKETS = ['spread', 'total', 'ml_home', 'ml_away']
VOTE_THRESH = 0.1

# Elite stack weights (tunable, regime-aware)
BASE_WEIGHTS = dict(
    ADAPTIVEMA=0.20, EV=0.18, MOM=0.10, VOL=0.08, EMA=0.06, SMA=0.05,
    MACD=0.07, RSI=0.06, BB=0.05, ZSCORE=0.05, ROC=0.02, FIB=0.02, STEAM=0.03, BREAKOUT=0.03
)

# ========== SAFE JSON ==========
def safe_json(obj):
    if isinstance(obj, (np.integer, np.int_)): return int(obj)
    if isinstance(obj, (np.floating, np.float_)): return float(obj)
    if isinstance(obj, (np.ndarray,)): return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime)): return str(obj)
    if isinstance(obj, dict): return {k: safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list): return [safe_json(v) for v in obj]
    return obj

# ========== RECENCY ==========
def recency_weights(times, half_life_hrs=18):
    now = max(times)
    deltas = np.array([(now-t).total_seconds()/3600 for t in times])
    decay = np.log(2)/half_life_hrs
    weights = np.exp(-decay * deltas)
    weights = weights / (weights.sum() + 1e-12)
    return weights

# ========== PARSER ==========
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
            if all(v is not None for v in [ts, spread, spread_vig, total, total_vig, ml_away, ml_home]):
                blocks.append(block)
            else:
                errors.append({"block": lines[i:i+5], "error": "Missing required field"})
        except Exception as e:
            errors.append({"block": lines[i:i+5], "error": str(e)})
        i += 5
    return blocks, errors

# ========== INDICATORS ==========
def EMA(series, n): return pd.Series(series).ewm(span=n, adjust=False).mean().values
def SMA(series, n): return pd.Series(series).rolling(n).mean().values
def MACD(series, fast=4, slow=8, signal=3):
    ema_fast = EMA(series, fast)
    ema_slow = EMA(series, slow)
    macd = ema_fast - ema_slow
    macd_signal = pd.Series(macd).ewm(span=signal,adjust=False).mean().values
    return macd, macd_signal
def RSI(series, n=5):
    series = pd.Series(series)
    delta = series.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    roll_up = up.rolling(n).mean()
    roll_down = down.rolling(n).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))
def BB(series, n=6, mult=2):
    sma = SMA(series, n)
    std = pd.Series(series).rolling(n).std().values
    upper = sma + mult*std
    lower = sma - mult*std
    return upper, lower
def zscore(series, n=6):
    ser = pd.Series(series)
    return ((ser - ser.rolling(n).mean()) / (ser.rolling(n).std() + 1e-12)).values
def ROC(series, n=2):
    arr = np.array(series)
    out = np.full_like(arr, np.nan, dtype=np.float64)
    for i in range(n, len(arr)):
        if arr[i-n] != 0:
            out[i] = 100*(arr[i]-arr[i-n])/abs(arr[i-n])
    return out
def fib_levels(series, window=6):
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
def adaptiveMA(series, fast=2, slow=8, efficiency_lookback=4):
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
def MOM(series, n=4):
    arr = np.array(series)
    out = np.full_like(arr, np.nan, dtype=np.float64)
    for i in range(n, len(arr)):
        out[i] = arr[i] - arr[i-n]
    return out
def VOL(series, n=6):
    return pd.Series(series).rolling(n).std().values

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

# ========== EV & KELLY ==========
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
    return max(0, min(f_star, 1))

# ========== PROJECT LINE ==========
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

# ========== GOD MACHINE MARKET ANALYSIS ==========
def analyze_market_god(df, market_type, recency_wt, ama_params, dynamic_weights=True):
    if market_type == 'spread':
        series = df['spread'].values
        odds = df['spread_vig'].values
    elif market_type == 'total':
        series = df['total'].values
        odds = df['total_vig'].values
    elif market_type == 'ml_home':
        series = df['ml_home'].values
        odds = df['ml_home'].values
    elif market_type == 'ml_away':
        series = df['ml_away'].values
        odds = df['ml_away'].values
    else:
        return None
    N = len(series)
    if N < max(LOOKBACKS)+3: return None

    # Multi-horizon indicators
    signals_hist = []
    for i in range(N):
        sigs = {}
        for L in LOOKBACKS:
            if i < L: continue
            ema = EMA(series[:i+1], L)[-1]
            sma = SMA(series[:i+1], L)[-1]
            macd, macd_signal = MACD(series[:i+1], fast=L//2, slow=L, signal=3)
            macd_val = macd[-1] - macd_signal[-1]
            rsi = RSI(series[:i+1], L)
            rsi_val = (rsi.iloc[-1]-50)/50 if not np.isnan(rsi.iloc[-1]) else 0
            bb_upper, bb_lower = BB(series[:i+1], L)
            zs = zscore(series[:i+1], L)
            roc = ROC(series[:i+1], 2)
            fib = fib_levels(series[:i+1], L)
            steam = int(abs(series[i]-np.mean(series[max(0,i-L):i+1])) > 2*np.std(series[max(0,i-L):i+1]))
            ama = adaptiveMA(series[:i+1], fast=ama_params['fast'], slow=L, efficiency_lookback=ama_params['eff'])[-1]
            mom = MOM(series[:i+1], 4)[-1] if i >= 4 else 0
            vol = VOL(series[:i+1], 6)[-1] if i >= 6 else 0
            cross_ama_ema = crossover([ama], [ema])[-1]
            cross_ema_sma = crossover([ema], [sma])[-1]
            # Normalized signals
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
            # Add FIB object for projection at last lookback
            if L == LOOKBACKS[-1]: fib_obj = fib
        # Recency weighting
        for k in sigs:
            if isinstance(sigs[k], (float,int)): sigs[k] *= recency_wt[i]
        signals_hist.append(sigs.copy())
    # --- Dynamic regime detection and weights ---
    weights = BASE_WEIGHTS.copy()
    # Regime: If volatility > 2*median, decrease trend weights, increase mean reversion
    vols = [sh.get('vol_8',0) for sh in signals_hist if 'vol_8' in sh]
    if vols and np.nanmedian(vols)>0 and np.nanmean(vols[-5:]) > 2*np.nanmedian(vols):
        weights['ADAPTIVEMA'] *= 0.6
        weights['ZSCORE'] *= 1.7
        weights['VOL'] *= 1.7
        weights['MOM'] *= 0.7
    # Compose weighted scores for each tick, ensemble
    scores = []
    evs, model_probs, kellys = [], [], []
    for i, sh in enumerate(signals_hist):
        # Weighted ensemble: AMA, EV, MOM, VOL, others
        score = 0
        for k0 in BASE_WEIGHTS:
            # Find best horizon
            best = 0
            for L in LOOKBACKS:
                name = {
                    "EMA": f"ema_{L}", "SMA": f"sma_{L}", "MACD": f"macd_{L}", "RSI": f"rsi_{L}",
                    "BB": f"bb_{L}", "ZSCORE": f"zs_{L}", "ROC": f"roc_{L}", "FIB": f"fib_{L}",
                    "STEAM": f"steam_{L}", "ADAPTIVEMA": f"ama_{L}", "MOM": f"mom_{L}", "VOL": f"vol_{L}",
                    "BREAKOUT": f"breakout_{L}"
                }[k0]
                if name in sh and np.isfinite(sh[name]) and abs(sh[name]) > abs(best): best = sh[name]
            score += weights[k0]*best
        # Add EV (will get updated next)
        model_prob = 0.5 + 0.4*score if score>0 else 0.5 + 0.4*score
        model_prob = np.clip(model_prob, 0.05, 0.95)
        odds = df['spread_vig'].values[i] if market_type == 'spread' else \
               df['total_vig'].values[i] if market_type == 'total' else \
               df['ml_home'].values[i] if market_type == 'ml_home' else \
               df['ml_away'].values[i]
        ev = expected_value(model_prob, odds)
        kelly = kelly_fraction(model_prob, odds)
        scores.append(score)
        model_probs.append(model_prob)
        evs.append(ev)
        kellys.append(kelly)
        sh['score'] = score
        sh['ev'] = ev
        sh['model_prob'] = model_prob
        sh['kelly'] = kelly
    # Compose output DataFrame
    df_out = pd.DataFrame(signals_hist)
    df_out['time'] = df['time'].values
    df_out['line'] = series
    df_out['score'] = scores
    df_out['model_prob'] = model_probs
    df_out['ev'] = evs
    df_out['kelly'] = kellys
    # Last recommendation
    last_obj = signals_hist[-1]
    final_score = last_obj['score']
    rec = "Bet Favorite" if final_score > 0.2 else "Bet Underdog" if final_score < -0.2 else "Hold"
    proj, retrace, narrative = project_line(series, fib_obj if 'fib_obj' in locals() else None, final_score)
    # Regime meta
    regime = "High Volatility" if weights['VOL']>BASE_WEIGHTS['VOL'] else "Normal"
    return df_out, dict(
        recommendation=rec, projected_line=proj, retrace_zone=retrace, projection_narrative=narrative,
        meta_regime=regime, weights=safe_json(weights), narrative=narrative, last_obj=last_obj
    )

# ========== STREAMLIT UI ==========
st.set_page_config(layout="wide")
st.markdown("<h1 style='color:#d97706'>Omniscience Oracle Syndicate — The God Machine</h1>", unsafe_allow_html=True)
col1, col2 = st.columns([1.2,2])

with col1:
    st.subheader("Paste Odds Feed")
    odds_feed = st.text_area("Header, then 5-line blocks for each tick. Each block: time [team?] spread, spread vig, total (e.g. o154/u154.5), total vig, awayML homeML", height=320)
    c1, c2, c3 = st.columns(3)
    analyze = c1.button("Analyze")
    clear = c3.button("Clear")
    st.markdown("AMA Parameters (syndicate-tuned):")
    ama_fast = st.slider("AMA Fast", 1, 8, 2)
    ama_slow = st.slider("AMA Slow", 4, 16, 8)
    ama_eff = st.slider("AMA Efficiency Lookback", 2, 8, 4)
    st.markdown("Malformed blocks are reported in the analysis.")
    if odds_feed.strip():
        blocks, errors = parse_blocks(odds_feed)
        if blocks:
            dfprev = pd.DataFrame(blocks)
            st.markdown("**Parsed Preview (last 25, valid only)**")
            st.dataframe(dfprev.tail(25), hide_index=True)
        else:
            st.warning("No valid blocks parsed.")
    else:
        blocks, errors = [], []

with col2:
    if clear:
        st.experimental_rerun()
    if analyze and blocks:
        df = pd.DataFrame(blocks)
        df = df.sort_values("time").reset_index(drop=True)
        cutoff = df.time.max() - timedelta(hours=48)
        df = df[df.time >= cutoff]
        rec_weights = recency_weights(df["time"].tolist(), half_life_hrs=RECENCY_HALFLIFE_HRS)
        for mkt in MARKETS:
            mktlabel = {"spread":"Spread","total":"Total","ml_home":"Home ML","ml_away":"Away ML"}[mkt]
            df_out, meta = analyze_market_god(df, mkt, rec_weights, {"fast":ama_fast,"slow":ama_slow,"eff":ama_eff})
            if df_out is None: continue
            st.markdown(f"## {mktlabel} — {meta['recommendation']}")
            st.markdown(f"**Meta Regime:** {meta['meta_regime']}")
            st.line_chart(df_out.set_index('time')[['line','score','ev','model_prob','kelly']])
            st.line_chart(df_out.set_index('time')[[c for c in df_out.columns if c.startswith('ama_') or c.startswith('ema_') or c.startswith('sma_')]].fillna(0))
            st.line_chart(df_out.set_index('time')[[c for c in df_out.columns if c.startswith('mom_') or c.startswith('vol_')]].fillna(0))
            st.dataframe(df_out.tail(25))
            st.markdown(f"**Projection:** {meta['projection_narrative']}")
            st.markdown(f"**Weights:** {meta['weights']}")
            st.markdown(f"**Last-Stack:** {safe_json(meta['last_obj'])}")
        if errors:
            st.markdown("### Parser Warnings (malformed blocks):")
            st.code(safe_json(errors[:5]))
    elif not blocks and analyze:
        st.warning("No valid blocks parsed. Check your feed formatting.")
