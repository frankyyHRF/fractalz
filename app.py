import gc
import io
import re
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import requests

# ==============================================================================
# --- UNIFIED SYSTEM INITIALIZATION ---
# ==============================================================================
st.set_page_config(page_title="HRF QUANT PLATFORM", layout="wide")

st.markdown("""
    <style>
    .reportview-container { background: #0d0d11; }
    .sidebar .sidebar-content { background: #13131a; }
    h1, h2, h3, p, label { color: #ffffff !important; }
    div.stButton > button:first-child {
        background-color: #8a2be2; color: white; border-radius: 8px; font-weight: bold; width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🏛️ HRF QUANT MASTER PLATFORM V5.0")
st.caption("The Complete Unified Multi-Engine Ecosystem — Model By HRF")
st.divider()

app_mode = st.sidebar.selectbox(
    "🚀 Choose Analysis Core Engine",
    ["Algorithmic Fractal Scan", "Structural Capitulation Wave", "Statistical Framework Core"]
)

csv_url_input = st.sidebar.text_input("🔗 Optional Data Link (.parquet / .csv / Google Drive)", value="")
max_short_bars = st.sidebar.number_input(
    "Max Short-TF Bars (RAM Guard)",
    min_value=500,
    max_value=25000,
    value=4000,
    step=500
)

if st.sidebar.button("🧹 Clear Cache & Free RAM"):
    st.cache_data.clear()
    gc.collect()
    st.rerun()

data_status_box = st.sidebar.empty()

ticker_map = {
    'Bitcoin (BTC)': 'XBTUSD',
}

def percentage_return_scale(series):
    arr = np.asarray(series, dtype=np.float32).flatten()
    if arr.size == 0 or arr[0] == 0:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - arr[0]) / arr[0]) * np.float32(100.0)

def get_election_regime(year):
    if year % 4 == 0: return 'Election Year'
    elif year % 4 == 1: return 'Post-Election'
    elif year % 4 == 2: return 'Midterm Year'
    else: return 'Pre-Election'

# ==============================================================================
# DATA PIPELINE (KRAKEN PUBLIC OHLC + PARQUET / CSV LOW-RAM RESAMPLER)
# ==============================================================================
def _extract_gdrive_id(url):
    m = re.search(r"/file/d/([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)
    m2 = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
    if m2:
        return m2.group(1)
    return None

def _open_remote_file_response(raw_url):
    u = raw_url.strip()
    file_id = _extract_gdrive_id(u)
    session = requests.Session()
    headers = {"User-Agent": "Mozilla/5.0"}

    if file_id:
        url = f"https://drive.usercontent.google.com/download?export=download&id={file_id}&confirm=t"
        resp = session.get(url, headers=headers, stream=True, timeout=60)
        resp.raise_for_status()
        ct = resp.headers.get("Content-Type", "")
        if "text/html" in ct:
            html_text = resp.text
            action_match = re.search(r'action="([^"]+)"', html_text)
            inputs = dict(re.findall(r'name="([^"]+)"\s+value="([^"]*)"', html_text))
            if action_match and inputs:
                action_url = action_match.group(1).replace("&amp;", "&")
                resp = session.get(action_url, params=inputs, headers=headers, stream=True, timeout=60)
                resp.raise_for_status()
                ct = resp.headers.get("Content-Type", "")
        if "text/html" in ct:
            raise ValueError("Google Drive returned HTML instead of data file. Check link sharing.")
        return resp

    resp = session.get(u, headers=headers, stream=True, timeout=60)
    resp.raise_for_status()
    return resp

def _normalize_ohlcv_df(df):
    out = df.copy()

    if not isinstance(out.index, pd.DatetimeIndex):
        out.columns = [str(c).strip().lower() for c in out.columns]
        time_col = next(
            (c for c in ["time", "timestamp", "date", "open_time", "datetime"] if c in out.columns),
            None,
        )
        if time_col is not None:
            ts = out[time_col]
            if pd.api.types.is_numeric_dtype(ts):
                s_vals = pd.to_numeric(ts, errors="coerce")
                max_v = float(s_vals.max()) if not s_vals.dropna().empty else 0.0
                unit = "us" if max_v > 1e14 else ("ms" if max_v > 1e11 else "s")
                idx = pd.to_datetime(s_vals, unit=unit, errors="coerce")
            else:
                idx = pd.to_datetime(ts, errors="coerce")
            out.index = idx
        else:
            out.index = pd.to_datetime(out.index, errors="coerce")
    else:
        out.columns = [str(c).strip().lower() for c in out.columns]
        out.index = pd.to_datetime(out.index, errors="coerce")

    out.index.name = "time"
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in out.columns:
            out[col] = out["close"] if (col != "volume" and "close" in out.columns) else 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(np.float32)

    out = out[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
    out = out[~out.index.isna()].sort_index()
    return out[~out.index.duplicated(keep="last")]

@st.cache_data(ttl=1800, max_entries=2, show_spinner="Loading & resampling remote data...")
def _load_and_resample_remote_file(raw_url, interval_str, bar_cap):
    rule_map = {'1m': '1min', '5m': '5min', '15m': '15min', '1h': '1h', '4h': '4h'}
    rule = rule_map.get(interval_str)
    if not rule:
        return None

    cap = int(bar_cap)
    keep_tail = max(cap * 2, 5000)
    resp = _open_remote_file_response(raw_url)
    content_bytes = resp.content
    if not content_bytes:
        raise ValueError("Remote file returned empty content.")

    is_parquet = (
        ".parquet" in raw_url.lower()
        or content_bytes[:4] == b"PAR1"
        or content_bytes[-4:] == b"PAR1"
    )

    if is_parquet:
        buf = io.BytesIO(content_bytes)
        del content_bytes
        gc.collect()

        raw_df = pd.read_parquet(buf, engine="pyarrow")
        del buf
        gc.collect()

        cleaned = _normalize_ohlcv_df(raw_df)
        del raw_df
        gc.collect()

        if cleaned.empty:
            return None

        resampled = (
            cleaned.resample(rule)
            .agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
            .dropna(subset=['close'])
            .iloc[-cap:]
        )
        del cleaned
        gc.collect()
        return resampled[['open', 'high', 'low', 'close', 'volume']]

    buf = io.BytesIO(content_bytes)
    del content_bytes
    gc.collect()

    reader = pd.read_csv(buf, chunksize=25000)
    rolling_df = None

    for part in reader:
        cleaned = _normalize_ohlcv_df(part)
        if not cleaned.empty:
            r_part = cleaned.resample(rule).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna(subset=['close'])
            rolling_df = r_part if rolling_df is None else pd.concat([rolling_df, r_part])
            rolling_df = rolling_df.sort_index()
            rolling_df = rolling_df[~rolling_df.index.duplicated(keep='last')].iloc[-keep_tail:]
            del cleaned, r_part

    del reader, buf
    gc.collect()

    if rolling_df is None or rolling_df.empty:
        return None

    final_df = rolling_df.iloc[-cap:].copy()
    del rolling_df
    gc.collect()
    return final_df[['open', 'high', 'low', 'close', 'volume']]

@st.cache_data(ttl=120, max_entries=4, show_spinner=False)
def _fetch_kraken_btc_ohlc(interval_str, bar_cap):
    tf_map = {
        '1m': 1, '5m': 5, '15m': 15, '1h': 60,
        '4h': 240, '1day': 1440, '1 week': 10080, '1month': 21600,
    }
    k_interval = tf_map.get(interval_str, 1440)
    url = f"https://api.kraken.com/0/public/OHLC?pair=XBTUSD&interval={k_interval}"
    resp = requests.get(url, timeout=12)
    resp.raise_for_status()
    payload = resp.json()

    if payload.get("error"):
        raise ValueError(", ".join(payload["error"]))

    result = payload.get("result", {})
    candle_key = next((k for k in result if k != "last"), None)
    if not candle_key or not result[candle_key]:
        return None

    rows = result[candle_key]
    df = pd.DataFrame(rows, columns=['ts', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'])
    df['time'] = pd.to_datetime(pd.to_numeric(df['ts'], errors='coerce'), unit='s')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)

    df = df.dropna(subset=['time', 'close']).set_index('time').sort_index()
    df = df[~df.index.duplicated(keep='last')]

    if interval_str == '1month':
        df = df.resample('1ME').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna(subset=['close'])

    if interval_str in {'1m', '5m', '15m', '1h', '4h'}:
        df = df.iloc[-int(bar_cap):]

    return df[['open', 'high', 'low', 'close', 'volume']]

def fetch_legacy_market_data(asset_name, interval_str):
    df_live = None
    source_note = []
    cap = int(max_short_bars)

    try:
        df_live = _fetch_kraken_btc_ohlc(interval_str, cap)
        if df_live is not None and not df_live.empty:
            source_note.append(f"Kraken BTC ({len(df_live)} bars)")
    except Exception as e:
        source_note.append(f"Kraken err: {e}")
        df_live = None

    intraday_frames = {'1m', '5m', '15m', '1h', '4h'}
    if csv_url_input.strip() and interval_str in intraday_frames:
        try:
            df_file = _load_and_resample_remote_file(csv_url_input.strip(), interval_str, cap)
            if df_file is not None and not df_file.empty:
                source_note.append(f"File ({len(df_file)} bars)")
                if df_live is None or df_live.empty:
                    data_status_box.caption("📡 Source: " + " | ".join(source_note))
                    return df_file
                hist_gap = df_file[df_file.index < df_live.index.min()]
                if not hist_gap.empty:
                    combined = pd.concat([hist_gap, df_live])
                    merged = combined[~combined.index.duplicated(keep='last')].sort_index().iloc[-cap:]
                    del hist_gap, combined
                    gc.collect()
                    data_status_box.caption(f"📡 Source: File + Kraken ({len(merged)} bars)")
                    return merged
        except Exception as e:
            source_note.append(f"File skipped: {e}")

    if source_note:
        data_status_box.caption("📡 Source: " + " | ".join(source_note))
    return df_live

# ==============================================================================
# NATIVE VECTORIZED MATH AND INDICATORS CALCULATOR
# ==============================================================================
def calculate_custom_indicators(df, cfg):
    close = df['close']
    high = df['high']
    low = df['low']
    vol = df['volume']

    df['SMA'] = close.rolling(window=cfg['sma_p']).mean()
    df['EMA'] = close.ewm(span=cfg['ema_p'], adjust=False).mean()
    w = np.arange(1, cfg['wma_p'] + 1, dtype=np.float32)
    w_sum = w.sum()
    df['WMA'] = close.rolling(window=cfg['wma_p']).apply(lambda x: np.dot(x, w) / w_sum, raw=True)

    delta = close.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=cfg['rsi_p']).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=cfg['rsi_p']).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = 100.0 - (100.0 / (1.0 + rs))

    ema_fast = close.ewm(span=cfg['macd_f'], adjust=False).mean()
    ema_slow = close.ewm(span=cfg['macd_s'], adjust=False).mean()
    df['MACD'] = ema_fast - ema_slow

    df['BB_Mid'] = close.rolling(window=cfg['bb_p']).mean()
    bb_std = close.rolling(window=cfg['bb_p']).std()
    df['BB_Upper'] = df['BB_Mid'] + (cfg['bb_std'] * bb_std)
    df['BB_Lower'] = df['BB_Mid'] - (cfg['bb_std'] * bb_std)

    h_l = high - low
    h_pc = (high - close.shift(1)).abs()
    l_pc = (low - close.shift(1)).abs()
    tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=cfg['atr_p']).mean()

    cum_vol_price = (close * vol).cumsum()
    cum_vol = vol.cumsum()
    df['VWAP'] = cum_vol_price / (cum_vol + 1e-10)

    atr = df['ATR'].fillna(0)
    hl2 = (high + low) / 2.0
    upper_band = hl2 + (cfg['st_m'] * atr)
    lower_band = hl2 - (cfg['st_m'] * atr)

    st_arr = np.zeros(len(df), dtype=np.float32)
    in_uptrend = True
    for i in range(1, len(df)):
        if close.iloc[i] > upper_band.iloc[i-1]:
            in_uptrend = True
        elif close.iloc[i] < lower_band.iloc[i-1]:
            in_uptrend = False
        st_arr[i] = lower_band.iloc[i] if in_uptrend else upper_band.iloc[i]
    df['SuperTrend'] = st_arr
    return df

def calculate_rolling_correlation(series_a, series_b, lookback_window):
    arr_a = np.asarray(series_a, dtype=np.float32).flatten()
    arr_b = np.asarray(series_b, dtype=np.float32).flatten()
    min_len = min(len(arr_a), len(arr_b))
    if min_len == 0:
        return []
    df = pd.DataFrame({'Target_Curve': arr_a[:min_len], 'Match_Curve': arr_b[:min_len]})
    df['Ret_A'] = df['Target_Curve'].pct_change()
    df['Ret_C'] = df['Match_Curve'].pct_change()
    return df['Ret_A'].rolling(window=int(lookback_window)).corr(df['Ret_C']).fillna(0.0).tolist()

def calculate_hrf_path_drawdown(price_series):
    prices = np.asarray(price_series, dtype=np.float32)
    n = len(prices)
    drawdowns = np.zeros(n, dtype=np.float32)
    if n == 0:
        return drawdowns
    current_local_high = prices[0]
    for i in range(1, n):
        if prices[i] >= current_local_high:
            current_local_high = prices[i]
            drawdowns[i] = 0.0
        else:
            drawdowns[i] = (prices[i] - current_local_high) / current_local_high
    return drawdowns

# ==============================================================================
# ENGINE MODULE 1: ALGORITHMIC FRACTAL SCANNER
# ==============================================================================
if app_mode == "Algorithmic Fractal Scan":
    st.header("🎯 TradingView Core Fractal Scanner")

    t_asset = st.sidebar.selectbox("Baseline Target Asset", list(ticker_map.keys()), index=0)
    s_pool = st.sidebar.selectbox("Scan Matching Pool Range", list(ticker_map.keys()), index=0)
    i_choice = st.sidebar.selectbox("Sequence Time Frame Interval", ['1month', '1 week', '1day', '4h', '1h', '15m', '5m', '1m'], index=2)
    f_mode = st.sidebar.selectbox("Framework Processing Mode", ["Calculate Fractals", "Manual Compare"], index=0)

    st.sidebar.markdown("### 🎛️ Noise & Forward Regime Filters")
    corr_min, corr_max = st.sidebar.slider(
        "Correlation Filter Range (Min to Max)",
        min_value=-1.0,
        max_value=1.0,
        value=(0.30, 1.00),
        step=0.05
    )
    mse_min, mse_max = st.sidebar.slider(
        "MSE Error Filter Range (Min to Max)",
        min_value=0.0,
        max_value=200.0,
        value=(0.0, 30.0),
        step=0.5
    )
    forward_regime = st.sidebar.selectbox(
        "Forward-Looking Regime (After Matching Window)",
        ["All", "Bullish", "Bearish"],
        index=0
    )

    st.sidebar.markdown("### 📊 Inter-Asset Correlation Layer")
    enable_corr = st.sidebar.checkbox("Overlay Rolling Macro Asset Correlation", value=False)
    corr_window = st.sidebar.text_input("Correlation Lookback Window (Bars)", value="10")

    start_d = st.sidebar.text_input("Analysis Target Window Start", value="latest")
    end_d = st.sidebar.text_input("Analysis Target Window End", value="latest")

    ov_starts = st.sidebar.text_input("Manual Overlay Window Starts", value="2020-03-01, 2022-11-01")
    ov_ends = st.sidebar.text_input("Manual Overlay Window Ends", value="2020-05-01, 2023-01-01")

    t_bars = st.sidebar.text_input("Scanning Sequence Width (Bars)", value="30")
    f_bars = st.sidebar.text_input("Forward Projection Target (Bars)", value="15")
    n_fractals = st.sidebar.text_input("Maximum Displayed Fractals", value="5")
    iso_path = st.sidebar.text_input("Focus Highlight Mode Target (Number/All/Mean)", value="all")

    c_filter = st.sidebar.selectbox("US Cycle Macro Regime Filter", ['All Cycles', 'Election Year', 'Post-Election', 'Midterm Year', 'Pre-Election'], index=0)
    spec_years = st.sidebar.text_input("Restrict Matching to Specific Calendar Years", value="all")
    g_bars = st.sidebar.text_input("Temporal Separation Buffer Width (Bars)", value="20")
    v_boost = st.sidebar.text_input("Amplitude Volatility Boost Multiplier (%)", value="0")
    s_bands = st.sidebar.text_input("Standard Deviation Pipeline Bands (+/-)", value="1.0")

    try:
        target_bars_num = int(t_bars)
        forecast_bars_num = int(f_bars)
        num_fractals_num = int(n_fractals)
        gap_bars_num = int(g_bars)
        vol_boost_pct = float(v_boost)
        std_dev_multiplier = float(s_bands)
        c_win = int(corr_window)
    except ValueError:
        st.error("⚠️ Input Parse Warning: Confirm all dynamic inputs contain clean numeric integers.")
        st.stop()

    try:
        df_target = fetch_legacy_market_data(t_asset, i_choice)
        if df_target is None or df_target.empty:
            st.error("❌ Data download error from market data backend.")
            st.stop()

        close_target = df_target['close'].dropna()
        start_clean = start_d.strip().lower()
        end_clean = end_d.strip().lower()

        if start_clean == 'latest' or end_clean == 'latest':
            target_df = close_target.iloc[-target_bars_num:]
        else:
            try:
                target_df = close_target.loc[pd.to_datetime(start_clean):pd.to_datetime(end_clean)]
                if len(target_df) == 0:
                    target_df = close_target.iloc[-target_bars_num:]
            except Exception:
                target_df = close_target.iloc[-target_bars_num:]

        target_bars_num = len(target_df)
        if target_bars_num == 0:
            st.error("Selected timeline coordinates do not contain market bar metrics.")
            st.stop()

        target_scaled = percentage_return_scale(target_df.to_numpy(dtype=np.float32))
        target_norm = target_scaled - np.mean(target_scaled)
        target_std = float(np.std(target_norm))

        st.caption(
            f"Forward Regime: **{forward_regime}** | Corr: **[{corr_min:.2f}, {corr_max:.2f}]** | "
            f"MSE: **[{mse_min:.1f}, {mse_max:.1f}]** | Forward Bars: **{forecast_bars_num}**"
        )

        plt.style.use('dark_background')
        if enable_corr:
            fig_frac, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True, gridspec_kw={'height_ratios': [2.5, 1]})
        else:
            fig_frac, ax1 = plt.subplots(1, 1, figsize=(14, 6.5))
            ax2 = None

        ax1.plot(target_scaled, color='#00ffcc', linewidth=3.5, label=f'TARGET: {t_asset} ({i_choice})', zorder=5)

        if f_mode == "Manual Compare":
            starts = [s.strip() for s in ov_starts.split(',') if s.strip()]
            ends = [e.strip() for e in ov_ends.split(',') if e.strip()]
            manual_paths_list = []
            isolate_clean = iso_path.strip().lower()

            for idx, (st_val, ed_val) in enumerate(zip(starts, ends)):
                try:
                    ov_df = close_target.loc[pd.to_datetime(st_val):pd.to_datetime(ed_val)]
                    if len(ov_df) == 0:
                        continue
                    ov_scaled = percentage_return_scale(ov_df.to_numpy(dtype=np.float32))
                    manual_paths_list.append(ov_scaled)

                    if isolate_clean == 'all' or (isolate_clean.isdigit() and int(isolate_clean) == idx + 1):
                        ax1.plot(ov_scaled, linewidth=2, linestyle='--', alpha=0.65, label=f'Manual #{idx+1} [{st_val}]')
                except Exception:
                    pass

            if manual_paths_list and isolate_clean in ['all', 'mean']:
                max_len = max(len(p) for p in manual_paths_list)
                padded_list = [np.pad(p, (0, max_len - len(p)), 'edge') if len(p) < max_len else p for p in manual_paths_list]
                manual_matrix = np.vstack(padded_list) * (1.0 + (vol_boost_pct / 100.0))
                m_mean = np.mean(manual_matrix, axis=0)
                m_std = np.std(manual_matrix, axis=0)

                if std_dev_multiplier > 0:
                    ax1.fill_between(range(len(m_mean)), m_mean - (m_std * std_dev_multiplier), m_mean + (m_std * std_dev_multiplier), color='#ff00ff', alpha=0.12)
                ax1.plot(m_mean, color='#ff00ff', linewidth=3.5, label='MANUAL COMPOSITE MEAN TRACK', zorder=6)

        else:
            if start_clean == 'latest' or end_clean == 'latest':
                history_pool = close_target.iloc[:-target_bars_num].to_numpy(dtype=np.float32)
                history_dates = close_target.index[:-target_bars_num]
            else:
                history_pool = close_target.to_numpy(dtype=np.float32)
                history_dates = close_target.index

            win_len = target_bars_num + forecast_bars_num
            max_search_index = len(history_pool) - win_len
            step_size = 5 if i_choice in {'1m', '5m', '15m'} else (2 if i_choice in {'1h', '4h'} else 1)
            parsed_years = [int(y.strip()) for y in spec_years.split(',') if y.strip() and y.lower() != 'all']

            candidates = []
            if max_search_index > 0 and target_std > 1e-12:
                for i in range(0, max_search_index, step_size):
                    hist_year = history_dates[i].year
                    if hist_year == 2026:
                        continue
                    hist_regime = get_election_regime(hist_year)
                    if c_filter != 'All Cycles' and hist_regime != c_filter:
                        continue
                    if parsed_years and hist_year not in parsed_years:
                        continue

                    window_slice = history_pool[i : i + win_len]
                    base_p = window_slice[0]
                    if base_p == 0:
                        continue

                    scaled_full = ((window_slice - base_p) / base_p) * 100.0
                    hist_scaled = scaled_full[:target_bars_num]

                    # Forward-looking price behavior AFTER the matching window ends
                    anchor_val = scaled_full[target_bars_num - 1]
                    fwd_slice = scaled_full[target_bars_num:]
                    if fwd_slice.size == 0:
                        continue
                    fwd_net_change = float(fwd_slice[-1] - anchor_val)

                    if forward_regime == "Bullish" and not (fwd_net_change > 0.0):
                        continue
                    if forward_regime == "Bearish" and not (fwd_net_change < 0.0):
                        continue

                    mse = float(np.mean((target_scaled - hist_scaled) ** 2))
                    if not (mse_min <= mse <= mse_max):
                        continue

                    h_norm = hist_scaled - np.mean(hist_scaled)
                    denom = target_std * float(np.std(h_norm))
                    corr = float(np.dot(target_norm, h_norm) / (target_bars_num * denom)) if denom > 1e-12 else 0.0
                    if not (corr_min <= corr <= corr_max):
                        continue

                    f_tag = "Bullish" if fwd_net_change > 0 else ("Bearish" if fwd_net_change < 0 else "Flat")
                    candidates.append((
                        mse, corr, i, i + target_bars_num,
                        history_dates[i].strftime('%Y-%m-%d %H:%M'), f_tag, fwd_net_change
                    ))

            if candidates:
                candidates.sort(key=lambda x: x[0])
                unique_matches = []
                for mse_val, corr_val, s_idx, e_idx, d_str, f_tag, fwd_chg in candidates:
                    if not any(max(s_idx, es - gap_bars_num) < min(e_idx, ee + gap_bars_num) for _, _, es, ee, _, _, _ in unique_matches):
                        unique_matches.append((mse_val, corr_val, s_idx, e_idx, d_str, f_tag, fwd_chg))
                    if len(unique_matches) >= num_fractals_num:
                        break

                all_scaled_paths = []
                for _, _, s_idx, _, _, _, _ in unique_matches:
                    raw_full = history_pool[s_idx : s_idx + win_len]
                    scaled_full = ((raw_full - raw_full[0]) / raw_full[0]) * 100.0 if raw_full[0] != 0 else np.zeros_like(raw_full)
                    all_scaled_paths.append(scaled_full * (1.0 + (vol_boost_pct / 100.0)))

                stacked_paths = np.vstack(all_scaled_paths)
                mean_path_array = np.mean(stacked_paths, axis=0)
                isolate_clean = iso_path.strip().lower()

                if isolate_clean != 'mean':
                    for idx, (mse_val, corr_val, _, _, d_str, f_tag, fwd_chg) in enumerate(unique_matches):
                        if isolate_clean.isdigit() and int(isolate_clean) != idx + 1:
                            continue
                        ax1.plot(
                            all_scaled_paths[idx],
                            linestyle='--',
                            alpha=0.65,
                            label=f"#{idx+1} [{f_tag} {fwd_chg:+.1f}%] r={corr_val:.2f} mse={mse_val:.1f} ({d_str})"
                        )

                if isolate_clean in ['all', 'mean']:
                    if std_dev_multiplier > 0:
                        std_path = np.std(stacked_paths, axis=0)
                        ax1.fill_between(range(len(mean_path_array)), mean_path_array - (std_path * std_dev_multiplier), mean_path_array + (std_path * std_dev_multiplier), color='#ff00ff', alpha=0.1)
                    ax1.plot(mean_path_array, color='#ff00ff', linewidth=3.5, label=f'COMPOSITE MEAN ({forward_regime})', zorder=6)
                ax1.axvline(x=target_bars_num - 1, color='#ffffff', linestyle=':', alpha=0.6, label='Matching Window End')

                if enable_corr and ax2 is not None:
                    r_wave = calculate_rolling_correlation(target_scaled, mean_path_array, c_win)
                    ax2.plot(r_wave, color='#ff00ff', linewidth=2.5, label=f"Rolling {c_win}-Bar Correlation")
                    ax2.fill_between(range(len(r_wave)), r_wave, 0, where=(np.array(r_wave) >= 0), color='#00ff88', alpha=0.15)
                    ax2.fill_between(range(len(r_wave)), r_wave, 0, where=(np.array(r_wave) < 0), color='#ff0055', alpha=0.15)
            else:
                st.warning("No fractals matched your current Forward Regime, Correlation, and MSE filters. Adjust sliders or switch Forward Regime to 'All'.")

        ax1.axhline(y=0.0, color='#555555', linestyle='-', linewidth=1.2)
        ax1.set_title("HRF MATRIX ENGINE — Model By HRF", color='#ffffff', fontsize=12, fontweight='bold')
        ax1.set_ylabel("Percentage Performance Shift (%)")
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:+.1f}%'))
        ax1.legend(loc='upper left', facecolor='#111111', edgecolor='#333333')

        if enable_corr and ax2 is not None:
            ax2.axhline(0.0, color='#ffffff', linestyle='-', linewidth=0.8, alpha=0.5)
            ax2.set_ylim(-1.05, 1.05)
            ax2.set_ylabel("Correlation R-Scale")
            ax2.grid(True, color='#222222')

        plt.tight_layout()
        st.pyplot(fig_frac)
        plt.close(fig_frac)
        gc.collect()
    except Exception as e:
        st.error(f"Ecosystem halted: {e}")

# ==============================================================================
# ENGINE MODULE 2: STRUCTURAL CAPITULATION & MATHEMATICAL OPERATIONS ENGINE
# ==============================================================================
elif app_mode == "Structural Capitulation Wave":
    st.header("⏱️ Structural Capitulation Wave & Mathematical Matrix")

    stat_asset = st.sidebar.selectbox("Asset", list(ticker_map.keys()), index=0)
    stat_interval = st.sidebar.selectbox("Time Interval", ['1m', '5m', '15m', '1h', '4h', '1day', '1 week', '1month'], index=3)

    start_time_str = st.sidebar.text_input("Time Frame Start (YYYY-MM-DD:HH:MM)", value="2026-01-01:00:00")
    end_time_str = st.sidebar.text_input("Time Frame End (YYYY-MM-DD:HH:MM)", value="2026-07-14:00:00")

    target_indicator = st.sidebar.selectbox(
        "Indicators Matrix Selection",
        ["Moving Average (Simple)", "Moving Average (Exponential)", "Moving Average (Weighted)", "RSI", "MACD", "Bollinger Band (Middle)", "Super Trend", "ATR", "VWAP"]
    )

    target_operation = st.sidebar.selectbox(
        "Mathematical Programmatic Operations",
        ["None (Just Plot Asset & Indicator)", "Add (+)", "Subtract (-)", "Multiply (*)", "Divide (/)", "Power (^)", "Square Root (√)"]
    )

    with st.sidebar.expander("⚙️ Indicator Core Settings Parameters"):
        sma_p = st.number_input("Simple Moving Average Length", value=20, min_value=1)
        ema_p = st.number_input("Exponential Moving Average Length", value=20, min_value=1)
        wma_p = st.number_input("Weighted Moving Average Length", value=20, min_value=1)
        rsi_p = st.number_input("RSI Window Lookback", value=14, min_value=1)
        macd_f = st.number_input("MACD Fast Loop EMA Length", value=12, min_value=1)
        macd_s = st.number_input("MACD Slow Loop EMA Length", value=26, min_value=1)
        bb_p = st.number_input("Bollinger Bands Simple Period", value=20, min_value=1)
        bb_std = st.number_input("Bollinger Bands Dev Multiplier", value=2.0, min_value=0.1)
        atr_p = st.number_input("ATR Volatility Bands Period", value=14, min_value=1)
        st_m = st.number_input("SuperTrend Factor Multiplier", value=3.0, min_value=0.1)

    indicator_config = {
        'sma_p': int(sma_p), 'ema_p': int(ema_p), 'wma_p': int(wma_p),
        'rsi_p': int(rsi_p), 'macd_f': int(macd_f), 'macd_s': int(macd_s),
        'bb_p': int(bb_p), 'bb_std': float(bb_std), 'atr_p': int(atr_p), 'st_m': float(st_m)
    }

    try:
        df_raw = fetch_legacy_market_data(stat_asset, stat_interval)
        if df_raw is None or df_raw.empty:
            st.error("❌ Deep data stream failed to initiate for the selected parameters.")
            st.stop()

        df_calculated = calculate_custom_indicators(df_raw.copy(), indicator_config)

        ind_key_map = {
            "Moving Average (Simple)": "SMA", "Moving Average (Exponential)": "EMA", "Moving Average (Weighted)": "WMA",
            "RSI": "RSI", "MACD": "MACD", "Bollinger Band (Middle)": "BB_Mid", "Super Trend": "SuperTrend",
            "ATR": "ATR", "VWAP": "VWAP"
        }
        selected_ind_col = ind_key_map[target_indicator]

        def parse_custom_date(dt_str):
            try:
                parts = dt_str.strip().split(':')
                date_part = parts[0]
                hour = parts[1] if len(parts) > 1 else "00"
                minute = parts[2] if len(parts) > 2 else "00"
                return pd.to_datetime(f"{date_part} {hour}:{minute}:00")
            except Exception:
                return pd.to_datetime(dt_str)

        start_dt = parse_custom_date(start_time_str)
        end_dt = parse_custom_date(end_time_str)

        df_filtered = df_calculated.loc[start_dt:end_dt].copy()
        if df_filtered.empty:
            st.warning("Date filter had no overlap with loaded bars—showing latest available bars instead.")
            df_filtered = df_calculated.iloc[-min(500, len(df_calculated)):].copy()

        asset_val = df_filtered['close']
        ind_val = df_filtered[selected_ind_col]

        op_title = "Raw Plot Analysis"
        output_curve = asset_val.copy()

        if target_operation == "Add (+)":
            output_curve = asset_val + ind_val
            op_title = f"Sum Layer: [Asset + {selected_ind_col}]"
        elif target_operation == "Subtract (-)":
            output_curve = asset_val - ind_val
            op_title = f"Spread Layer: [Asset - {selected_ind_col}]"
        elif target_operation == "Multiply (*)":
            output_curve = asset_val * ind_val
            op_title = f"Product Matrix: [Asset * {selected_ind_col}]"
        elif target_operation == "Divide (/)":
            output_curve = asset_val / (ind_val + 1e-10)
            op_title = f"Ratio Signal Shift: [Asset / {selected_ind_col}]"
        elif target_operation == "Power (^)":
            output_curve = asset_val ** ind_val
            op_title = f"Exponential Growth: [Asset ^ {selected_ind_col}]"
        elif target_operation == "Square Root (√)":
            output_curve = np.sqrt(asset_val.abs())
            op_title = "Absolute Asset Vector Square Root (√)"

        plt.style.use('dark_background')

        if target_operation == "None (Just Plot Asset & Indicator)":
            fig, ax1 = plt.subplots(1, 1, figsize=(14, 6))
            ax1.plot(df_filtered.index, asset_val, color='#9400d3', linewidth=2.5, label=f"Asset Close ({stat_asset})")
            ax1.plot(df_filtered.index, ind_val, color='#ff007f', linewidth=2.0, linestyle="--", label=f"Indicator ({target_indicator})")
            ax1.set_title("STRUCTURAL VISUAL METRIC MATRIX — Model By HRF", color='#ff007f', fontweight='bold', fontsize=12)
            ax1.grid(True, color='#222222', alpha=0.5)
            ax1.legend(loc='upper left', facecolor='#111111')
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1.5]})
            ax1.plot(df_filtered.index, asset_val, color='#9400d3', linewidth=2.5, label=f"Asset Close ({stat_asset})")
            ax1.plot(df_filtered.index, ind_val, color='#ffffff', linewidth=1.5, linestyle=":", label=f"Indicator ({target_indicator})")
            ax1.set_title("STRUCTURAL CAPITULATION WAVE COMPONENTS", color='#ff007f', fontweight='bold', fontsize=11)
            ax1.grid(True, color='#222222', alpha=0.4)
            ax1.legend(loc='upper left', facecolor='#111111')

            ax2.plot(df_filtered.index, output_curve, color='#ff0055', linewidth=3.0, label=op_title)
            ax2.fill_between(df_filtered.index, output_curve, output_curve.mean(), color='#9400d3', alpha=0.12)
            ax2.set_title("ALGEBRAIC OUTPUT COEFFICIENT WAVE", color='#ffffff', fontweight='bold', fontsize=10)
            ax2.grid(True, color='#222222', alpha=0.4)
            ax2.legend(loc='upper left', facecolor='#111111')

        plt.tight_layout(pad=2.0)
        st.pyplot(fig)
        plt.close(fig)

        st.subheader(f"📊 Matrix Metrics Sheet ({stat_asset} Engine Processing)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Asset Execution Price", f"${asset_val.iloc[-1]:,.2f}")
        c2.metric(f"Active {selected_ind_col} Terminal", f"{ind_val.iloc[-1]:,.4f}")
        c3.metric("Final Transformed Yield Value", f"{output_curve.iloc[-1]:,.4f}")
        gc.collect()

    except Exception as ex:
        st.error(f"Execution Error inside Statistical Pipeline Horizon: {ex}")

# ==============================================================================
# ENGINE MODULE 3: STATISTICAL FRAMEWORK CORE (PATH RESET DRAWDOWN)
# ==============================================================================
elif app_mode == "Statistical Framework Core":
    st.header("📈 Path-Dependent Statistical Framework Engine")

    stat_asset = st.sidebar.selectbox("Select Target Framework Asset", list(ticker_map.keys()), index=0)
    stat_interval = st.sidebar.selectbox("Framework Time Interval", ['1m', '5m', '15m', '1h', '4h', '1day', '1 week', '1month'], index=5)

    start_time_str = st.sidebar.text_input("Frame Start (YYYY-MM-DD)", value="2025-01-01")
    end_time_str = st.sidebar.text_input("Frame End (YYYY-MM-DD)", value="2026-07-14")

    try:
        df_raw = fetch_legacy_market_data(stat_asset, stat_interval)
        if df_raw is None or df_raw.empty:
            st.error("❌ Deep data stream failed to initiate for the statistical framework.")
            st.stop()

        start_dt = pd.to_datetime(start_time_str)
        end_dt = pd.to_datetime(end_time_str)

        df_filtered = df_raw.loc[start_dt:end_dt].copy()
        if df_filtered.empty:
            st.warning("Date filter had no overlap with loaded bars—showing latest available bars instead.")
            df_filtered = df_raw.iloc[-min(500, len(df_raw)):].copy()

        df_filtered['HRF_Drawdown'] = calculate_hrf_path_drawdown(df_filtered['close'])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_filtered.index,
            y=df_filtered['close'],
            mode='lines',
            name=f'{stat_asset} Price',
            line=dict(color='#8A2BE2', width=2.5)
        ))
        fig.add_trace(go.Scatter(
            x=df_filtered.index,
            y=df_filtered['HRF_Drawdown'] * 100,
            mode='lines',
            name='Path-Dependent Drawdown (%)',
            line=dict(color='#FF0055', width=1.5),
            yaxis='y2'
        ))
        fig.update_layout(
            title=f"<b>{stat_asset} Framework - Model By HRF</b>",
            xaxis=dict(title="Timeline Series", gridcolor='#222222'),
            yaxis=dict(title="Asset Price ($)", side="left", gridcolor='#222222'),
            yaxis2=dict(title="Conditional Drawdown (%)", side="right", overlaying="y", showgrid=False, range=[-100, 5]),
            template="plotly_dark",
            paper_bgcolor='#0d0d11',
            plot_bgcolor='#0d0d11',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📊 Path Analysis Metrics")
        m1, m2 = st.columns(2)
        m1.metric("Current Asset Close", f"${df_filtered['close'].iloc[-1]:,.2f}")
        m2.metric("Max Trapped Path Drawdown", f"{df_filtered['HRF_Drawdown'].min() * 100:.2f}%")
        gc.collect()

    except Exception as ex:
        st.error(f"Ecosystem failed inside statistical block calculations: {ex}")

st.divider()
st.markdown("<p style='text-align: center; color: #555555;'>--- Model By HRF ---</p>", unsafe_allow_html=True)
