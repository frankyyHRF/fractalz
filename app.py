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
# DATA PIPELINE (KRAKEN PUBLIC OHLC + FULL PARQUET / CSV RESAMPLER)
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
        resp = session.get(url, headers=headers, stream=True, timeout=120)
        resp.raise_for_status()
        ct = resp.headers.get("Content-Type", "")
        if "text/html" in ct:
            html_text = resp.text
            action_match = re.search(r'action="([^"]+)"', html_text)
            inputs = dict(re.findall(r'name="([^"]+)"\s+value="([^"]*)"', html_text))
            if action_match and inputs:
                action_url = action_match.group(1).replace("&amp;", "&")
                resp = session.get(action_url, params=inputs, headers=headers, stream=True, timeout=120)
                resp.raise_for_status()
                ct = resp.headers.get("Content-Type", "")
        if "text/html" in ct:
            raise ValueError("Google Drive returned HTML instead of data file. Check link sharing.")
        return resp

    resp = session.get(u, headers=headers, stream=True, timeout=120)
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

@st.cache_data(ttl=1800, max_entries=2, show_spinner="Loading & resampling full remote dataset...")
def _load_and_resample_remote_file(raw_url, interval_str):
    rule_map = {'1m': '1min', '5m': '5min', '15m': '15min', '1h': '1h', '4h': '4h'}
    rule = rule_map.get(interval_str)
    if not rule:
        return None

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
        )
        del cleaned
        gc.collect()
        return resampled[['open', 'high', 'low', 'close', 'volume']]

    buf = io.BytesIO(content_bytes)
    del content_bytes
    gc.
