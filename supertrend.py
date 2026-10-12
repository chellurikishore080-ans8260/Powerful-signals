"""
supertrend.py - Supertrend(10,3) WICK-TOUCH scanner (second dashboard of POWERFUL SIGNALS).

Rule set
  * Supertrend(period, multiplier) per timeframe (Wilder ATR, same band logic as TradingView ta.supertrend).
  * WICK TOUCH  : uptrend (line below price)  -> candle LOW  <= line   (pull-back wick to support)
                  downtrend (line above price)-> candle HIGH >= line   (rally wick to resistance)
                  The candle still closes on the trend side (a close through the line is a trend FLIP, not a touch).
  * Evaluated on the LIVE (still forming) candle + the last `look` closed candles -> alert the moment a wick touches.
  * FRESH    : first touch of the current trend run (since the last flip)  -> top priority.
  * RETOUCH  : a later touch, allowed again only after >= N candles since the previous touch
               (N configurable per timeframe, default 5). Touches inside the cool-down are suppressed.
Educational decision-support tool, not financial advice.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import feeds

TIMEFRAMES = ["1m", "3m", "5m", "15m", "30m", "1h", "4h"]
DEFAULT_COOLDOWN = {tf: 5 for tf in TIMEFRAMES}


# ----------------------------------------------------------------------------------------
# Logic (pure)
# ----------------------------------------------------------------------------------------
def supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0) -> pd.DataFrame:
    """Adds `st` (line) and `dir` (+1 uptrend / -1 downtrend). Rows before the ATR warm-up have st = NaN."""
    h, l, c = df["high"].values.astype(float), df["low"].values.astype(float), df["close"].values.astype(float)
    n = len(df)
    pc = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
    atr = pd.Series(tr).ewm(alpha=1 / period, adjust=False, min_periods=period).mean().values
    hl2 = (h + l) / 2
    up, lo = hl2 + mult * atr, hl2 - mult * atr
    fu, fl = np.full(n, np.nan), np.full(n, np.nan)
    st, dr = np.full(n, np.nan), np.zeros(n, dtype=int)
    valid = np.where(~np.isnan(atr))[0]
    if len(valid):
        s = int(valid[0])
        fu[s], fl[s] = up[s], lo[s]
        dr[s] = 1 if c[s] > hl2[s] else -1
        st[s] = fl[s] if dr[s] == 1 else fu[s]
        for i in range(s + 1, n):
            fu[i] = up[i] if (up[i] < fu[i - 1] or c[i - 1] > fu[i - 1]) else fu[i - 1]
            fl[i] = lo[i] if (lo[i] > fl[i - 1] or c[i - 1] < fl[i - 1]) else fl[i - 1]
            if dr[i - 1] == -1:
                dr[i] = 1 if c[i] > fu[i] else -1
            else:
                dr[i] = -1 if c[i] < fl[i] else 1
            st[i] = fl[i] if dr[i] == 1 else fu[i]
    out = df.copy().reset_index(drop=True)
    out["st"], out["dir"] = st, dr
    return out


def detect_touch(d: pd.DataFrame, cooldown_n: int, tol_pct: float = 0.0, look: int = 1):
    """
    d needs columns high, low, st, dir (rows with st NaN are ignored). Returns the newest actionable touch among the
    live candle and `look` previous candles, or None. status: FRESH | RETOUCH (suppressed touches are skipped).
    """
    n = len(d)
    if n < 3:
        return None
    hi, lo = d["high"].values, d["low"].values
    st, dr = d["st"].values, d["dir"].values
    tol = tol_pct / 100.0
    valid = ~np.isnan(st)
    touch = np.zeros(n, dtype=bool)
    up = valid & (dr == 1)
    dn = valid & (dr == -1)
    touch[up] = lo[up] <= st[up] * (1 + tol)
    touch[dn] = hi[dn] >= st[dn] * (1 - tol)
    seg = np.r_[0, np.cumsum(dr[1:] != dr[:-1])]                       # trend-run id, +1 on every flip
    for i in range(n - 1, max(n - 2 - look, -1), -1):
        if not touch[i]:
            continue
        same = np.where(touch[:i] & (seg[:i] == seg[i]))[0]
        prev = int(same[-1]) if len(same) else None
        if prev is None:
            status = "FRESH"
        elif i - prev >= cooldown_n:
            status = "RETOUCH"
        else:
            continue                                                   # inside the cool-down -> suppressed
        start = int(np.where(seg == seg[i])[0][0])
        return {"idx": i, "status": status, "dir": int(dr[i]), "line": float(st[i]), "candles_ago": n - 1 - i,
                "live": i == n - 1, "prev_touch_gap": None if prev is None else i - prev,
                "touches_in_trend": int((touch & (seg == seg[i]))[: i + 1].sum()), "trend_age": i - start}
    return None


# ----------------------------------------------------------------------------------------
# Scan
# ----------------------------------------------------------------------------------------
def scan_one(ex, symbol: str, tf: str, period: int, mult: float, cooldown_n: int, tol_pct: float, look: int):
    raw = feeds.fetch_candles(ex, symbol, tf, 200)                    # last row = live (forming) candle
    if raw is None:
        return None
    d = supertrend(raw, period, mult)
    d = d[d["st"].notna()].reset_index(drop=True)
    hit = detect_touch(d, cooldown_n, tol_pct, look)
    if not hit:
        return None
    last = float(d["close"].iloc[-1])
    return {**hit, "exchange": ex.id, "symbol": symbol, "tf": tf, "price": last,
            "dist_pct": abs(last - hit["line"]) / last * 100,
            "touch_time": d["datetime"].iloc[hit["idx"]], "candle_ts": int(d["timestamp"].iloc[hit["idx"]]),
            "_df": d.tail(150).reset_index(drop=True)}


def run_scan(ex, symbols: list[str], tfs: list[str], period: int, mult: float, cooldowns: dict, tol_pct: float,
             look: int, workers: int = 6):
    hits, errors = [], []
    jobs = [(s, tf) for s in symbols for tf in tfs]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(scan_one, ex, s, tf, period, mult, cooldowns.get(tf, 5), tol_pct, look): (s, tf)
                for s, tf in jobs}
        for fut in as_completed(futs):
            try:
                r = fut.result()
                if r:
                    hits.append(r)
            except Exception as e:                                        # noqa: BLE001
                errors.append({"Exchange": ex.id, "Symbol": futs[fut][0], "Error": f"{futs[fut][1]}: {str(e)[:140]}"})
    hits.sort(key=lambda r: (r["status"] != "FRESH", r["candles_ago"], r["dist_pct"]))
    return hits, errors


def to_row(r: dict) -> dict:
    up = r["dir"] == 1
    return {
        "Priority": "🔥 FRESH" if r["status"] == "FRESH" else "↻ RETOUCH",
        "Symbol": r["symbol"], "TF": r["tf"],
        "Zone": "Support (uptrend, wick down to line)" if up else "Resistance (downtrend, wick up to line)",
        "Candle": "LIVE" if r["live"] else f'closed {r["candles_ago"]} ago',
        "Price": r["price"], "Supertrend": r["line"], "Dist %": round(r["dist_pct"], 3),
        "Touch # in trend": r["touches_in_trend"],
        "Candles since prev touch": r["prev_touch_gap"] if r["prev_touch_gap"] is not None else "—",
        "Trend age (candles)": r["trend_age"],
        "Touch candle (UTC)": r["touch_time"].strftime("%m-%d %H:%M"),
    }


# ----------------------------------------------------------------------------------------
# Streamlit page
# ----------------------------------------------------------------------------------------
def page(get_exchange, friendly_error):
    import plotly.graph_objects as go
    import streamlit as st

    try:
        from streamlit_autorefresh import st_autorefresh
    except Exception:                                                 # noqa: BLE001
        st_autorefresh = None

    with st.sidebar:
        st.header("Supertrend wick-touch")
        venue = st.selectbox("Exchange", feeds.EXCHANGE_IDS, index=0, format_func=lambda x: feeds.EXCHANGE_LABELS[x],
                             key="st_venue")
        st.caption("Only candles are needed, so every venue works (KuCoin too). If Binance futures is blocked here, "
                   "it falls back to the Binance data mirror, then OKX.")
        tfs = st.multiselect("Timeframes", TIMEFRAMES, default=["15m"], key="st_tfs")
        period = st.number_input("ATR period", 2, 50, 10, key="st_period")
        mult = st.number_input("Multiplier", 0.5, 10.0, 3.0, step=0.5, key="st_mult")
        tol = st.number_input("Touch tolerance % (0 = wick must reach the line)", 0.0, 1.0, 0.0, step=0.01, key="st_tol")
        look = st.slider("Also check last N closed candles", 0, 3, 1, key="st_look")
        with st.expander("Cool-down N per timeframe (candles)"):
            st.caption("A coin that touched before is eligible again only after N candles since its last touch.")
            cds = {tf: st.number_input(f"{tf}", 1, 50, DEFAULT_COOLDOWN[tf], key=f"st_cd_{tf}") for tf in tfs}
        max_symbols = st.slider("Max symbols (by 24h volume)", 20, 300, 100, step=20, key="st_maxsym")
        min_qv = st.number_input("Min 24h quote volume (USD)", min_value=0, value=5_000_000, step=1_000_000, key="st_minqv")
        refresh = st.slider("Auto refresh (seconds)", 15, 300, 30, step=15, key="st_refresh")
        go_btn = st.button("🔎 Scan now", type="primary", width="stretch", key="st_go")

    st.markdown("# 📈 Supertrend wick-touch")
    st.markdown(f"##### Supertrend({int(period)}, {mult:g}) · alert the moment a wick touches the line · "
                "FRESH touches first")
    st.caption("Public candle data only • No orders • Screening tool, not a trade instruction.")
    if st_autorefresh is not None:
        st_autorefresh(interval=refresh * 1000, key="st_autorefresh")

    if "st_last" not in st.session_state:
        st.session_state.st_last = 0.0
    if tfs and (go_btn or time.time() - st.session_state.st_last > refresh):
        st.session_state.st_last = time.time()
        status = st.status("Scanning…", expanded=True)
        queue, note, ex, syms, errors = [venue], None, None, [], []
        qi = 0
        while qi < len(queue) and ex is None:
            exid = queue[qi]
            qi += 1
            try:
                cand = get_exchange(exid)
                allsyms = feeds.linear_usdt_swaps(cand)
                try:
                    tk = cand.fetch_tickers()
                    ranked = sorted([(s, (tk.get(s) or {}).get("quoteVolume") or 0) for s in allsyms],
                                    key=lambda z: z[1], reverse=True)
                    allsyms = [s for s, q in ranked if q >= min_qv][:max_symbols]
                except Exception as e:                                    # noqa: BLE001
                    if feeds.is_geo_block(e):
                        raise
                    allsyms = allsyms[:max_symbols]
                ex, syms = cand, allsyms
            except Exception as e:                                        # noqa: BLE001
                nxt = feeds.next_fallback(exid, queue, e)
                if nxt:
                    queue.append(nxt)
                    note = ((note + " ") if note else "") + (
                        f"{feeds.EXCHANGE_LABELS[exid]} unavailable from this server "
                        f"({'blocked 451' if feeds.is_geo_block(e) else str(e)[:50]}). "
                        f"Using {feeds.EXCHANGE_LABELS[nxt]}.")
                else:
                    errors.append({"Exchange": exid, "Symbol": "—", "Error": friendly_error(e)})
        hits = []
        if ex is not None:
            status.write(f"{feeds.EXCHANGE_LABELS[ex.id]}: {len(syms)} symbols × {len(tfs)} timeframe(s)…")
            hits, errs = run_scan(ex, syms, tfs, int(period), float(mult), cds, float(tol), int(look))
            errors += errs
        book = st.session_state.get("st_book", {})
        new_fresh = []
        for r in hits:
            key = (r["exchange"], r["symbol"], r["tf"], r["candle_ts"], r["dir"])
            r["is_new"] = key not in book
            book.setdefault(key, datetime.now(timezone.utc).strftime("%H:%M:%S UTC"))
            if r["is_new"] and r["status"] == "FRESH":
                new_fresh.append(r)
        for r in new_fresh[:5]:
            st.toast(f'🔥 FRESH touch: {r["symbol"]} {r["tf"]} ({"support" if r["dir"] == 1 else "resistance"})')
        st.session_state.update(st_book=book, st_hits=hits, st_errors=errors, st_note=note,
                                st_scanned=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                                st_universe=len(syms))
        status.update(label=f"Scan complete · {len(hits)} touches", state="complete", expanded=False)

    hits = st.session_state.get("st_hits", [])
    if st.session_state.get("st_note"):
        st.warning(st.session_state["st_note"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Symbols scanned", st.session_state.get("st_universe", 0))
    c2.metric("🔥 Fresh touches", sum(r["status"] == "FRESH" for r in hits))
    c3.metric("↻ Re-touches", sum(r["status"] == "RETOUCH" for r in hits))
    if st.session_state.get("st_scanned"):
        st.caption(f'Last scan: {st.session_state.st_scanned}')

    if hits:
        rows = []
        for r in hits:
            row = to_row(r)
            row["NEW"] = "🆕" if r.get("is_new") else ""
            rows.append(row)
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, width="stretch",
                     column_config={"Price": st.column_config.NumberColumn(format="%.6g"),
                                    "Supertrend": st.column_config.NumberColumn(format="%.6g")})
        st.download_button("Download CSV", df.to_csv(index=False).encode(), file_name="supertrend_touches.csv",
                           mime="text/csv")
        labels = [f'{r["symbol"]} | {r["tf"]} | {r["status"]}' for r in hits]
        pick = st.selectbox("Chart", labels, key="st_pick")
        r = hits[labels.index(pick)]
        d = r["_df"]
        fig = go.Figure(go.Candlestick(x=d["datetime"], open=d["open"], high=d["high"], low=d["low"], close=d["close"],
                                       name="Price"))
        fig.add_trace(go.Scatter(x=d["datetime"], y=d["st"].where(d["dir"] == 1), name="Supertrend ↑",
                                 line=dict(color="#2e9e5b", width=2)))
        fig.add_trace(go.Scatter(x=d["datetime"], y=d["st"].where(d["dir"] == -1), name="Supertrend ↓",
                                 line=dict(color="#d0021b", width=2)))
        t = d.iloc[len(d) - 1 - r["candles_ago"]]
        fig.add_trace(go.Scatter(x=[t["datetime"]], y=[t["low"] if r["dir"] == 1 else t["high"]], mode="markers",
                                 marker=dict(size=13, color="#f5a623", symbol="diamond"), name="Wick touch"))
        fig.update_layout(height=520, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=40, b=10),
                          title=f'{r["symbol"]} · {r["tf"]} · {r["status"]}', legend=dict(orientation="h"))
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No wick touches right now. The app re-scans automatically.")

    if st.session_state.get("st_errors"):
        with st.expander(f'Warnings ({len(st.session_state.st_errors)})'):
            st.dataframe(pd.DataFrame(st.session_state.st_errors), hide_index=True, width="stretch")

    st.markdown("---")
    st.markdown("""
**How to read it**
- 🔥 **FRESH** = first wick touch since the trend flipped (highest priority). ↻ **RETOUCH** = touched again, allowed only after
  N candles since the previous touch (set per timeframe in the sidebar).
- *Support* = uptrend and a wick dipped to the line; *Resistance* = downtrend and a wick rallied to the line.
- **LIVE** = the candle is still forming (the touch can still turn into a flip if it closes through the line).
- Binance data mirror = Binance **spot** candles (close to futures prices). Heuristic screening, not a proven edge.
""")
