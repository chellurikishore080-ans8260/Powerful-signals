"""
POWERFUL SIGNALS v2 - Liquidity Sweep + Real Footprint + True CVD (Binance USDⓈ-M first)
Educational decision-support tool, not financial advice. Public market data only - no API keys, no orders.
"""
import time
from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import feeds
import orderflow as of
import scanner

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:                                             # app still works without auto-refresh
    st_autorefresh = None

st.set_page_config(page_title="POWERFUL SIGNALS v2 | Sweep + Footprint", page_icon="⚡", layout="wide")


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


@st.cache_resource(ttl=3600, show_spinner=False)
def get_exchange(exchange_id: str):
    ex = feeds.make_exchange(exchange_id)
    ex.load_markets()
    return ex


def friendly_error(e: Exception) -> str:
    t = str(e)
    if "451" in t or "restricted location" in t.lower() or "eligibility" in t.lower():
        return ("Binance blocked this server's location (HTTP 451). Streamlit Community Cloud runs in the US, "
                "which Binance futures blocks. Run the app locally / on a VPS outside the US, or deploy on a host in "
                "Asia/EU.")
    return t[:200]


# ------------------------------------------------------------------------------------ dashboard switch
with st.sidebar:
    mode = st.radio("Dashboard", ["⚡ Sweep + Footprint", "📈 Supertrend wick-touch"], key="mode")

# ------------------------------------------------------------------------------------ sidebar
with st.sidebar:
    if st.button("🔌 Test connections", width="stretch"):
        with st.spinner("Pinging venues from this server…"):
            st.session_state["conn_test"] = feeds.test_connections()
    if st.session_state.get("conn_test"):
        st.dataframe(pd.DataFrame(st.session_state["conn_test"]), hide_index=True, width="stretch")
    if mode.startswith("📈"):
        import supertrend
        supertrend.page(get_exchange, friendly_error)
        st.stop()
    st.header("Scanner settings")
    selected = st.multiselect("Exchanges (Binance = main)", feeds.EXCHANGE_IDS, default=[feeds.PRIMARY],
                              format_func=lambda x: feeds.EXCHANGE_LABELS[x])
    st.caption("If Binance is blocked on this server (e.g. US cloud hosts), the app switches to OKX automatically.")
    if feeds.MIRROR in selected:
        st.caption("Mirror = Binance public data host (spot). Real taker-buy delta + aggTrades footprint, works from cloud.")
    if "okx" in selected:
        st.caption("OKX: real footprint (history-trades), but no taker-buy candle delta → CVD divergence unavailable.")
    if "kucoinfutures" in selected:
        st.caption("KuCoin: public API has no historical trades → no footprint; signals max out at SETUP unless you "
                   "untick 'CONFIRMED requires real footprint'.")
    tf = st.selectbox("Sweep timeframe", ["1m", "3m", "5m", "15m"], index=2)
    st.caption(f"HTF bias timeframe: **{feeds.HTF_FOR[tf]}**")
    max_symbols = st.slider("Max symbols per exchange (by 24h volume)", 10, 150, 50, step=10)
    min_qv = st.number_input("Min 24h quote volume (USD)", min_value=0, value=20_000_000, step=5_000_000)
    n_cand = st.slider("Footprint deep-scan: top N setups", 1, 20, 8)
    max_pages = st.slider("Max trade pages per footprint (1000 trades each)", 2, 20, 8)
    refresh = st.slider("Auto refresh (seconds)", 30, 300, 60, step=15)
    with st.expander("Strategy parameters"):
        sweep_k = st.slider("Reclaim must be within last N candles", 1, 6, 3)
        max_age = st.slider("Max candles since reclaim", 0, 5, 2)
        min_score = st.slider("Min score for CONFIRMED", 40, 95, 65)
        min_rr = st.slider("Min risk:reward", 1.0, 4.0, 1.5, step=0.25)
        min_hits = st.slider("Order-flow confirmations needed (of 4)", 1, 4, 2)
        imb_ratio = st.slider("Imbalance ratio", 2.0, 6.0, 3.0, step=0.5)
        min_stack = st.slider("Stacked imbalance levels", 2, 5, 3)
        req_fp = st.checkbox("CONFIRMED requires real footprint", value=True)
    scan_button = st.button("🔎 Scan now", type="primary", width="stretch")
    st.markdown("---")
    st.markdown("**Rule:** sweep + reclaim, then ≥N order-flow confirmations (absorption, stacked imbalance, "
                "true CVD divergence, reclaim delta). CVD divergence alone is never an entry.")

cfg = of.Config(sweep_last_k=sweep_k, max_age=max_age, min_score=min_score, min_rr=min_rr, min_of_hits=min_hits,
                imbalance_ratio=imb_ratio, min_stack=min_stack, require_footprint=req_fp)

st.markdown("# ⚡ POWERFUL SIGNALS v2")
st.markdown("##### Liquidity sweep · Real footprint (bid/ask traded volume) · True CVD · Binance Futures first")
st.caption("Public market data only • No orders • Signals are rule-based candidates, not guaranteed calls.")
if st_autorefresh is not None:
    st_autorefresh(interval=refresh * 1000, key="ps_autorefresh")

# ------------------------------------------------------------------------------------ scan
if "last_scan" not in st.session_state:
    st.session_state.last_scan = 0.0
if selected and (scan_button or time.time() - st.session_state.last_scan > refresh):
    st.session_state.last_scan = time.time()
    records, errors, universe = [], [], {}
    to_scan, fallback_note = list(selected), None
    status = st.status("Scanning…", expanded=True)
    qi = 0
    while qi < len(to_scan):
        exid = to_scan[qi]
        qi += 1
        try:
            ex = get_exchange(exid)
            syms = feeds.linear_usdt_swaps(ex)
            ranked = []
            try:
                tickers = ex.fetch_tickers()
                for s in syms:
                    qv = (tickers.get(s) or {}).get("quoteVolume")
                    if qv is not None and qv >= min_qv:
                        ranked.append((s, qv))
                ranked.sort(key=lambda z: z[1], reverse=True)
                syms = [s for s, _ in ranked[:max_symbols]]
            except Exception as e:                                        # noqa: BLE001
                if feeds.is_geo_block(e):
                    raise
                syms = syms[:max_symbols]
            universe[exid] = syms
            status.write(f"{feeds.EXCHANGE_LABELS[exid]}: stage 1 on {len(syms)} perpetuals…")
            recs, errs = scanner.run_scan(ex, syms, tf, cfg, n_cand, max_pages)
            records += recs
            errors += errs
        except Exception as e:                                            # noqa: BLE001
            nxt = feeds.next_fallback(exid, to_scan, e)
            if nxt:
                to_scan.append(nxt)
                why = ("blocked from this server's location (HTTP 451 - typical for US hosts like Streamlit Cloud)"
                       if feeds.is_geo_block(e) else f"unavailable ({str(e)[:60]})")
                fallback_note = (fallback_note + " " if fallback_note else "") + (
                    f"{feeds.EXCHANGE_LABELS[exid]} {why}. Switched automatically to {feeds.EXCHANGE_LABELS[nxt]}. "
                    + ("Mirror = Binance SPOT data (close to futures prices). " if nxt == feeds.MIRROR else "")
                    + ("For true futures data run locally / on a non-US host." if exid == feeds.PRIMARY else ""))
            else:
                errors.append({"Exchange": exid, "Symbol": "—", "Error": friendly_error(e)})
    book = st.session_state.get("signal_book", {})
    for r in records:
        s = r["setup"]
        if s and s["state"] == "CONFIRMED":
            key = (r["exchange"], r["symbol"], tf, s["direction"], str(s["sweep_time"]))
            r["is_new"] = key not in book
            book.setdefault(key, utc_now())
            r["first_seen"] = book[key]
    st.session_state.update(signal_book=book, records=records, errors=errors, universe=universe,
                            scan_tf=tf, scanned_at=utc_now(), fallback_note=fallback_note)
    status.update(label=f"Scan complete · {len(records)} markets analysed", state="complete", expanded=False)

records = st.session_state.get("records", [])
errors = st.session_state.get("errors", [])
if st.session_state.get("fallback_note"):
    st.warning(st.session_state["fallback_note"])
scan_tf = st.session_state.get("scan_tf", tf)


# ------------------------------------------------------------------------------------ charts
def price_chart(rec):
    s, df = rec["setup"], rec["_df"]
    d = df.tail(90).reset_index(drop=True)
    off = len(df) - len(d)
    real = of.has_real_delta(df)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.04)
    fig.add_trace(go.Candlestick(x=d["datetime"], open=d["open"], high=d["high"], low=d["low"], close=d["close"],
                                 name="Price"), row=1, col=1)
    x0, x1 = d["datetime"].iloc[0], d["datetime"].iloc[-1]
    fig.add_shape(type="line", x0=x0, x1=x1, y0=s["level"], y1=s["level"], line=dict(color="#f5a623", dash="dash"),
                  row=1, col=1)
    for name, price, color in (("Entry", s["entry"], "#4a90d9"), ("Stop", s["stop"], "#d0021b"),
                               ("Target", s["target"], "#2e9e5b")):
        fig.add_shape(type="line", x0=x0, x1=x1, y0=price, y1=price, line=dict(color=color, width=1), row=1, col=1)
        fig.add_annotation(x=x1, y=price, text=name, showarrow=False, xanchor="left", font=dict(color=color, size=11),
                           row=1, col=1)
    j = s["sweep_idx"] - off
    if 0 <= j < len(d):
        fig.add_vrect(x0=d["datetime"].iloc[j] - pd.Timedelta(seconds=feeds.TF_MS[rec["tf"]] / 2000),
                      x1=d["datetime"].iloc[j] + pd.Timedelta(seconds=feeds.TF_MS[rec["tf"]] / 2000),
                      fillcolor="rgba(245,166,35,0.18)", line_width=0, row=1, col=1)
    if real:
        colors = ["#2e9e5b" if v >= 0 else "#d0021b" for v in d["delta"]]
        fig.add_trace(go.Bar(x=d["datetime"], y=d["delta"], marker_color=colors, name="Delta (real)"), row=2, col=1)
        cvd = d["cvd"] - d["cvd"].iloc[0]
        fig.add_trace(go.Scatter(x=d["datetime"], y=cvd, name="CVD (window)", line=dict(color="#4a90d9")),
                      row=2, col=1)
    else:
        fig.add_annotation(text="Candle delta unavailable on this venue — use the footprint", row=2, col=1,
                           xref="x domain", yref="y domain", x=0.5, y=0.5, showarrow=False)
    fig.update_layout(height=620, xaxis_rangeslider_visible=False, margin=dict(l=10, r=60, t=40, b=10),
                      title=f'{rec["symbol"]} · {rec["tf"]} · {s["direction"]} idea · sweep zone shaded',
                      legend=dict(orientation="h"))
    return fig


def footprint_chart(fpm, level=None, title=""):
    f = fpm["fp"]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=f["price"], x=-f["sell"], orientation="h", name="Aggressive SELL (hit bid)",
                         marker_color=["#ff6b6b" if i else "#b83a3a" for i in f["sell_imb"]]))
    fig.add_trace(go.Bar(y=f["price"], x=f["buy"], orientation="h", name="Aggressive BUY (lift ask)",
                         marker_color=["#5de08b" if i else "#2e9e5b" for i in f["buy_imb"]]))
    fig.add_hline(y=fpm["poc"], line=dict(color="#4a90d9", dash="dot"), annotation_text="POC")
    if level is not None:
        fig.add_hline(y=level, line=dict(color="#f5a623", dash="dash"), annotation_text="swept level")
    fig.update_layout(barmode="overlay", height=560, title=title, bargap=0.05,
                      yaxis=dict(tickformat=".8g", title="price"), xaxis=dict(title="volume (sell ◄ | ► buy)"),
                      margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h"))
    return fig


def fp_table(fpm):
    f = fpm["fp"].copy()
    f["Flag"] = ""
    f.loc[f["buy_imb"], "Flag"] += "▲ buy-imb "
    f.loc[f["sell_imb"], "Flag"] += "▼ sell-imb "
    f.loc[f["price"] == fpm["poc"], "Flag"] += "POC"
    return f.sort_values("price", ascending=False).rename(
        columns={"price": "Price", "buy": "Buy (ask lifted)", "sell": "Sell (bid hit)", "delta": "Delta",
                 "total": "Total"})[["Price", "Sell (bid hit)", "Buy (ask lifted)", "Delta", "Total", "Flag"]]


def tick(v):
    return "—" if v is None else ("✅" if v else "❌")


# ------------------------------------------------------------------------------------ results
with_setup = [r for r in records if r.get("setup")]
confirmed = [r for r in with_setup if r["setup"]["state"] == "CONFIRMED"]
others = [r for r in with_setup if r["setup"]["state"] not in ("CONFIRMED", "INVALID")]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Markets scanned", len(records))
c2.metric("Sweeps + reclaim", len(with_setup))
c3.metric("✅ Confirmed", len(confirmed))
c4.metric("👀 Setups / watch", len(others))
if st.session_state.get("scanned_at"):
    st.caption(f'Last scan: {st.session_state.scanned_at} · timeframe {scan_tf}')

PRICE_COLS = {c: st.column_config.NumberColumn(format="%.6g") for c in
              ("Swept level", "Entry", "Stop", "Target")}

st.markdown("## 🚨 CONFIRMED signals")
if confirmed:
    rows = []
    for r in sorted(confirmed, key=lambda r: r["setup"]["score"], reverse=True):
        row = scanner.to_row(r)
        row["Status"] = "NEW" if r.get("is_new") else "ACTIVE"
        row["First detected"] = r.get("first_seen", "")
        rows.append(row)
    cdf = pd.DataFrame(rows)
    st.dataframe(cdf, width="stretch", hide_index=True, column_config=PRICE_COLS)
    st.download_button("Download CSV", cdf.to_csv(index=False).encode(), file_name="powerful_signals_v2.csv",
                       mime="text/csv")
else:
    st.info("No setup passed every rule (sweep → reclaim → order-flow → score → RR). Waiting is a valid result.")

st.markdown("## 👀 Setups & watchlist")
if others:
    odf = pd.DataFrame([scanner.to_row(r) for r in sorted(others, key=lambda r: r["setup"]["score"], reverse=True)])
    st.dataframe(odf, width="stretch", hide_index=True, column_config=PRICE_COLS)
    st.caption("SETUP (footprint pending) = beyond the top-N deep scan; raise N or use the Footprint viewer below.")
else:
    st.caption("Nothing on the watchlist.")

st.markdown("## 🔬 Inspect a setup")
pool_ = sorted([r for r in with_setup if r["setup"]["state"] != "INVALID"], key=lambda r: r["setup"]["score"],
               reverse=True)
if pool_:
    labels = [f'{r["symbol"]} | {r["setup"]["state"]} | {r["setup"]["direction"]} | score {r["setup"]["score"]}'
              for r in pool_]
    pick = st.selectbox("Setup", labels)
    rec = pool_[labels.index(pick)]
    s, fpm, meta = rec["setup"], rec.get("fpm"), rec.get("fp_meta")
    left, right = st.columns([1, 2])
    with left:
        st.markdown(f'**{s["state"]}** · **{s["direction"]}** · score **{s["score"]}/100**')
        st.markdown(f'HTF bias **{s["bias"]}** · RR **{s["rr"]:.2f}** (target: {s["target_src"]})')
        for k, v in s["checks"].items():
            st.markdown(f"{tick(v)} {k}")
        if meta:
            cov = f'{meta["coverage"] * 100:.0f}% of candle volume' if meta["coverage"] is not None else "n/a"
            st.caption(f'Footprint: {meta["trades"]} trades · {meta["pages"]} pages · coverage {cov}'
                       + (" · ⚠ PARTIAL (raise max pages)" if meta["partial"] else ""))
        if fpm and fpm.get("fp_ok"):
            st.markdown(f'Candle delta **{fpm["delta"]:+.2f}** ({fpm["delta_pct"]:+.1f}%) · wick zone holds '
                        f'**{fpm["wick_share"] * 100:.0f}%** of volume, trapped side **{fpm["trapped_share"] * 100:.0f}%**')
            if fpm.get("stack_zone"):
                st.markdown(f'Stacked imbalance zone: `{fpm["stack_zone"][0]:.6g}` – `{fpm["stack_zone"][1]:.6g}`')
    with right:
        st.plotly_chart(price_chart(rec), width="stretch")
    if fpm and fpm.get("fp_ok"):
        a, b = st.columns([3, 2])
        a.plotly_chart(footprint_chart(fpm, s["level"], f'Footprint of sweep candle · {rec["symbol"]}'),
                       width="stretch")
        b.dataframe(fp_table(fpm), width="stretch", hide_index=True, height=560)
    else:
        st.info("Footprint not built for this setup (outside top-N, partial, or no trades). Use the viewer below.")
else:
    st.caption("Run a scan to inspect setups.")

# ------------------------------------------------------------------------------------ footprint viewer
st.markdown("## 🧪 Footprint viewer (any symbol, any recent candle)")
uni = st.session_state.get("universe", {})
ex_opts = list(uni.keys()) or selected
if ex_opts:
    v1, v2, v3 = st.columns([1, 2, 1])
    v_ex = v1.selectbox("Exchange", ex_opts, format_func=lambda x: feeds.EXCHANGE_LABELS[x], key="v_ex")
    v_syms = uni.get(v_ex, []) or ["BTC/USDT:USDT", "ETH/USDT:USDT"]
    v_sym = v2.selectbox("Symbol", v_syms, key="v_sym")
    v_back = v3.number_input("Candles back (1 = last closed)", 1, 30, 1, key="v_back")
    if st.button("Build footprint"):
        try:
            ex = get_exchange(v_ex)
            cdf_ = feeds.fetch_candles(ex, v_sym, tf, 120)
            closed = of.add_flow_columns(cdf_.iloc[:-1])
            idx = len(closed) - int(v_back)
            t0 = int(closed["timestamp"].iloc[idx])
            t1 = t0 + feeds.TF_MS[tf]
            trades, reached, pages = feeds.fetch_trades_window(ex, v_sym, t0, t1, max_pages=max_pages)
            rng = float(closed["high"].iloc[idx] - closed["low"].iloc[idx])
            bs = of.pick_bin_size(feeds.tick_size(ex, v_sym), rng)
            fp = of.build_footprint(trades, bs, t0, t1)
            if fp.empty:
                st.warning("No trades returned for that candle (too old for this exchange's trade history, or illiquid).")
            else:
                f = of.add_imbalances(fp, cfg.imbalance_ratio)
                poc = float(f.loc[f["total"].idxmax(), "price"])
                fpm_ = {"fp": f, "poc": poc}
                kline_v = float(closed["volume"].iloc[idx])
                st.caption(f'{closed["datetime"].iloc[idx]:%Y-%m-%d %H:%M UTC} · {len(trades)} trades · '
                           f'delta {f["delta"].sum():+.2f} · coverage {f["total"].sum() / kline_v * 100:.0f}% of candle volume'
                           + ("" if reached else " · ⚠ partial (raise max pages)"))
                a, b = st.columns([3, 2])
                a.plotly_chart(footprint_chart(fpm_, None, f"{v_sym} · {tf} candle footprint"), width="stretch")
                b.dataframe(fp_table(fpm_), width="stretch", hide_index=True, height=560)
        except Exception as e:                                            # noqa: BLE001
            st.error(friendly_error(e))

if errors:
    with st.expander(f"Warnings / data errors ({len(errors)})"):
        st.dataframe(pd.DataFrame(errors), width="stretch", hide_index=True)

st.markdown("---")
st.markdown("""
### How the signal is built
1. **HTF bias** (15m for a 5m scan): EMA20/50 + swing structure (HH/HL vs LH/LL).
2. **Liquidity pools**: swing highs/lows and *equal highs/lows* (stops rest beyond them).
3. **Sweep + reclaim**: wick pierces an untouched pool, a candle **closes back inside** (same or next candle).
4. **Footprint of the sweep candle** (real aggressor trades, per price level): *absorption* in the wick zone, *stacked imbalances*.
5. **True CVD divergence** from Binance taker-buy volume (not a candle-direction proxy).
6. **Price confirmation**: reclaim candle closes in the favourable half; entry only while fresh and not extended.
7. **Stop** beyond the sweep extreme (+0.15 ATR). **Target** = next opposing liquidity pool that pays ≥ min RR.

### Data & risk notes
- Binance candle delta = `2 × taker_buy_volume − volume` (exchange-reported aggressor volume). Footprint uses `aggTrades`
  (aggregated trades; side = taker side). Coverage vs candle volume is shown — a partial footprint can't produce CONFIRMED.
- Footprint is built only for the top-N setups to respect Binance rate limits (aggTrades weight = 20 per call).
- CVD is a rolling window anchored at the first loaded candle, not a session CVD.
- Heuristic rules, not a proven edge. Backtest/paper-trade with fees, funding and slippage before risking money.
""")
