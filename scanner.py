"""
scanner.py - two-stage scan pipeline (no Streamlit).

Stage 1 (cheap, every symbol):  candles (+ real taker-buy delta on Binance) -> liquidity levels -> sweep+reclaim
Stage 2 (expensive, top N only): public trades of the SWEEP candle -> real footprint -> re-score
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

import feeds
import orderflow as of


def analyze_stage1(ex, symbol: str, tf: str, cfg: of.Config):
    raw = feeds.fetch_candles(ex, symbol, tf, 300)
    htf_raw = feeds.fetch_candles(ex, symbol, feeds.HTF_FOR[tf], 150)
    if raw is None:
        return None
    df = of.add_flow_columns(raw.iloc[:-1])                    # drop the still-forming candle
    htf = of.add_flow_columns(htf_raw.iloc[:-1]) if htf_raw is not None else None
    if len(df) < 70 or not pd.notna(df["atr"].iloc[-1]):
        return None
    levels = of.liquidity_levels(df, cfg)
    sweeps = of.detect_sweeps(df, levels, cfg)
    bias = of.htf_bias(htf)
    rec = {"exchange": ex.id, "symbol": symbol, "tf": tf, "bias": bias, "_df": df, "_levels": levels,
           "setup": None, "fpm": None, "fp_meta": None}
    if sweeps:
        rec["setup"] = of.evaluate(df, sweeps[0], bias, levels, None, cfg)
    return rec


def attach_footprint(ex, rec: dict, cfg: of.Config, max_pages: int):
    """Stage 2: pull the sweep candle's trades, build footprint, re-evaluate."""
    s, df = rec["setup"], rec["_df"]
    j = s["sweep_idx"]
    t0 = int(df["timestamp"].iloc[j])
    t1 = t0 + feeds.TF_MS[rec["tf"]]
    trades, reached, pages = feeds.fetch_trades_window(ex, rec["symbol"], t0, t1, max_pages=max_pages)
    cand_rng = float(df["high"].iloc[j] - df["low"].iloc[j])
    bin_size = of.pick_bin_size(feeds.tick_size(ex, rec["symbol"]), cand_rng)
    fp = of.build_footprint(trades, bin_size, t0, t1)
    fpm = of.footprint_metrics(fp, s["direction"], s["level"], bin_size, cfg)
    coverage = None
    if ex.id in feeds.BINANCE_IDS and float(df["volume"].iloc[j]) > 0 and len(fp):
        coverage = float(fp["total"].sum() / df["volume"].iloc[j])
    rec["fp_meta"] = {"trades": int(len(trades)), "pages": pages, "reached_end": reached,
                      "coverage": coverage, "bin_size": bin_size, "t0": t0, "t1": t1}
    # An incomplete footprint (page cap hit) must not produce a CONFIRMED signal.
    partial = (coverage is not None and coverage < 0.85) or (not reached)
    rec["fp_meta"]["partial"] = partial
    rec["fpm"] = fpm
    use = None if (partial and cfg.require_footprint) else fpm
    rec["setup"] = of.evaluate(df, {k: s[k] for k in ("direction", "level", "touches", "level_idx", "sweep_idx",
                                                       "reclaim_idx", "extreme", "penetration_atr")},
                               rec["bias"], rec["_levels"], use, cfg)
    return rec


def run_scan(ex, symbols: list[str], tf: str, cfg: of.Config, n_candidates: int, max_pages: int,
             progress=None, workers: int = 6):
    records, errors = [], []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(analyze_stage1, ex, s, tf, cfg): s for s in symbols}
        for fut in as_completed(futs):
            done += 1
            try:
                r = fut.result()
                if r:
                    records.append(r)
            except Exception as e:                                        # noqa: BLE001
                errors.append({"Exchange": ex.id, "Symbol": futs[fut], "Error": str(e)[:160]})
            if progress:
                progress(done, len(symbols))
    cands = [r for r in records if r["setup"] and r["setup"]["state"] != "INVALID" and r["setup"]["fresh"]]
    cands.sort(key=lambda r: r["setup"]["score"], reverse=True)
    for r in cands[:n_candidates]:
        if not feeds.FOOTPRINT_SUPPORT.get(ex.id, True):
            r["fp_unavailable"] = True                                    # e.g. KuCoin: no historical trades API
            continue
        try:
            attach_footprint(ex, r, cfg, max_pages)
        except Exception as e:                                            # noqa: BLE001
            errors.append({"Exchange": ex.id, "Symbol": r["symbol"], "Error": "footprint: " + str(e)[:140]})
    return records, errors


def to_row(rec: dict) -> dict:
    s = rec["setup"]
    meta = rec.get("fp_meta") or {}
    fpm = rec.get("fpm") or {}
    return {
        "Exchange": rec["exchange"], "Symbol": rec["symbol"], "Signal": s["state"], "Dir": s["direction"],
        "Score": s["score"], "HTF bias": s["bias"], "Order-flow hits": f'{s["of_hits"]}/4',
        "Swept level": s["level"], "Pool touches": s["touches"], "Entry": s["entry"], "Stop": s["stop"],
        "Target": s["target"], "RR": round(s["rr"], 2), "Candles since reclaim": s["age"],
        "Absorption": fpm.get("absorption") if fpm.get("fp_ok") else None,
        "Stacked imb.": fpm.get("stacked") if fpm.get("fp_ok") else None,
        "CVD div (real)": s["cvd_div"],
        "Footprint": ("partial" if meta.get("partial") else "full") if meta
        else ("n/a (venue)" if rec.get("fp_unavailable") or not feeds.FOOTPRINT_SUPPORT.get(rec["exchange"], True)
              else "pending"),
        "Sweep candle": s["sweep_time"].strftime("%H:%M UTC") if s["sweep_time"] is not None else "",
    }
