"""
orderflow.py - pure trading logic (no network, no Streamlit).

Strategy (Liquidity Sweep + Footprint + true CVD):
  1. HTF (15m) bias           -> EMA trend + swing structure
  2. 5m liquidity sweep       -> wick through a swing high/low (or equal highs/lows) and CLOSE back inside
  3. Footprint of sweep candle-> absorption in the wick zone, stacked imbalances (real bid/ask traded volume)
  4. True CVD                 -> divergence between price extreme and cumulative delta
  5. Price confirmation       -> reclaim candle closes back inside AND in the favourable half of its range

CVD divergence alone is NEVER an entry. A CONFIRMED signal needs the sweep + reclaim
+ at least `min_of_hits` order-flow confirmations + score/RR thresholds.

Delta sources:
  * Binance klines -> taker_buy volume (real aggressor volume):  delta = 2*taker_buy - volume
  * Footprint      -> public trades with aggressor side, binned per price level
Educational decision-support tool, not financial advice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------------------
@dataclass
class Config:
    # liquidity levels / sweep
    swing_left: int = 3
    swing_right: int = 3
    level_lookback: int = 80          # bars used to find liquidity pools
    equal_tol_atr: float = 0.20       # highs/lows within this many ATR = "equal highs/lows"
    sweep_last_k: int = 3             # reclaim must have happened within the last k closed candles
    min_pen_atr: float = 0.03         # wick must pierce level by at least this (ATR)
    max_pen_atr: float = 1.50         # deeper than this = breakout, not a sweep
    # footprint
    imbalance_ratio: float = 3.0
    min_stack: int = 3
    absorb_sell_share: float = 0.55   # wick-zone aggressor share on the "trapped" side
    absorb_density: float = 0.80      # wick-zone volume per level vs candle average
    absorb_wick_share: float = 0.08   # wick zone must hold >= 8% of candle volume
    # confirmation / scoring
    vol_mult: float = 1.3
    max_age: int = 2                  # candles since reclaim
    max_extension_atr: float = 1.5    # don't chase: price already > 1.5 ATR from level
    stop_buffer_atr: float = 0.15
    min_of_hits: int = 2              # order-flow confirmations needed (of 4)
    min_delta_pct: float = 0.10       # reclaim-candle delta must be >= 10% of its volume (not noise)
    min_score: int = 65
    min_rr: float = 1.5
    require_footprint: bool = True    # CONFIRMED needs a real footprint


# ----------------------------------------------------------------------------------------
# Basics
# ----------------------------------------------------------------------------------------
def wilder_atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def add_flow_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Adds ema20/ema50/atr and, when taker_buy exists, REAL delta + CVD."""
    d = df.copy().reset_index(drop=True)
    d["ema20"] = d["close"].ewm(span=20, adjust=False).mean()
    d["ema50"] = d["close"].ewm(span=50, adjust=False).mean()
    d["atr"] = wilder_atr(d)
    d["vol_sma"] = d["volume"].rolling(20).mean()
    if "taker_buy" in d.columns and d["taker_buy"].notna().all():
        d["delta"] = 2 * d["taker_buy"] - d["volume"]
        d["cvd"] = d["delta"].cumsum()
    else:
        d["delta"] = np.nan
        d["cvd"] = np.nan
    return d


def has_real_delta(df: pd.DataFrame) -> bool:
    return "delta" in df.columns and df["delta"].notna().all()


# ----------------------------------------------------------------------------------------
# Liquidity pools
# ----------------------------------------------------------------------------------------
def swing_points(df: pd.DataFrame, left: int = 3, right: int = 3):
    hi, lo, n = df["high"].values, df["low"].values, len(df)
    highs, lows = [], []
    for i in range(left, n - right):
        if hi[i] > hi[i - left:i].max() and hi[i] >= hi[i + 1:i + right + 1].max():
            highs.append((i, float(hi[i])))
        if lo[i] < lo[i - left:i].min() and lo[i] <= lo[i + 1:i + right + 1].min():
            lows.append((i, float(lo[i])))
    return highs, lows


def _cluster(points, tol, side):
    """Group swing points within tol -> one liquidity pool (stops sit beyond the extreme)."""
    if not points:
        return []
    pts = sorted(points, key=lambda p: p[1])
    groups, cur = [], [pts[0]]
    for p in pts[1:]:
        if p[1] - cur[-1][1] <= tol:
            cur.append(p)
        else:
            groups.append(cur)
            cur = [p]
    groups.append(cur)
    out = []
    for g in groups:
        price = max(x[1] for x in g) if side == "high" else min(x[1] for x in g)
        out.append({"side": side, "price": price, "touches": len(g), "idx": max(x[0] for x in g)})
    return out


def liquidity_levels(df: pd.DataFrame, cfg: Config) -> list[dict]:
    n = len(df)
    start = max(0, n - cfg.level_lookback)
    sub = df.iloc[start:].reset_index(drop=True)
    atr = float(df["atr"].iloc[-1]) if pd.notna(df["atr"].iloc[-1]) else float((sub["high"] - sub["low"]).mean())
    highs, lows = swing_points(sub, cfg.swing_left, cfg.swing_right)
    tol = cfg.equal_tol_atr * atr
    levels = _cluster(highs, tol, "high") + _cluster(lows, tol, "low")
    for lv in levels:
        lv["idx"] += start
    return levels


# ----------------------------------------------------------------------------------------
# Sweep detection
# ----------------------------------------------------------------------------------------
def detect_sweeps(df: pd.DataFrame, levels: list[dict], cfg: Config) -> list[dict]:
    """
    A sweep = a candle whose wick pierces an UNTOUCHED liquidity level, followed (same candle or next)
    by a CLOSE back inside. Sweep of a HIGH -> SHORT idea, sweep of a LOW -> LONG idea.
    """
    n = len(df)
    last = n - 1
    atr = float(df["atr"].iloc[-1])
    if not np.isfinite(atr) or atr <= 0:
        return []
    min_pen, max_pen = cfg.min_pen_atr * atr, cfg.max_pen_atr * atr
    hi, lo, cl = df["high"].values, df["low"].values, df["close"].values
    out = []
    for lv in levels:
        p = lv["price"]
        for j in range(lv["idx"] + 1, n):
            if lv["side"] == "high":
                if j - 1 > lv["idx"] and hi[lv["idx"] + 1:j].max() > p:
                    break                                    # level already taken earlier
                if hi[j] - p < min_pen:
                    continue
                r = j if cl[j] < p else (j + 1 if j + 1 <= last and cl[j + 1] < p else None)
                if r is None:
                    break                                    # accepted above level = breakout
                extreme = float(hi[j:r + 1].max())
                if extreme - p > max_pen:
                    break
                direction = "SHORT"
            else:
                if j - 1 > lv["idx"] and lo[lv["idx"] + 1:j].min() < p:
                    break
                if p - lo[j] < min_pen:
                    continue
                r = j if cl[j] > p else (j + 1 if j + 1 <= last and cl[j + 1] > p else None)
                if r is None:
                    break
                extreme = float(lo[j:r + 1].min())
                if p - extreme > max_pen:
                    break
                direction = "LONG"
            if r < n - cfg.sweep_last_k:
                break                                        # stale
            out.append({"direction": direction, "level": p, "touches": lv["touches"], "level_idx": lv["idx"],
                        "sweep_idx": j, "reclaim_idx": r, "extreme": extreme,
                        "penetration_atr": abs(extreme - p) / atr})
            break
    out.sort(key=lambda s: (s["reclaim_idx"], s["touches"]), reverse=True)
    return out


# ----------------------------------------------------------------------------------------
# HTF bias
# ----------------------------------------------------------------------------------------
def htf_bias(htf: pd.DataFrame) -> str:
    if htf is None or len(htf) < 60:
        return "NEUTRAL"
    c = htf.iloc[-1]
    ema_s = 1 if (c["ema20"] > c["ema50"] and c["close"] > c["ema50"]) else (
        -1 if (c["ema20"] < c["ema50"] and c["close"] < c["ema50"]) else 0)
    highs, lows = swing_points(htf, 2, 2)
    st = 0
    if len(highs) >= 2 and len(lows) >= 2:
        hh, hl = highs[-1][1] > highs[-2][1], lows[-1][1] > lows[-2][1]
        lh, ll = highs[-1][1] < highs[-2][1], lows[-1][1] < lows[-2][1]
        st = 1 if (hh and hl) else (-1 if (lh and ll) else 0)
    s = ema_s + st
    return "BULL" if s >= 1 else ("BEAR" if s <= -1 else "NEUTRAL")


# ----------------------------------------------------------------------------------------
# Footprint
# ----------------------------------------------------------------------------------------
def pick_bin_size(tick: float, candle_range: float, target_levels: int = 20) -> float:
    tick = tick if tick and tick > 0 else max(candle_range / 100, 1e-8)
    raw = max(candle_range / target_levels, tick)
    return round(np.ceil(raw / tick - 1e-9) * tick, 12)


def build_footprint(trades: pd.DataFrame, bin_size: float, t0: int, t1: int) -> pd.DataFrame:
    """
    trades: columns ts(ms), price, amount, side ('buy' = aggressive buyer lifted the ask, 'sell' = aggressor sold into bid).
    Returns one row per price level (continuous grid): price, buy, sell, delta, total.
    """
    cols = ["price", "buy", "sell", "delta", "total"]
    if trades is None or trades.empty:
        return pd.DataFrame(columns=cols)
    t = trades[(trades["ts"] >= t0) & (trades["ts"] < t1)]
    if t.empty:
        return pd.DataFrame(columns=cols)
    k = np.floor(t["price"].values / bin_size + 1e-9).astype(np.int64)
    g = pd.DataFrame({"k": k,
                      "buy": np.where(t["side"].values == "buy", t["amount"].values, 0.0),
                      "sell": np.where(t["side"].values == "sell", t["amount"].values, 0.0)}).groupby("k").sum()
    g = g.reindex(np.arange(g.index.min(), g.index.max() + 1), fill_value=0.0)
    fp = pd.DataFrame({"price": np.round(g.index.values * bin_size, 10), "buy": g["buy"].values, "sell": g["sell"].values})
    fp["delta"] = fp["buy"] - fp["sell"]
    fp["total"] = fp["buy"] + fp["sell"]
    return fp


def _runs(mask: np.ndarray):
    runs, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if (not m) and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def add_imbalances(fp: pd.DataFrame, ratio: float) -> pd.DataFrame:
    """Diagonal imbalances: buy@P vs sell@(P-1 tick) and sell@P vs buy@(P+1 tick)."""
    f = fp.copy()
    buy, sell, n = f["buy"].values, f["sell"].values, len(f)
    nz = f.loc[f["total"] > 0, "total"]
    min_vol = 0.5 * float(nz.mean()) if len(nz) else 0.0
    bi, si = np.zeros(n, bool), np.zeros(n, bool)
    for i in range(1, n):
        bi[i] = buy[i] >= min_vol and buy[i] >= ratio * max(sell[i - 1], 1e-12)
    for i in range(0, n - 1):
        si[i] = sell[i] >= min_vol and sell[i] >= ratio * max(buy[i + 1], 1e-12)
    f["buy_imb"], f["sell_imb"] = bi, si
    return f


def footprint_metrics(fp: pd.DataFrame, direction: str, level: float, bin_size: float, cfg: Config) -> dict:
    """Order-flow reading of the SWEEP candle's footprint."""
    m = {"fp_ok": False}
    if fp is None or len(fp) < 3 or fp["total"].sum() <= 0:
        return m
    f = add_imbalances(fp, cfg.imbalance_ratio)
    total, delta = float(f["total"].sum()), float(f["delta"].sum())
    poc = float(f.loc[f["total"].idxmax(), "price"])
    center = f["price"] + bin_size / 2
    long_ = direction == "LONG"

    wick = f[center < level] if long_ else f[center > level]          # zone where stops were hunted
    wick_total = float(wick["total"].sum()) if len(wick) else 0.0
    wick_buy = float(wick["buy"].sum()) if len(wick) else 0.0
    wick_sell = float(wick["sell"].sum()) if len(wick) else 0.0
    trapped = wick_sell if long_ else wick_buy                         # aggressors who got absorbed
    density = (wick_total / len(wick)) / (total / len(f)) if len(wick) else 0.0
    wick_share = wick_total / total
    trapped_share = trapped / wick_total if wick_total > 0 else 0.0
    absorption = bool(len(wick) and trapped_share >= cfg.absorb_sell_share and density >= cfg.absorb_density
                      and wick_share >= cfg.absorb_wick_share)

    # "initiative at the extreme": extreme 25% of levels flips to the reversal side
    q = max(1, int(round(len(f) * 0.25)))
    ext = f.iloc[:q] if long_ else f.iloc[-q:]
    ext_delta = float(ext["delta"].sum())
    extreme_aligned = ext_delta > 0 if long_ else ext_delta < 0

    imb = f["buy_imb"].values if long_ else f["sell_imb"].values
    runs = [r for r in _runs(imb) if (r[1] - r[0] + 1) >= cfg.min_stack]
    stacked = bool(runs)
    zone = None
    if runs:
        a, b = max(runs, key=lambda r: r[1] - r[0])
        zone = (float(f["price"].iloc[a]), float(f["price"].iloc[b] + bin_size))

    m.update({"fp_ok": True, "fp": f, "bin_size": bin_size, "total": total, "delta": delta,
              "delta_pct": delta / total * 100 if total else 0.0, "poc": poc,
              "wick_total": wick_total, "wick_share": wick_share, "trapped_share": trapped_share,
              "wick_density": density, "absorption": absorption, "extreme_aligned": extreme_aligned,
              "ext_delta": ext_delta, "stacked": stacked, "stack_zone": zone})
    return m


# ----------------------------------------------------------------------------------------
# CVD divergence (needs real delta)
# ----------------------------------------------------------------------------------------
def cvd_divergence(df: pd.DataFrame, s: dict, lookback: int = 25, gap: int = 3) -> Optional[bool]:
    if not has_real_delta(df):
        return None
    j, r = s["sweep_idx"], s["reclaim_idx"]
    a, b = max(0, j - lookback), max(1, j - gap + 1)
    if b - a < 3:
        return None
    seg_lo, seg_hi = df["low"].values[a:b], df["high"].values[a:b]
    cvd = df["cvd"].values
    if s["direction"] == "LONG":
        k = a + int(np.argmin(seg_lo))
        return bool(df["low"].values[j:r + 1].min() < df["low"].values[k] and cvd[j:r + 1].min() > cvd[k:k + 2].min())
    k = a + int(np.argmax(seg_hi))
    return bool(df["high"].values[j:r + 1].max() > df["high"].values[k] and cvd[j:r + 1].max() < cvd[k:k + 2].max())


# ----------------------------------------------------------------------------------------
# Full evaluation
# ----------------------------------------------------------------------------------------
def evaluate(df: pd.DataFrame, s: dict, bias: str, levels: list[dict], fpm: Optional[dict], cfg: Config) -> dict:
    """Scores one sweep setup. `fpm` = footprint_metrics() result or None (stage-1, footprint pending)."""
    n = len(df)
    last = n - 1
    long_ = s["direction"] == "LONG"
    atr = float(df["atr"].iloc[-1])
    j, r = s["sweep_idx"], s["reclaim_idx"]
    rc, sc = df.iloc[r], df.iloc[j]
    level, extreme = s["level"], s["extreme"]
    close = float(df["close"].iloc[-1])

    # --- price confirmation ---
    rng = max(float(rc["high"] - rc["low"]), 1e-12)
    pos = float(rc["close"] - rc["low"]) / rng
    reclaim_strong = pos >= 0.5 if long_ else pos <= 0.5
    after = df["close"].values[r:]
    failed = bool((after < level).any()) if long_ else bool((after > level).any())
    age = last - r
    ext_atr = (close - level) / atr if long_ else (level - close) / atr
    not_extended = ext_atr <= cfg.max_extension_atr
    fresh = age <= cfg.max_age
    invalidated = failed or (close <= extreme if long_ else close >= extreme)

    # --- order-flow items ---
    real_delta = has_real_delta(df)
    cvd_div = cvd_divergence(df, s)
    reclaim_delta = None
    if real_delta:
        d_r = float(df["delta"].iloc[r])
        v_r = max(float(df["volume"].iloc[r]), 1e-12)
        thr = cfg.min_delta_pct * v_r
        reclaim_delta = d_r >= thr if long_ else d_r <= -thr
    elif fpm and fpm.get("fp_ok"):
        thr = cfg.min_delta_pct * max(fpm["total"], 1e-12)
        reclaim_delta = (fpm["delta"] >= thr) if long_ else (fpm["delta"] <= -thr)
    vol_ok = bool(pd.notna(sc["vol_sma"]) and sc["volume"] >= cfg.vol_mult * sc["vol_sma"])
    fp_ok = bool(fpm and fpm.get("fp_ok"))
    absorption = bool(fp_ok and (fpm["absorption"] or fpm["extreme_aligned"] and fpm["wick_share"] >= 0.05))
    stacked = bool(fp_ok and fpm["stacked"])
    of_hits = int(bool(absorption)) + int(bool(stacked)) + int(bool(cvd_div)) + int(bool(reclaim_delta))

    # --- bias ---
    want = "BULL" if long_ else "BEAR"
    aligned = bias == want
    opposed = bias not in (want, "NEUTRAL")

    # --- levels / RR ---
    entry = close
    stop = extreme - cfg.stop_buffer_atr * atr if long_ else extreme + cfg.stop_buffer_atr * atr
    risk = abs(entry - stop)
    opp = [lv["price"] for lv in levels if lv["side"] == ("high" if long_ else "low")
           and ((lv["price"] > entry) if long_ else (lv["price"] < entry))]
    if opp:
        target = min(opp) if long_ else max(opp)
        t_src = "liquidity"
    else:
        target = entry + 2 * risk if long_ else entry - 2 * risk
        t_src = "2R"
    rr = abs(target - entry) / risk if risk > 0 else 0.0
    if rr < cfg.min_rr:                       # nearest pool too close -> next liquidity pool that pays min RR, else 2R
        far = [p for p in opp if abs(p - entry) / max(risk, 1e-12) >= cfg.min_rr]
        if far:
            target = min(far) if long_ else max(far)
            t_src = "liquidity"
        else:
            target = entry + 2 * risk if long_ else entry - 2 * risk
            t_src = "2R"
        rr = abs(target - entry) / risk if risk > 0 else 0.0

    # --- score (100) ---
    score = 20 + (5 if s["touches"] >= 2 else 0)                         # sweep + reclaim (+equal highs/lows)
    score += 15 if aligned else (0 if opposed else 7)                    # HTF bias
    score += 20 if absorption else 0                                     # footprint absorption
    score += 10 if stacked else 0                                        # stacked imbalances
    score += 10 if cvd_div else 0                                        # true CVD divergence
    score += 10 if reclaim_delta else 0                                  # reclaim candle delta
    score += 5 if vol_ok else 0
    score += 5 if rr >= cfg.min_rr else 0

    checks = {
        "Liquidity sweep + reclaim": True,
        "Equal highs/lows pool (2+ touches)": s["touches"] >= 2,
        "HTF bias aligned": aligned,
        "Footprint absorption": absorption if fp_ok else None,
        "Stacked imbalances": stacked if fp_ok else None,
        "True CVD divergence": cvd_div,
        "Reclaim delta aligned": reclaim_delta,
        "Volume expansion": vol_ok,
        "Reclaim candle strong close": reclaim_strong,
        "RR >= min": rr >= cfg.min_rr,
    }

    # with require_footprint: real footprint evidence (absorption or stacked imbalance) is mandatory,
    # CVD / delta alone can never confirm an entry.
    fp_evidence = (absorption or stacked) if cfg.require_footprint else True
    confirmed = (fresh and not_extended and not invalidated and reclaim_strong and of_hits >= cfg.min_of_hits
                 and score >= cfg.min_score and rr >= cfg.min_rr and fp_evidence)
    if invalidated:
        state = "INVALID"
    elif confirmed:
        state = "CONFIRMED"
    elif fresh and of_hits >= 1:
        state = "SETUP" if fp_ok or not cfg.require_footprint else "SETUP (footprint pending)"
    else:
        state = "WATCH"

    return {**s, "state": state, "score": int(score), "bias": bias, "of_hits": of_hits, "age": age,
            "entry": entry, "stop": stop, "target": target, "target_src": t_src, "rr": rr,
            "ext_atr": ext_atr, "atr": atr, "checks": checks, "real_delta": real_delta,
            "cvd_div": cvd_div, "fresh": fresh, "reclaim_strong": reclaim_strong,
            "sweep_time": df["datetime"].iloc[j] if "datetime" in df else None}
