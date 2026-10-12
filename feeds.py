"""
feeds.py - exchange data access (ccxt). Binance USDⓈ-M is the primary venue.

Binance:  /fapi/v1/klines  -> includes taker-buy volume (REAL aggressor delta per candle)
          /fapi/v1/aggTrades -> every aggregated trade with aggressor side (REAL footprint)
OKX / KuCoin (optional, beta): candles via ccxt, footprint via ccxt fetch_trades (side = taker side).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import requests
import ccxt

PRIMARY = "binanceusdm"
MIRROR = "binancespot"                # Binance public market-data mirror (spot), not geo-blocked
BINANCE_IDS = (PRIMARY, MIRROR)
EXCHANGE_IDS = ["binanceusdm", "binancespot", "okx", "kucoinfutures"]
EXCHANGE_LABELS = {
    "binanceusdm": "Binance USDⓈ-M Futures (main)",
    "binancespot": "Binance data mirror (spot, works on cloud)",
    "okx": "OKX Swaps (beta)",
    "kucoinfutures": "KuCoin Futures (beta)",
}
# Which venues can give a REAL historical footprint (trades with aggressor side for an exact time window)
FOOTPRINT_SUPPORT = {"binanceusdm": True, "binancespot": True, "okx": True, "kucoinfutures": False}
FALLBACK_CHAIN = [MIRROR, "okx"]      # tried in order when Binance futures is geo-blocked / unreachable
MIRROR_HOST = "https://data-api.binance.vision"


def is_geo_block(e: Exception) -> bool:
    t = str(e).lower()
    return "451" in t or "restricted location" in t or "eligibility" in t or "403 forbidden" in t


TF_MS = {"1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000}
HTF_FOR = {"1m": "5m", "3m": "15m", "5m": "15m", "15m": "1h"}


STABLES = {"USDC", "FDUSD", "TUSD", "USDP", "DAI", "BUSD", "EUR", "AEUR", "USDE", "XUSD", "USD1", "RLUSD", "UST"}

CONNECTION_TESTS = [
    ("Binance USDⓈ-M Futures", "https://fapi.binance.com/fapi/v1/ping"),
    ("Binance data mirror (spot)", MIRROR_HOST + "/api/v3/ping"),
    ("OKX", "https://www.okx.com/api/v5/public/time"),
    ("KuCoin Futures", "https://api-futures.kucoin.com/api/v1/timestamp"),
]


def test_connections(session=None) -> list[dict]:
    """Ping each venue FROM THE SERVER THIS APP RUNS ON (what matters for deployment)."""
    s = session or requests.Session()
    out = []
    for name, url in CONNECTION_TESTS:
        try:
            r = s.get(url, timeout=8)
            ok = r.status_code == 200
            note = "OK" if ok else ("blocked from this location (451)" if r.status_code == 451 else f"HTTP {r.status_code}")
        except Exception as e:                                            # noqa: BLE001
            ok, note = False, str(e)[:80]
        out.append({"Venue": name, "Reachable": "✅" if ok else "❌", "Detail": note})
    return out


def next_fallback(exid: str, queue: list[str], error: Exception):
    """Which venue to try next when `exid` fails. Binance futures: only on geo-block. Mirror: on any failure."""
    if exid == PRIMARY and not is_geo_block(error):
        return None
    if exid not in (PRIMARY, MIRROR):
        return None
    start = FALLBACK_CHAIN.index(MIRROR) if exid == PRIMARY else FALLBACK_CHAIN.index(MIRROR) + 1
    for cand in FALLBACK_CHAIN[start:]:
        if cand not in queue:
            return cand
    return None


class BinanceMirror:
    """
    Minimal client for Binance's public market-data host (data-api.binance.vision): spot klines (with taker-buy volume),
    24h tickers and aggTrades (with aggressor side). Same data shapes the scanner needs from the futures API.
    NOTE: spot market, not USDⓈ-M futures - prices track closely but volume/delta are spot-only.
    """
    id = MIRROR
    precisionMode = ccxt.TICK_SIZE
    is_mirror = True

    def __init__(self, session=None, host: str = MIRROR_HOST):
        self.s = session or requests.Session()
        self.host = host.rstrip("/")
        self.markets: dict = {}

    def _get(self, path: str, params: dict | None = None):
        r = self.s.get(self.host + path, params=params or {}, timeout=20)
        r.raise_for_status()
        return r.json()

    def load_markets(self):
        info = self._get("/api/v3/exchangeInfo")
        out = {}
        for m in info.get("symbols", []):
            if m.get("status") != "TRADING" or m.get("quoteAsset") != "USDT":
                continue
            base = m["baseAsset"]
            if base.endswith(("UP", "DOWN", "BULL", "BEAR")) and len(base) > 4:
                continue                                         # leveraged tokens
            if base in STABLES:
                continue
            tick = next((float(f["tickSize"]) for f in m.get("filters", []) if f.get("filterType") == "PRICE_FILTER"), 0.0)
            out[f"{base}/USDT"] = {"id": m["symbol"], "symbol": f"{base}/USDT", "active": True,
                                   "precision": {"price": tick}}
        self.markets = out
        return out

    def market(self, symbol: str):
        return self.markets[symbol]

    def fetch_tickers(self):
        rows = self._get("/api/v3/ticker/24hr")
        by_id = {v["id"]: k for k, v in self.markets.items()}
        return {by_id[r["symbol"]]: {"quoteVolume": float(r["quoteVolume"])} for r in rows if r["symbol"] in by_id}

    def klines(self, symbol: str, interval: str, limit: int):
        return self._get("/api/v3/klines", {"symbol": self.market(symbol)["id"], "interval": interval, "limit": limit})

    def fetch_trades(self, symbol: str, since=None, limit: int = 1000, params=None):
        q = {"symbol": self.market(symbol)["id"], "limit": limit}
        params = params or {}
        if "fromId" in params:
            q["fromId"] = int(params["fromId"])
        elif since is not None:
            q["startTime"] = int(since)
        rows = self._get("/api/v3/aggTrades", q)
        return [{"id": str(r["a"]), "timestamp": int(r["T"]), "price": float(r["p"]), "amount": float(r["q"]),
                 "side": "sell" if r["m"] else "buy"} for r in rows]       # m = buyer is maker -> taker sold


def make_exchange(exchange_id: str):
    if exchange_id == MIRROR:
        return BinanceMirror()
    opts = {"enableRateLimit": True, "timeout": 20000}
    if exchange_id in ("okx", "kucoinfutures"):
        opts["options"] = {"defaultType": "swap"}
    return getattr(ccxt, exchange_id)(opts)


def linear_usdt_swaps(ex) -> list[str]:
    if getattr(ex, "is_mirror", False):
        return sorted(ex.markets)
    out = []
    for sym, m in ex.markets.items():
        if m.get("active") is False:
            continue
        if not (m.get("swap") or m.get("future")):
            continue
        if m.get("settle") != "USDT" or not m.get("linear", False):
            continue
        if m.get("expiry"):                      # skip dated futures, keep perpetuals
            continue
        out.append(sym)
    return sorted(set(out))


def tick_size(ex, symbol: str) -> float:
    try:
        p = ex.market(symbol)["precision"].get("price")
        if p is None or p <= 0:
            return 0.0
        return float(p) if ex.precisionMode == ccxt.TICK_SIZE else float(10 ** (-int(p)))
    except Exception:
        return 0.0


def fetch_candles(ex, symbol: str, tf: str, limit: int = 300) -> pd.DataFrame | None:
    """Closed + forming candles. Binance rows carry taker_buy (real aggressor buy volume)."""
    df = None
    if ex.id in BINANCE_IDS:
        if ex.id == PRIMARY:
            raw = ex.fapiPublicGetKlines({"symbol": ex.market(symbol)["id"], "interval": tf, "limit": limit})
        else:
            raw = ex.klines(symbol, tf, limit)
        if raw:
            df = pd.DataFrame({
                "timestamp": [int(r[0]) for r in raw], "open": [float(r[1]) for r in raw],
                "high": [float(r[2]) for r in raw], "low": [float(r[3]) for r in raw],
                "close": [float(r[4]) for r in raw], "volume": [float(r[5]) for r in raw],
                "taker_buy": [float(r[9]) for r in raw]})
    else:
        rows = ex.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
        if rows:
            df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    if df is None or len(df) < 70:
        return None
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def _fetch_trades_window_okx(ex, symbol: str, t0: int, t1: int, max_pages: int):
    """
    OKX /market/history-trades with type=2 (timestamp paging): `after=ts` returns records OLDER than ts,
    newest first, max 100 per request, side = taker side. We walk backwards from t1 to t0.
    `max_pages` counts 1000-trade pages for parity with Binance -> up to max_pages*10 requests of 100.
    Sizes are contracts; converted to base-coin amount with ctVal so ratios/volumes are comparable.
    """
    m = ex.market(symbol)
    ct = float(m.get("contractSize") or 1.0)
    inst = m["id"]
    rows, seen, reqs, reached = [], set(), 0, False
    after = t1
    while reqs < max_pages * 10:
        resp = ex.publicGetMarketHistoryTrades({"instId": inst, "type": "2", "after": str(after), "limit": "100"})
        reqs += 1
        data = (resp or {}).get("data") or []
        if not data:
            reached = True
            break
        new = [d for d in data if d["tradeId"] not in seen]
        for d in new:
            seen.add(d["tradeId"])
        rows.extend(new)
        oldest = min(int(d["ts"]) for d in data)
        if oldest <= t0:
            reached = True
            break
        if not new:
            after = oldest                      # same-ms cluster larger than a page: step past it
        else:
            after = oldest + 1                  # overlap by 1 ms, de-duplicated by tradeId
        if len(data) < 100:
            reached = True
            break
    df = pd.DataFrame([{"ts": int(d["ts"]), "price": float(d["px"]), "amount": float(d["sz"]) * ct, "side": d["side"]}
                       for d in rows if d.get("side") in ("buy", "sell")])
    if df.empty:
        return pd.DataFrame(columns=["ts", "price", "amount", "side"]), reached, reqs
    df = df[(df["ts"] >= t0) & (df["ts"] < t1)].sort_values("ts").reset_index(drop=True)
    return df, reached, reqs


def fetch_trades_window(ex, symbol: str, t0: int, t1: int, max_pages: int = 8, page_limit: int = 1000):
    """
    Public trades with aggressor side in [t0, t1). Returns (DataFrame[ts,price,amount,side], reached_end, pages).
    Binance: pages with fromId (no gaps/dupes). Others: pages with `since` + de-dup.
    """
    if ex.id == "okx":
        return _fetch_trades_window_okx(ex, symbol, t0, t1, max_pages)
    if not FOOTPRINT_SUPPORT.get(ex.id, True):
        raise NotImplementedError(f"{ex.id}: public API has no historical trades, footprint unavailable")
    rows, seen, pages, reached = [], set(), 0, False
    params, since = {}, t0
    while pages < max_pages:
        page = ex.fetch_trades(symbol, since=since if not params else None, limit=page_limit, params=params)
        pages += 1
        if not page:
            reached = True
            break
        new = []
        for t in page:
            key = t.get("id") or (t["timestamp"], t["price"], t["amount"])
            if key in seen:
                continue
            seen.add(key)
            new.append(t)
        if not new:
            break
        rows.extend(new)
        last_ts = page[-1]["timestamp"]
        if last_ts >= t1:
            reached = True
            break
        if len(page) < page_limit:
            reached = True
            break
        if ex.id in BINANCE_IDS and page[-1].get("id") is not None:
            params = {"fromId": int(page[-1]["id"]) + 1}
        else:
            since = last_ts
    df = pd.DataFrame([{"ts": t["timestamp"], "price": t["price"], "amount": t["amount"], "side": t.get("side")}
                       for t in rows if t.get("side") in ("buy", "sell")])
    if df.empty:
        return pd.DataFrame(columns=["ts", "price", "amount", "side"]), reached, pages
    df = df[(df["ts"] >= t0) & (df["ts"] < t1)].reset_index(drop=True)
    return df, reached, pages
