# POWERFUL SIGNALS v2 — Liquidity Sweep + Real Footprint + True CVD

Streamlit scanner for **Binance USDⓈ-M perpetuals (main)**. OKX / KuCoin are optional (beta) in the exchange selector.
Public data only — no API keys, no orders.

## What changed vs v1
| v1 | v2 |
|---|---|
| "CVD" = candle-direction proxy | **True delta** from Binance klines (`2 × taker_buy_volume − volume`) |
| No footprint | **Real footprint** from `aggTrades` (aggressor side) per price level: bid/ask volume, delta, POC, imbalances, stacked imbalances, absorption |
| Sweep = 20-bar high/low | Liquidity **pools** (swing highs/lows + equal highs/lows), untouched-level check, sweep + **reclaim** |
| 4 of 6 votes = signal | Gated: sweep+reclaim → order-flow confirmations → score → RR. CVD alone never enters |
| Fake ATR (avg range) | Wilder ATR; stop beyond sweep extreme; target = next opposing liquidity pool |
| Needed missing `streamlit_autorefresh` | Dependency added; app also runs if it is missing |

## Signal pipeline (5m sweep, 15m bias by default)
1. **HTF bias** — EMA20/50 + swing structure (HH/HL vs LH/LL).
2. **Liquidity pools** — swing highs/lows, clustered into *equal highs/lows* (touches ≥ 2 scores higher).
3. **Sweep + reclaim** — wick pierces an untouched pool; same/next candle **closes back inside**.
4. **Footprint of the sweep candle**
   - *Absorption*: wick zone holds real volume, ≥55% of it from the trapped aggressors, and price still reclaimed.
   - *Stacked imbalances*: ≥3 consecutive diagonal imbalances (ratio 3.0) in the reversal direction.
5. **True CVD divergence** — price makes a lower low / higher high, cumulative delta does not.
6. **Price confirmation** — reclaim candle closes in the favourable half; entry only while fresh (≤2 candles) and not extended (>1.5 ATR).
7. **Levels** — stop = sweep extreme ± 0.15 ATR; target = nearest opposing liquidity pool paying ≥ min RR (else 2R).

States: `WATCH` → `SETUP` → **`CONFIRMED`** (needs footprint evidence + ≥2 of 4 order-flow hits + score ≥ 65 + RR ≥ 1.5).
Order-flow hits: absorption · stacked imbalance · CVD divergence · reclaim-candle delta (≥10% of volume).

## Rate limits (important)
Stage 1 scans every symbol with klines (cheap). Stage 2 pulls **trades only for the top-N setups** and only for the sweep candle
(aggTrades weight = 20/call). Coverage (footprint volume ÷ candle volume) is shown; a partial footprint (<85% or page cap hit)
**cannot** produce CONFIRMED — raise "max trade pages" if you see ⚠ PARTIAL on very liquid coins (BTC/ETH on 5m).

## Two dashboards (switch in the sidebar)
1. **⚡ Sweep + Footprint** — liquidity sweep + real footprint + true CVD (above).
2. **📈 Supertrend wick-touch** — Supertrend(10, 3) per timeframe; alerts the moment a wick touches the line.
   - Uptrend: candle **low ≤ line** (support touch). Downtrend: candle **high ≥ line** (resistance touch). Close stays on the trend side.
   - Checks the **live (forming) candle** + the last N closed candles. 🆕 + toast for new FRESH touches.
   - 🔥 **FRESH** = first touch since the trend flipped (sorted first). ↻ **RETOUCH** = allowed again only after **N candles** since
     the previous touch; **N is set per timeframe** (default 5). Touches inside the cool-down are suppressed.
   - Needs candles only, so it works on every venue (KuCoin included).

## Venues & automatic fallback
Order tried when Binance futures is unreachable/blocked: **Binance futures → Binance data mirror (spot) → OKX**.
| Venue | Candles | Real candle delta / CVD | Real footprint | Notes |
|---|---|---|---|---|
| **Binance USDⓈ-M** (main) | klines | ✅ taker-buy volume | ✅ `aggTrades` | Best data. Blocked for US IPs (HTTP 451) |
| **Binance data mirror** (`data-api.binance.vision`) | klines | ✅ taker-buy volume | ✅ `aggTrades` | **Spot** market (prices ≈ futures, volume/delta are spot). Market-data host, intended to work from cloud |
| **OKX swaps** | ccxt | ❌ (no taker-buy in candles) | ✅ `history-trades` (timestamp paging) | Works from US hosts; CVD divergence skipped, never faked |
| **KuCoin futures** | ccxt | ❌ | ❌ public API only returns the last 100 trades | Sweep/bias only; max state SETUP unless you untick "requires real footprint" |

Use **🔌 Test connections** (sidebar) to see which hosts the *server running the app* can reach. This is the real answer to
"does Binance work from Streamlit?" — it depends on the server's location, not on your phone.

## Running without a VPS
Binance futures API returns **HTTP 451 for US IPs** and Streamlit Community Cloud is believed to run in the US. The app then
switches automatically (banner shown). Options, all free: (1) deploy on Streamlit Cloud and use the mirror/OKX fallback;
(2) run on your own PC (`streamlit run app.py`) for true futures data; (3) a host with a non-US region.

## Run
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Files
- `app.py` — UI, dashboard switch · `supertrend.py` — Supertrend wick-touch logic + page · `scanner.py` — 2-stage sweep pipeline · `feeds.py` — Binance / mirror / OKX data
- `orderflow.py` — all trading logic (pure functions) · `tests/` — offline tests with synthetic klines + trades

```bash
python tests/test_pipeline.py     # sweep/footprint logic, paging, partial-footprint guard, OKX + mirror paths
python tests/test_supertrend.py   # supertrend properties, FRESH/RETOUCH/cool-down, flip reset
python tests/smoke_app.py         # headless UI runs with mock exchanges: both dashboards, 451 → mirror → OKX fallback
```

## Limitations
- `aggTrades` are *aggregated* trades (same-price, same-side fills merged) — standard for Binance footprints, not tick-by-tick.
- CVD is a rolling window from the first loaded candle, not a session CVD.
- OKX footprint is built from raw trades converted with contract size; OKX history keeps ~3 months. OKX, KuCoin and mirror paths are tested offline with mocks only (live APIs were not reachable from the build sandbox).
- Rules are heuristic, **not a proven edge**. Backtest / paper-trade with fees, funding, slippage first. Not financial advice.
