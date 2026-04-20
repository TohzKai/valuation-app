import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Valuation Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        text-align: center;
    }
    .metric-card .label { font-size: 0.75rem; color: #64748b; margin-bottom: 4px; }
    .metric-card .value { font-size: 1.5rem; font-weight: 700; }
    .bear  { color: #dc2626; }
    .base  { color: #2563eb; }
    .bull  { color: #16a34a; }
    .buy   { background: #dcfce7; border-color: #86efac; }
    .hold  { background: #fef9c3; border-color: #fde047; }
    .avoid { background: #fee2e2; border-color: #fca5a5; }
    .section-header {
        font-size: 1.1rem; font-weight: 700; color: #1e293b;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.4rem; margin-bottom: 1rem; margin-top: 1.5rem;
    }
    div[data-testid="stMetric"] label { font-size: 0.75rem !important; }
    /* Force dark text on all coloured signal banners — readable in dark mode */
    .signal-banner { color: #111827 !important; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ── SGX stock reference table (Yahoo Finance missing data fallback) ──────────
# Yahoo Finance does not reliably return sector/PE/fundamentals for SGX stocks.
# This table fills in known data for common SG stocks. Update after each earnings.
SGX_DATA = {
    # Banks
    "O39.SI": {"name": "OCBC Bank",         "sector": "Banking",          "pe": 9.8,  "book": 9.80,  "roe": 0.135, "dps": 0.88, "beta": 0.75},
    "D05.SI": {"name": "DBS Group",          "sector": "Banking",          "pe": 10.2, "book": 19.20, "roe": 0.180, "dps": 2.16, "beta": 0.85},
    "U11.SI": {"name": "UOB",                "sector": "Banking",          "pe": 9.5,  "book": 25.10, "roe": 0.130, "dps": 1.70, "beta": 0.80},
    # REITs
    "C38U.SI": {"name": "CapitaLand Int. REIT", "sector": "REIT - Retail", "pe": 18.5, "book": 2.10,  "roe": 0.055, "dps": 0.108, "beta": 0.70},
    "A17U.SI": {"name": "CapitaLand Ascendas REIT", "sector": "REIT - Industrial", "pe": 20.1, "book": 1.85, "roe": 0.050, "dps": 0.153, "beta": 0.65},
    "ME8U.SI": {"name": "Mapletree Ind. Trust", "sector": "REIT - Industrial", "pe": 17.8, "book": 1.75, "roe": 0.052, "dps": 0.134, "beta": 0.68},
    "M44U.SI": {"name": "Mapletree Log. Trust", "sector": "REIT - Industrial", "pe": 16.5, "book": 1.42, "roe": 0.048, "dps": 0.090, "beta": 0.62},
    "N2IU.SI": {"name": "Mapletree Pan Asia REIT", "sector": "REIT - Diversified", "pe": 15.2, "book": 1.55, "roe": 0.045, "dps": 0.085, "beta": 0.70},
    "BUOU.SI": {"name": "Frasers L&I Trust",  "sector": "REIT - Industrial", "pe": 14.8, "book": 1.10, "roe": 0.047, "dps": 0.076, "beta": 0.65},
    "J91U.SI": {"name": "Parkway Life REIT",  "sector": "REIT - Healthcare", "pe": 22.0, "book": 2.35, "roe": 0.060, "dps": 0.142, "beta": 0.45},
    # Telcos / Others
    "Z74.SI":  {"name": "Singtel",            "sector": "Telecom",          "pe": 22.0, "book": 1.85,  "roe": 0.062, "dps": 0.15,  "beta": 0.55},
    "BN4.SI":  {"name": "Keppel Corp",        "sector": "Industrials",      "pe": 12.5, "book": 6.20,  "roe": 0.095, "dps": 0.33,  "beta": 0.90},
    "Y92.SI":  {"name": "Thai Bev",           "sector": "Consumer Staples", "pe": 14.2, "book": 0.52,  "roe": 0.120, "dps": 0.045, "beta": 0.60},
    # US stocks (Yahoo works fine but added for completeness)
    "O":    {"name": "Realty Income",         "sector": "REIT - Retail",    "pe": 42.0, "book": 16.50, "roe": 0.030, "dps": 3.07,  "beta": 0.85},
}


def sgx_enrich(ticker: str, data: dict) -> dict:
    """Patch any missing fields from the SGX lookup table."""
    ref = SGX_DATA.get(ticker.upper())
    if not ref:
        return data
    if data is None:
        data = {}
    # Only fill in what Yahoo couldn't provide
    for key, val in ref.items():
        if not data.get(key) or data.get(key) == "—":
            data[key] = val
    return data


# ── Helpers ───────────────────────────────────────────────────────────────────
def _finnhub_symbol(ticker: str) -> str:
    """Convert SGX ticker format for Finnhub. O39.SI -> O39:SP, D05.SI -> D05:SP"""
    if ticker.endswith(".SI"):
        return ticker.replace(".SI", ":SP")
    return ticker  # US tickers work as-is


def _get_api_key() -> str:
    """Get Finnhub API key from Streamlit secrets or session state."""
    try:
        return st.secrets["FINNHUB_API_KEY"]
    except Exception:
        return st.session_state.get("finnhub_key", "")


@st.cache_data(ttl=300)
def fetch_ticker(ticker: str, api_key: str = ""):
    """
    Fetch price + fundamentals via Finnhub (works on Streamlit Cloud).
    Falls back to yfinance for local use if no API key.
    """
    import requests

    key = api_key or _get_api_key()

    # ── Method 1: Finnhub (cloud-safe) ───────────────────────────────────────
    if key:
        fh_sym = _finnhub_symbol(ticker)
        base = "https://finnhub.io/api/v1"
        h = {"X-Finnhub-Token": key}
        try:
            # Quote (live price)
            q = requests.get(f"{base}/quote?symbol={fh_sym}", headers=h, timeout=8).json()
            price = q.get("c") or q.get("pc")  # current or previous close

            # Company profile (name, sector, currency)
            p = requests.get(f"{base}/stock/profile2?symbol={fh_sym}", headers=h, timeout=8).json()
            name     = p.get("name", ticker)
            sector   = p.get("finnhubIndustry") or p.get("gsector") or "—"
            currency = p.get("currency", "")

            # Basic financials (PE, book, beta, etc.)
            f = requests.get(f"{base}/stock/metric?symbol={fh_sym}&metric=all", headers=h, timeout=8).json()
            m = f.get("metric", {})

            pe   = m.get("peBasicExclExtraTTM") or m.get("peTTM")
            book = m.get("bookValuePerShareQuarterly") or m.get("bookValuePerShareAnnual")
            roe  = m.get("roeTTM")
            beta = m.get("beta")
            dps  = m.get("dividendPerShareAnnual")
            pb   = m.get("pbQuarterly") or m.get("pbAnnual")
            eps  = m.get("epsTTM")

            if roe: roe = roe / 100  # Finnhub gives ROE as %, convert to decimal

            return {
                "price":    price,
                "name":     name,
                "sector":   sector,
                "eps":      eps,
                "dps":      dps,
                "book":     book,
                "roe":      roe,
                "beta":     beta,
                "pe":       pe,
                "pb":       pb,
                "mktcap":   p.get("marketCapitalization"),
                "currency": currency,
            }
        except Exception:
            pass

    # ── Method 2: yfinance (local fallback, often blocked on cloud) ───────────
    try:
        t = yf.Ticker(ticker)
        info = t.info
        if info and len(info) > 5:
            price = (info.get("currentPrice") or info.get("regularMarketPrice")
                     or info.get("previousClose"))
            if not price:
                hist = t.history(period="1d")
                if not hist.empty:
                    price = float(hist["Close"].iloc[-1])
            return {
                "price":    price,
                "name":     info.get("longName") or info.get("shortName") or ticker,
                "sector":   info.get("sector", "—"),
                "eps":      info.get("trailingEps"),
                "dps":      info.get("dividendRate") or info.get("trailingAnnualDividendRate"),
                "book":     info.get("bookValue"),
                "roe":      info.get("returnOnEquity"),
                "beta":     info.get("beta"),
                "pe":       info.get("trailingPE"),
                "pb":       info.get("priceToBook"),
                "mktcap":   info.get("marketCap"),
                "currency": info.get("currency", ""),
            }
    except Exception:
        pass

    return None


@st.cache_data(ttl=3600)
def fetch_price_history(ticker: str, period: str = "2y", api_key: str = ""):
    """
    Fetch daily close history. Finnhub first, yfinance fallback.
    """
    import requests
    from datetime import datetime, timedelta

    key = api_key or _get_api_key()

    # ── Method 1: Finnhub candles ─────────────────────────────────────────────
    if key:
        fh_sym = _finnhub_symbol(ticker)
        t_to   = int(datetime.now().timestamp())
        days   = 730 if period == "2y" else 365
        t_from = int((datetime.now() - timedelta(days=days)).timestamp())
        try:
            url = f"https://finnhub.io/api/v1/stock/candle?symbol={fh_sym}&resolution=D&from={t_from}&to={t_to}"
            r = requests.get(url, headers={"X-Finnhub-Token": key}, timeout=15)
            js = r.json()
            if js.get("s") == "ok":
                dates  = pd.to_datetime(js["t"], unit="s")
                closes = js["c"]
                df = pd.DataFrame({"Close": closes}, index=dates)
                df.index = df.index.tz_localize(None)
                return df.dropna()
        except Exception:
            pass

    # ── Method 2: yfinance fallback ───────────────────────────────────────────
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=period)
        if not hist.empty:
            hist.index = hist.index.tz_localize(None)
            return hist[["Close"]].copy()
    except Exception:
        pass

    return None


def position_sizing(current_price, bear, base, bull, max_alloc=100.0):
    """
    Kelly-lite position sizing:
    Edge  = (base - current) / current   -- how much upside to base case
    Risk  = (current - bear) / current   -- how much downside to bear case
    Size  = (Edge / Risk) * max_alloc, capped at max_alloc
    If price already below bear: max position (best entry).
    If price above base: no new position.
    """
    if None in (current_price, bear, base, bull):
        return None, None, None
    edge = (base - current_price) / current_price
    risk = (current_price - bear) / current_price if current_price > bear else 0.001
    if risk <= 0:
        ratio = 1.0
    else:
        ratio = max(0.0, min(1.0, edge / risk))
    size_pct = round(ratio * max_alloc, 1)
    return edge * 100, risk * 100, size_pct


def render_action_panel(current_price, bear, base, bull, ticker, currency=""):
    """Renders the Buy/Hold/Sell price panel + position sizing + price history chart."""

    if None in (bear, base, bull) or bear <= 0 or base <= 0 or bull <= 0:
        st.info("Complete the valuation inputs above to see buy/sell prices and position sizing.")
        return

    cur = current_price or 0.0

    # ── Action prices ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🎯 Action Prices</div>', unsafe_allow_html=True)

    # Sanity check — warn if fair values seem totally off vs current price
    ratio_check = max(bear, bull) / cur if cur > 0 else 1
    if ratio_check < 0.3 or ratio_check > 5:
        st.warning(
            f"⚠️ Fair values ({fmt_price(bear, currency)} – {fmt_price(bull, currency)}) look very different "
            f"from current price ({fmt_price(cur, currency)}). "
            f"Double-check your inputs match the correct asset type — e.g. OCBC (O39.SI) should use **Bank**, not REIT."
        )

    a1, a2, a3, a4 = st.columns(4)

    # BUY price: how far DOWN from current to reach buy zone
    buy_gap = (bear - cur) / cur * 100 if cur else 0
    buy_label = f"{buy_gap:+.1f}% vs current" if bear < cur else f"✅ Already in buy zone!"
    a1.metric("BUY below", fmt_price(bear, currency), delta=buy_label,
              delta_color="inverse")   # red when bear > cur (need to wait), green when <=

    # HOLD target: upside from current to base case
    hold_gap = (base - cur) / cur * 100 if cur else 0
    a2.metric("HOLD / target", fmt_price(base, currency),
              delta=f"{hold_gap:+.1f}% upside to here")

    # TRIM: upside from current to bull case
    trim_gap = (bull - cur) / cur * 100 if cur else 0
    a3.metric("TRIM / sell", fmt_price(bull, currency),
              delta=f"{trim_gap:+.1f}% upside to here")

    a4.metric("Current price", fmt_price(cur, currency))

    # Colour-coded interpretation
    if cur <= bear:
        colour, msg = "#dcfce7", f"✅ STRONG BUY — price is below bear case ({fmt_price(bear, currency)}). Maximum margin of safety."
    elif cur <= base:
        colour, msg = "#fef9c3", f"👀 WATCH — price between buy zone and target. Reasonable entry with smaller position."
    elif cur <= bull:
        colour, msg = "#fef3c7", f"⚠️ HOLD — price above target. Hold existing. Wait for pullback to {fmt_price(bear, currency)} before adding."
    else:
        colour, msg = "#fee2e2", f"🚫 AVOID / TRIM — price {fmt_price(cur, currency)} exceeds bull case {fmt_price(bull, currency)}. Consider reducing position."

    st.markdown(
        f'<div style="background:{colour};border-radius:10px;padding:0.9rem 1.2rem;'
        f'margin:0.5rem 0 1rem;font-weight:600;font-size:0.95rem;color:#111827;">{msg}</div>',
        unsafe_allow_html=True
    )

    # ── Position sizing ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📐 Position Sizing</div>', unsafe_allow_html=True)
    st.caption("How much of your planned allocation to deploy now, based on upside vs downside.")

    max_alloc = st.slider(
        "Your max allocation to this stock (%)", 5, 50, 20, 5,
        help="E.g. if you never want more than 20% in one stock, set to 20."
    )

    edge_pct, risk_pct, size_pct = position_sizing(cur, bear, base, bull, max_alloc)

    if edge_pct is not None:
        p1, p2, p3 = st.columns(3)
        p1.metric("Upside to base case", f"{edge_pct:+.1f}%")
        p2.metric("Downside to bear case", f"{risk_pct:.1f}%")
        p3.metric("Suggested position size", f"{size_pct:.0f}% of portfolio")

        # Visual bar
        bar_pct = min(int(size_pct), 100)
        bar_color = "#16a34a" if bar_pct >= 15 else "#ca8a04" if bar_pct >= 5 else "#dc2626"
        st.markdown(
            f'<div style="background:#e2e8f0;border-radius:99px;height:12px;margin:0.3rem 0 0.6rem;">'
            f'<div style="background:{bar_color};width:{bar_pct}%;height:100%;border-radius:99px;"></div></div>'
            f'<div style="font-size:0.8rem;color:#64748b;">Deploy {size_pct:.0f}% of your portfolio in this stock now '
            f'(based on {edge_pct:+.1f}% upside vs {risk_pct:.1f}% downside)</div>',
            unsafe_allow_html=True
        )

        with st.expander("📖 How position sizing works"):
            st.markdown("""
**Kelly-lite formula used here:**

`Position size = (Upside to base ÷ Downside to bear) × Max allocation`

- **Upside to base** = how much the stock can gain if the base case plays out
- **Downside to bear** = how much you could lose if the bear case plays out
- High upside + low downside = large position. Equal upside/downside = half max allocation.

**Example:** Base case $20, Bear case $14, Current price $15, Max alloc 20%
- Upside = (20−15)/15 = 33%
- Downside = (15−14)/15 = 6.7%
- Ratio = 33/6.7 = 4.9 → capped at 1.0 → deploy 20% (full allocation)

**Rule of thumb:**
- Price at or below bear case → deploy full allocation
- Price between bear and base → deploy proportionally
- Price above base → deploy 0% (wait for pullback)
            """)

    # ── Price vs Fair Value Band chart ───────────────────────────────────────
    st.markdown('<div class="section-header">📈 Price vs Fair Value Band (2 Years)</div>', unsafe_allow_html=True)

    hist = fetch_price_history(ticker, api_key=st.session_state.get('finnhub_key',''))
    if hist is not None and not hist.empty:
        hist["Buy zone (bear)"] = bear
        hist["Target (base)"] = base
        hist["Trim (bull)"] = bull

        import altair as alt

        hist_reset = hist.reset_index()
        hist_reset.columns = ["Date", "Price", "Buy zone", "Target", "Trim"]

        base_chart = alt.Chart(hist_reset)

        band = base_chart.mark_area(opacity=0.15, color="#2563eb").encode(
            x=alt.X("Date:T", title=""),
            y=alt.Y("Buy zone:Q", title="Price", scale=alt.Scale(zero=False)),
            y2="Trim:Q"
        )

        price_line = base_chart.mark_line(color="#1e293b", strokeWidth=2).encode(
            x="Date:T",
            y=alt.Y("Price:Q", scale=alt.Scale(zero=False)),
            tooltip=["Date:T", "Price:Q"]
        )

        bear_line = base_chart.mark_line(color="#dc2626", strokeDash=[4,2], strokeWidth=1.5).encode(
            x="Date:T", y="Buy zone:Q"
        )

        base_line = base_chart.mark_line(color="#2563eb", strokeDash=[4,2], strokeWidth=1.5).encode(
            x="Date:T", y="Target:Q"
        )

        bull_line = base_chart.mark_line(color="#16a34a", strokeDash=[4,2], strokeWidth=1.5).encode(
            x="Date:T", y="Trim:Q"
        )

        chart = (band + price_line + bear_line + base_line + bull_line).properties(
            height=320
        ).interactive()

        st.altair_chart(chart, use_container_width=True)
        st.caption("🔴 BUY line (bear case)  ·  🔵 TARGET line (base case)  ·  🟢 TRIM line (bull case)  ·  ⬛ Actual price")

        # How many days was stock in buy zone?
        days_in_buy = (hist["Close"] <= bear).sum()
        days_total = len(hist)
        days_in_hold = ((hist["Close"] > bear) & (hist["Close"] <= base)).sum()
        days_above_sell = (hist["Close"] > bull).sum()

        d1, d2, d3 = st.columns(3)
        d1.metric("Days in buy zone (2yr)", f"{days_in_buy} / {days_total}",
                  delta=f"{days_in_buy/days_total*100:.0f}% of time")
        d2.metric("Days in hold zone", f"{days_in_hold} / {days_total}",
                  delta=f"{days_in_hold/days_total*100:.0f}% of time")
        d3.metric("Days above trim price", f"{days_above_sell} / {days_total}",
                  delta=f"{days_above_sell/days_total*100:.0f}% of time")
    else:
        st.info("Could not load price history for this ticker. Check the ticker symbol includes .SI for SGX stocks.")



def fmt_price(v, currency=""):
    if v is None:
        return "—"
    return f"{currency} {v:,.2f}".strip()


def signal(price, bear, bull):
    if price is None:
        return "—", ""
    if price <= bear:
        return "BUY — price is at or below bear case", "buy"
    elif price <= (bear + bull) / 2:
        return "WATCH — approaching fair value", "hold"
    elif price <= bull:
        return "HOLD — inside fair value range", "hold"
    else:
        return "AVOID — price exceeds bull case", "avoid"


def dcf_price(fcf, growth_bear, growth_base, growth_bull,
              wacc_bear, wacc_base, wacc_bull,
              terminal_growth, years=10):
    results = {}
    for label, g, w in [
        ("Bear", growth_bear / 100, wacc_bear / 100),
        ("Base", growth_base / 100, wacc_base / 100),
        ("Bull", growth_bull / 100, wacc_bull / 100),
    ]:
        flows = [fcf * (1 + g) ** t for t in range(1, years + 1)]
        pv = sum(f / (1 + w) ** t for t, f in enumerate(flows, 1))
        tv = flows[-1] * (1 + terminal_growth / 100) / (w - terminal_growth / 100)
        pv_tv = tv / (1 + w) ** years
        results[label] = pv + pv_tv
    return results


def ddm_price(dps, growth_bear, growth_base, growth_bull,
              coe_bear, coe_base, coe_bull):
    results = {}
    for label, g, r in [
        ("Bear", growth_bear / 100, coe_bear / 100),
        ("Base", growth_base / 100, coe_base / 100),
        ("Bull", growth_bull / 100, coe_bull / 100),
    ]:
        if r <= g:
            results[label] = None
        else:
            results[label] = dps * (1 + g) / (r - g)
    return results


def ffo_price(ffo_per_unit, pb_bear, pb_base, pb_bull,
              yield_bear, yield_base, yield_bull,
              dpu=None):
    """
    For REITs: two methods
    1. Price / FFO multiple approach
    2. Dividend Yield approach (if DPU provided)
    """
    results_ffo = {}
    for label, multiple in [("Bear", pb_bear), ("Base", pb_base), ("Bull", pb_bull)]:
        results_ffo[label] = ffo_per_unit * multiple

    results_yield = {}
    if dpu:
        for label, y in [("Bear", yield_bear / 100), ("Base", yield_base / 100), ("Bull", yield_bull / 100)]:
            results_yield[label] = dpu / y if y > 0 else None

    return results_ffo, results_yield


def pb_roe_price(book, roe, coe_bear, coe_base, coe_bull):
    """Gordon Growth variant for banks: P/B = (ROE - g) / (COE - g)"""
    results = {}
    g = 0.03  # assumed long-run growth
    roe_dec = (roe or 0.10)
    for label, coe in [
        ("Bear", coe_bear / 100),
        ("Base", coe_base / 100),
        ("Bull", coe_bull / 100),
    ]:
        if coe <= g:
            results[label] = None
        else:
            justified_pb = (roe_dec - g) / (coe - g)
            results[label] = book * max(justified_pb, 0)
    return results


# ── Sidebar: ticker + asset type ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Valuation Tool")
    st.markdown("*Bear / Base / Bull scenarios*")
    st.divider()

    ticker_input = st.text_input("Ticker symbol", value="O39.SI",
                                 help="SGX: e.g. C38U.SI (CapitaLand REIT)\nUS: e.g. O (Realty Income)\nBank: e.g. D05.SI (DBS)")
    asset_type = st.selectbox("Asset type", ["REIT", "Bank", "Company (DCF)", "Company (DDM)"])

    st.divider()

    st.markdown("**🔑 Finnhub API Key**")
    st.caption("Free key at [finnhub.io](https://finnhub.io) — required for live data on cloud")
    api_key_input = st.text_input("API Key", value=st.session_state.get("finnhub_key",""),
                                   type="password", placeholder="Paste your free Finnhub key")
    if api_key_input:
        st.session_state["finnhub_key"] = api_key_input

    fetch_btn = st.button("🔄 Fetch live price", use_container_width=True)

    st.markdown("---")
    st.caption("Data via Finnhub · For educational use only · Not financial advice")


# ── Fetch data ────────────────────────────────────────────────────────────────
ticker = ticker_input.strip().upper()
_key = st.session_state.get("finnhub_key", "")
data = fetch_ticker(ticker, api_key=_key) if ticker else None
data = sgx_enrich(ticker, data)  # fill missing sector/PE/fundamentals from lookup table

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📊 Stock Valuation Tool")
st.caption(f"Analysing: **{ticker}** — {asset_type}")

if data:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Company", data["name"][:28] if data["name"] else "—")
    live_price = data.get("price")
    if live_price:
        price_display = fmt_price(live_price, data["currency"])
    else:
        price_display = "—"
    c2.metric("Live price", price_display)
    c3.metric("Sector", data["sector"] if data.get("sector") and data["sector"] != "—" else "—")
    c4.metric("P/E ratio", f"{data['pe']:.1f}x" if data.get("pe") else "—")
    if not live_price:
        st.info("💡 **Live price unavailable** — enter your Finnhub API key in the sidebar to enable live prices and the price history chart. Get a free key at [finnhub.io](https://finnhub.io) (takes 1 minute, no credit card).")
else:
    st.info("💡 **Enter your Finnhub API key** in the sidebar for live prices. Get a free key at [finnhub.io](https://finnhub.io). You can still use the valuation models manually without it.")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# REIT VALUATION
# ═══════════════════════════════════════════════════════════════════════════════
if asset_type == "REIT":
    st.markdown('<div class="section-header">REIT Inputs</div>', unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown("**Fundamentals**")
        ffo = st.number_input("FFO per unit / share (annual)", value=0.12, step=0.01, format="%.3f",
                              help="Funds From Operations per unit. Found in REIT annual reports.")
        dpu = st.number_input("DPU / Distribution per unit (annual)", value=0.10, step=0.01, format="%.3f",
                              help="Total annual distribution per unit paid to unitholders.")
        current_price = st.number_input("Current price (override)", value=float(data["price"]) if data and data["price"] else 1.20,
                                        step=0.01, format="%.3f")

    with col_r:
        st.markdown("**Method 1: Price / FFO multiple**")
        pb_bear = st.slider("Bear FFO multiple (x)", 8.0, 20.0, 12.0, 0.5,
                            help="Low end — stressed market, rising rates")
        pb_base = st.slider("Base FFO multiple (x)", 8.0, 25.0, 15.0, 0.5)
        pb_bull = st.slider("Bull FFO multiple (x)", 10.0, 30.0, 18.0, 0.5)

        st.markdown("**Method 2: Dividend yield**")
        yield_bear = st.slider("Bear yield target (%)", 3.0, 10.0, 6.5, 0.1,
                               help="Higher yield = lower implied price (stressed)")
        yield_base = st.slider("Base yield target (%)", 2.0, 9.0, 5.0, 0.1)
        yield_bull = st.slider("Bull yield target (%)", 1.5, 8.0, 3.8, 0.1)

    ffo_results, yield_results = ffo_price(ffo, pb_bear, pb_base, pb_bull,
                                           yield_bear, yield_base, yield_bull, dpu)

    st.markdown('<div class="section-header">Results: REIT Valuation</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Method 1 — Price/FFO Multiple", "Method 2 — Dividend Yield"])

    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Bear fair value", fmt_price(ffo_results["Bear"]), delta=f"{(ffo_results['Bear'] - current_price) / current_price * 100:.1f}% vs price")
        c2.metric("Base fair value", fmt_price(ffo_results["Base"]), delta=f"{(ffo_results['Base'] - current_price) / current_price * 100:.1f}% vs price")
        c3.metric("Bull fair value", fmt_price(ffo_results["Bull"]), delta=f"{(ffo_results['Bull'] - current_price) / current_price * 100:.1f}% vs price")
        sig, css = signal(current_price, ffo_results["Bear"], ffo_results["Bull"])
        c4.metric("Current price", fmt_price(current_price))
        st.markdown(f'<div class="metric-card {css}" style="margin-top:0.5rem;color:#111827;"><div class="label">Signal</div><div class="value" style="color:#111827;">{sig}</div></div>', unsafe_allow_html=True)

    with tab2:
        if yield_results:
            c1, c2, c3, c4 = st.columns(4)
            bv, bav, blv = yield_results.get("Bear"), yield_results.get("Base"), yield_results.get("Bull")
            c1.metric("Bear fair value (yield)", fmt_price(bv))
            c2.metric("Base fair value (yield)", fmt_price(bav))
            c3.metric("Bull fair value (yield)", fmt_price(blv))
            if bv and blv:
                sig, css = signal(current_price, bv, blv)
                c4.metric("Current price", fmt_price(current_price))
                st.markdown(f'<div class="metric-card {css}" style="margin-top:0.5rem;color:#111827;"><div class="label">Signal</div><div class="value" style="color:#111827;">{sig}</div></div>', unsafe_allow_html=True)

    # Range chart
    st.markdown('<div class="section-header">Fair Value Range</div>', unsafe_allow_html=True)
    chart_data = pd.DataFrame({
        "Scenario": ["Bear", "Base", "Bull", "Current price"],
        "Price/FFO method": [ffo_results["Bear"], ffo_results["Base"], ffo_results["Bull"], current_price],
        "Yield method": [yield_results.get("Bear"), yield_results.get("Base"), yield_results.get("Bull"), current_price],
    }).set_index("Scenario")
    st.bar_chart(chart_data)

    render_action_panel(current_price, ffo_results["Bear"], ffo_results["Base"], ffo_results["Bull"], ticker, data["currency"] if data else "")

    with st.expander("📖 How to read REIT valuation"):
        st.markdown("""
**Price/FFO multiple** is the REIT equivalent of P/E. A multiple of 15x means investors pay 15x annual FFO.
- Singapore REITs typically trade between 12x–20x FFO in normal markets
- Rising interest rate environments compress multiples toward 10–14x
- Strong-sponsor, large-cap REITs command premium multiples (18x+)

**Dividend yield method** anchors on what yield the market demands. If a REIT pays $0.10 DPU and the market demands 5% yield, fair value = $0.10 / 0.05 = $2.00.
- Bear case: market demands higher yield (risk-off) → lower price
- Bull case: market accepts lower yield (risk-on) → higher price

**Buy zone**: price at or below bear case fair value = maximum margin of safety.
        """)


# ═══════════════════════════════════════════════════════════════════════════════
# BANK VALUATION
# ═══════════════════════════════════════════════════════════════════════════════
elif asset_type == "Bank":
    st.markdown('<div class="section-header">Bank Inputs</div>', unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown("**Fundamentals**")
        book = st.number_input("Book value per share", value=float(data["book"]) if data and data["book"] else 10.0,
                               step=0.1, format="%.2f", help="Net assets / shares outstanding. In annual report.")
        roe = st.number_input("Return on Equity — ROE (%)", value=float(data["roe"] * 100) if data and data["roe"] else 12.0,
                              step=0.5, format="%.1f") / 100
        dps = st.number_input("Dividend per share (annual)", value=float(data["dps"]) if data and data["dps"] else 0.50,
                              step=0.05, format="%.2f")
        current_price = st.number_input("Current price (override)", value=float(data["price"]) if data and data["price"] else 10.0,
                                        step=0.05, format="%.2f")

    with col_r:
        st.markdown("**Cost of Equity (COE) by scenario**")
        st.caption("COE = risk-free rate + beta × equity risk premium. DBS/OCBC typically 8–11%.")
        coe_bear = st.slider("Bear COE — %", 8.0, 16.0, 12.0, 0.5,
                             help="High rates / low growth environment")
        coe_base = st.slider("Base COE — %", 7.0, 14.0, 9.5, 0.5)
        coe_bull = st.slider("Bull COE — %", 5.0, 12.0, 7.5, 0.5)

        st.markdown("**DDM (dividend cross-check)**")
        ddm_g_bear = st.slider("Dividend growth — Bear %", 0.0, 8.0, 1.0, 0.5)
        ddm_g_base = st.slider("Dividend growth — Base %", 0.0, 10.0, 4.0, 0.5)
        ddm_g_bull = st.slider("Dividend growth — Bull %", 0.0, 12.0, 6.0, 0.5)

    pb_results = pb_roe_price(book, roe, coe_bear, coe_base, coe_bull)
    ddm_results = ddm_price(dps, ddm_g_bear, ddm_g_base, ddm_g_bull, coe_bear, coe_base, coe_bull)

    st.markdown('<div class="section-header">Results: Bank Valuation</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Method 1 — Justified P/B (ROE model)", "Method 2 — DDM cross-check"])

    with tab1:
        st.caption(f"ROE = {roe*100:.1f}% · Book value = {book:.2f} · Assumed long-run growth = 3%")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Bear fair value", fmt_price(pb_results["Bear"]),
                  delta=f"{(pb_results['Bear'] - current_price)/current_price*100:.1f}% vs price" if pb_results["Bear"] else "—")
        c2.metric("Base fair value", fmt_price(pb_results["Base"]),
                  delta=f"{(pb_results['Base'] - current_price)/current_price*100:.1f}% vs price" if pb_results["Base"] else "—")
        c3.metric("Bull fair value", fmt_price(pb_results["Bull"]),
                  delta=f"{(pb_results['Bull'] - current_price)/current_price*100:.1f}% vs price" if pb_results["Bull"] else "—")
        if pb_results["Bear"] and pb_results["Bull"]:
            sig, css = signal(current_price, pb_results["Bear"], pb_results["Bull"])
            c4.metric("Current price", fmt_price(current_price))
            st.markdown(f'<div class="metric-card {css}" style="margin-top:0.5rem;color:#111827;"><div class="label">Signal</div><div class="value" style="color:#111827;">{sig}</div></div>', unsafe_allow_html=True)

        # Show P/B ratios
        st.markdown("**Implied P/B multiples**")
        pb_cols = st.columns(3)
        for col, label in zip(pb_cols, ["Bear", "Base", "Bull"]):
            v = pb_results[label]
            col.metric(f"{label} P/B", f"{v/book:.2f}x" if v else "—")

    with tab2:
        c1, c2, c3, c4 = st.columns(4)
        bv = ddm_results.get("Bear")
        bav = ddm_results.get("Base")
        blv = ddm_results.get("Bull")
        c1.metric("Bear DDM value", fmt_price(bv))
        c2.metric("Base DDM value", fmt_price(bav))
        c3.metric("Bull DDM value", fmt_price(blv))
        c4.metric("Current price", fmt_price(current_price))
        if bv and blv:
            sig, css = signal(current_price, bv, blv)
            st.markdown(f'<div class="metric-card {css}" style="margin-top:0.5rem;color:#111827;"><div class="label">Signal</div><div class="value" style="color:#111827;">{sig}</div></div>', unsafe_allow_html=True)

    chart_data = pd.DataFrame({
        "Scenario": ["Bear", "Base", "Bull", "Current"],
        "P/B model": [pb_results["Bear"], pb_results["Base"], pb_results["Bull"], current_price],
        "DDM": [ddm_results.get("Bear"), ddm_results.get("Base"), ddm_results.get("Bull"), current_price],
    }).set_index("Scenario")
    st.markdown('<div class="section-header">Fair Value Range</div>', unsafe_allow_html=True)
    st.bar_chart(chart_data)

    render_action_panel(current_price, pb_results["Bear"], pb_results["Base"], pb_results["Bull"], ticker, data["currency"] if data else "")

    with st.expander("📖 How to read Bank valuation"):
        st.markdown("""
**Justified P/B model** is the primary tool for banks. Logic: if a bank earns ROE > cost of equity, it deserves to trade above book (P/B > 1). If ROE < COE, it should trade below book.

Formula: **Justified P/B = (ROE − g) / (COE − g)**

- DBS typically trades at P/B 1.5–2.0x because ROE (~15%) well exceeds COE (~9%)
- A bank with ROE = COE should theoretically trade at exactly 1x book
- Bear case: higher COE (rates rise, risk premium expands) → lower justified P/B

**DDM** acts as a cross-check using dividend history. If both methods give similar answers, you have higher conviction.
        """)


# ═══════════════════════════════════════════════════════════════════════════════
# COMPANY — DCF
# ═══════════════════════════════════════════════════════════════════════════════
elif asset_type == "Company (DCF)":
    st.markdown('<div class="section-header">DCF Inputs</div>', unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown("**Fundamentals**")
        fcf = st.number_input("Free Cash Flow per share (annual, $)", value=2.50, step=0.10, format="%.2f",
                              help="FCF = Operating Cash Flow − Capex. Divide by shares outstanding.")
        current_price = st.number_input("Current price (override)", value=float(data["price"]) if data and data["price"] else 50.0,
                                        step=0.50, format="%.2f")
        years = st.slider("Projection period (years)", 5, 15, 10)
        terminal_g = st.slider("Terminal growth rate (%)", 0.5, 4.0, 2.5, 0.1,
                               help="Long-run GDP growth. Very sensitive — stress-test this carefully.")

        st.warning(f"⚠️ Changing terminal growth by 0.5% can move fair value by 20–30%. This is your biggest risk.")

    with col_r:
        st.markdown("**FCF Growth by scenario**")
        g_bear = st.slider("Bear FCF growth (%/yr)", -5.0, 15.0, 3.0, 0.5)
        g_base = st.slider("Base FCF growth (%/yr)", 0.0, 25.0, 8.0, 0.5)
        g_bull = st.slider("Bull FCF growth (%/yr)", 5.0, 35.0, 14.0, 0.5)

        st.markdown("**WACC / Discount Rate by scenario**")
        st.caption("WACC = weighted cost of capital. Higher rate → lower fair value.")
        wacc_bear = st.slider("Bear WACC (%)", 8.0, 18.0, 12.0, 0.5,
                              help="Stressed environment: high rates, high risk premium")
        wacc_base = st.slider("Base WACC (%)", 6.0, 15.0, 9.0, 0.5)
        wacc_bull = st.slider("Bull WACC (%)", 4.0, 12.0, 7.0, 0.5)

    dcf_results = dcf_price(fcf, g_bear, g_base, g_bull,
                            wacc_bear, wacc_base, wacc_bull,
                            terminal_g, years)

    st.markdown('<div class="section-header">Results: DCF Valuation</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Bear fair value", fmt_price(dcf_results["Bear"]),
              delta=f"{(dcf_results['Bear'] - current_price)/current_price*100:.1f}% vs price")
    c2.metric("Base fair value", fmt_price(dcf_results["Base"]),
              delta=f"{(dcf_results['Base'] - current_price)/current_price*100:.1f}% vs price")
    c3.metric("Bull fair value", fmt_price(dcf_results["Bull"]),
              delta=f"{(dcf_results['Bull'] - current_price)/current_price*100:.1f}% vs price")
    c4.metric("Current price", fmt_price(current_price))

    sig, css = signal(current_price, dcf_results["Bear"], dcf_results["Bull"])
    st.markdown(f'<div class="metric-card {css}" style="margin-top:0.5rem;color:#111827;"><div class="label">Signal</div><div class="value" style="color:#111827;">{sig}</div></div>', unsafe_allow_html=True)

    # Terminal value breakdown
    st.markdown('<div class="section-header">Value Breakdown (Base Case)</div>', unsafe_allow_html=True)
    w = wacc_base / 100
    g = g_base / 100
    flows_base = [fcf * (1 + g) ** t for t in range(1, years + 1)]
    pv_flows = sum(f / (1 + w) ** t for t, f in enumerate(flows_base, 1))
    tv = flows_base[-1] * (1 + terminal_g / 100) / (w - terminal_g / 100)
    pv_tv = tv / (1 + w) ** years
    total = pv_flows + pv_tv

    b1, b2, b3 = st.columns(3)
    b1.metric("PV of cash flows", f"${pv_flows:.2f}", delta=f"{pv_flows/total*100:.0f}% of total")
    b2.metric("PV of terminal value", f"${pv_tv:.2f}", delta=f"{pv_tv/total*100:.0f}% of total")
    b3.metric("Total intrinsic value", f"${total:.2f}")

    if pv_tv / total > 0.70:
        st.warning(f"⚠️ Terminal value is {pv_tv/total*100:.0f}% of your base case valuation — highly sensitive to terminal growth assumption. Stress-test it.")

    chart_data = pd.DataFrame({
        "Scenario": ["Bear", "Base", "Bull", "Current price"],
        "Fair value ($)": [dcf_results["Bear"], dcf_results["Base"], dcf_results["Bull"], current_price],
    }).set_index("Scenario")
    st.markdown('<div class="section-header">Fair Value Range</div>', unsafe_allow_html=True)
    st.bar_chart(chart_data)

    render_action_panel(current_price, dcf_results["Bear"], dcf_results["Base"], dcf_results["Bull"], ticker, data["currency"] if data else "")

    with st.expander("📖 How to read DCF valuation"):
        st.markdown("""
**DCF** discounts all future free cash flows back to today's value. Two components:

1. **PV of cash flows** — projected FCF for your chosen period, discounted by WACC
2. **PV of terminal value** — all cash flows beyond the projection period (using Gordon Growth model)

**Key sensitivities to watch:**
- Terminal growth rate: the single most dangerous assumption. Never use > nominal GDP growth (~3.5%)
- WACC: if risk-free rate (US 10yr / SG 10yr) rises, your WACC should rise too
- FCF quality: make sure FCF is genuine (not inflated by working capital timing or low maintenance capex)

**Buy rule**: only buy at or below your *bear case* — that gives you margin of safety even if the bad scenario plays out.
        """)


# ═══════════════════════════════════════════════════════════════════════════════
# COMPANY — DDM
# ═══════════════════════════════════════════════════════════════════════════════
elif asset_type == "Company (DDM)":
    st.markdown('<div class="section-header">DDM Inputs</div>', unsafe_allow_html=True)
    st.info("DDM works best for stable dividend-paying companies with predictable payout ratios (utilities, consumer staples, telcos).")

    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown("**Fundamentals**")
        dps_val = st.number_input("Dividend per share — current (annual)", value=float(data["dps"]) if data and data["dps"] else 1.20,
                                  step=0.05, format="%.2f")
        current_price = st.number_input("Current price (override)", value=float(data["price"]) if data and data["price"] else 25.0,
                                        step=0.25, format="%.2f")
        payout = st.slider("Payout ratio (%)", 20, 100, 60,
                           help="% of earnings paid as dividends. Helps assess sustainability.")
        st.caption(f"Implied EPS = ${dps_val / (payout/100):.2f}")

    with col_r:
        st.markdown("**Dividend growth by scenario**")
        g_bear = st.slider("Bear growth (%/yr)", 0.0, 8.0, 1.0, 0.5,
                           help="Low growth: recession, sector headwinds")
        g_base = st.slider("Base growth (%/yr)", 0.0, 12.0, 4.0, 0.5)
        g_bull = st.slider("Bull growth (%/yr)", 2.0, 15.0, 7.0, 0.5)

        st.markdown("**Cost of equity (COE) by scenario**")
        coe_bear = st.slider("Bear COE (%)", 7.0, 16.0, 11.0, 0.5)
        coe_base = st.slider("Base COE (%)", 5.0, 14.0, 8.0, 0.5)
        coe_bull = st.slider("Bull COE (%)", 3.0, 12.0, 6.0, 0.5)

    ddm_results = ddm_price(dps_val, g_bear, g_base, g_bull, coe_bear, coe_base, coe_bull)

    st.markdown('<div class="section-header">Results: DDM Valuation</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    bv = ddm_results.get("Bear")
    bav = ddm_results.get("Base")
    blv = ddm_results.get("Bull")

    c1.metric("Bear fair value", fmt_price(bv) if bv else "N/A — COE ≤ growth",
              delta=f"{(bv - current_price)/current_price*100:.1f}% vs price" if bv else None)
    c2.metric("Base fair value", fmt_price(bav) if bav else "N/A",
              delta=f"{(bav - current_price)/current_price*100:.1f}% vs price" if bav else None)
    c3.metric("Bull fair value", fmt_price(blv) if blv else "N/A",
              delta=f"{(blv - current_price)/current_price*100:.1f}% vs price" if blv else None)
    c4.metric("Current price", fmt_price(current_price))

    if bv and blv:
        sig, css = signal(current_price, bv, blv)
        st.markdown(f'<div class="metric-card {css}" style="margin-top:0.5rem;color:#111827;"><div class="label">Signal</div><div class="value" style="color:#111827;">{sig}</div></div>', unsafe_allow_html=True)
    else:
        st.error("One or more scenarios invalid — cost of equity must be greater than dividend growth rate.")

    # Current yield context
    curr_yield = dps_val / current_price * 100
    st.markdown('<div class="section-header">Yield Context</div>', unsafe_allow_html=True)
    y1, y2, y3 = st.columns(3)
    y1.metric("Current yield", f"{curr_yield:.2f}%")
    y2.metric("Bear yield (at bear price)", f"{dps_val/bv*100:.2f}%" if bv else "—")
    y3.metric("Bull yield (at bull price)", f"{dps_val/blv*100:.2f}%" if blv else "—")

    chart_data = pd.DataFrame({
        "Scenario": ["Bear", "Base", "Bull", "Current price"],
        "Fair value": [bv, bav, blv, current_price],
    }).set_index("Scenario")
    st.markdown('<div class="section-header">Fair Value Range</div>', unsafe_allow_html=True)
    st.bar_chart(chart_data)

    if bv and bav and blv:
        render_action_panel(current_price, bv, bav, blv, ticker, data["currency"] if data else "")

    with st.expander("📖 How to read DDM valuation"):
        st.markdown("""
**Gordon Growth Model (DDM)**: Fair value = Next year's DPS / (COE − growth rate)

Works best when:
- Company has stable, predictable dividend history (10+ years)
- Payout ratio is sustainable (below 80% for most sectors)
- Growth rate is materially lower than cost of equity

**Warning signs**:
- If growth ≥ COE, the model breaks (outputs infinity) — this means the stock is likely overvalued
- High payout ratios (>90%) with high growth assumptions are often unsustainable
- DDM understates value for companies that retain and reinvest earnings well (use DCF instead)
        """)


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("⚠️ This tool is for educational and research purposes only. It is not financial advice. Always do your own due diligence before investing. Price data via Yahoo Finance and may be delayed.")
