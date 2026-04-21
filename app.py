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
        font-size: 1rem; font-weight: 700;
        color: #2563eb !important;
        border-bottom: 2px solid #2563eb;
        padding: 0.5rem 0 0.4rem 0;
        margin-bottom: 1rem; margin-top: 1.5rem;
        display: block; width: 100%;
        background: transparent;
        letter-spacing: 0.01em;
    }
    div[data-testid="stMetric"] label { font-size: 0.75rem !important; }
    /* Force dark text on all coloured signal banners — readable in dark mode */
    .signal-banner { color: #111827 !important; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ── Stock reference table (Yahoo Finance missing data fallback) ──────────────
# Covers SGX stocks (Yahoo rarely returns fundamentals) and major US stocks.
# FCF/share is the key DCF input. Update after each earnings season.
# Last updated: April 2026
SGX_DATA = {
    # ── SGX Banks — updated FY2025 ───────────────────────────────────────────
    # OCBC FY2025: profit $7.8B, ROE ~13%, DPS $0.99 ordinary + $0.16 special = $1.15 total
    # DBS FY2024: profit $11.4B, ROE 18%, DPS $2.22 ordinary + capital return
    # UOB FY2025: ROE ~13%, DPS includes $0.50 special (90th anniversary)
    "O39.SI": {"name": "OCBC Bank",              "sector": "Banking",           "pe": 10.5, "book": 11.20, "roe": 0.130, "dps": 1.15,  "beta": 0.75, "fcf_per_share": None, "_updated": "FY2025"},
    "D05.SI": {"name": "DBS Group",              "sector": "Banking",           "pe": 11.0, "book": 23.00, "roe": 0.180, "dps": 2.22,  "beta": 0.85, "fcf_per_share": None, "_updated": "FY2024"},
    "U11.SI": {"name": "UOB",                    "sector": "Banking",           "pe": 10.2, "book": 27.50, "roe": 0.130, "dps": 2.00,  "beta": 0.80, "fcf_per_share": None, "_updated": "FY2025"},
    # ── SGX REITs ─────────────────────────────────────────────────────────────
    "C38U.SI": {"name": "CapitaLand Int. REIT",  "sector": "REIT - Retail",     "pe": 18.5, "book": 2.10,  "roe": 0.055, "dps": 0.108, "beta": 0.70, "fcf_per_share": None},
    "A17U.SI": {"name": "CapitaLand Ascendas",   "sector": "REIT - Industrial", "pe": 20.1, "book": 1.85,  "roe": 0.050, "dps": 0.153, "beta": 0.65, "fcf_per_share": None},
    "ME8U.SI": {"name": "Mapletree Ind. Trust",  "sector": "REIT - Industrial", "pe": 17.8, "book": 1.75,  "roe": 0.052, "dps": 0.134, "beta": 0.68, "fcf_per_share": None},
    "M44U.SI": {"name": "Mapletree Log. Trust",  "sector": "REIT - Industrial", "pe": 16.5, "book": 1.42,  "roe": 0.048, "dps": 0.090, "beta": 0.62, "fcf_per_share": None},
    "N2IU.SI": {"name": "Mapletree Pan Asia",    "sector": "REIT - Diversified","pe": 15.2, "book": 1.55,  "roe": 0.045, "dps": 0.085, "beta": 0.70, "fcf_per_share": None},
    "BUOU.SI": {"name": "Frasers L&I Trust",     "sector": "REIT - Industrial", "pe": 14.8, "book": 1.10,  "roe": 0.047, "dps": 0.076, "beta": 0.65, "fcf_per_share": None},
    "J91U.SI": {"name": "Parkway Life REIT",     "sector": "REIT - Healthcare", "pe": 22.0, "book": 2.35,  "roe": 0.060, "dps": 0.142, "beta": 0.45, "fcf_per_share": None},
    "Z74.SI":  {"name": "Singtel",               "sector": "Telecom",           "pe": 22.0, "book": 1.85,  "roe": 0.062, "dps": 0.150, "beta": 0.55, "fcf_per_share": 0.12},
    "BN4.SI":  {"name": "Keppel Corp",           "sector": "Industrials",       "pe": 12.5, "book": 6.20,  "roe": 0.095, "dps": 0.330, "beta": 0.90, "fcf_per_share": 0.45},
    "Y92.SI":  {"name": "Thai Bev",              "sector": "Consumer Staples",  "pe": 14.2, "book": 0.52,  "roe": 0.120, "dps": 0.045, "beta": 0.60, "fcf_per_share": 0.04},
    # ── US REITs ──────────────────────────────────────────────────────────────
    "O":    {"name": "Realty Income",            "sector": "REIT - Retail",     "pe": 42.0, "book": 16.50, "roe": 0.030, "dps": 3.07,  "beta": 0.85, "fcf_per_share": 3.20},
    "PLD":  {"name": "Prologis",                 "sector": "REIT - Industrial", "pe": 35.0, "book": 52.00, "roe": 0.045, "dps": 3.84,  "beta": 0.90, "fcf_per_share": 4.10},
    "AMT":  {"name": "American Tower",           "sector": "REIT - Telecom",    "pe": 40.0, "book": 5.00,  "roe": 0.080, "dps": 6.48,  "beta": 0.85, "fcf_per_share": 7.20},
    # ── US Big Tech ───────────────────────────────────────────────────────────
    # FCF/share = Annual FCF ÷ diluted shares outstanding (FY2024 actuals)
    "AAPL":  {"name": "Apple Inc.",              "sector": "Technology",        "pe": 33.0, "book": 4.00,  "roe": 1.600, "dps": 1.00,  "beta": 1.20, "fcf_per_share": 7.20, "fcf_g_bear": 5.0, "fcf_g_base": 10.0, "fcf_g_bull": 15.0, "wacc_bear": 11.0, "wacc_base": 9.0, "wacc_bull": 7.5},
    # Apple FY2024: FCF ~$108B ÷ 15.4B shares = $6.90/share; EPS $6.11
    "MSFT":  {"name": "Microsoft Corp",          "sector": "Technology",        "pe": 35.0, "book": 36.00, "roe": 0.380, "dps": 3.00,  "beta": 0.90, "fcf_per_share": 12.50, "fcf_g_bear": 8.0, "fcf_g_base": 15.0, "fcf_g_bull": 22.0, "wacc_bear": 10.0, "wacc_base": 8.5, "wacc_bull": 7.0},
    # MSFT FY2024: FCF ~$74B ÷ 7.4B shares = ~$10-11/share
    "GOOGL": {"name": "Alphabet Inc. (Google)",  "sector": "Technology",        "pe": 22.0, "book": 24.00, "roe": 0.357, "dps": 0.00,  "beta": 1.13, "fcf_per_share": 6.05, "fcf_g_bear": 8.0, "fcf_g_base": 15.0, "fcf_g_bull": 22.0, "wacc_bear": 11.0, "wacc_base": 9.0, "wacc_bull": 7.5},
    # GOOGL FY2024: FCF $73.3B ÷ 12.1B shares = $6.05/share; EPS $10.80
    "GOOG":  {"name": "Alphabet Inc. (Google)",  "sector": "Technology",        "pe": 22.0, "book": 24.00, "roe": 0.357, "dps": 0.00,  "beta": 1.13, "fcf_per_share": 6.05, "fcf_g_bear": 8.0, "fcf_g_base": 15.0, "fcf_g_bull": 22.0, "wacc_bear": 11.0, "wacc_base": 9.0, "wacc_bull": 7.5},
    "AMZN":  {"name": "Amazon.com Inc.",         "sector": "Technology",        "pe": 38.0, "book": 22.00, "roe": 0.230, "dps": 0.00,  "beta": 1.30, "fcf_per_share": 5.50, "fcf_g_bear": 8.0, "fcf_g_base": 18.0, "fcf_g_bull": 28.0, "wacc_bear": 11.0, "wacc_base": 9.0, "wacc_bull": 7.5},
    # AMZN FY2024: FCF ~$50B ÷ 10.6B shares = ~$4.80/share
    "META":  {"name": "Meta Platforms",          "sector": "Technology",        "pe": 28.0, "book": 26.00, "roe": 0.340, "dps": 2.00,  "beta": 1.25, "fcf_per_share": 22.00, "fcf_g_bear": 8.0, "fcf_g_base": 18.0, "fcf_g_bull": 28.0, "wacc_bear": 11.0, "wacc_base": 9.0, "wacc_bull": 7.5},
    # META FY2024: FCF ~$52B ÷ 2.56B shares = ~$19.80/share
    "NVDA":  {"name": "NVIDIA Corp",             "sector": "Technology",        "pe": 40.0, "book": 4.00,  "roe": 0.730, "dps": 0.16,  "beta": 1.70, "fcf_per_share": 2.40, "fcf_g_bear": 10.0, "fcf_g_base": 25.0, "fcf_g_bull": 40.0, "wacc_bear": 12.0, "wacc_base": 10.0, "wacc_bull": 8.0},
    # NVDA FY2025: FCF ~$42B ÷ 24.4B shares = ~$1.70/share (post 10:1 split)
    "TSLA":  {"name": "Tesla Inc.",              "sector": "Automotive/Tech",   "pe": 120.0,"book": 20.00, "roe": 0.110, "dps": 0.00,  "beta": 2.30, "fcf_per_share": 0.90},
    # ── US Banks ──────────────────────────────────────────────────────────────
    "JPM":   {"name": "JPMorgan Chase",          "sector": "Banking",           "pe": 13.0, "book": 105.0, "roe": 0.170, "dps": 4.60,  "beta": 1.10, "fcf_per_share": None},
    "BAC":   {"name": "Bank of America",         "sector": "Banking",           "pe": 14.0, "book": 35.00, "roe": 0.095, "dps": 1.00,  "beta": 1.35, "fcf_per_share": None},
    "WFC":   {"name": "Wells Fargo",             "sector": "Banking",           "pe": 13.0, "book": 52.00, "roe": 0.115, "dps": 1.40,  "beta": 1.15, "fcf_per_share": None},
    # ── US Consumer / Other ───────────────────────────────────────────────────
    "KO":    {"name": "Coca-Cola",               "sector": "Consumer Staples",  "pe": 27.0, "book": 6.00,  "roe": 0.420, "dps": 1.94,  "beta": 0.55, "fcf_per_share": 2.30},
    "JNJ":   {"name": "Johnson & Johnson",       "sector": "Healthcare",        "pe": 15.0, "book": 26.00, "roe": 0.220, "dps": 4.96,  "beta": 0.55, "fcf_per_share": 7.50},
    "V":     {"name": "Visa Inc.",               "sector": "Financials",        "pe": 33.0, "book": 18.00, "roe": 0.520, "dps": 2.08,  "beta": 0.95, "fcf_per_share": 10.50},
    "MA":    {"name": "Mastercard",              "sector": "Financials",        "pe": 38.0, "book": 8.00,  "roe": 2.000, "dps": 2.64,  "beta": 1.05, "fcf_per_share": 12.00},
    "BRK.B": {"name": "Berkshire Hathaway B",   "sector": "Diversified",       "pe": 22.0, "book": 230.0, "roe": 0.140, "dps": 0.00,  "beta": 0.90, "fcf_per_share": 15.00},
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


@st.cache_data(ttl=300, show_spinner=False)
def fetch_ticker(ticker: str, api_key: str = ""):
    """
    Fetch price + fundamentals via Finnhub (works on Streamlit Cloud).
    Falls back to yfinance for local use if no API key.
    """
    import requests

    key = api_key.strip() if api_key else ""

    # ── Method 1: Finnhub (cloud-safe) ───────────────────────────────────────
    if key:
        # Finnhub uses exchange suffix format: O39.SI -> O39:SP
        fh_sym = _finnhub_symbol(ticker)
        base_url = "https://finnhub.io/api/v1"
        h = {"X-Finnhub-Token": key}
        try:
            # Quote — live price
            q = requests.get(f"{base_url}/quote?symbol={fh_sym}", headers=h, timeout=10)
            q.raise_for_status()
            qj = q.json()
            # c = current price, pc = previous close, t = timestamp
            # c=current price (0 when market closed), pc=previous close
            price = qj.get("c") or qj.get("pc")  # always use prev close if current=0

            # Company profile — name, sector, currency
            p = requests.get(f"{base_url}/stock/profile2?symbol={fh_sym}", headers=h, timeout=10)
            p.raise_for_status()
            pj = p.json()
            name     = pj.get("name") or ticker
            sector   = pj.get("finnhubIndustry") or "—"
            currency = pj.get("currency") or "SGD"

            # Basic financials — PE, book, beta, ROE, etc.
            fm = requests.get(f"{base_url}/stock/metric?symbol={fh_sym}&metric=all", headers=h, timeout=10)
            fm.raise_for_status()
            fmj = fm.json()
            m = fmj.get("metric", {})

            pe   = m.get("peBasicExclExtraTTM") or m.get("peTTM") or m.get("peAnnual")
            book = m.get("bookValuePerShareQuarterly") or m.get("bookValuePerShareAnnual")
            roe  = m.get("roeTTM") or m.get("roeAnnual")
            beta = m.get("beta")
            dps  = m.get("dividendPerShareAnnual") or m.get("dividendPerShareTTM")
            pb   = m.get("pbQuarterly") or m.get("pbAnnual")
            eps  = m.get("epsTTM") or m.get("epsAnnual")

            # Finnhub returns ROE as percentage (e.g. 13.5), convert to decimal
            if roe and abs(roe) > 1:
                roe = roe / 100

            result = {
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
                "mktcap":   pj.get("marketCapitalization"),
                "currency": currency,
            }
            # Return if we got at least a price
            if result.get("price"):
                return result
            # Even if price is 0 (market closed), return with sgx_enrich filling price
            if pj.get("name"):  # profile loaded = valid ticker
                return result
        except Exception as e:
            pass  # fall through to yfinance

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


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_price_history(ticker: str, period: str = "2y", api_key: str = ""):
    """
    Fetch daily close history. Finnhub first, yfinance fallback.
    """
    import requests
    from datetime import datetime, timedelta

    key = api_key.strip() if api_key else ""

    # ── Method 1: Finnhub candles ─────────────────────────────────────────────
    if key:
        fh_sym = _finnhub_symbol(ticker)
        t_to   = int(datetime.now().timestamp())
        days   = 730 if period == "2y" else 365
        t_from = int((datetime.now() - timedelta(days=days)).timestamp())
        try:
            url = f"https://finnhub.io/api/v1/stock/candle?symbol={fh_sym}&resolution=D&from={t_from}&to={t_to}"
            r = requests.get(url, headers={"X-Finnhub-Token": key}, timeout=15)
            r.raise_for_status()
            js = r.json()
            if js.get("s") == "ok" and js.get("t"):
                dates  = pd.to_datetime(js["t"], unit="s")
                closes = js["c"]
                df = pd.DataFrame({"Close": closes}, index=dates)
                df.index = df.index.tz_localize(None)
                return df.dropna()
            # s == "no_data" means invalid symbol for this exchange
        except Exception:
            pass

    # ── Method 2: Yahoo Finance chart API (works for SGX history) ──────────────
    import requests as _req
    try:
        yf_url = f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range={period}"
        _h = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
        _r = _req.get(yf_url, headers=_h, timeout=15)
        if _r.status_code == 200:
            _js = _r.json()
            _res = _js.get("chart", {}).get("result", [None])[0]
            if _res:
                _ts = _res.get("timestamp", [])
                _cl = _res.get("indicators", {}).get("quote", [{}])[0].get("close", [])
                if _ts and _cl:
                    _dates = pd.to_datetime(_ts, unit="s")
                    _df = pd.DataFrame({"Close": _cl}, index=_dates)
                    _df = _df.dropna()
                    _df.index = _df.index.tz_localize(None)
                    if not _df.empty:
                        return _df
    except Exception:
        pass

    # ── Method 3: yfinance library fallback ──────────────────────────────────
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



def fetch_buy_sell_pressure(ticker: str) -> dict:
    """Fetch basic buy/sell pressure from Yahoo Finance quote data."""
    import requests
    try:
        url = f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range=5d"
        h = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
        r = requests.get(url, headers=h, timeout=8)
        if r.status_code == 200:
            js = r.json()
            meta = js.get("chart", {}).get("result", [{}])[0].get("meta", {})
            quote = js.get("chart", {}).get("result", [{}])[0].get("indicators", {}).get("quote", [{}])[0]
            closes = [c for c in (quote.get("close") or []) if c]
            volumes = [v for v in (quote.get("volume") or []) if v]
            opens = [o for o in (quote.get("open") or []) if o]
            if len(closes) >= 2:
                # Simple up/down days for buy/sell pressure
                up_vol = sum(v for c, o, v in zip(closes, opens, volumes) if c > o)
                down_vol = sum(v for c, o, v in zip(closes, opens, volumes) if c <= o)
                total_vol = up_vol + down_vol
                buy_pct = up_vol / total_vol * 100 if total_vol > 0 else 50
                # 52-week range position
                low52 = meta.get("fiftyTwoWeekLow", 0)
                high52 = meta.get("fiftyTwoWeekHigh", 0)
                current = meta.get("regularMarketPrice") or closes[-1]
                if high52 > low52 > 0:
                    range_pos = (current - low52) / (high52 - low52) * 100
                else:
                    range_pos = 50
                return {
                    "buy_pct": round(buy_pct, 1),
                    "sell_pct": round(100 - buy_pct, 1),
                    "up_vol": up_vol,
                    "down_vol": down_vol,
                    "range_pos": round(range_pos, 1),
                    "low52": low52,
                    "high52": high52,
                    "current": current,
                    "prev_close": meta.get("previousClose") or meta.get("chartPreviousClose"),
                    "ask": meta.get("ask"),
                    "bid": meta.get("bid"),
                    "day_high": meta.get("regularMarketDayHigh"),
                    "day_low": meta.get("regularMarketDayLow"),
                    "volume": meta.get("regularMarketVolume", 0),
                    "avg_vol": meta.get("averageDailyVolume10Day", 0),
                }
    except Exception:
        pass
    return {}


def calculate_technicals(hist_df):
    """Calculate RSI(14), MA50, MA200 from price history dataframe."""
    if hist_df is None or hist_df.empty or len(hist_df) < 20:
        return {}
    closes = hist_df["Close"].dropna()
    result = {}
    # Moving averages
    if len(closes) >= 50:
        result["ma50"] = round(closes.rolling(50).mean().iloc[-1], 2)
    if len(closes) >= 200:
        result["ma200"] = round(closes.rolling(200).mean().iloc[-1], 2)
    # RSI(14)
    if len(closes) >= 15:
        delta = closes.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, float('nan'))
        rsi_series = 100 - (100 / (1 + rs))
        result["rsi"] = round(rsi_series.iloc[-1], 1)
    # Last 30 days RSI for mini chart
    if len(closes) >= 30:
        delta = closes.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, float('nan'))
        rsi_all = (100 - (100 / (1 + rs))).dropna()
        result["rsi_series"] = rsi_all.tail(60).tolist()
        result["rsi_dates"] = [str(d.date()) for d in rsi_all.tail(60).index]
    return result


@st.cache_data(ttl=3600)
def fetch_div_yield_history(ticker: str) -> dict:
    """Get current and average historical dividend yield for REITs/dividend stocks."""
    import requests
    try:
        # Get 3yr price history and compute avg yield
        url = f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1mo&range=3y"
        h = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
        r = requests.get(url, headers=h, timeout=10)
        if r.status_code != 200:
            return {}
        js = r.json()
        meta = js.get("chart",{}).get("result",[{}])[0].get("meta",{})
        closes = js.get("chart",{}).get("result",[{}])[0].get("indicators",{}).get("quote",[{}])[0].get("close",[])
        closes = [c for c in closes if c]
        annual_dps = meta.get("dividendRate") or meta.get("trailingAnnualDividendRate")
        if not annual_dps or not closes:
            return {}
        avg_price_3yr = sum(closes) / len(closes)
        avg_yield_3yr = annual_dps / avg_price_3yr * 100
        current_price = meta.get("regularMarketPrice") or closes[-1]
        current_yield = annual_dps / current_price * 100 if current_price else 0
        yield_discount = (current_yield - avg_yield_3yr) / avg_yield_3yr * 100
        return {
            "current_yield": round(current_yield, 2),
            "avg_yield_3yr": round(avg_yield_3yr, 2),
            "yield_discount": round(yield_discount, 1),
            "annual_dps": annual_dps,
            "avg_price_3yr": round(avg_price_3yr, 2),
        }
    except Exception:
        return {}


def send_telegram_alert(bot_token: str, chat_id: str, message: str) -> bool:
    """Send a Telegram message. Returns True if successful."""
    import requests
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        r = requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"}, timeout=8)
        return r.status_code == 200
    except Exception:
        return False

def render_price_alerts(ticker, bear, base, bull, current_price, currency=""):
    """Render price alert setup section."""
    st.markdown('<div class="section-header">🔔 Price Alerts (Telegram)</div>', unsafe_allow_html=True)

    with st.expander("Set up price alerts — free via Telegram", expanded=False):
        st.markdown("""
**How to set up (one-time, 3 minutes):**
1. Open Telegram → search **@BotFather** → send `/newbot` → follow steps → copy the **bot token**
2. Search **@userinfobot** in Telegram → send any message → copy your **Chat ID**
3. Paste both below and click Test to confirm it works
        """)
        # Load from Streamlit Secrets first (preferred — token never typed in app)
        try:
            _secret_token = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
            _secret_chat  = st.secrets.get("TELEGRAM_CHAT_ID", "")
        except Exception:
            _secret_token, _secret_chat = "", ""

        if _secret_token:
            st.success("✅ Bot token loaded from Streamlit Secrets — no need to enter here.")
            bot_token = _secret_token
            st.session_state["tg_token"] = _secret_token
        else:
            bot_token = st.text_input("Bot token", value=st.session_state.get("tg_token",""),
                                      type="password", placeholder="123456:ABC-DEF...")
            if bot_token: st.session_state["tg_token"] = bot_token
            st.caption("💡 Better: add TELEGRAM_BOT_TOKEN to Streamlit Secrets so it never appears here.")

        if _secret_chat:
            chat_id = _secret_chat
            st.session_state["tg_chat"] = _secret_chat
            st.success(f"✅ Chat ID loaded from Secrets.")
        else:
            chat_id = st.text_input("Chat ID", value=st.session_state.get("tg_chat",""),
                                    placeholder="123456789")
            if chat_id: st.session_state["tg_chat"] = chat_id

        if st.button("📱 Test connection"):
            if bot_token and chat_id:
                ok = send_telegram_alert(bot_token, chat_id,
                    f"✅ <b>Valuation Tool</b> connected!\n\nYou will receive alerts for <b>{ticker}</b>.")
                st.success("✅ Test message sent!") if ok else st.error("❌ Failed — check token and chat ID.")
            else:
                st.warning("Enter both bot token and chat ID first.")

        # Alert settings
        st.divider()
        st.markdown("**Set price triggers:**")
        a1, a2, a3 = st.columns(3)
        alert_buy  = a1.checkbox(f"Alert when price drops to BUY zone (≤ {fmt_price(bear, currency)})", value=True)
        alert_base = a2.checkbox(f"Alert when price reaches TARGET ({fmt_price(base, currency)})", value=False)
        alert_trim = a3.checkbox(f"Alert when price hits TRIM ({fmt_price(bull, currency)})", value=False)

        if st.button("💾 Save alerts for this session"):
            if bot_token and chat_id:
                st.session_state[f"alert_{ticker}"] = {
                    "buy": alert_buy, "base": alert_base, "trim": alert_trim,
                    "bear": bear, "base_price": base, "bull": bull
                }
                st.success(f"Alerts saved for {ticker}. They are active while this app is open.")
                st.caption("⚠️ Alerts only fire while the app is open in your browser. For persistent alerts, keep a browser tab open or deploy on a server.")
            else:
                st.warning("Set up your Telegram connection first.")

    # Check and fire any pending alerts
    _alert_cfg = st.session_state.get(f"alert_{ticker}", {})
    _tg_token  = st.session_state.get("tg_token", "")
    _tg_chat   = st.session_state.get("tg_chat", "")
    if _alert_cfg and _tg_token and _tg_chat and current_price:
        msgs = []
        if _alert_cfg.get("buy") and current_price <= _alert_cfg.get("bear", 0):
            msgs.append(f"🟢 <b>BUY ZONE ALERT</b>\n{ticker} is now at <b>{fmt_price(current_price, currency)}</b>\nBelow buy zone: {fmt_price(_alert_cfg['bear'], currency)}\nConsider entering a position!")
        if _alert_cfg.get("base") and current_price >= _alert_cfg.get("base_price", 999999):
            msgs.append(f"🎯 <b>TARGET REACHED</b>\n{ticker} hit <b>{fmt_price(current_price, currency)}</b>\nBase case target: {fmt_price(_alert_cfg['base_price'], currency)}")
        if _alert_cfg.get("trim") and current_price >= _alert_cfg.get("bull", 999999):
            msgs.append(f"✂️ <b>TRIM ALERT</b>\n{ticker} hit <b>{fmt_price(current_price, currency)}</b>\nAbove bull case: {fmt_price(_alert_cfg['bull'], currency)}. Consider reducing position.")
        for msg in msgs:
            sent = send_telegram_alert(_tg_token, _tg_chat, msg)
            if sent:
                st.toast(f"🔔 Alert sent to Telegram!", icon="📱")

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

    # Colour-coded interpretation + distance to buy zone
    pct_to_buy = (cur - bear) / cur * 100 if cur > bear else 0
    pct_to_bull = (bull - cur) / cur * 100 if cur < bull else 0

    if cur <= bear:
        colour = "#dcfce7"
        msg = f"✅ STRONG BUY — price is below the buy zone ({fmt_price(bear, currency)}). Maximum margin of safety. Deploy full allocation."
        action_tip = ""
    elif cur <= base:
        colour = "#fef9c3"
        msg = f"👀 WATCH — price is {pct_to_buy:.1f}% above the buy zone. Reasonable entry with a smaller position (50% of max allocation)."
        action_tip = f"Set a price alert at {fmt_price(bear, currency)} for a full-position entry."
    elif cur <= bull:
        colour = "#fef3c7"
        dist = pct_to_buy
        msg = f"⚠️ HOLD — price is {dist:.1f}% above the buy zone ({fmt_price(bear, currency)}). Hold existing positions. Do not add at current price."
        if dist < 20:
            action_tip = f"Only {dist:.1f}% above buy zone — a small pullback could create a good entry. Set alert at {fmt_price(bear * 1.05, currency)}."
        elif dist < 50:
            action_tip = f"Consider DCA (dollar-cost average) monthly if this is a long-term hold. Next buy target: {fmt_price(bear, currency)}."
        else:
            action_tip = f"Price has run significantly. Wait for a meaningful pullback to {fmt_price(bear * 1.1, currency)} before considering entry."
    else:
        colour = "#fee2e2"
        msg = f"🚫 AVOID / TRIM — price {fmt_price(cur, currency)} exceeds the bull case {fmt_price(bull, currency)}. Risk/reward unfavourable."
        action_tip = f"Consider trimming 25–50% of position. Re-enter if price falls back to {fmt_price(base, currency)}."

    st.markdown(
        f'<div style="background:{colour};border-radius:10px;padding:0.9rem 1.2rem;'
        f'margin:0.5rem 0 0.25rem;font-weight:600;font-size:0.95rem;color:#111827;">{msg}</div>',
        unsafe_allow_html=True
    )
    if action_tip:
        st.markdown(
            f'<div style="background:{colour};border-radius:0 0 10px 10px;padding:0.5rem 1.2rem;'
            f'font-size:0.82rem;color:#374151;margin-bottom:1rem;border-top:1px solid rgba(0,0,0,0.06);">'
            f'💡 {action_tip}</div>',
            unsafe_allow_html=True
        )

    # ── Market sentiment / buy-sell pressure ────────────────────────────────
    st.markdown('<div class="section-header">📊 Market Sentiment (5-day)</div>', unsafe_allow_html=True)
    bs = fetch_buy_sell_pressure(ticker)
    if bs:
        s1, s2, s3 = st.columns(3)
        buy_c = "#16a34a" if bs["buy_pct"] >= 50 else "#dc2626"
        sell_c = "#dc2626" if bs["sell_pct"] >= 50 else "#16a34a"
        s1.metric("Buying volume", f"{bs['buy_pct']}%",
                  delta="Bullish" if bs["buy_pct"] >= 55 else "Neutral" if bs["buy_pct"] >= 45 else "Bearish")
        s2.metric("Selling volume", f"{bs['sell_pct']}%")
        s3.metric("52-week position", f"{bs['range_pos']}%",
                  delta=f"Low {bs['low52']:.2f} → High {bs['high52']:.2f}")
        # Visual buy/sell bar
        bp = int(bs["buy_pct"])
        sp = 100 - bp
        st.markdown(
            f'''<div style="margin:0.5rem 0 0.25rem;font-size:0.75rem;color:#64748b;">Volume-weighted buy vs sell pressure (5 days)</div>
<div style="display:flex;border-radius:6px;overflow:hidden;height:20px;">
  <div style="width:{bp}%;background:#16a34a;display:flex;align-items:center;justify-content:center;font-size:11px;color:white;font-weight:600;">
    {bp}% buy
  </div>
  <div style="width:{sp}%;background:#dc2626;display:flex;align-items:center;justify-content:center;font-size:11px;color:white;font-weight:600;">
    {sp}% sell
  </div>
</div>''',
            unsafe_allow_html=True
        )
        st.caption("⚠️ Sentiment is 5-day volume approximation only — not a trading signal. Always base decisions on valuation fundamentals above.")
    else:
        st.info("Market sentiment unavailable for this ticker.")

    # ── Dividend yield vs historical average ──────────────────────────────────
    _dy = fetch_div_yield_history(ticker)
    if _dy and _dy.get("current_yield") and _dy.get("avg_yield_3yr"):
        st.markdown('<div class="section-header">💰 Dividend Yield vs 3-Year Average</div>', unsafe_allow_html=True)
        dy1, dy2, dy3 = st.columns(3)
        dy1.metric("Current yield", f"{_dy['current_yield']}%",
                   delta=f"{_dy['yield_discount']:+.1f}% vs 3yr avg")
        dy2.metric("3-year avg yield", f"{_dy['avg_yield_3yr']}%")
        _signal_txt = (
            "🟢 HIGH yield vs history — price may be oversold. Good entry if fundamentals intact."
            if _dy["yield_discount"] > 20 else
            "🟡 Slightly above average yield — mildly attractive."
            if _dy["yield_discount"] > 5 else
            "⚪ Yield near historical average — fairly priced on income basis."
            if _dy["yield_discount"] > -5 else
            "🟠 Below average yield — price has run up. Income less attractive."
            if _dy["yield_discount"] > -20 else
            "🔴 LOW yield vs history — price significantly above historical norms on income basis."
        )
        _sig_col = ("#dcfce7" if _dy["yield_discount"] > 20
                    else "#fef9c3" if _dy["yield_discount"] > 5
                    else "#f8fafc" if _dy["yield_discount"] > -5
                    else "#fef3c7" if _dy["yield_discount"] > -20
                    else "#fee2e2")
        dy3.metric("Annual DPS", f"${_dy['annual_dps']:.3f}")
        st.markdown(f'<div style="background:{_sig_col};border-radius:8px;padding:0.6rem 1rem;font-size:0.85rem;color:#111827;margin-top:0.3rem;">{_signal_txt}</div>', unsafe_allow_html=True)
        st.caption("Based on 3-year monthly price history. Higher yield than average = cheaper price relative to income.")

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
    st.markdown('<div class="section-header">📈 Price vs Fair Value Band + Technical Indicators</div>', unsafe_allow_html=True)

    hist = fetch_price_history(ticker, api_key=st.session_state.get('finnhub_key',''))
    ta = calculate_technicals(hist)
    if hist is not None and not hist.empty:
        # Calculate technical indicators
        # RSI (14-day)
        delta = hist["Close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, float('nan'))
        hist["RSI"] = 100 - (100 / (1 + rs))

        # Moving averages
        hist["MA50"]  = hist["Close"].rolling(50).mean()
        hist["MA200"] = hist["Close"].rolling(200).mean()

        # Current values for summary
        latest_rsi  = hist["RSI"].dropna().iloc[-1] if not hist["RSI"].dropna().empty else None
        latest_ma50 = hist["MA50"].dropna().iloc[-1] if not hist["MA50"].dropna().empty else None
        latest_ma200= hist["MA200"].dropna().iloc[-1] if not hist["MA200"].dropna().empty else None
        latest_price= hist["Close"].iloc[-1]

        # Technical signal summary
        t1, t2, t3 = st.columns(3)
        if latest_rsi:
            rsi_signal = "Oversold — potential entry" if latest_rsi < 35 else ("Overbought — wait" if latest_rsi > 65 else "Neutral")
            rsi_color  = "green" if latest_rsi < 35 else ("red" if latest_rsi > 65 else "orange")
            t1.metric("RSI (14-day)", f"{latest_rsi:.1f}", delta=rsi_signal)
        if latest_ma200:
            ma_signal = "Above 200MA — uptrend" if latest_price > latest_ma200 else "Below 200MA — caution"
            t2.metric("vs 200-day MA", fmt_price(latest_ma200), delta=ma_signal)
        if latest_ma50 and latest_ma200:
            cross = "Golden cross — bullish" if latest_ma50 > latest_ma200 else "Death cross — bearish"
            t3.metric("50MA vs 200MA", f"{latest_ma50/latest_ma200*100:.1f}%", delta=cross)

        # Combined signal
        tech_signals = []
        if latest_rsi and latest_rsi < 40:
            tech_signals.append("RSI oversold")
        if latest_ma200 and latest_price < latest_ma200:
            tech_signals.append("below 200MA")
        range_pos_val = (latest_price - hist["Close"].min()) / (hist["Close"].max() - hist["Close"].min()) * 100
        if range_pos_val < 30:
            tech_signals.append("near 52w low")

        if tech_signals and cur <= bear:
            tech_banner = f"🟢 STRONG ENTRY SIGNAL — Valuation BUY + {', '.join(tech_signals)}"
            tech_colour = "#dcfce7"
        elif tech_signals:
            tech_banner = f"👀 TECHNICAL OVERSOLD ({', '.join(tech_signals)}) — watch for valuation entry"
            tech_colour = "#fef9c3"
        elif cur <= bull:
            tech_banner = "📊 No technical oversold signal — wait for RSI < 40 or price near 52w low before adding"
            tech_colour = "#f1f5f9"
        else:
            tech_banner = "⚠️ Technically extended — RSI elevated and price above trend. Not a good entry point."
            tech_colour = "#fee2e2"

        st.markdown(f'<div style="background:{tech_colour};border-radius:10px;padding:0.75rem 1rem;margin:0.5rem 0;font-size:0.85rem;font-weight:500;color:#111827;">{tech_banner}</div>', unsafe_allow_html=True)

        hist["Buy zone (bear)"] = bear
        hist["Target (base)"] = base
        hist["Trim (bull)"] = bull

        import altair as alt

        hist_reset = hist.reset_index()
        # Rename columns carefully based on what's in hist now
        base_cols = ["Date", "Price", "Buy zone", "Target", "Trim"]
        extra_cols = [c for c in hist_reset.columns[len(base_cols):]]
        hist_reset.columns = base_cols + extra_cols

        # Add MA columns if calculated
        if "MA50" in hist.columns:
            hist_reset["MA50"]  = hist["MA50"].values
        if "MA200" in hist.columns:
            hist_reset["MA200"] = hist["MA200"].values

        base_chart = alt.Chart(hist_reset)

        band = base_chart.mark_area(opacity=0.12, color="#3b82f6").encode(
            x=alt.X("Date:T", title=""),
            y=alt.Y("Buy zone:Q", title="Price", scale=alt.Scale(zero=False)),
            y2="Trim:Q"
        )
        cur_df = hist_reset.copy()
        cur_df["Current"] = cur
        cur_line = alt.Chart(cur_df).mark_line(
            color="#f59e0b", strokeDash=[6,3], strokeWidth=1.5, opacity=0.6
        ).encode(x="Date:T", y="Current:Q")
        price_line = base_chart.mark_line(color="#f59e0b", strokeWidth=2.5).encode(
            x="Date:T",
            y=alt.Y("Price:Q", scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip("Date:T", title="Date"), alt.Tooltip("Price:Q", title="Price", format=".2f")]
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
        layers = [band, cur_line, price_line, bear_line, base_line, bull_line]
        if ta.get("ma50"):
            ma50_df = hist_reset.copy()
            ma50_df["MA50"] = hist["Close"].rolling(50).mean().values[:len(hist_reset)]
            layers.append(alt.Chart(ma50_df).mark_line(color="#a78bfa", strokeWidth=1.5, opacity=0.9).encode(
                x="Date:T", y=alt.Y("MA50:Q", scale=alt.Scale(zero=False))))
        if ta.get("ma200"):
            ma200_df = hist_reset.copy()
            ma200_df["MA200"] = hist["Close"].rolling(200).mean().values[:len(hist_reset)]
            layers.append(alt.Chart(ma200_df).mark_line(color="#fb923c", strokeWidth=1.5, opacity=0.9).encode(
                x="Date:T", y=alt.Y("MA200:Q", scale=alt.Scale(zero=False))))
        chart = alt.layer(*layers).properties(height=320).encode(
            x=alt.X("Date:T", scale=alt.Scale(domain=[
                hist_reset["Date"].min().isoformat(),
                hist_reset["Date"].max().isoformat()
            ]))
        ).interactive()
        st.altair_chart(chart, use_container_width=True)
        ma_caption = "🟡 Price  ·  🔴 BUY  ·  🔵 TARGET  ·  🟢 TRIM"
        if ta.get("ma50"): ma_caption += "  ·  🟣 MA50"
        if ta.get("ma200"): ma_caption += "  ·  🟠 MA200"

        # ── RSI + Technical Panel ────────────────────────────────────────────
        if ta.get("rsi"):
            rsi_val = ta["rsi"]
            st.markdown('<div class="section-header">📉 RSI (14-day) & Moving Averages</div>', unsafe_allow_html=True)
            r1, r2, r3 = st.columns(3)
            r1.metric("RSI (14)", f"{rsi_val}",
                      delta="Oversold — watch for entry" if rsi_val < 35 else
                            "Overbought — avoid adding" if rsi_val > 65 else "Neutral")
            r2.metric("MA50", fmt_price(ta.get("ma50"), currency) if ta.get("ma50") else "—",
                      delta=f"{(cur/ta['ma50']-1)*100:+.1f}% price vs MA50" if ta.get("ma50") else None)
            r3.metric("MA200", fmt_price(ta.get("ma200"), currency) if ta.get("ma200") else "—",
                      delta=f"{(cur/ta['ma200']-1)*100:+.1f}% price vs MA200" if ta.get("ma200") else None)

            if rsi_val < 30:
                rsi_msg, rsi_col = f"🟢 RSI {rsi_val} — OVERSOLD. Best entry zone when valuation also says BUY.", "#dcfce7"
            elif rsi_val < 40:
                rsi_msg, rsi_col = f"🟡 RSI {rsi_val} — Approaching oversold. Watch closely. Consider partial entry.", "#fef9c3"
            elif rsi_val < 60:
                rsi_msg, rsi_col = f"⚪ RSI {rsi_val} — Neutral. No strong technical signal either way.", "#f8fafc"
            elif rsi_val < 70:
                rsi_msg, rsi_col = f"🟠 RSI {rsi_val} — Approaching overbought. Momentum strong but stretched.", "#fef3c7"
            else:
                rsi_msg, rsi_col = f"🔴 RSI {rsi_val} — OVERBOUGHT. Do not add. Wait for RSI to fall below 50.", "#fee2e2"
            st.markdown(f'<div style="background:{rsi_col};border-radius:8px;padding:0.7rem 1rem;font-size:0.85rem;color:#111827;margin:0.5rem 0;">{rsi_msg}</div>', unsafe_allow_html=True)

            # ── Combined signal ───────────────────────────────────────────────
            st.markdown('<div class="section-header">🎯 Combined Signal (Valuation + RSI)</div>', unsafe_allow_html=True)
            val_z = "buy" if cur <= bear else "watch" if cur <= base else "hold" if cur <= bull else "avoid"
            rsi_z = "buy" if rsi_val < 40 else "neutral" if rsi_val < 60 else "avoid"
            combos = {
                ("buy","buy"):     ("#dcfce7","🟢 STRONG ENTRY — Valuation cheap + RSI oversold. Best combination. Deploy full allocation."),
                ("buy","neutral"): ("#dcfce7","🟢 GOOD ENTRY — Valuation cheap, neutral RSI. Good risk/reward. Enter 50–75% of allocation."),
                ("buy","avoid"):   ("#fef9c3","🟡 Valuation BUY but RSI overbought — wait for RSI to cool below 55 before entering."),
                ("watch","buy"):   ("#fef9c3","🟡 Near buy zone + RSI oversold — good for averaging in. Start with 25–50% of allocation."),
                ("watch","neutral"):("#fef9c3","🟡 WATCH — Between buy and target, neutral RSI. DCA monthly if long-term conviction."),
                ("watch","avoid"): ("#fef3c7","⚠️ Near buy zone but overbought — mixed signal. Wait for RSI to normalise."),
                ("hold","buy"):    ("#fef9c3","🟡 HOLD + RSI oversold — price pulled back within fair value. Small entry acceptable."),
                ("hold","neutral"):("#fef3c7","⚠️ HOLD — Fair valued, neutral RSI. No new buying. Collect dividends and wait for buy zone."),
                ("hold","avoid"):  ("#fee2e2","🔴 HOLD + Overbought — Do not add. Pullback likely. Set alert at buy zone price."),
                ("avoid","buy"):   ("#fef9c3","🟡 Expensive + RSI oversold — short-term bounce possible only. Not for long-term accumulation."),
                ("avoid","neutral"):("#fee2e2","🔴 AVOID — Above bull case, neutral RSI. Trim 25% if held. No new buying."),
                ("avoid","avoid"): ("#fee2e2","🔴 STRONG AVOID — Above bull case + overbought. Worst entry point. Trim position."),
            }
            c_col, c_msg = combos.get((val_z, rsi_z), ("#f8fafc","⚪ Mixed signals — use business quality and conviction to decide."))
            st.markdown(f'<div style="background:{c_col};border-radius:10px;padding:1rem 1.2rem;font-weight:600;font-size:0.95rem;color:#111827;margin-bottom:0.5rem;">{c_msg}</div>', unsafe_allow_html=True)
            st.caption("Combined signal = valuation zone × RSI zone. Always verify business fundamentals independently. Not financial advice.")
        st.caption("🟡 Price  ·  🔴 BUY (bear)  ·  🔵 TARGET (base)  ·  🟢 TRIM (bull)  ·  🟣 50-day MA  ·  🔶 200-day MA")

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

    # ── Price alerts ──────────────────────────────────────────────────────
    render_price_alerts(ticker, bear, base, bull, cur, currency)





def fmt_price(v, currency=""):
    if v is None:
        return "—"
    return f"{currency} {v:,.2f}".strip()

def render_ai_analysis(ticker, asset_type, current_price, bear, base, bull,
                        sector, pe, roe, book, dps, beta,
                        buy_pct, range_pos, days_in_buy, days_in_hold, days_above_sell, days_total,
                        signal_text):
    """
    100% free rule-based investment analysis engine.
    No API calls. No cost. Runs entirely in Python.
    """
    st.markdown('<div class="section-header">🤖 Investment Analysis</div>', unsafe_allow_html=True)

    # ── Scoring engine ────────────────────────────────────────────────────────
    # Each indicator contributes points: positive = bullish, negative = bearish
    # Final score → overall verdict

    score = 0
    max_score = 100
    signals = []

    # 1. VALUATION (40 pts max)
    if current_price > 0 and bear > 0 and bull > 0:
        vs_bear = (current_price - bear) / bear * 100
        vs_base = (current_price - base) / base * 100
        vs_bull = (current_price - bull) / bull * 100

        if current_price <= bear:
            score += 40
            val_verdict = f"STRONGLY UNDERVALUED — trading {abs(vs_bear):.1f}% below the bear case fair value of {fmt_price(bear)}. This is maximum margin of safety territory."
            val_colour = "#16a34a"
        elif current_price <= base * 0.95:
            score += 28
            val_verdict = f"UNDERVALUED — price is {abs(vs_base):.1f}% below the base case fair value of {fmt_price(base)}. A reasonable entry with good upside."
            val_colour = "#16a34a"
        elif current_price <= base * 1.05:
            score += 15
            val_verdict = f"FAIRLY VALUED — price is close to the base case fair value of {fmt_price(base)} (within 5%). Limited upside from here unless assumptions improve."
            val_colour = "#ca8a04"
        elif current_price <= bull:
            score += 5
            val_verdict = f"SLIGHTLY EXPENSIVE — price is {vs_base:.1f}% above base case. Still within bull case range ({fmt_price(bull)}) but margin of safety is thin."
            val_colour = "#ca8a04"
        else:
            score -= 10
            val_verdict = f"OVERVALUED — price exceeds the bull case fair value of {fmt_price(bull)}. Risk/reward is unfavourable at current levels."
            val_colour = "#dc2626"
    else:
        val_verdict = "Valuation data incomplete — enter bear/base/bull inputs to generate verdict."
        val_colour = "#64748b"

    # 2. FUNDAMENTAL QUALITY (30 pts max)
    fund_points = []
    fund_score = 0

    # ROE quality
    if roe:
        roe_pct = roe * 100
        if roe_pct >= 15:
            fund_score += 10
            fund_points.append(f"ROE of {roe_pct:.1f}% is excellent — well above the 10% threshold that indicates strong capital efficiency.")
        elif roe_pct >= 10:
            fund_score += 6
            fund_points.append(f"ROE of {roe_pct:.1f}% is solid, indicating the business generates reasonable returns on shareholder equity.")
        elif roe_pct >= 5:
            fund_score += 2
            fund_points.append(f"ROE of {roe_pct:.1f}% is modest — the business is profitable but not generating exceptional returns.")
        else:
            fund_score -= 5
            fund_points.append(f"ROE of {roe_pct:.1f}% is weak — the business is struggling to generate adequate returns on equity.")

    # P/E assessment
    if pe:
        if asset_type == "Bank":
            cheap_pe, fair_pe, rich_pe = 8, 12, 16
        elif asset_type == "REIT":
            cheap_pe, fair_pe, rich_pe = 12, 18, 25
        else:
            cheap_pe, fair_pe, rich_pe = 12, 20, 30

        if pe <= cheap_pe:
            fund_score += 10
            fund_points.append(f"P/E of {pe:.1f}x is cheap for a {asset_type} — historically this sector trades at {fair_pe}x, suggesting potential re-rating upside.")
        elif pe <= fair_pe:
            fund_score += 6
            fund_points.append(f"P/E of {pe:.1f}x is fair for a {asset_type}. The market is pricing in steady growth without excessive optimism.")
        elif pe <= rich_pe:
            fund_score += 2
            fund_points.append(f"P/E of {pe:.1f}x is on the higher side for a {asset_type}. Growth expectations are elevated — any miss could hurt the price.")
        else:
            fund_score -= 5
            fund_points.append(f"P/E of {pe:.1f}x looks expensive relative to {asset_type} peers. Requires exceptional growth to justify.")

    # Dividend
    if dps and current_price > 0:
        div_yield = dps / current_price * 100
        if div_yield >= 5:
            fund_score += 10
            fund_points.append(f"Dividend yield of {div_yield:.1f}% is attractive — provides strong income while you wait for price appreciation.")
        elif div_yield >= 3:
            fund_score += 6
            fund_points.append(f"Dividend yield of {div_yield:.1f}% is decent — offers meaningful income return on top of capital gains potential.")
        elif div_yield >= 1:
            fund_score += 2
            fund_points.append(f"Dividend yield of {div_yield:.1f}% is modest but confirms the company returns capital to shareholders.")
        else:
            fund_points.append("Low or no dividend — total return depends primarily on price appreciation.")

    # Beta / risk
    if beta:
        if beta < 0.7:
            fund_score += 5
            fund_points.append(f"Beta of {beta:.2f} means this stock is significantly less volatile than the market — good defensive characteristics.")
        elif beta < 1.0:
            fund_score += 3
            fund_points.append(f"Beta of {beta:.2f} indicates below-market volatility — relatively stable compared to the broader market.")
        elif beta < 1.3:
            fund_points.append(f"Beta of {beta:.2f} is close to market — expect similar swings to the index in both directions.")
        else:
            fund_score -= 3
            fund_points.append(f"Beta of {beta:.2f} means this stock is more volatile than the market — larger swings up and down.")

    score += min(fund_score, 30)
    fund_text = " ".join(fund_points) if fund_points else "Insufficient fundamental data to assess quality."

    # 3. MARKET CONTEXT (20 pts max)
    mkt_score = 0
    mkt_points = []

    # 52-week range position
    if range_pos is not None:
        if range_pos <= 25:
            mkt_score += 10
            mkt_points.append(f"Price is near a 52-week low (bottom {range_pos:.0f}% of range) — historically a good entry zone for patient investors.")
        elif range_pos <= 50:
            mkt_score += 6
            mkt_points.append(f"Price is in the lower half of the 52-week range ({range_pos:.0f}%) — neither at extreme low nor high.")
        elif range_pos <= 75:
            mkt_score += 2
            mkt_points.append(f"Price is in the upper half of the 52-week range ({range_pos:.0f}%) — momentum is positive but limited room to the upside.")
        else:
            mkt_score -= 5
            mkt_points.append(f"Price is near a 52-week high ({range_pos:.0f}% of range) — chasing momentum at the top carries more risk.")

    # Buy/sell pressure
    if buy_pct is not None:
        if buy_pct >= 60:
            mkt_score += 10
            mkt_points.append(f"5-day volume is {buy_pct:.0f}% buy-dominated — institutional accumulation likely underway.")
        elif buy_pct >= 50:
            mkt_score += 5
            mkt_points.append(f"5-day volume is slightly buy-weighted ({buy_pct:.0f}%) — mild positive momentum.")
        elif buy_pct >= 40:
            mkt_score += 0
            mkt_points.append(f"5-day volume is balanced ({buy_pct:.0f}% buy / {100-buy_pct:.0f}% sell) — no clear direction from recent activity.")
        else:
            mkt_score -= 5
            mkt_points.append(f"5-day volume is {100-buy_pct:.0f}% sell-dominated — near-term selling pressure could push price lower.")

    score += min(mkt_score, 20)
    mkt_text = " ".join(mkt_points) if mkt_points else "Insufficient market data."

    # 4. HISTORICAL OPPORTUNITY (10 pts max)
    hist_score = 0
    hist_points = []

    if days_total > 0:
        buy_pct_hist = days_in_buy / days_total * 100
        hold_pct_hist = days_in_hold / days_total * 100
        above_pct_hist = days_above_sell / days_total * 100

        if buy_pct_hist == 0:
            hist_points.append(f"In the past 2 years, this stock has NEVER traded in the buy zone (below {fmt_price(bear)}) — the current price is the cheapest relative to fair value in recent history.")
            hist_score += 5 if current_price > bear else 10
        elif buy_pct_hist <= 10:
            hist_score += 8
            hist_points.append(f"Buy zone opportunities are rare — only {buy_pct_hist:.0f}% of trading days in 2 years. Current level is historically cheap.")
        elif buy_pct_hist <= 30:
            hist_score += 4
            hist_points.append(f"Stock has spent {buy_pct_hist:.0f}% of the past 2 years in the buy zone — opportunities exist but aren't exceptionally rare.")
        else:
            hist_score += 0
            hist_points.append(f"Stock spends {buy_pct_hist:.0f}% of time in the buy zone — the current valuation inputs may be too conservative or the stock is structurally cheap.")

        if above_pct_hist <= 5:
            hist_points.append(f"Stock rarely exceeds the bull case — only {above_pct_hist:.0f}% of days. The bull target of {fmt_price(bull)} is realistic.")
        elif above_pct_hist >= 30:
            hist_points.append(f"Stock has spent {above_pct_hist:.0f}% of time above the trim price — consider raising your bull case assumptions.")

    score += min(hist_score, 10)
    hist_text = " ".join(hist_points) if hist_points else "No historical data available."

    # ── Final verdict ─────────────────────────────────────────────────────────
    score = max(0, min(score, 100))  # clamp 0-100

    if score >= 75:
        verdict = "STRONG BUY"
        verdict_colour = "#16a34a"
        verdict_icon = "🟢"
        action = f"Consider building a full position. Target price: {fmt_price(base)} (base case). Add more if price falls to {fmt_price(bear)}."
    elif score >= 55:
        verdict = "BUY / ACCUMULATE"
        verdict_colour = "#22c55e"
        verdict_icon = "🟢"
        action = f"Good risk/reward. Start with a partial position, add on weakness toward {fmt_price(bear)}. Target: {fmt_price(base)}."
    elif score >= 40:
        verdict = "HOLD / WATCH"
        verdict_colour = "#f59e0b"
        verdict_icon = "🟡"
        action = f"Fair value — hold existing positions. Wait for a pullback to {fmt_price(bear)} before adding. Trim toward {fmt_price(bull)}."
    elif score >= 25:
        verdict = "AVOID / WAIT"
        verdict_colour = "#f97316"
        verdict_icon = "🟠"
        action = f"Risk/reward is unfavourable at current price. Wait for a pullback to {fmt_price(bear)} before considering entry."
    else:
        verdict = "SELL / TRIM"
        verdict_colour = "#dc2626"
        verdict_icon = "🔴"
        action = f"Price exceeds fair value. Consider trimming. Re-enter only if price falls back to {fmt_price(bear)}."

    # ── Render ────────────────────────────────────────────────────────────────
    # Score gauge
    gauge_colour = verdict_colour
    st.markdown(f"""
<div style="background:var(--color-background-secondary,#f8fafc);border:1px solid var(--color-border-tertiary,#e2e8f0);border-radius:12px;padding:1.2rem 1.5rem;margin-bottom:1rem;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.75rem;">
    <div>
      <div style="font-size:0.75rem;color:#64748b;margin-bottom:2px;">Overall score</div>
      <div style="font-size:2rem;font-weight:700;color:{gauge_colour};">{verdict_icon} {verdict}</div>
    </div>
    <div style="text-align:right;">
      <div style="font-size:0.75rem;color:#64748b;margin-bottom:2px;">Confidence score</div>
      <div style="font-size:2rem;font-weight:700;color:{gauge_colour};">{score}/100</div>
    </div>
  </div>
  <div style="background:var(--color-border-tertiary,#e2e8f0);border-radius:99px;height:10px;">
    <div style="background:{gauge_colour};width:{score}%;height:100%;border-radius:99px;"></div>
  </div>
  <div style="margin-top:0.75rem;font-size:0.9rem;color:var(--color-text-primary,#1e293b);font-weight:500;">{action}</div>
</div>""", unsafe_allow_html=True)

    # 4 analysis cards
    cards = [
        ("📊 Valuation verdict", val_colour, val_verdict),
        ("🏦 Fundamental quality", "#3b82f6", fund_text),
        ("📈 Market context", "#8b5cf6", mkt_text),
        ("📅 Historical opportunity", "#06b6d4", hist_text),
    ]

    for title, colour, body in cards:
        st.markdown(f"""
<div style="border-left:3px solid {colour};background:var(--color-background-secondary, #f8fafc);
     padding:0.8rem 1rem;margin:0.5rem 0;border-radius:0 8px 8px 0;
     border-top:0.5px solid #e2e8f0;border-right:0.5px solid #e2e8f0;border-bottom:0.5px solid #e2e8f0;">
  <div style="font-size:0.75rem;font-weight:600;color:{colour};margin-bottom:4px;">{title}</div>
  <div style="font-size:0.875rem;color:var(--color-text-secondary,#475569);line-height:1.65;">{body}</div>
</div>""", unsafe_allow_html=True)

    st.caption("⚠️ Rule-based analysis only. Not financial advice. Always verify with primary sources and your own research.")



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
    g = 0.04  # assumed long-run growth for SG banks (conservative but not extreme)
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



# ── Asset type lookup (auto-detect from ticker) ──────────────────────────────
ASSET_TYPE_MAP = {
    # Banks
    "O39.SI": "Bank", "D05.SI": "Bank", "U11.SI": "Bank",
    # REITs — identified by U suffix or known REIT tickers
    "C38U.SI": "REIT", "A17U.SI": "REIT", "ME8U.SI": "REIT",
    "M44U.SI": "REIT", "N2IU.SI": "REIT", "BUOU.SI": "REIT",
    "J91U.SI": "REIT", "OXMU.SI": "REIT", "SK6U.SI": "REIT",
    "T82U.SI": "REIT", "RW0U.SI": "REIT", "SV3U.SI": "REIT",
    # US REITs
    "O": "REIT", "PLD": "REIT", "AMT": "REIT", "SPG": "REIT",
    "VICI": "REIT", "WPC": "REIT", "NNN": "REIT",
    # Companies (DDM — dividend payers)
    "Z74.SI": "Company (DDM)", "Y92.SI": "Company (DDM)",
    "BN4.SI": "Company (DCF)",
    # US Companies
    "AAPL": "Company (DCF)", "MSFT": "Company (DCF)",
    "GOOGL": "Company (DCF)", "AMZN": "Company (DCF)",
    "JPM": "Bank", "BAC": "Bank", "WFC": "Bank",
}

def auto_detect_asset_type(ticker: str) -> str:
    """Guess asset type from ticker. SGX .U suffix = REIT."""
    t = ticker.upper().strip()
    if t in ASSET_TYPE_MAP:
        return ASSET_TYPE_MAP[t]
    # SGX REITs almost always end in U before .SI
    if t.endswith(".SI") and len(t) > 4 and t[-4] == "U":
        return "REIT"
    # SGX banks — 3 char + .SI
    if t.endswith(".SI") and len(t) == 6:
        return "Bank"
    return "Company (DCF)"


# ── Watchlist helpers ─────────────────────────────────────────────────────────
def save_to_watchlist(ticker, asset_type, notes=""):
    """Save ticker to watchlist in session state."""
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = {}
    st.session_state.watchlist[ticker.upper()] = {
        "asset_type": asset_type,
        "notes": notes,
        "added": pd.Timestamp.now().strftime("%d %b %Y")
    }

def remove_from_watchlist(ticker):
    if "watchlist" in st.session_state and ticker in st.session_state.watchlist:
        del st.session_state.watchlist[ticker]

def render_watchlist_sidebar():
    """Render compact watchlist in sidebar."""
    wl = st.session_state.get("watchlist", {})
    if not wl:
        return
    st.markdown("**📋 Watchlist**")
    for t, info in list(wl.items()):
        col_a, col_b = st.columns([3, 1])
        if col_a.button(f"{t} · {info['asset_type'][:4]}", key=f"wl_{t}",
                        use_container_width=True, help=f"Added {info['added']}"):
            st.session_state["load_ticker"] = t
            st.session_state["load_asset_type"] = info["asset_type"]
            st.rerun()
        if col_b.button("✕", key=f"wl_rm_{t}"):
            remove_from_watchlist(t)
            st.rerun()

# ── Sidebar: ticker + asset type ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Valuation Tool")
    st.markdown("*Bear / Base / Bull scenarios*")
    st.divider()

    # Load from watchlist click if triggered
    _default_ticker = st.session_state.pop("load_ticker", "O39.SI")
    _default_asset  = st.session_state.pop("load_asset_type", None)

    ticker_input = st.text_input("Ticker symbol", value=_default_ticker,
                                 help="SGX: e.g. C38U.SI (CapitaLand REIT)\nUS: e.g. O (Realty Income)\nBank: e.g. D05.SI (DBS)")

    # Auto-detect asset type
    _auto_type = auto_detect_asset_type(ticker_input.strip().upper())
    _type_options = ["REIT", "Bank", "Company (DCF)", "Company (DDM)"]
    _auto_idx = _type_options.index(_auto_type) if _auto_type in _type_options else 0

    with st.expander(f"Asset type: **{_auto_type}** (auto)", expanded=False):
        st.caption("Auto-detected. Override if wrong:")
        asset_type = st.selectbox("Asset type", _type_options, index=_auto_idx, label_visibility="collapsed")
    # asset_type comes from selectbox above (already set)

    st.divider()

    # Load API key — secrets first, then session state
    _saved_key = ""
    try:
        _saved_key = st.secrets.get("FINNHUB_API_KEY", "")
    except Exception:
        pass
    _saved_key = _saved_key or st.session_state.get("finnhub_key", "")

    if _saved_key:
        # Key already loaded — show minimal status, no input needed
        st.session_state["finnhub_key"] = _saved_key
        with st.expander("🔑 API Key ✅", expanded=False):
            st.caption("Key loaded from Streamlit Secrets. Live prices and charts enabled.")
            if st.button("Change key", use_container_width=True):
                st.session_state["show_key_input"] = True
    else:
        st.markdown("**🔑 Finnhub API Key**")
        st.caption("Free key at [finnhub.io](https://finnhub.io) — needed for live prices")
        api_key_input = st.text_input(
            "API Key", value="", type="password",
            placeholder="Paste your free Finnhub key"
        )
        if api_key_input:
            st.session_state["finnhub_key"] = api_key_input
            st.caption("⚡ Active this session. Save in Streamlit Secrets to make permanent.")

    col_a, col_b = st.columns(2)
    fetch_btn = col_a.button("🔄 Fetch", use_container_width=True)
    if col_b.button("🗑️ Clear cache", use_container_width=True):
        st.cache_data.clear()
        st.success("Cache cleared!")

    # Save to watchlist
    _cur_ticker = ticker_input.strip().upper()
    _cur_asset  = _default_asset or auto_detect_asset_type(_cur_ticker)
    if st.button("⭐ Save to watchlist", use_container_width=True):
        save_to_watchlist(_cur_ticker, _cur_asset)
        st.success(f"{_cur_ticker} saved!")

    st.divider()
    render_watchlist_sidebar()

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

# Fetch market data for header (ask/bid/prev close) — uses Yahoo, no key needed
_mkt = fetch_buy_sell_pressure(ticker)

if data:
    # Use prev close from market data if Finnhub price is empty
    live_price = data.get("price") or (_mkt.get("prev_close") if _mkt else None)
    if live_price and not data.get("price"):
        data["price"] = live_price  # backfill so current_price override uses it

    curr = data.get("currency", "SGD")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Company", data.get("name","")[:28] or "—")
    _now_sgt = pd.Timestamp.now(tz="Asia/Singapore")
    _is_sgx  = ticker.endswith(".SI")
    _mkt_open = (9 <= _now_sgt.hour < 17) and (_now_sgt.weekday() < 5)
    if live_price:
        price_label = "Live (20min delay)" if _is_sgx else "Price (real-time)"
    else:
        price_label = "Prev close"
    c2.metric(price_label, fmt_price(live_price, curr) if live_price else "—")
    c3.metric("Sector", data.get("sector") or "—")
    c4.metric("P/E ratio", f"{data.get('pe'):.1f}x" if data.get("pe") else "—")

    # Market data row — always-available data only
    if _mkt:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Prev close",   fmt_price(_mkt.get("prev_close"), curr) if _mkt.get("prev_close") else "—")
        m2.metric("Day high",     fmt_price(_mkt.get("day_high"), curr) if _mkt.get("day_high") else "—")
        m3.metric("Day low",      fmt_price(_mkt.get("day_low"), curr) if _mkt.get("day_low") else "—")
        m4.metric("52w high",     fmt_price(_mkt.get("high52"), curr) if _mkt.get("high52") else "—")
        m5.metric("52w low",      fmt_price(_mkt.get("low52"), curr) if _mkt.get("low52") else "—")
        # Bid/ask removed — only available during market hours, too confusing when empty

    if not live_price:
        st.info("💡 Live price unavailable — enter your Finnhub key in sidebar or use manual override below.")
    # Data freshness strip
    _ref_check = SGX_DATA.get(ticker, {})
    _ref_year = _ref_check.get("_updated", "FY2024") if _ref_check else "live"
    _fund_src = f"Reference table ({_ref_year}) — verify after next earnings" if _ref_check else "Finnhub live data"
    _price_src = "SGX Exchange: 20-min delayed" if ticker.endswith(".SI") else "US market: real-time"
    st.markdown(
        f'<div style="background:#f8fafc;border:0.5px solid #e2e8f0;border-radius:8px;padding:5px 14px;'
        f'font-size:0.72rem;color:#64748b;margin-bottom:0.25rem;">'
        f'🕐 Price data: {_price_src} &nbsp;·&nbsp; 📋 Fundamentals: {_fund_src}'
        f'</div>',
        unsafe_allow_html=True
    )
else:
    st.info("💡 Enter Finnhub API key in sidebar for live prices. Get free key at [finnhub.io](https://finnhub.io).")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# REIT VALUATION
# ═══════════════════════════════════════════════════════════════════════════════
if asset_type == "REIT":
    st.markdown('<div class="section-header">🏢 REIT Inputs</div>', unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown("**Fundamentals**")
        ffo = st.number_input("FFO per unit / share (annual)", value=0.12, step=0.01, format="%.3f",
                              help="Funds From Operations per unit. Found in REIT annual reports.")
        dpu = st.number_input("DPU / Distribution per unit (annual)", value=0.10, step=0.01, format="%.3f",
                              help="Total annual distribution per unit paid to unitholders.")
        current_price = st.number_input("Current price (override)", value=float(data.get("price") or 1.20) if data and data.get("price") else 1.20,
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

    st.markdown('<div class="section-header">📊 Results: REIT Valuation</div>', unsafe_allow_html=True)

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
    st.markdown('<div class="section-header">📈 Fair Value Range</div>', unsafe_allow_html=True)
    chart_data = pd.DataFrame({
        "Scenario": ["Bear", "Base", "Bull", "Current price"],
        "Price/FFO method": [ffo_results["Bear"], ffo_results["Base"], ffo_results["Bull"], current_price],
        "Yield method": [yield_results.get("Bear"), yield_results.get("Base"), yield_results.get("Bull"), current_price],
    }).set_index("Scenario")
    st.bar_chart(chart_data)

    render_action_panel(current_price, ffo_results["Bear"], ffo_results["Base"], ffo_results["Bull"], ticker, data.get("currency","") if data else "")

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
    # ── Analysis ─────────────────────────────────────────────────────────────
    _bs_reit = fetch_buy_sell_pressure(ticker)
    _sig_reit, _ = signal(current_price, ffo_results.get("Bear",0), ffo_results.get("Bull",0)) if ffo_results.get("Bear") and ffo_results.get("Bull") else ("—", "")
    _hist_reit = fetch_price_history(ticker, api_key=st.session_state.get("finnhub_key",""))
    _dib_reit = int((_hist_reit["Close"] <= ffo_results.get("Bear",0)).sum()) if _hist_reit is not None and not _hist_reit.empty and ffo_results.get("Bear") else 0
    _dih_reit = int(((_hist_reit["Close"] > ffo_results.get("Bear",0)) & (_hist_reit["Close"] <= ffo_results.get("Base",0))).sum()) if _hist_reit is not None and not _hist_reit.empty and ffo_results.get("Bear") and ffo_results.get("Base") else 0
    _dat_reit = int((_hist_reit["Close"] > ffo_results.get("Bull",0)).sum()) if _hist_reit is not None and not _hist_reit.empty and ffo_results.get("Bull") else 0
    _dtot_reit = len(_hist_reit) if _hist_reit is not None and not _hist_reit.empty else 502
    render_ai_analysis(
        ticker=ticker, asset_type=asset_type, current_price=current_price,
        bear=ffo_results.get("Bear",0), base=ffo_results.get("Base",0), bull=ffo_results.get("Bull",0),
        sector=data.get("sector","—") if data else "—",
        pe=data.get("pe") if data else None,
        roe=data.get("roe") if data else None,
        book=data.get("book") if data else None,
        dps=dpu, beta=data.get("beta") if data else None,
        buy_pct=_bs_reit.get("buy_pct",50) if _bs_reit else 50,
        range_pos=_bs_reit.get("range_pos",50) if _bs_reit else 50,
        days_in_buy=_dib_reit, days_in_hold=_dih_reit, days_above_sell=_dat_reit, days_total=_dtot_reit,
        signal_text=_sig_reit
    )



# ═══════════════════════════════════════════════════════════════════════════════
# BANK VALUATION
# ═══════════════════════════════════════════════════════════════════════════════
elif asset_type == "Bank":
    st.markdown('<div class="section-header">🏦 Bank Inputs</div>', unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown("**Fundamentals**")
        book = st.number_input("Book value per share", value=float(data.get("book")) if data and data.get("book") else 10.0,
                               step=0.1, format="%.2f", help="Net assets / shares outstanding. In annual report.")
        roe = st.number_input("Return on Equity — ROE (%)", value=float(data.get("roe") * 100) if data and data.get("roe") else 12.0,
                              step=0.5, format="%.1f") / 100
        dps = st.number_input("Dividend per share (annual)", value=float(data.get("dps")) if data and data.get("dps") else 0.50,
                              step=0.05, format="%.2f")
        current_price = st.number_input("Current price (override)", value=float(data.get("price") or 10.0) if data and data.get("price") else 10.0,
                                        step=0.05, format="%.2f")

    with col_r:
        st.markdown("**Market scenario assumptions**")
        st.caption("These set how much return investors demand. Higher = lower fair value. Defaults set for Singapore 2026.")
        coe_bear = st.slider("😰 Stressed (rates rise / recession) %", 8.0, 14.0, 10.5, 0.5,
                             help="Example: MAS raises rates, Singapore economy slows, investors demand more return = lower P/B")
        coe_base = st.slider("😐 Normal market conditions %", 6.5, 12.0, 8.5, 0.5,
                             help="Current SG environment: 10yr bond ~2.8% + risk premium ~6% = ~8.5%")
        coe_bull = st.slider("😊 Optimistic (rate cuts / strong growth) %", 5.0, 10.0, 7.0, 0.5,
                             help="Rate cuts materialise, risk appetite high, investors accept lower return = higher P/B")

        st.markdown("**DDM (dividend cross-check)**")
        ddm_g_bear = st.slider("Dividend growth — Bear %", 0.0, 8.0, 1.0, 0.5)
        ddm_g_base = st.slider("Dividend growth — Base %", 0.0, 10.0, 4.0, 0.5)
        ddm_g_bull = st.slider("Dividend growth — Bull %", 0.0, 12.0, 6.0, 0.5)

    pb_results = pb_roe_price(book, roe, coe_bear, coe_base, coe_bull)
    ddm_results = ddm_price(dps, ddm_g_bear, ddm_g_base, ddm_g_bull, coe_bear, coe_base, coe_bull)

    st.markdown('<div class="section-header">📊 Results: Bank Valuation</div>', unsafe_allow_html=True)

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
    st.markdown('<div class="section-header">📈 Fair Value Range</div>', unsafe_allow_html=True)
    st.bar_chart(chart_data)

    render_action_panel(current_price, pb_results["Bear"], pb_results["Base"], pb_results["Bull"], ticker, data.get("currency","") if data else "")

    with st.expander("📖 How to read Bank valuation"):
        st.markdown("""
**Justified P/B model** is the primary tool for banks. Logic: if a bank earns ROE > cost of equity, it deserves to trade above book (P/B > 1). If ROE < COE, it should trade below book.

Formula: **Justified P/B = (ROE − g) / (COE − g)**

- DBS typically trades at P/B 1.5–2.0x because ROE (~15%) well exceeds COE (~9%)
- A bank with ROE = COE should theoretically trade at exactly 1x book
- Bear case: higher COE (rates rise, risk premium expands) → lower justified P/B

**DDM** acts as a cross-check using dividend history. If both methods give similar answers, you have higher conviction.
        """)
    # ── AI Analysis ──────────────────────────────────────────────────────────
    _bs2 = fetch_buy_sell_pressure(ticker)
    _sig2, _ = signal(current_price, pb_results.get("Bear",0), pb_results.get("Bull",0)) if pb_results.get("Bear") and pb_results.get("Bull") else ("—", "")
    _hist2 = fetch_price_history(ticker, api_key=st.session_state.get("finnhub_key",""))
    _dib = int((_hist2["Close"] <= pb_results.get("Bear",0)).sum()) if _hist2 is not None and not _hist2.empty and pb_results.get("Bear") else 0
    _dih = int(((_hist2["Close"] > pb_results.get("Bear",0)) & (_hist2["Close"] <= pb_results.get("Base",0))).sum()) if _hist2 is not None and not _hist2.empty and pb_results.get("Bear") and pb_results.get("Base") else 0
    _dat = int((_hist2["Close"] > pb_results.get("Bull",0)).sum()) if _hist2 is not None and not _hist2.empty and pb_results.get("Bull") else 0
    _dtot = len(_hist2) if _hist2 is not None and not _hist2.empty else 502
    render_ai_analysis(
        ticker=ticker, asset_type=asset_type, current_price=current_price,
        bear=pb_results.get("Bear",0), base=pb_results.get("Base",0), bull=pb_results.get("Bull",0),
        sector=data.get("sector","—") if data else "—",
        pe=data.get("pe") if data else None,
        roe=roe/100, book=book, dps=dps, beta=data.get("beta") if data else None,
        buy_pct=_bs2.get("buy_pct",50) if _bs2 else 50,
        range_pos=_bs2.get("range_pos",50) if _bs2 else 50,
        days_in_buy=_dib, days_in_hold=_dih, days_above_sell=_dat, days_total=_dtot,
        signal_text=_sig2
    )



# ═══════════════════════════════════════════════════════════════════════════════
# COMPANY — DCF
# ═══════════════════════════════════════════════════════════════════════════════
elif asset_type == "Company (DCF)":
    st.markdown('<div class="section-header">💹 DCF Inputs</div>', unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1], gap="large")

    # Auto-populate FCF from lookup table, then EPS fallback
    _ref = SGX_DATA.get(ticker, {})
    _fcf_default = _ref.get("fcf_per_share")  # None if not in table

    # Fallback: use EPS from live data as FCF proxy (conservative but directionally correct)
    if not _fcf_default and data:
        _eps = data.get("eps")
        if _eps and _eps > 0:
            _fcf_default = round(_eps * 0.85, 2)  # FCF typically ~85% of EPS for quality companies

    # Last resort: price/30 gives a rough P/FCF of 30x as starting point
    _price_default = float(data.get("price") or 50.0) if data and data.get("price") else 50.0
    if not _fcf_default:
        _fcf_default = round(_price_default / 30, 2)  # assumes 30x P/FCF as placeholder

    # Use per-stock growth/WACC if in lookup table, else sector defaults
    _sector = (data.get("sector") or _ref.get("sector") or "").lower()
    if _ref.get("fcf_g_base"):
        # Use stock-specific assumptions from lookup table
        _g_bear  = _ref.get("fcf_g_bear", 5.0)
        _g_base  = _ref.get("fcf_g_base", 12.0)
        _g_bull  = _ref.get("fcf_g_bull", 20.0)
        _wacc_bear = _ref.get("wacc_bear", 11.0)
        _wacc_base = _ref.get("wacc_base", 9.0)
        _wacc_bull = _ref.get("wacc_bull", 7.5)
    elif "tech" in _sector:
        _g_bear, _g_base, _g_bull = 5.0, 15.0, 25.0
        _wacc_bear, _wacc_base, _wacc_bull = 12.0, 10.0, 8.0
    elif "bank" in _sector or "financ" in _sector:
        _g_bear, _g_base, _g_bull = 2.0, 5.0, 8.0
        _wacc_bear, _wacc_base, _wacc_bull = 11.0, 9.0, 7.5
    elif "reit" in _sector:
        _g_bear, _g_base, _g_bull = 1.0, 3.0, 5.0
        _wacc_bear, _wacc_base, _wacc_bull = 9.0, 7.5, 6.0
    elif "consumer" in _sector or "staple" in _sector:
        _g_bear, _g_base, _g_bull = 2.0, 5.0, 8.0
        _wacc_bear, _wacc_base, _wacc_bull = 10.0, 8.5, 7.0
    else:
        _g_bear, _g_base, _g_bull = 3.0, 8.0, 14.0
        _wacc_bear, _wacc_base, _wacc_bull = 12.0, 9.0, 7.0

    with col_l:
        st.markdown("**Fundamentals**")
        if _ref.get("fcf_per_share"):
            st.caption(f"✅ FCF/share loaded from reference data for {ticker}. Verify against latest annual report.")
        elif data and data.get("eps") and data.get("eps", 0) > 0:
            st.caption(f"⚡ FCF estimated from live EPS (${data.get('eps'):.2f} × 85%). Adjust if you have the exact FCF from the annual report.")
        else:
            st.caption(f"ℹ️ Enter FCF/share manually: find it in the annual report → Cash Flow Statement → Free Cash Flow ÷ shares outstanding.")
        fcf = st.number_input("Free Cash Flow per share (annual, $)", value=float(_fcf_default), step=0.10, format="%.2f",
                              help="FCF = Operating Cash Flow − Capex ÷ shares outstanding. Find in annual report cash flow statement.")
        current_price = st.number_input("Current price (override)", value=_price_default, step=0.50, format="%.2f")
        years = st.slider("Projection period (years)", 5, 15, 10)
        terminal_g = st.slider("Terminal growth rate (%)", 0.5, 4.0, 2.5, 0.1,
                               help="Long-run GDP growth rate. 2.0–2.5% is standard. Never use above 3.5%.")
        st.warning("⚠️ Terminal growth rate is the most sensitive input — a 0.5% change moves fair value by 20–30%.")

    with col_r:
        st.markdown("**FCF Growth by scenario**")
        st.caption(f"Pre-set for {data.get('sector') or _ref.get('sector','this sector')} — adjust if needed")
        g_bear = st.slider("Bear FCF growth (%/yr)", -5.0, 20.0, _g_bear, 0.5)
        g_base = st.slider("Base FCF growth (%/yr)", 0.0, 30.0, _g_base, 0.5)
        g_bull = st.slider("Bull FCF growth (%/yr)", 5.0, 40.0, _g_bull, 0.5)

        st.markdown("**Market scenario assumptions**")
        st.caption("Higher % = more conservative fair value. US tech typically 9–12%. SGX blue chips 7–9%.")
        wacc_bear = st.slider("😰 Stressed (high rates / slow growth) %", 8.0, 18.0, _wacc_bear, 0.5)
        wacc_base = st.slider("😐 Normal conditions %", 6.0, 15.0, _wacc_base, 0.5)
        wacc_bull = st.slider("😊 Optimistic (rate cuts / strong growth) %", 4.0, 12.0, _wacc_bull, 0.5)

    dcf_results = dcf_price(fcf, g_bear, g_base, g_bull,
                            wacc_bear, wacc_base, wacc_bull,
                            terminal_g, years)

    st.markdown('<div class="section-header">📊 Results: DCF Valuation</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="section-header">🔍 Value Breakdown (Base Case)</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="section-header">📈 Fair Value Range</div>', unsafe_allow_html=True)
    st.bar_chart(chart_data)

    render_action_panel(current_price, dcf_results["Bear"], dcf_results["Base"], dcf_results["Bull"], ticker, data.get("currency","") if data else "")

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
    # ── Analysis ─────────────────────────────────────────────────────────────
    _bs_dcf = fetch_buy_sell_pressure(ticker)
    _sig_dcf, _ = signal(current_price, dcf_results.get("Bear",0), dcf_results.get("Bull",0)) if dcf_results.get("Bear") and dcf_results.get("Bull") else ("—", "")
    _hist_dcf = fetch_price_history(ticker, api_key=st.session_state.get("finnhub_key",""))
    _dib_dcf = int((_hist_dcf["Close"] <= dcf_results.get("Bear",0)).sum()) if _hist_dcf is not None and not _hist_dcf.empty and dcf_results.get("Bear") else 0
    _dih_dcf = int(((_hist_dcf["Close"] > dcf_results.get("Bear",0)) & (_hist_dcf["Close"] <= dcf_results.get("Base",0))).sum()) if _hist_dcf is not None and not _hist_dcf.empty and dcf_results.get("Bear") and dcf_results.get("Base") else 0
    _dat_dcf = int((_hist_dcf["Close"] > dcf_results.get("Bull",0)).sum()) if _hist_dcf is not None and not _hist_dcf.empty and dcf_results.get("Bull") else 0
    _dtot_dcf = len(_hist_dcf) if _hist_dcf is not None and not _hist_dcf.empty else 502
    _roe_dcf = data.get("roe") if data else None
    render_ai_analysis(
        ticker=ticker, asset_type=asset_type, current_price=current_price,
        bear=dcf_results.get("Bear",0), base=dcf_results.get("Base",0), bull=dcf_results.get("Bull",0),
        sector=data.get("sector","—") if data else "—",
        pe=data.get("pe") if data else None,
        roe=_roe_dcf, book=data.get("book") if data else None,
        dps=data.get("dps") if data else None,
        beta=data.get("beta") if data else None,
        buy_pct=_bs_dcf.get("buy_pct",50) if _bs_dcf else 50,
        range_pos=_bs_dcf.get("range_pos",50) if _bs_dcf else 50,
        days_in_buy=_dib_dcf, days_in_hold=_dih_dcf, days_above_sell=_dat_dcf, days_total=_dtot_dcf,
        signal_text=_sig_dcf
    )



# ═══════════════════════════════════════════════════════════════════════════════
# COMPANY — DDM
# ═══════════════════════════════════════════════════════════════════════════════
elif asset_type == "Company (DDM)":
    st.markdown('<div class="section-header">💰 DDM Inputs</div>', unsafe_allow_html=True)
    st.info("DDM works best for stable dividend-paying companies with predictable payout ratios (utilities, consumer staples, telcos).")

    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown("**Fundamentals**")
        dps_val = st.number_input("Dividend per share — current (annual)", value=float(data.get("dps")) if data and data.get("dps") else 1.20,
                                  step=0.05, format="%.2f")
        current_price = st.number_input("Current price (override)", value=float(data.get("price") or 25.0) if data and data.get("price") else 25.0,
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

        st.markdown("**Market scenario assumptions**")
        coe_bear = st.slider("😰 Stressed scenario %", 7.0, 16.0, 11.0, 0.5)
        coe_base = st.slider("😐 Normal conditions %", 5.0, 14.0, 8.0, 0.5)
        coe_bull = st.slider("😊 Optimistic scenario %", 3.0, 12.0, 6.0, 0.5)

    ddm_results = ddm_price(dps_val, g_bear, g_base, g_bull, coe_bear, coe_base, coe_bull)

    st.markdown('<div class="section-header">📊 Results: DDM Valuation</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="section-header">💰 Yield Context</div>', unsafe_allow_html=True)
    y1, y2, y3 = st.columns(3)
    y1.metric("Current yield", f"{curr_yield:.2f}%")
    y2.metric("Bear yield (at bear price)", f"{dps_val/bv*100:.2f}%" if bv else "—")
    y3.metric("Bull yield (at bull price)", f"{dps_val/blv*100:.2f}%" if blv else "—")

    chart_data = pd.DataFrame({
        "Scenario": ["Bear", "Base", "Bull", "Current price"],
        "Fair value": [bv, bav, blv, current_price],
    }).set_index("Scenario")
    st.markdown('<div class="section-header">📈 Fair Value Range</div>', unsafe_allow_html=True)
    st.bar_chart(chart_data)

    if bv and bav and blv:
        render_action_panel(current_price, bv, bav, blv, ticker, data.get("currency","") if data else "")

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
    # ── Analysis ─────────────────────────────────────────────────────────────
    _bs_ddm = fetch_buy_sell_pressure(ticker)
    _bv_ddm = ddm_results.get("Bear")
    _base_ddm = ddm_results.get("Base")
    _bull_ddm = ddm_results.get("Bull")
    if _bv_ddm and _bull_ddm:
        _sig_ddm, _ = signal(current_price, _bv_ddm, _bull_ddm)
    else:
        _sig_ddm = "—"
    _hist_ddm = fetch_price_history(ticker, api_key=st.session_state.get("finnhub_key",""))
    _dib_ddm = int((_hist_ddm["Close"] <= _bv_ddm).sum()) if _hist_ddm is not None and not _hist_ddm.empty and _bv_ddm else 0
    _dih_ddm = int(((_hist_ddm["Close"] > _bv_ddm) & (_hist_ddm["Close"] <= _base_ddm)).sum()) if _hist_ddm is not None and not _hist_ddm.empty and _bv_ddm and _base_ddm else 0
    _dat_ddm = int((_hist_ddm["Close"] > _bull_ddm).sum()) if _hist_ddm is not None and not _hist_ddm.empty and _bull_ddm else 0
    _dtot_ddm = len(_hist_ddm) if _hist_ddm is not None and not _hist_ddm.empty else 502
    render_ai_analysis(
        ticker=ticker, asset_type=asset_type, current_price=current_price,
        bear=_bv_ddm or 0, base=_base_ddm or 0, bull=_bull_ddm or 0,
        sector=data.get("sector","—") if data else "—",
        pe=data.get("pe") if data else None,
        roe=data.get("roe") if data else None,
        book=data.get("book") if data else None,
        dps=dps_val, beta=data.get("beta") if data else None,
        buy_pct=_bs_ddm.get("buy_pct",50) if _bs_ddm else 50,
        range_pos=_bs_ddm.get("range_pos",50) if _bs_ddm else 50,
        days_in_buy=_dib_ddm, days_in_hold=_dih_ddm, days_above_sell=_dat_ddm, days_total=_dtot_ddm,
        signal_text=_sig_ddm
    )



# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("⚠️ This tool is for educational and research purposes only. It is not financial advice. Always do your own due diligence before investing. Price data via Yahoo Finance and may be delayed.")
