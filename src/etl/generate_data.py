"""
FinTech AI - Synthetic Financial Data Generator
Mirrors the EduAI generate_data.py pattern but for hedge fund / market data.
Generates: portfolio positions, market data, earnings transcripts, macro indicators.
No real financial data used.
"""

import json
import random
import os
from datetime import datetime, timedelta
from faker import Faker

fake = Faker()
random.seed(42)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../../data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

#  Config 

TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",
           "JPM", "GS", "BRK.B", "UNH", "JNJ", "XOM", "CVX", "LLY",
           "V", "MA", "AVGO", "ORCL", "AMD"]

SECTORS = {
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "GOOGL": "Technology", "AMZN": "Consumer Discretionary", "META": "Technology",
    "TSLA": "Consumer Discretionary", "JPM": "Financials", "GS": "Financials",
    "BRK.B": "Financials", "UNH": "Healthcare", "JNJ": "Healthcare",
    "XOM": "Energy", "CVX": "Energy", "LLY": "Healthcare",
    "V": "Financials", "MA": "Financials", "AVGO": "Technology",
    "ORCL": "Technology", "AMD": "Technology"
}

MACRO_INDICATORS = [
    "Federal Funds Rate", "CPI YoY", "Core PCE", "10Y Treasury Yield",
    "2Y Treasury Yield", "Unemployment Rate", "GDP Growth QoQ",
    "ISM Manufacturing PMI", "Consumer Confidence Index", "VIX"
]


#  Portfolio Positions 

def generate_portfolio(n_positions=15):
    positions = []
    selected = random.sample(TICKERS, n_positions)
    for ticker in selected:
        shares = random.randint(100, 10000)
        avg_cost = round(random.uniform(50, 800), 2)
        current_price = round(avg_cost * random.uniform(0.6, 1.8), 2)
        market_value = round(shares * current_price, 2)
        unrealized_pnl = round((current_price - avg_cost) * shares, 2)
        pnl_pct = round((current_price - avg_cost) / avg_cost * 100, 2)

        positions.append({
            "ticker": ticker,
            "sector": SECTORS.get(ticker, "Unknown"),
            "shares": shares,
            "avg_cost": avg_cost,
            "current_price": current_price,
            "market_value": market_value,
            "unrealized_pnl": unrealized_pnl,
            "pnl_pct": pnl_pct,
            "weight_pct": 0.0,  # filled below
            "beta": round(random.uniform(0.4, 2.2), 2),
            "52w_high": round(current_price * random.uniform(1.0, 1.6), 2),
            "52w_low": round(current_price * random.uniform(0.5, 1.0), 2),
            "pe_ratio": round(random.uniform(8, 60), 1),
            "analyst_rating": random.choice(["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"]),
        })

    total_value = sum(p["market_value"] for p in positions)
    for p in positions:
        p["weight_pct"] = round(p["market_value"] / total_value * 100, 2)

    return {
        "portfolio_id": "HF-ALPHA-001",
        "fund_name": "Horner Capital Alpha Fund",
        "aum": total_value,
        "positions": positions,
        "as_of_date": datetime.today().strftime("%Y-%m-%d"),
    }


#  Historical Market Data 

def generate_market_data(days=90):
    records = []
    base_date = datetime.today() - timedelta(days=days)

    for ticker in TICKERS:
        price = random.uniform(50, 800)
        for i in range(days):
            date = base_date + timedelta(days=i)
            if date.weekday() >= 5:
                continue
            change_pct = random.gauss(0, 0.02)
            price = max(1.0, price * (1 + change_pct))
            volume = random.randint(1_000_000, 50_000_000)
            records.append({
                "ticker": ticker,
                "sector": SECTORS.get(ticker, "Unknown"),
                "date": date.strftime("%Y-%m-%d"),
                "open": round(price * random.uniform(0.98, 1.0), 2),
                "high": round(price * random.uniform(1.0, 1.03), 2),
                "low": round(price * random.uniform(0.97, 1.0), 2),
                "close": round(price, 2),
                "volume": volume,
                "rsi_14": round(random.uniform(20, 80), 1),
                "sma_50": round(price * random.uniform(0.90, 1.10), 2),
                "sma_200": round(price * random.uniform(0.80, 1.20), 2),
                "macd_signal": round(random.uniform(-5, 5), 3),
            })
    return records


#  Earnings Transcripts 

TRANSCRIPT_TEMPLATES = [
    "Revenue for Q{q} {year} came in at ${rev}B, {beat} analyst estimates of ${est}B. "
    "EPS of ${eps} {beat2} the consensus of ${eps_est}. Management guided {guide} for next quarter.",

    "The CEO highlighted strength in {segment}, noting {pct}% YoY growth. "
    "Gross margins expanded to {margin}%, driven by {driver}. "
    "Free cash flow reached ${fcf}B for the quarter.",

    "On the macro environment, CFO noted {macro_comment}. "
    "The company is focused on {strategy} to drive long-term shareholder value. "
    "Share buybacks of ${buyback}B were authorized for fiscal year {year}.",
]

MACRO_COMMENTS = [
    "cautious consumer sentiment impacting discretionary spending",
    "strong enterprise demand despite rising interest rates",
    "FX headwinds reducing international revenue by approximately 3%",
    "supply chain normalization contributing to margin improvement",
    "AI-driven demand acceleration across all product lines",
]

STRATEGIES = [
    "AI integration across core product offerings",
    "international market expansion in APAC and EMEA",
    "operational efficiency and cost discipline",
    "M&A in adjacent technology verticals",
    "cloud migration and recurring revenue growth",
]


def generate_earnings_transcripts(n=40):
    transcripts = []
    for ticker in TICKERS:
        for quarter in range(1, 3):
            year = 2024 if quarter < 3 else 2025
            rev = round(random.uniform(5, 120), 1)
            est = round(rev * random.uniform(0.93, 1.07), 1)
            beat = "beat" if rev > est else "missed"
            eps = round(random.uniform(0.5, 8.0), 2)
            eps_est = round(eps * random.uniform(0.92, 1.08), 2)
            beat2 = "beat" if eps > eps_est else "missed"
            guide = random.choice(["above", "in-line with", "below"])
            transcript_body = " ".join([
                t.format(
                    q=quarter, year=year, rev=rev, est=est, beat=beat,
                    eps=eps, eps_est=eps_est, beat2=beat2, guide=guide,
                    segment=random.choice(["cloud", "hardware", "services", "advertising", "enterprise"]),
                    pct=random.randint(5, 45),
                    margin=round(random.uniform(30, 75), 1),
                    driver=random.choice(["pricing power", "product mix shift", "scale efficiencies"]),
                    fcf=round(random.uniform(1, 30), 1),
                    macro_comment=random.choice(MACRO_COMMENTS),
                    strategy=random.choice(STRATEGIES),
                    buyback=round(random.uniform(1, 20), 1),
                ) for t in TRANSCRIPT_TEMPLATES
            ])
            transcripts.append({
                "ticker": ticker,
                "sector": SECTORS.get(ticker, "Unknown"),
                "quarter": f"Q{quarter} {year}",
                "date": f"{year}-{'02' if quarter == 1 else '05' if quarter == 2 else '08' if quarter == 3 else '11'}-15",
                "revenue_actual": rev,
                "revenue_estimate": est,
                "eps_actual": eps,
                "eps_estimate": eps_est,
                "beat_miss": "beat" if (rev > est and eps > eps_est) else "miss",
                "transcript_summary": transcript_body,
            })
    return transcripts


#  Macro Indicators 

def generate_macro_indicators(months=12):
    records = []
    base_date = datetime.today() - timedelta(days=365)
    for i in range(months):
        date = base_date + timedelta(days=i * 30)
        records.append({
            "date": date.strftime("%Y-%m"),
            "Federal Funds Rate": round(random.uniform(4.5, 5.5), 2),
            "CPI YoY": round(random.uniform(2.5, 5.5), 2),
            "Core PCE": round(random.uniform(2.2, 4.5), 2),
            "10Y Treasury Yield": round(random.uniform(3.8, 5.0), 2),
            "2Y Treasury Yield": round(random.uniform(4.0, 5.5), 2),
            "Unemployment Rate": round(random.uniform(3.5, 4.5), 2),
            "GDP Growth QoQ": round(random.uniform(-0.5, 4.0), 2),
            "ISM Manufacturing PMI": round(random.uniform(44, 56), 1),
            "Consumer Confidence Index": round(random.uniform(60, 115), 1),
            "VIX": round(random.uniform(12, 35), 2),
            "commentary": random.choice([
                "Fed holds rates steady amid mixed inflation signals.",
                "Labor market remains resilient; jobless claims tick up slightly.",
                "PMI contraction signals slowing manufacturing activity.",
                "Consumer spending beats expectations despite high rates.",
                "Yield curve inversion deepens; recession watch elevated.",
                "Inflation prints below expectations; rate cut bets increase.",
            ])
        })
    return records


#  Main 

if __name__ == "__main__":
    print("Generating synthetic financial data...")

    portfolio = generate_portfolio()
    with open(f"{OUTPUT_DIR}/portfolio.json", "w") as f:
        json.dump(portfolio, f, indent=2)
    print(f"  [OK] portfolio.json  ({len(portfolio['positions'])} positions)")

    market = generate_market_data()
    with open(f"{OUTPUT_DIR}/market_data.json", "w") as f:
        json.dump(market, f, indent=2)
    print(f"  [OK] market_data.json  ({len(market)} daily records)")

    transcripts = generate_earnings_transcripts()
    with open(f"{OUTPUT_DIR}/earnings_transcripts.json", "w") as f:
        json.dump(transcripts, f, indent=2)
    print(f"  [OK] earnings_transcripts.json  ({len(transcripts)} transcripts)")

    macro = generate_macro_indicators()
    with open(f"{OUTPUT_DIR}/macro_indicators.json", "w") as f:
        json.dump(macro, f, indent=2)
    print(f"  [OK] macro_indicators.json  ({len(macro)} monthly records)")

    print("\nAll synthetic data generated in /data/")
