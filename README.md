
# --- Imports ---
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from google.colab import files, drive
import warnings

# --- Ignore warnings ---
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- Mount Google Drive ---
drive.mount('/content/drive')

# --- Set tickers ---
ticker = "BSE.NS"             # Stock of interest
benchmark_ticker = "INDIGO.NS"  # Benchmark

# --- Download stock and benchmark data ---
data = yf.download(ticker, start="2020-01-01", end="2025-09-06", auto_adjust=False).dropna()
benchmark_data = yf.download(benchmark_ticker, start="2020-01-01", end="2025-09-06", auto_adjust=False).dropna()

# --- Prices and returns ---
prices = data["Close"]
returns = prices.pct_change().dropna().squeeze()
benchmark_prices = benchmark_data["Close"]
benchmark_returns = benchmark_prices.pct_change().dropna().squeeze()

# --- Metrics for stock ---
expected_annual_return = returns.mean() * 252
annual_volatility = returns.std(ddof=0) * np.sqrt(252)
annual_volatility_val = float(annual_volatility)
sharpe_ratio = expected_annual_return / annual_volatility_val if annual_volatility_val != 0 else np.nan

downside_returns = returns[returns < 0]
downside_std = downside_returns.std(ddof=0) * np.sqrt(252)
downside_std_val = float(downside_std)
sortino_ratio = expected_annual_return / downside_std_val if downside_std_val != 0 else np.nan

cumulative_returns = (1 + returns).cumprod()
rolling_max = cumulative_returns.cummax()
drawdowns = (cumulative_returns - rolling_max) / rolling_max
max_drawdown = drawdowns.min()

# --- Metrics for benchmark ---
expected_annual_return_b = benchmark_returns.mean() * 252
annual_volatility_b = benchmark_returns.std(ddof=0) * np.sqrt(252)
annual_volatility_val_b = float(annual_volatility_b)
sharpe_ratio_b = expected_annual_return_b / annual_volatility_val_b if annual_volatility_val_b != 0 else np.nan

downside_returns_b = benchmark_returns[benchmark_returns < 0]
downside_std_b = downside_returns_b.std(ddof=0) * np.sqrt(252)
downside_std_val_b = float(downside_std_b)
sortino_ratio_b = expected_annual_return_b / downside_std_val_b if downside_std_val_b != 0 else np.nan

cumulative_returns_b = (1 + benchmark_returns).cumprod()
rolling_max_b = cumulative_returns_b.cummax()
drawdowns_b = (cumulative_returns_b - rolling_max_b) / rolling_max_b
max_drawdown_b = drawdowns_b.min()

# --- Current PE ratio, Dividend Yield, EPS ---
stock_info = yf.Ticker(ticker).info
benchmark_info = yf.Ticker(benchmark_ticker).info

current_pe = stock_info.get("trailingPE", np.nan)
dividend_yield = stock_info.get("dividendYield", np.nan)
eps = stock_info.get("trailingEps", np.nan)

benchmark_pe = benchmark_info.get("trailingPE", np.nan)
benchmark_dividend_yield = benchmark_info.get("dividendYield", np.nan)
benchmark_eps = benchmark_info.get("trailingEps", np.nan)

# --- EMAs ---
ema_50 = prices.ewm(span=50, adjust=False).mean()
ema_200 = prices.ewm(span=200, adjust=False).mean()
ema_50_b = benchmark_prices.ewm(span=50, adjust=False).mean()
ema_200_b = benchmark_prices.ewm(span=200, adjust=False).mean()

# --- PDF path in Drive ---
pdf_path = '/content/drive/MyDrive/quant_report_BSE_vs_INDIGO.pdf'

# --- Scoring function ---
def calculate_score(pe, sharpe, sortino, vol, mdd, div, eps, ema_50, ema_200, price):
    score = 0
    if pe < 50: score += 3
    if sharpe > 1: score += 3
    if sortino > 1: score += 1
    if vol < 0.4: score += 2
    if mdd > -0.4: score += 1
    if div > 0.01: score += 1
    if eps > 1: score += 1
    # EMA based scoring
    if float(ema_50.iloc[-1]) > float(ema_200.iloc[-1]):  # EMA Golden Cross
        score += 3
    if float(price.iloc[-1]) > float(ema_50.iloc[-1]) and float(price.iloc[-1]) > float(ema_200.iloc[-1]):
        score += 2
    return score

stock_score = calculate_score(current_pe, sharpe_ratio, sortino_ratio, annual_volatility_val, 
                              max_drawdown, dividend_yield, eps, ema_50, ema_200, prices)
benchmark_score = calculate_score(benchmark_pe, sharpe_ratio_b, sortino_ratio_b, annual_volatility_val_b, 
                                  max_drawdown_b, benchmark_dividend_yield, benchmark_eps, ema_50_b, ema_200_b, benchmark_prices)

def verdict(score):
    if score >= 15:
        return "BUY"
    elif score >= 9:
        return "HOLD"
    else:
        return "SELL"

stock_verdict = verdict(stock_score)
benchmark_verdict = verdict(benchmark_score)

# --- Create PDF ---
with PdfPages(pdf_path) as pdf:

    # --- Page 1: Metrics Table ---
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    metrics_df = pd.DataFrame({
        "Metric": ["Expected Annual Return", "Annual Volatility", "Sharpe Ratio", "Sortino Ratio", "Max Drawdown", "P/E Ratio"],
        "Stock": [f"{expected_annual_return:.2%}", f"{annual_volatility_val:.2%}", f"{sharpe_ratio:.2f}", f"{sortino_ratio:.2f}", f"{max_drawdown:.2%}", f"{current_pe:.2f}"],
        "Benchmark": [f"{expected_annual_return_b:.2%}", f"{annual_volatility_val_b:.2%}", f"{sharpe_ratio_b:.2f}", f"{sortino_ratio_b:.2f}", f"{max_drawdown_b:.2%}", f"{benchmark_pe:.2f}"]
    })
    table = ax.table(cellText=metrics_df.values, colLabels=metrics_df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    ax.set_title(f"Quantitative Analysis Report - {ticker} vs {benchmark_ticker}", fontsize=16, pad=20)
    pdf.savefig(fig)
    plt.close(fig)

    # --- Pages 2-5: OHLC Charts ---
    price_columns = ["Open", "High", "Low", "Close"]
    colors = ["orange", "green", "red", "blue"]
    for col, color in zip(price_columns, colors):
        if col in data.columns and col in benchmark_data.columns:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(data.index, data[col], label=f"{ticker} {col}", color=color, linestyle="-")
            ax.plot(benchmark_data.index, benchmark_data[col], label=f"{benchmark_ticker} {col}", color=color, linestyle="--")
            ax.set_title(f"{ticker} vs {benchmark_ticker} - {col} Price History")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price (INR)")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.6)
            pdf.savefig(fig)
            plt.close(fig)

    # --- Page 6: Combined OHLC ---
    fig, ax = plt.subplots(figsize=(12, 6))
    for col, color in zip(price_columns, colors):
        if col in data.columns and col in benchmark_data.columns:
            ax.plot(data.index, data[col], label=f"{ticker} {col}", color=color, linestyle="-")
            ax.plot(benchmark_data.index, benchmark_data[col], label=f"{benchmark_ticker} {col}", color=color, linestyle="--")
    ax.set_title(f"{ticker} vs {benchmark_ticker} - Combined OHLC Price History")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (INR)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    pdf.savefig(fig)
    plt.close(fig)

    # --- Page 7: Moving Averages ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(prices.index, prices, label="Close Price", color="blue")
    ax.plot(prices.index, prices.rolling(50).mean(), label="50-day MA", color="orange")
    ax.plot(prices.index, prices.rolling(200).mean(), label="200-day MA", color="green")
    ax.set_title(f"{ticker} - Moving Averages")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (INR)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    pdf.savefig(fig)
    plt.close(fig)

    # --- Page 8: Cumulative Returns ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(cumulative_returns.index, cumulative_returns, label=ticker, color="blue")
    ax.plot(cumulative_returns_b.index, cumulative_returns_b, label=benchmark_ticker, color="orange")
    ax.set_title("Cumulative Returns: Stock vs Benchmark")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    pdf.savefig(fig)
    plt.close(fig)

    # --- Page 9: Drawdowns ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(drawdowns.index, drawdowns, label=ticker, color="purple")
    ax.plot(drawdowns_b.index, drawdowns_b, label=benchmark_ticker, color="brown")
    ax.set_title("Drawdowns: Stock vs Benchmark")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    pdf.savefig(fig)
    plt.close(fig)

    # --- Page 10: Returns Histogram ---
    fig, ax = plt.subplots(figsize=(12, 6))
    returns.hist(bins=50, ax=ax, color="cyan")
    ax.set_title(f"{ticker} Daily Returns Distribution")
    ax.set_xlabel("Return")
    ax.set_ylabel("Frequency")
    ax.grid(True, linestyle="--", alpha=0.6)
    pdf.savefig(fig)
    plt.close(fig)

    # --- Page 11: Performance + Dividend Yield & EPS ---
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    compare_df = pd.DataFrame({
        "Metric": ["Expected Annual Return", "Annual Volatility", "Sharpe Ratio", "Sortino Ratio", "Max Drawdown", "P/E Ratio", "Dividend Yield", "EPS"],
        "Stock": [f"{expected_annual_return:.2%}", f"{annual_volatility_val:.2%}", f"{sharpe_ratio:.2f}", f"{sortino_ratio:.2f}", f"{max_drawdown:.2%}",
                  f"{current_pe:.2f}", f"{dividend_yield:.2%}" if pd.notna(dividend_yield) else "N/A", f"{eps:.2f}" if pd.notna(eps) else "N/A"],
        "Benchmark": [f"{expected_annual_return_b:.2%}", f"{annual_volatility_val_b:.2%}", f"{sharpe_ratio_b:.2f}", f"{sortino_ratio_b:.2f}", f"{max_drawdown_b:.2%}",
                      f"{benchmark_pe:.2f}", f"{benchmark_dividend_yield:.2%}" if pd.notna(benchmark_dividend_yield) else "N/A", f"{benchmark_eps:.2f}" if pd.notna(benchmark_eps) else "N/A"]
    })
    table = ax.table(cellText=compare_df.values, colLabels=compare_df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    ax.set_title("Stock vs Benchmark: Performance + Dividend Yield & EPS", fontsize=16, pad=20)
    pdf.savefig(fig)
    plt.close(fig)

    # --- Page 12: Scoring & Verdict ---
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    score_df = pd.DataFrame({
        "Metric": ["Total Score", "Verdict"],
        "Stock": [stock_score, stock_verdict],
        "Benchmark": [benchmark_score, benchmark_verdict]
    })
    table = ax.table(cellText=score_df.values, colLabels=score_df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 3)
    ax.set_title("Stock vs Benchmark: Scoring & Verdict", fontsize=16, pad=20)
    pdf.savefig(fig)
    plt.close(fig)

print(f"✅ Report saved successfully in your Google Drive at: {pdf_path}")
files.download(pdf_path)
