
import yfinance as yf
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# --- Streamlit App ---
st.title("💼 Portfolio Optimizer with Live Market Data")

# Step 1: User Inputs
st.sidebar.header("Portfolio Settings")
tickers = st.sidebar.text_input("Enter stock tickers (comma-separated):", "AAPL,MSFT,GOOGL").upper().split(',')
start_date = st.sidebar.date_input("Start date", pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End date", pd.to_datetime("today"))
risk_free_rate = 0.01

# Step 2: Fetch Data
@st.cache_data
def get_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)
    st.write("Raw Yahoo Finance Data:", data.head())  # Debug output
    if "Adj Close" in data.columns:
        return data["Adj Close"].dropna()
    elif "Close" in data.columns:
        return data["Close"].dropna()
    else:
        raise ValueError("No 'Adj Close' or 'Close' column found. Check ticker symbols or date range.")

try:
    price_data = get_data(tickers, start_date, end_date)
    st.write("Filtered Price Data:", price_data.head())  # Debug output
    returns = price_data.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

# Step 3: Portfolio Optimization
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.dot(weights, mean_returns) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    sharpe_ratio = (returns - risk_free_rate) / std
    return std, returns, sharpe_ratio

def neg_sharpe_ratio(weights, mean_returns, cov_matrix):
    return -portfolio_performance(weights, mean_returns, cov_matrix)[2]

def optimize_portfolio(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    result = minimize(neg_sharpe_ratio, num_assets * [1. / num_assets], args=args,
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result

# Step 4: Display Results
result = optimize_portfolio(mean_returns, cov_matrix)
optimal_weights = result.x
std, ret, sharpe = portfolio_performance(optimal_weights, mean_returns, cov_matrix)

st.subheader("📊 Optimized Portfolio")
for i, ticker in enumerate(tickers):
    st.write(f"{ticker}: {optimal_weights[i]:.2%}")

st.write(f"Expected annual return: {ret:.2%}")
st.write(f"Annual volatility: {std:.2%}")
st.write(f"Sharpe Ratio: {sharpe:.2f}")

# Plot allocation
fig, ax = plt.subplots()
ax.pie(optimal_weights, labels=tickers, autopct='%1.1f%%', startangle=90)
ax.axis('equal')
st.pyplot(fig)
