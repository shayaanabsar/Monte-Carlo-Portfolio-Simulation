import yfinance as yf
import streamlit as st
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Check if tickers exist
# -----------------------------
def tickers_exist(tickers: list[str]) -> bool:
    """Check if each ticker exists in Yahoo Finance."""
    all_exist = True
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            if info is None or info == {} or info.get('regularMarketPrice') is None:
                st.warning(f'Ticker {ticker} could not be found')
                all_exist = False
        except Exception:
            st.warning(f'Ticker {ticker} could not be found')
            all_exist = False
    return all_exist


# -----------------------------
# Download historical stock data
# -----------------------------
def download_stock_data(tickers: list[str], days: int = 365) -> pd.DataFrame:
    """Download historical stock prices and compute daily returns."""
    stock_prices = yf.download(
        tickers,
        start=dt.datetime.now() - dt.timedelta(days=days),
        end=dt.datetime.now()
    )
    close_prices = stock_prices['Close']
    returns = close_prices.pct_change().dropna()
    return returns


# -----------------------------
# Calculate mean returns & covariance
# -----------------------------
def calculate_statistics(stock_prices: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return mean returns and covariance matrix of stock returns."""
    mean_returns = stock_prices.mean().values
    covariance = stock_prices.cov().values
    return mean_returns, covariance


# -----------------------------
# Calculate annualized Sharpe ratio
# -----------------------------
def calculate_sharpe_ratio(final_returns, num_days):
    """
    Compute the annualized Sharpe ratio.
    
    final_returns: array of portfolio returns
    num_days: simulation horizon in days
    """
    final_returns = np.array(final_returns)

    # Download 3-month T-bill yield as risk-free rate
    rf_data = yf.download('^IRX', period='1y', progress=False)['Close'].dropna()
    rf_annual = float(rf_data.iloc[-1] / 100)

    # Excess returns over risk-free
    excess_returns = final_returns - rf_annual

    # Sharpe ratio: mean / std, annualized for simulation horizon
    sharpe_ratio = excess_returns.mean() / excess_returns.std(ddof=1)
    return sharpe_ratio * np.sqrt(252 / num_days)


# -----------------------------
# Risk metrics
# -----------------------------
def calculate_expected_value(final_values):
    """Mean portfolio value at the end of simulations."""
    return final_values.mean()


def calculate_var(final_values, alpha):
    """Value at Risk (percentile of final values)."""
    return np.percentile(final_values, alpha * 100)


def calculate_cvar(final_values, alpha):
    """Conditional Value at Risk (mean of worst alpha% outcomes)."""
    relevant_data = final_values <= calculate_var(final_values, alpha)
    return final_values[relevant_data].mean()


# -----------------------------
# Plot Monte Carlo simulations
# -----------------------------
def plot_graph(portfolio_simulations, alpha, num_days, initial_portfolio_value):
    """Plot all simulation paths and calculate statistics."""
    final_values = portfolio_simulations[-1, :]
    final_returns = (final_values - initial_portfolio_value) / initial_portfolio_value

    var = calculate_var(final_values, alpha)
    cvar = calculate_cvar(final_values, alpha)
    expected_value = calculate_expected_value(final_values)
    sharpe_ratio = calculate_sharpe_ratio(final_returns, num_days)

    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each simulation path
    for i in range(portfolio_simulations.shape[1]):
        ax.plot(portfolio_simulations[:, i], alpha=0.2)

    # Add VaR and CVaR thresholds
    ax.axhline(var, color="red", linestyle="--", linewidth=2, label=f"VaR (Portfolio): ${var:.2f}")
    ax.axhline(cvar, color="orange", linestyle="--", linewidth=2, label=f"CVaR (Portfolio): ${cvar:.2f}")

    # Labels & grid
    ax.set_title("Monte Carlo Portfolio Simulations")
    ax.set_xlabel("Day")
    ax.set_ylabel("Portfolio Value")
    ax.legend()
    ax.grid(True)

    # Return statistics for dashboard
    statistics = {
        'var': var,
        'cvar': cvar,
        'expected_value': expected_value,
        'sharpe_ratio': sharpe_ratio
    }
    return fig, statistics


# -----------------------------
# Perform Monte Carlo simulation
# -----------------------------
def perform_mc_simulation(tickers: [str], weights: np.ndarray, num_simulations: int,
                          num_days: int, initial_portfolio_value: int, alpha: float):
    """Run Monte Carlo simulation for a portfolio given tickers, weights, and parameters."""
    if not tickers_exist(tickers):
        st.warning("One or more tickers not found.")
        return

    # Download returns and compute stats
    mean_returns, covariance = calculate_statistics(download_stock_data(tickers))

    # Generate correlated random returns
    random_samples = np.random.multivariate_normal(mean_returns, covariance,
                                                   size=(num_simulations, num_days))
    daily_portfolio_returns = np.einsum('sdt,t->sd', random_samples, weights)

    # Compute cumulative portfolio paths
    portfolio_paths = np.cumprod(daily_portfolio_returns + 1, axis=1).T * initial_portfolio_value

    return plot_graph(portfolio_paths, alpha, num_days, initial_portfolio_value)
