import yfinance as yf
import streamlit as st
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def tickers_exist(tickers: list[str]) -> bool:
    all_exist = True
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info  = stock.info
            if info is None or info == {} or info.get('regularMarketPrice') is None:
                st.warning(f'Ticker {ticker} could not be found')
                all_exist = False
        except Exception:
            st.warning(f'Ticker {ticker} could not be found')
            all_exist = False
    return all_exist

def download_stock_data(tickers: list[str], days: int = 365) -> pd.DataFrame:
    stock_prices = yf.download(
        tickers,
        start=dt.datetime.now() - dt.timedelta(days=days),
        end=dt.datetime.now()
    )
    
    close_prices = stock_prices['Close']
    returns = close_prices.pct_change().dropna()
    return returns

def calculate_statistics(stock_prices: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    mean_returns = stock_prices.mean().values
    covariance = stock_prices.cov().values
    
    return mean_returns, covariance

def calculate_var(final_returns, alpha):
    return np.percentile(final_returns, alpha * 100)

def calculate_cvar(final_returns, alpha):
    relevant_data = final_returns <= calculate_var(final_returns, alpha)

    return final_returns[relevant_data].mean()

def plot_graph(portfolio_simulations, alpha, num_days):
    final_returns = portfolio_simulations[-1, :]

    var  = calculate_var(final_returns, alpha)
    cvar = calculate_cvar(final_returns, alpha)

    plt.style.use('seaborn-v0_8')

    fig, ax = plt.subplots(figsize=(10,6))

    for i in range(portfolio_simulations.shape[1]):  # each simulation
        ax.plot(portfolio_simulations[:, i], alpha=0.2)

    ax.axhline(var, color="red", linestyle="--", linewidth=2, label=f"VaR (Portfolio): ${var:.2f}")
    ax.axhline(cvar, color="orange", linestyle="--", linewidth=2, label=f"CVaR (Portfolio): ${cvar:.2f}")



    ax.set_title("Monte Carlo Portfolio Simulations")
    ax.set_xlabel("Day")
    ax.set_ylabel("Portfolio Value")
    ax.legend()
    ax.grid(True)

    return fig, var, cvar

def perform_mc_simulation(tickers: [str], weights: np.ndarray, num_simulations: int, num_days: int, portfolio_value: int, alpha: int):
    if not tickers_exist(tickers):
        st.warning("One or more tickers not found.")
        return

    mean_returns, covariance = calculate_statistics(download_stock_data(tickers))
    
    random_samples = np.random.multivariate_normal(mean_returns, covariance, size=(num_simulations, num_days)) 
    daily_portfolio_returns = daily_portfolio_returns = np.einsum('sdt,t->sd', random_samples, weights)
    portfolio_paths = np.cumprod(daily_portfolio_returns + 1, axis=1).T * portfolio_value
    
    return plot_graph(portfolio_paths, alpha, num_days)
