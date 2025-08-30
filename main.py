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

def plot_graph(portfolio_simulations):
    plt.style.use('seaborn-v0_8')

    fig, ax = plt.subplots(figsize=(10,6))

    for i in range(portfolio_simulations.shape[1]):  # each simulation
        ax.plot(portfolio_simulations[:, i], alpha=0.2)

    ax.set_title("Monte Carlo Portfolio Simulations")
    ax.set_xlabel("Day")
    ax.set_ylabel("Portfolio Value")
    ax.grid(True)

    return fig


def perform_mc_simulation(tickers: [str], weights: np.ndarray, num_simulations: int, num_days: int):
    if not tickers_exist(tickers):
        st.warning("One or more tickers not found.")
        return

    # Download data and compute statistics
    mean_returns, covariance = calculate_statistics(download_stock_data(tickers))
    
    random_samples = np.random.multivariate_normal(mean_returns, covariance, size=(num_simulations, num_days))  # shape: (T, n_stocks)
    daily_portfolio_returns = daily_portfolio_returns = np.einsum('sdt,t->sd', random_samples, weights)
    portfolio_paths = np.cumprod(daily_portfolio_returns + 1, axis=1).T
    
    return plot_graph(portfolio_paths)
