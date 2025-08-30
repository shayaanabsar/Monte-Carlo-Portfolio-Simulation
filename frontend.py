import streamlit as st
from main import *
import numpy as np

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Monte Carlo Portfolio Simulation",
    page_icon="📈",
)

st.title('Monte Carlo Portfolio Simulation')
st.write("Simulate portfolio performance and risk using Monte Carlo methods.")

# -----------------------------
# Stock tickers input
# -----------------------------
stock_input = st.text_input(
    label='Enter stock tickers separated by commas',
    help='Example: AAPL, TSLA, MSFT'
)

# Clean and parse tickers
stock_tickers = [ticker.strip().upper() for ticker in stock_input.split(',') if ticker.strip()]
num_stocks = len(stock_tickers)

# -----------------------------
# Simulation configuration sliders
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    num_simulations = st.slider(
        label='Number of simulations',
        min_value=10,
        max_value=10_000,
        value=1000,
        step=10,
        help='Higher numbers give smoother distributions but take longer to run'
    )

with col2:
    num_days = st.slider(
        label='Number of days to simulate',
        min_value=1,
        max_value=365,
        value=100,
        step=1,
        help='How many days into the future to simulate portfolio performance'
    )

# -----------------------------
# Weight input and normalization
# -----------------------------
if tickers_exist(stock_tickers) and num_stocks > 0:
    weights_input = st.text_input(
        "Enter weights separated by commas (e.g. 0.3,0.5,0.2)",
        help="Leave blank for equal weights"
    )

    # Default equal weights
    weights = np.array([1/num_stocks]*num_stocks)

    if weights_input:
        try:
            weights = np.array([float(w.strip()) for w in weights_input.split(",")])
            if len(weights) != num_stocks:
                st.warning('Please enter the same number of weights as stocks.')
            else:
                weights /= weights.sum()  # normalize
        except ValueError:
            st.warning('Please enter numeric values as weights.')

    col1, col2 = st.columns(2)
    with col1:
        st.write("Normalized weights:", dict(zip(stock_tickers, np.round(weights, 3))))
    with col2:
        st.write("Selected tickers:", stock_tickers)

    perform_mc_simulation(stock_tickers,
                          weights,
                          num_simulations,
                          num_days)