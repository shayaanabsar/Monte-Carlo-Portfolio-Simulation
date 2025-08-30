import streamlit as st
import numpy as np
from main import *

# --------------------------------
# Page Configuration
# --------------------------------
st.set_page_config(
    page_title="Monte Carlo Portfolio Simulation",
    page_icon="📈",
    layout="wide"
)

# --------------------------------
# Title & Description
# --------------------------------
st.title("📈 Monte Carlo Portfolio Simulation")
st.markdown(
    "Use Monte Carlo methods to simulate portfolio performance and assess risk. "
    "Configure your stocks, weights, and simulation parameters on the left."
)

# Main layout
col1, col2 = st.columns([1.1, 1.5])
figure = None

with col1:
    # --------------------------------
    # Stock Tickers
    # --------------------------------
    with st.expander("1️⃣ Stock Selection", expanded=True):
        stock_input = st.text_input(
            label="Enter stock tickers (comma-separated)",
            placeholder="Example: AAPL, TSLA, MSFT"
        )
        stock_tickers = [t.strip().upper() for t in stock_input.split(",") if t.strip()]
        num_stocks = len(stock_tickers)

    # --------------------------------
    # Simulation Settings
    # --------------------------------
    with st.expander("2️⃣ Simulation Settings", expanded=True):
        nested_col1, nested_col2 = st.columns(2)

        with nested_col1:
            num_simulations = st.slider(
                "Number of simulations",
                min_value=10,
                max_value=10_000,
                value=1000,
                step=10,
                help="Higher numbers give smoother distributions but take longer to run."
            )

        with nested_col2:
            num_days = st.slider(
                "Days to simulate",
                min_value=1,
                max_value=365,
                value=100,
                step=1,
                help="How many days into the future to simulate portfolio performance."
            )

    # --------------------------------
    # Weights
    # --------------------------------
    if tickers_exist(stock_tickers) and num_stocks > 0:
        with st.expander("3️⃣ Portfolio Weights", expanded=True):
            weights_input = st.text_input(
                "Weights (comma-separated, leave blank for equal weights)",
                placeholder="e.g. 0.3, 0.5, 0.2"
            )

            # Default equal weights
            weights = np.array([1 / num_stocks] * num_stocks)

            if weights_input:
                try:
                    weights = np.array([float(w.strip()) for w in weights_input.split(",")])
                    if len(weights) != num_stocks:
                        st.warning("⚠️ Please enter the same number of weights as stocks.")
                    else:
                        weights /= weights.sum()
                except ValueError:
                    st.warning("⚠️ Please enter numeric values for weights.")

            # Display weights & tickers neatly
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.write("Normalized weights:", dict(zip(stock_tickers, np.round(weights, 3))))
            with metric_col2:
                st.write("Selected tickers:", stock_tickers)

            # Run Monte Carlo simulation
            figure = perform_mc_simulation(
                stock_tickers,
                weights,
                num_simulations,
                num_days
            )

# --------------------------------
# Display Results
# --------------------------------
with col2:
    with st.spinner('Running Simulations...'):
        if figure:
            st.subheader("📊 Simulation Results")
            st.pyplot(figure)
        else:
            st.info("Enter tickers and parameters on the left to run a simulation.")
