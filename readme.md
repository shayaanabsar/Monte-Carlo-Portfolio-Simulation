# Monte Carlo Portfolio Simulation

## Project Overview

This project simulates the performance of a portfolio of stocks using **Monte Carlo methods**. The goal is to estimate potential portfolio returns, risk metrics such as **Value at Risk (VaR)** and **Conditional Value at Risk (CVaR)**, and visualize multiple possible future scenarios.

---

## Features

1. **Dynamic Stock Selection**  
   - Users can input any list of stock tickers to include in the portfolio.

2. **Portfolio Weights**  
   - Specify weights for each stock.  
   - Automatically normalizes weights to sum to 1.

3. **Monte Carlo Simulation**  
   - Uses historical stock returns to calculate mean returns and covariance matrix.  
   - Simulates multiple future paths of portfolio performance by sampling from a multivariate normal distribution.

4. **Risk Metrics**  
   - **Value at Risk (VaR)**: The worst expected loss at a given confidence level (e.g., 5%).  
   - **Conditional VaR (CVaR)**: The average of losses exceeding the VaR threshold.

5. **Visualization**  
   - Plots simulated portfolio paths.  
   - Displays VaR and CVaR as reference lines.  

---
