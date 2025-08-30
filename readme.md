# 📈 Monte Carlo Portfolio Simulation (link)[https://monte-carlo-portfolio-simulation-gh3lp6kb8kvqpctuztdmsc.streamlit.app/]

## 📝 Project Overview

Simulates the performance of a stock portfolio using **Monte Carlo methods** to estimate potential returns and risk metrics. Provides insights into portfolio risk and expected performance under uncertainty.

---

## ⚡ Features

### 1️⃣ Dynamic Stock Selection  
- Input any list of stock tickers to include in the portfolio.  

### 2️⃣ Flexible Portfolio Weights & Portfolio Value  
- Specify weights for each stock, with automatic normalization.  
- Set total portfolio value to scale simulations.  

### 3️⃣ Monte Carlo Simulation & Risk Metrics  
- Generates multiple portfolio paths using historical returns and covariance.  
- Calculates **Value at Risk (VaR)**, **Conditional VaR (CVaR)**, and **expected portfolio value**.  
- Computes **Sharpe Ratio** using real-world 3-month T-Bill yields.  

### 4️⃣ Interactive Dashboard  
- Visualizes simulated portfolio trajectories 📊.  
- Displays VaR, CVaR, expected value, and Sharpe ratio metrics for easy interpretation.

---

## 💻 Technical Stack

- **Python** – Core language for computation  
- **NumPy & Pandas** – Data manipulation and numerical computation  
- **yFinance** – Historical stock data and 3-month T-Bill rates  
- **Matplotlib** – Plotting portfolio trajectories and risk metrics  
- **Streamlit** – Interactive dashboard for user inputs and visualizations  
