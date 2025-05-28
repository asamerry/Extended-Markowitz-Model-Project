import numpy as np
import pandas as pd
import cvxpy as cp
from matplotlib import pyplot as plt
import os, glob

def import_data() -> tuple[dict[str, list[float]], int]:
    files = glob.glob("data/Full-Time/*.csv")
    prices = pd.DataFrame({ os.path.basename(file).split('.')[0]: pd.read_csv(file, index_col="Date", date_format="%m/%Y")["Close"] for file in files })
    return prices, len(files)

def model() -> None:
    prices, num_stocks = import_data()
    returns = prices.pct_change().dropna()
    mu_hat = np.reshape(returns.mean().values, (num_stocks, 1)) # Vector of expected return for each asset
    Sigma_hat = returns.cov().values # Covariance matrix for asset returns
    
    x = cp.Variable(num_stocks)
    f = cp.quad_form(x, Sigma_hat)

    print("-="*40 + "-")
    print(f"Min Return: {min(mu_hat)}; Max Return: {max(mu_hat)}")
    print("-="*40 + "-")

    objective_values = []
    
    r_range = np.linspace(min(mu_hat), max(mu_hat))
    for r in r_range:
        g = [
            mu_hat.T @ x == r, 
            sum(x) == 1, 
            x >= 0,
        ]
        prob = cp.Problem(cp.Minimize(f), g)
        prob.solve()
        if prob.status == "optimal":
            print(f"r-value: {r}; Status: {prob.status}")
            print(f"Objective: {prob.value}; x: {x.value}")
            x_val = np.array([x.value]).T
            objective_values.append(x_val.T @ (Sigma_hat @ x_val))
    
    if True:
        plt.figure(figsize=(12, 6))
        plt.plot(np.reshape(objective_values, (50, 1)), r_range)
        plt.grid(True)
        plt.ylabel("Expected Return")
        plt.xlabel("Expected Risk")
        plt.title("Efficient Frontier for Vanilla Model (without shorting)")
        plt.show()
    
if __name__ == '__main__':
    model()
