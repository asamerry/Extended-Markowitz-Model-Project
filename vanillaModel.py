# Data was collected from the top 20 weighted companies on the S&P 500 from nasdaq.com. Data is organized monthly and spans the previous 5 years starting from 05.26.25. Had to swap BRK.B (#9) for JNJ (#21) since Berkshire stock data was unavailable.

# For now we will obstain from adding a $x \geq 0$ constraint, meaning that we will allow shorting stocks. This can be adjusted later if time permits.

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os, glob

def import_data() -> tuple[dict[str, list[float]], int]:
    files = glob.glob("data/*.csv")
    prices = pd.DataFrame({ os.path.basename(file).split('.')[0]: pd.read_csv(file, index_col="Date", parse_dates=True, date_format="%m/%Y")["Close"] for file in files })
    return prices, len(files)

def model() -> None:
    prices, num_stocks = import_data()
    returns = prices.pct_change().dropna()
    mu_hat = np.reshape(returns.mean().values, (num_stocks, 1)) # Vector of expected return for each asset
    Sigma_hat = returns.cov().values # Covariance matrix for asset returns

    row1 = np.hstack( (2*Sigma_hat, mu_hat, np.ones((num_stocks, 1))) )
    row2 = np.hstack( (mu_hat.T, np.zeros((1, 2))) )
    row3 = np.hstack( (np.ones((num_stocks, 1)).T, np.zeros((1, 2))) )
    L_mat = np.vstack( (row1, row2, row3) ) # Lagrangian matrix - num_stock+2 x num_stocks+2 dimensional matrix; we will need to take the inverse

    objective_values = []
    r_range = [0.001*i for i in range(101)]

    for r in r_range: 
        x_lambda = np.linalg.inv(L_mat) @ np.vstack( (np.zeros((num_stocks, 1)), [[r]], [[1]]) )
        x = np.array([x_lambda[i].item() for i in range(num_stocks)])
        var = x.T @ ( Sigma_hat @ x )
        objective_values.append(var)

    if True:
        plt.figure(figsize=(12, 6))
        plt.plot(objective_values, r_range)
        plt.grid(True)
        plt.ylabel("Expected Return")
        plt.xlabel("Expected Risk")
        plt.title("Efficient Frontier for Vanilla Model (with shorting)")
        plt.show()

if __name__ == '__main__':
    model()
