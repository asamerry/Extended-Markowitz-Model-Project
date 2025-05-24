# Data was collected from the top 20 weighted companies on the S&P 500 from nasdaq.com. Data is organized weekly and spans the previous 5 years starting from 05.19.25. Had to swap BRK.B (#9) for HD (#21) since Berkshire stock data was unavailable.

# For now we will obstain from adding a $x \geq 0$ constraint, meaning that we will allow shorting stocks. This can be adjusted later if time permits.

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os, glob

def import_data() -> dict[str, list[float]]:
    files = glob.glob("data/*.csv")
    prices = pd.DataFrame({ os.path.basename(file).split('.')[0]: pd.read_csv(file, index_col="Date", parse_dates=True)["Close"] for file in files })
    return prices, len(files)

def model() -> None:
    prices, num_stocks = import_data()
    returns = prices.pct_change().dropna()
    mu_hat = np.reshape(returns.mean(), (num_stocks, 1)) # Vector of expected return for each asset
    Sigma_hat = returns.cov() # Covariance matrix for asset returns

    row1 = np.hstack( (Sigma_hat, mu_hat, np.ones((num_stocks, 1))) )
    row2 = np.hstack( (mu_hat.T, np.zeros((1, 2))) )
    row3 = np.hstack( (np.ones((num_stocks, 1)).T, np.zeros((1, 2))) )
    L_mat = np.vstack( (row1, row2, row3) )
    
    objective_values = []
    r_range = [0.001*i for i in range(101)]
    
    for r in r_range: 
        x_lambda = np.linalg.inv(L_mat) @ np.vstack( (np.zeros((num_stocks, 1)), r, 1) )
        x = np.array([x_lambda[i].item() for i in range(num_stocks)])
        var = x.T @ ( Sigma_hat @ x )
        objective_values.append(var)

    if True:
        plt.figure(figsize=(12, 6))
        plt.plot(r_range, objective_values)
        plt.grid(True)
        plt.xlabel("Expected Return")
        plt.ylabel("Expected Risk")
        plt.title("Return Value vs Risk Rate")
        plt.show()

if __name__ == '__main__':
    model()
