# (i) problem/model formulation for your chosen topic
# (ii) discussion of theory supporting use of the chosen optimization methods and their validation
# (iii) the optimization results you obtained in your study

# Data was collected from the top 20 weighted companies on the S&P 500 from nasdaq.com. Data is organized weekly and spans the previous 5 years starting from 05.19.25. Had to swap BRK.B (#9) for HD (#21) since Berkshire stock data was unavailable. 

import os
import glob
import pandas as pd
import numpy as np

def import_data() -> dict[str, list[float]]:
    universe = {}
    for path in glob.glob("data/*.csv"):
        df = pd.read_csv(path, index_col="Date")
        universe[os.path.basename(path).split(".csv")[0]] = df["Close"].astype(float).tolist()
    return universe

def f(x, Sigma) -> float:
    return x.T @ Sigma @ x 

def g1(x, mu, R) -> float:
    return mu.T @ x - R # R = 1.00 to 1.10 which represents a 0% to 10% ROI

def g2(x) -> float:
    return sum(x) - 1

def L(f, g1, g2):
    return f - lambda1*g1 - lambda2*g2

def model(universe) -> None:
    x = np.array([0.05] * 20) # Decision vector of portfolio weights
    mu = np.array([]) # Vector of expected return for each asset
    Sigma = np.array([]) # Covariance matrix for asset returns

if __name__ == '__main__':
    universe = import_data()
    print(universe)
