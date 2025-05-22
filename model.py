# (i) problem/model formulation for your chosen topic
# (ii) discussion of theory supporting use of the chosen optimization methods and their validation
# (iii) the optimization results you obtained in your study

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import linprog

def import_data() -> [[float]]:
    universe = []
    for file in directory:
        with open(file, 'r') as data:
            universe.append(data)
    return universe

def f(x, Sigma) -> float:
    return x.T @ Sigma @ x 

def g1(x, mu, R) -> float:
    return mu.T @ x - R

def g2(x) -> float:
    return sum(x) - 1

def main(universe) -> None:
    x = np.array([]) # Decision vector of portfolio weights
    mu = np.([]) # Vector of expected return for each asset
    Sigma = np.array([]) # Covariance matrix for asset returns

if __name__ == '__main__':
    universe = import_data()
    main(universe)
