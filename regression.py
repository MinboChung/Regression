import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import plotly.express as px

"""
    TODO: 
"""

class OrdinaryLeastSquare:
    """
        Ordinary Least square is a statistical method in linear regression anaylsis to estimate beta of a linear relationship between
        a dependent variable y and one or more independent variable x, i.e. y = f(x) + error_residual.

        The goal of OLS is to minimize the linear relationship by derivating hte linear expression by 0 with respect to the coefficients, beta.
    """
    def __init__(self):
        self.B = None
    
    def fit(self, X, y):
        X = np.column_stack((np.ones(len(X)), X))
        self.B = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X_test):
        if self.B is None:
            raise ValueError("OLS.predict: Model has not been trained.")
        X_test = np.column_stack((np.ones(len(X_test)), X_test))
        y_pred = X_test @ self.B
        return y_pred

class WeightedLeastSqaure:
    ...

class WeightedLinearRegression:
    ...

class PolynomialRegression:
    def __init__(self):
        ...

    def load_model_pipe(self):
        ...
    def load_optimal_model(self):
        ...