# hyper_regression.py

import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_module import load_data

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2

def run_grid_search(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_

if __name__ == "__main__":
    df = load_data()
    X = df.drop("MEDV", axis=1)
    y = df["MEDV"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("🔍 Performing hyperparameter tuning...\n")

    # Linear Regression (no real hyperparameters, included for consistency)
    lr_model = LinearRegression()
    mse, r2 = evaluate_model(lr_model, X_train, X_test, y_train, y_test)
    print(f"Linear Regression -> MSE: {mse:.2f}, R²: {r2:.2f}")

    # Decision Tree
    dt_params = {
        "max_depth": [3, 5, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }
    best_dt, dt_best_params = run_grid_search(DecisionTreeRegressor(random_state=42), dt_params, X_train, y_train)
    mse, r2 = evaluate_model(best_dt, X_train, X_test, y_train, y_test)
    print(f"Decision Tree (tuned) -> MSE: {mse:.2f}, R²: {r2:.2f}")
    print(f"Best Params: {dt_best_params}")

    # Random Forest
    rf_params = {
        "n_estimators": [50, 100],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5],
    }
    best_rf, rf_best_params = run_grid_search(RandomForestRegressor(random_state=42), rf_params, X_train, y_train)
    mse, r2 = evaluate_model(best_rf, X_train, X_test, y_train, y_test)
    print(f"Random Forest (tuned) -> MSE: {mse:.2f}, R²: {r2:.2f}")
    print(f"Best Params: {rf_best_params}")
