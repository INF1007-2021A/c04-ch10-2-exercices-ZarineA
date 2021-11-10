#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def read_file() -> pd.DataFrame:
    return pd.read_csv("data/winequality-white.csv", sep=";")


def extract_x_y(data: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    return data.drop("quality", axis=1), data["quality"]


def train_random_forest(X_train: pd.DataFrame, y_train:pd.DataFrame) -> RandomForestRegressor:
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    return model


def train_linear_regression(X_train: pd.DataFrame, y_train:pd.DataFrame) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test: pd.DataFrame) -> np.ndarray:
    return model.predict(X_test)


def plot_graph(target_values: np.array, predicted_values: np.array, title: str) -> None:
    plt.plot(range(len(target_values)), target_values, label="Target values")
    plt.plot(range(len(predicted_values)), predicted_values, "tab:orange", label="Predicted values")
    plt.xlabel("Number of samples")
    plt.ylabel("Quality")
    plt.title(title)
    plt.legend()
    plt.show()


if __name__ == '__main__':

    # Prepare the data
    data = read_file()
    X, y = extract_x_y(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

    # Create the models
    random_forest = train_random_forest(X_train, y_train)
    linear_regression = train_linear_regression(X_train, y_train)

    # Make predictions
    prediction_random_forest = evaluate(random_forest, X_test)
    prediction_linear_regression = evaluate(linear_regression, X_test)

    # Plot the predictions
    plot_graph(np.array(y_test), prediction_linear_regression, "LinearRegression predictions analysis")
    plot_graph(np.array(y_test), prediction_random_forest, "RandomForestRegressor predictions analysis")

    # Calculate mean squared error
    print("MSE de la régression linéaire :", mean_squared_error(y_test, prediction_linear_regression))
    print("MSE de l'arbre de décision :", mean_squared_error(y_test, prediction_random_forest))
