#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

boston_data = pd.read_csv("BostonHousing.csv")
print(boston_data.head())

X = boston_data.drop(columns=["MEDV"]) 
y = boston_data["MEDV"] 

correlation_coeffs = np.abs(boston_data.corr()["MEDV"]).drop("MEDV")
best_feature_name = correlation_coeffs.idxmax()

print(f"The attribute that best follows the linear relationship with the output price is: {best_feature_name}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_with_bias = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]

analytic_solution = np.linalg.inv(X_with_bias.T.dot(X_with_bias)).dot(X_with_bias.T).dot(y)

print("Analytic solution coefficients:", analytic_solution)

class LinearRegressionGradientDescent():
    def __init__(self, lr=0.0001, max_iter=10000, tolerance=1e-6):
        self.lr = lr
        self.max_iter = max_iter
        self.tolerance = tolerance

    def fit(self, X, y):
        X_with_bias = np.c_[np.ones((X.shape[0], 1)), X]
        self.weights = np.zeros(X_with_bias.shape[1])

        for _ in range(self.max_iter):
            gradients = -2 * X_with_bias.T.dot(y - X_with_bias.dot(self.weights))
            if np.all(np.abs(gradients) < self.tolerance):
                break
            self.weights -= self.lr * gradients

    def predict(self, X):
        X_with_bias = np.c_[np.ones((X.shape[0], 1)), X]
        return X_with_bias.dot(self.weights)

lr_full_batch_scaled = LinearRegressionGradientDescent()
lr_full_batch_scaled.fit(X_scaled, y)
print("Gradient descent (Full-batch) coefficients (scaled features):", lr_full_batch_scaled.weights)

class LinearRegressionStochasticGradientDescent():
    def __init__(self, lr=0.0001, max_iter=1000, tolerance=1e-6, batch_size=1):
        self.lr = lr
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.batch_size = batch_size

    def fit(self, X, y):
        X_with_bias = np.c_[np.ones((X.shape[0], 1)), X]
        self.weights = np.zeros(X_with_bias.shape[1])

        for _ in range(self.max_iter):
            indices = np.random.permutation(X_with_bias.shape[0])
            for start_idx in range(0, X_with_bias.shape[0], self.batch_size):
                batch_indices = indices[start_idx:start_idx + self.batch_size]
                X_batch = X_with_bias[batch_indices]
                y_batch = y[batch_indices]
                gradients = -2 * X_batch.T.dot(y_batch - X_batch.dot(self.weights))
                if np.all(np.abs(gradients) < self.tolerance):
                    break
                self.weights -= self.lr * gradients

    def predict(self, X):
        X_with_bias = np.c_[np.ones((X.shape[0], 1)), X]
        return X_with_bias.dot(self.weights)

lr_stochastic_scaled = LinearRegressionStochasticGradientDescent(lr=0.0001, max_iter=10000, batch_size=1)
lr_stochastic_scaled.fit(X_scaled, y)
print("Gradient descent (Stochastic) coefficients (scaled features):", lr_stochastic_scaled.weights)

