#Creating functions to compute various losses.
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

# Define various loss functions for regression
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# Load iris dataset
iris = load_iris()
data = pd.DataFrame(data= np.c_[iris['data'], iris['target']], columns= iris['feature_names'] + ['target'])
data['target'] = data['target'].astype(int)

# Split dataset into inputs and targets
X = data.drop('target', axis=1).values
y = data['target'].values

# Compute loss for regression using mean squared error
y_pred = np.random.rand(len(y))  # Replace with your own model predictions
mse_loss = mean_squared_error(y, y_pred)
print(f'Mean Squared Error: {mse_loss:.4f}')

# Compute loss for regression using mean absolute error
y_pred = np.random.rand(len(y))  # Replace with your own model predictions
mae_loss = mean_absolute_error(y, y_pred)
print(f'Mean Absolute Error: {mae_loss:.4f}')

# Compute loss for regression using root mean squared error
y_pred = np.random.rand(len(y))  # Replace with your own model predictions
rmse_loss = root_mean_squared_error(y, y_pred)
print(f'Root Mean Squared Error: {rmse_loss:.4f}')



