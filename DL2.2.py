import numpy as np
import pandas as pd

# Define various loss functions for multiclass classification
def categorical_crossentropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return np.mean(-np.sum(y_true * np.log(y_pred), axis=1))

def binary_crossentropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return np.mean(-np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred), axis=1))

# Load Iris dataset
data = pd.read_csv('iris.csv')

# Replace species names with binary labels
data['Species'] = data['Species'].replace(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], [0, 1, 2])

# Split dataset into inputs and targets
X = data.drop('Species', axis=1).values
y = data['Species'].values

# Compute loss for binary classification using binary crossentropy
y_pred = np.random.rand(len(y), 1)  # Replace with your own model predictions
binary_crossentropy_loss = binary_crossentropy(y.reshape(-1, 1), y_pred)
print(f'Binary Crossentropy Loss: {binary_crossentropy_loss:.4f}')

# Compute loss for multiclass classification using categorical crossentropy
y = pd.get_dummies(data['Species']).values
y_pred = np.random.rand(len(y), 3)  # Replace with your own model predictions
categorical_crossentropy_loss = categorical_crossentropy(y, y_pred)
categorical_accuracy_score = categorical_accuracy(y, y_pred)
print(f'Categorical Crossentropy Loss: {categorical_crossentropy_loss:.4f}')

