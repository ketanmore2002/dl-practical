#Implementing regression using deep neural network.
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

# Load the dataset
df = pd.read_csv('insurance.csv')

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df[['age', 'sex', 'bmi', 'children', 'smoker', 'region']], df['charges'], test_size=0.25, random_state=42)
X_train = X_train.astype('float32')
y_train = y_train.astype('float32')
# Create the model
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(6,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Evaluate the model
y_pred = model.predict(X_test)
print('Mean absolute error:', np.mean(np.abs(y_pred - y_test)))

# Plot the results
plt.scatter(y_test, y_pred)
plt.xlabel('True Value')
plt.ylabel('Predicted Value')
plt.show()