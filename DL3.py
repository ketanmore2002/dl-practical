#Feeding data to pretrained neural network and making predictions
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from sklearn.datasets import load_iris
from keras.utils import to_categorical
import numpy as np

# Load the iris dataset
data = load_iris()
x = data['data']
y = data['target']

# Preprocess the input data
x = np.array([preprocess_input(np.resize(img, (224, 224, 3))) for img in x])

# Convert the labels to one-hot encoded format
y = to_categorical(y)

# Load the pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# Make predictions on the test set
predictions = model.predict(x)

# Convert the predicted probabilities to class labels
predicted_labels = np.argmax(predictions, axis=1)

# Calculate the accuracy of the model on the test set
accuracy = np.mean(predicted_labels == y.argmax(axis=1))
print("Accuracy: {:.2f}%".format(accuracy * 100))
