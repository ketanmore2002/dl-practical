#Loading dataset into keras/pytorch, creating training and testing splits.
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape the input data to a 4D tensor
x_train = np.reshape(x_train, (x_train.shape[0], 28, 28, 1))
x_test = np.reshape(x_test, (x_test.shape[0], 28, 28, 1))

# Convert the data type to float32
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalize the data to a range between 0 and 1
x_train /= 255
x_test /= 255

# Convert the class labels to binary class matrices
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Print the shapes of the training and testing sets
print('Training set shape:', x_train.shape)
print('Testing set shape:', x_test.shape)
print('Training labels shape:', y_train.shape)
print('Testing labels shape:', y_test.shape)

