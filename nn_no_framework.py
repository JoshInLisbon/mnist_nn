import mnist
import numpy as np

mnist.datasets_url = 'https://storage.googleapis.com/cvdf-datasets/mnist/'

def reshape(images):
  # returns n rows
  # 28 * 28 = 784 columns
  return images.reshape((images.shape[0], images.shape[1] * images.shape[2]))

X_train = reshape(mnist.train_images()).T # (784 rows, 60000 columns)
X_train = X_train / 255. # normalise
y_train = mnist.train_labels() # (60000,)

X_test = reshape(mnist.test_images()).T # (784 rows, 10000 columns)
X_test = X_test / 255. # normalise
y_test = mnist.test_labels() # (10000,)

def init_params():
  W1 = np.random.rand(10, 784) - 0.5 # (10, 784)
  b1 = np.random.rand(10, 1) - 0.5 # (10, 1)

  W2 = np.random.rand(10,10) - 0.5 # (10, 10)
  b2 = np.random.rand(10, 1) - 0.5 # (10, 1)

  return W1, b1, W2, b2

def ReLU(Z):
  # applies to each element in Z
  return np.maximum(0, Z)

def ReLu_derivative(Z):
  # returns 1 if Z > 0, else 0
  return Z > 0

def softmax(Z, epsilon=1e-15):
    # shifting handles overflow runtime warnings
    # epsilon handles dividing by ~0
    Z_shifted = Z - np.max(Z, axis=0, keepdims=True)
    exp_Z = np.exp(Z_shifted)
    return exp_Z / (np.sum(exp_Z, axis=0, keepdims=True) + epsilon)

def forward_prop(W1, b1, W2, b2, X):
  Z1 = W1.dot(X) + b1 # (10, 784) . (784, 60000) + (10, 1) => (10, 60000)
  A1 = ReLU(Z1) # (10, 60000)
  Z2 = W2.dot(A1) + b2 # (10, 10) . (10, 60000) + (10, 1) => (10, 60000)
  A2 = softmax(Z2) # (10, 60000)

  return Z1, A1, Z2, A2

def one_hot(Y):
  one_hot = np.zeros((Y.size, 10)) # (60000, 10)
  one_hot[np.arange(Y.size), Y] = 1
  one_hot = one_hot.T

  return one_hot # (10, 60000)

def back_prop(Z1, A1, Z2, A2, W2, X, Y):
  m = Y.size
  one_hot_Y = one_hot(Y)
  dZ2 = A2 - one_hot_Y # (10, 60000)
  dW2 = 1/m * dZ2.dot(A1.T) # (10, 60000) . (60000, 10) => (10, 10)
  db2 = 1/m * np.sum(dZ2, 1).reshape(-1, 1) # (10, 1)
  dZ1 = W2.T.dot(dZ2) * ReLu_derivative(Z1) # (10, 10) . (10, 60000) => (10, 60000)
  dW1 = 1/m * dZ1.dot(X.T) # (10, 60000) . (60000, 784) => (10, 784)
  db1 = 1/m * np.sum(dZ1, 1).reshape(-1, 1) # (10, 1)

  return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
  W1 = W1 - learning_rate * dW1
  b1 = b1 - learning_rate * db1
  W2 = W2 - learning_rate * dW2
  b2 = b2 - learning_rate * db2

  return W1, b1, W2, b2

def get_predictions(A2):
  return np.argmax(A2, 0)

def print_vals(val_type, arr):
  frequencies = np.bincount(arr)
  print(f"{val_type}:")
  for value, count in enumerate(frequencies):
      print(f"Value {value}: {count} times")
  print("")

def get_accuracy(predictions, Y):
  print_vals("Predictions", predictions)
  print_vals("Truth", Y)
  return np.sum(predictions == Y) / Y.size

def gradient_decent(X_train, y_train, epochs, learning_rate):
  W1, b1, W2, b2 = init_params()

  _, _, _, A2 = forward_prop(W1, b1, W2, b2, X_train)
  print(f"Initial accuracy: {get_accuracy(get_predictions(A2), y_train)}\n")
  print("Training...\n")

  for i in range(1, epochs + 1):
    Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X_train)
    dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X_train, y_train)
    W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)

    if i % 100 == 0:
      print(f"Epoch: {i}\n")
      print(f"Accuracy: {get_accuracy(get_predictions(A2), y_train)}\n")

  return W1, b1, W2, b2

epochs = 1000
learning_rate = 0.1

W1, b1, W2, b2 = gradient_decent(X_train, y_train, epochs, learning_rate)

# Evaluate on test data
_, _, _, A2 = forward_prop(W1, b1, W2, b2, X_test)
print("Running on test data.\n")
print(f"Test accuracy: {get_accuracy(get_predictions(A2), y_test)}\n")

