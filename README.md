# Native MNIST Neural Network Project

The aim of this project is to reinforce my first principles understanding of neural networks.

## Method

Construct a neural network without relying on any popular frameworks (e.g., PyTorch, TensorFlow). Subsequently, build the same network using PyTorch and compare results.

## NN Structure

* Type: Feedforward
* Input layer: 784 units (28px * 28px)
* Hidden layer: 10 nodes, ReLU activation
* Output layer: 10 nodes, Softmax activation

### Loss Function:
* No framework: ~Cross entropy loss (prediction distribution, minus one-hot encoded true values)
* PyTorch: Cross entropy loss

### Optimisation
* No framework: Gradient computation using the chain rule
* PyTorch: Stochastic gradient descent

### Parameters

* Learning rate: 0.1
* No framework epochs: 1,000
* PyTorch epochs: 100

## Results

No framework:
```
Test accuracy: 0.8894

python3 nn_no_framework.py  404.51s user 24.97s system 377% cpu 1:53.72 total
```

PyTorch:
```
Test accuracy: 0.8499

python3 nn_torch.py  3646.52s user 87.24s system 384% cpu 16:12.01 total
```


