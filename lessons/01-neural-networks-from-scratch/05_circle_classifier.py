import numpy as np

# Nonlinear circle dataset
np.random.seed(0)
N = 500
X = np.random.randn(N, 2)
radius = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
y = (radius > 1.0).astype(int)

# Hyperparameters
input_dim = 2
hidden_dim = 16
output_dim = 2
reg = 0.01
lr = 0.01

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# He initialization for ReLU layers
np.random.seed(42)
W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
b1 = np.zeros((1, hidden_dim))
W2 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
b2 = np.zeros((1, hidden_dim))
W3 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
b3 = np.zeros((1, output_dim))

def forward(X, parameters):
    W1, b1, W2, b2, W3, b3 = parameters

    Z1 = X @ W1 + b1
    A1 = relu(Z1)

    Z2 = A1 @ W2 + b2
    A2 = relu(Z2)

    Z3 = A2 @ W3 + b3
    probs = softmax(Z3)

    cache = (X, Z1, A1, Z2, A2, Z3, probs)
    return probs, cache

def cross_entropy_loss(y_true, probs, W1, W2, W3, reg):
    n = y_true.shape[0]
    correct_probs = probs[np.arange(n), y_true]

    data_loss = -np.mean(np.log(correct_probs + 1e-12))
    reg_loss = 0.5 * reg * (
        np.sum(W1**2) +
        np.sum(W2**2) +
        np.sum(W3**2)
    )
    return data_loss + reg_loss

def backward(y_true, cache, parameters, reg):
    X, Z1, A1, Z2, A2, Z3, probs = cache
    W1, b1, W2, b2, W3, b3 = parameters
    n = X.shape[0]

    # Softmax + cross-entropy gradient
    dZ3 = probs.copy()
    dZ3[np.arange(n), y_true] -= 1
    dZ3 /= n

    dW3 = A2.T @ dZ3 + reg * W3
    db3 = np.sum(dZ3, axis=0, keepdims=True)

    dA2 = dZ3 @ W3.T
    dZ2 = dA2 * (Z2 > 0)

    dW2 = A1.T @ dZ2 + reg * W2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (Z1 > 0)

    dW1 = X.T @ dZ1 + reg * W1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    return dW1, db1, dW2, db2, dW3, db3

def update(parameters, gradients, lr):
    W1, b1, W2, b2, W3, b3 = parameters
    dW1, db1, dW2, db2, dW3, db3 = gradients

    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2
    W3 -= lr * dW3
    b3 -= lr * db3

    return W1, b1, W2, b2, W3, b3

parameters = (W1, b1, W2, b2, W3, b3)

for epoch in range(5000):
    probs, cache = forward(X, parameters)
    W1, b1, W2, b2, W3, b3 = parameters

    loss = cross_entropy_loss(y, probs, W1, W2, W3, reg)
    gradients = backward(y, cache, parameters, reg)
    parameters = update(parameters, gradients, lr)

    if epoch % 500 == 0:
        predictions = np.argmax(probs, axis=1)
        accuracy = np.mean(predictions == y)
        print(f"epoch={epoch}, loss={loss:.4f}, accuracy={accuracy:.3f}")
