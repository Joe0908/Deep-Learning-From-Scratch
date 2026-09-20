import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
N = 500
X = np.random.randn(N, 2)
r = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
y = (r > 1.0).astype(int)

idx = np.random.permutation(N)
train_idx, test_idx = idx[:400], idx[400:]
X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)

def relu(z):
    return np.maximum(0, z)

def accuracy(probs, y_true):
    return np.mean(np.argmax(probs, axis=1) == y_true)

# Linear softmax classifier
np.random.seed(42)
W_lin = np.random.randn(2, 2) * 0.01
b_lin = np.zeros((1, 2))

for _ in range(5000):
    probs = softmax(X_train @ W_lin + b_lin)
    n = len(X_train)

    dZ = probs.copy()
    dZ[np.arange(n), y_train] -= 1
    dZ /= n

    dW = X_train.T @ dZ + 0.01 * W_lin
    db = np.sum(dZ, axis=0, keepdims=True)

    W_lin -= 0.1 * dW
    b_lin -= 0.1 * db

# MLP 2 -> 16 -> 16 -> 2
np.random.seed(42)
W1 = np.random.randn(2, 16) * np.sqrt(2 / 2)
b1 = np.zeros((1, 16))
W2 = np.random.randn(16, 16) * np.sqrt(2 / 16)
b2 = np.zeros((1, 16))
W3 = np.random.randn(16, 2) * np.sqrt(2 / 16)
b3 = np.zeros((1, 2))

def mlp_forward(X_):
    Z1 = X_ @ W1 + b1
    A1 = relu(Z1)

    Z2 = A1 @ W2 + b2
    A2 = relu(Z2)

    probs = softmax(A2 @ W3 + b3)
    return probs, (X_, Z1, A1, Z2, A2)

for _ in range(5000):
    probs, (Xc, Z1, A1, Z2, A2) = mlp_forward(X_train)
    n = len(X_train)

    dZ3 = probs.copy()
    dZ3[np.arange(n), y_train] -= 1
    dZ3 /= n

    dW3 = A2.T @ dZ3 + 0.01 * W3
    db3 = np.sum(dZ3, axis=0, keepdims=True)

    dZ2 = (dZ3 @ W3.T) * (Z2 > 0)
    dW2 = A1.T @ dZ2 + 0.01 * W2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dZ1 = (dZ2 @ W2.T) * (Z1 > 0)
    dW1 = Xc.T @ dZ1 + 0.01 * W1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    W1 -= 0.01 * dW1
    b1 -= 0.01 * db1
    W2 -= 0.01 * dW2
    b2 -= 0.01 * db2
    W3 -= 0.01 * dW3
    b3 -= 0.01 * db3

lin_train = accuracy(softmax(X_train @ W_lin + b_lin), y_train)
lin_test = accuracy(softmax(X_test @ W_lin + b_lin), y_test)
mlp_train = accuracy(mlp_forward(X_train)[0], y_train)
mlp_test = accuracy(mlp_forward(X_test)[0], y_test)

print(f"Linear train accuracy: {lin_train:.3f}")
print(f"Linear test accuracy : {lin_test:.3f}")
print(f"MLP train accuracy   : {mlp_train:.3f}")
print(f"MLP test accuracy    : {mlp_test:.3f}")

# Decision-boundary plots
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)
grid = np.c_[xx.ravel(), yy.ravel()]

linear_grid = np.argmax(
    softmax(grid @ W_lin + b_lin),
    axis=1
).reshape(xx.shape)

plt.figure(figsize=(6, 5))
plt.contourf(xx, yy, linear_grid, alpha=0.25)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=30)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title(f"Linear classifier — test accuracy {lin_test:.3f}")
plt.show()

mlp_grid = np.argmax(
    mlp_forward(grid)[0],
    axis=1
).reshape(xx.shape)

plt.figure(figsize=(6, 5))
plt.contourf(xx, yy, mlp_grid, alpha=0.25)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=30)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title(f"MLP — test accuracy {mlp_test:.3f}")
plt.show()
