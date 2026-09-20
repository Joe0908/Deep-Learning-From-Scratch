import numpy as np

def relu(z):
    return np.maximum(0, z)

x = np.array([2.0, -1.0])
W1 = np.array([[1.0, 2.0], [-1.0, 3.0], [0.5, -2.0]])
b1 = np.array([1.0, 0.0, -1.0])
W2 = np.array([2.0, -1.0, 0.5])
b2 = 1.0

target = 2.0
lr = 0.01

for epoch in range(20):
    # Forward
    z = W1 @ x + b1
    a = relu(z)
    y_hat = W2 @ a + b2
    loss = (y_hat - target) ** 2

    # Backward
    dy_hat = 2 * (y_hat - target)
    dW2 = dy_hat * a
    db2 = dy_hat

    da = dy_hat * W2
    dz = da * (z > 0)

    dW1 = np.outer(dz, x)
    db1 = dz

    # Gradient descent update
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

    print(f"epoch {epoch:2d}  loss={loss:.6f}  y_hat={y_hat:.6f}")
