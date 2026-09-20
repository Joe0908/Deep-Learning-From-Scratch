import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([[0],[1],[1],[0]], dtype=float)

def relu(z):
    return np.maximum(0, z)

np.random.seed(42)
W1 = np.random.randn(2, 4) * 0.5
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1) * 0.5
b2 = np.zeros((1, 1))

lr = 0.1
N = X.shape[0]

for epoch in range(5000):
    # Forward
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    y_hat = A1 @ W2 + b2
    loss = np.mean((y_hat - y) ** 2)

    # Backward
    dY = 2 * (y_hat - y) / N
    dW2 = A1.T @ dY
    db2 = np.sum(dY, axis=0, keepdims=True)

    dA1 = dY @ W2.T
    dZ1 = dA1 * (Z1 > 0)
    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    # Update
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

    if epoch % 500 == 0:
        print(f"epoch={epoch}, loss={loss:.6f}")

# Final predictions
Z1 = X @ W1 + b1
A1 = relu(Z1)
y_hat = A1 @ W2 + b2

print("\nFinal predictions:")
print(y_hat)
print("\nRounded:")
print(np.round(y_hat))
