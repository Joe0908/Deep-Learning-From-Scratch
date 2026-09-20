import numpy as np

np.random.seed(42)

N = 1000
D = 100
layers = 20
X = np.random.randn(N, D)

def relu(x):
    return np.maximum(0, x)

def run_network(init_type):
    A = X.copy()
    print(f"\n--- {init_type} ---")
    print(f"input std: {A.std():.4f}")

    for layer in range(layers):
        if init_type == "tiny":
            W = np.random.randn(D, D) * 0.01
        elif init_type == "large":
            W = np.random.randn(D, D)
        elif init_type == "xavier":
            W = np.random.randn(D, D) * np.sqrt(1 / D)
        elif init_type == "he":
            W = np.random.randn(D, D) * np.sqrt(2 / D)
        else:
            raise ValueError(init_type)

        Z = A @ W
        A = relu(Z)

        print(
            f"layer {layer+1:2d}: "
            f"mean={A.mean():.4f}, std={A.std():.4f}"
        )

for name in ["tiny", "large", "xavier", "he"]:
    run_network(name)
