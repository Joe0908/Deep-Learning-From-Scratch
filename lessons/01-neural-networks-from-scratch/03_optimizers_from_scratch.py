import numpy as np

def sgd_update(param, grad, lr):
    return param - lr * grad

def momentum_update(param, grad, velocity, lr, beta=0.9):
    velocity = beta * velocity + (1 - beta) * grad
    param = param - lr * velocity
    return param, velocity

def adam_update(param, grad, m, v, t, lr=1e-3,
                beta1=0.9, beta2=0.999, eps=1e-8):
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad ** 2)

    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    param = param - lr * m_hat / (np.sqrt(v_hat) + eps)
    return param, m, v

if __name__ == "__main__":
    p = np.array([3.0, -2.0])
    g = np.array([0.5, -0.25])

    print("SGD:", sgd_update(p.copy(), g, lr=0.1))

    velocity = np.zeros_like(p)
    p_m, velocity = momentum_update(p.copy(), g, velocity, lr=0.1)
    print("Momentum:", p_m)

    m = np.zeros_like(p)
    v = np.zeros_like(p)
    p_a, m, v = adam_update(p.copy(), g, m, v, t=1)
    print("Adam:", p_a)
