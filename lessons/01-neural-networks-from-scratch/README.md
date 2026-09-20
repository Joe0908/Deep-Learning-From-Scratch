# Lesson 1 — Neural Networks from Scratch with NumPy

This lesson builds a neural network from first principles. The goal is to remove the black-box feeling by making every forward computation, loss, derivative, gradient and parameter update explicit.

## 1. Why can a neural network learn?

A neural network learns because:

> parameters define a function → the loss measures how poor that function is → gradients describe how parameters affect the loss → optimization repeatedly changes the parameters to reduce the loss.

If the parameters are collected in `θ`,

```text
θ(t+1) = θ(t) - η ∇θ L
```

where `η` is the learning rate.

The full learning loop is:

```text
parameters
   ↓
forward pass
   ↓
prediction
   ↓
loss
   ↓
backpropagation
   ↓
gradients
   ↓
parameter update
   ↓
repeat
```

---

## 2. A neuron is a weighted combination

For an input vector `x`, a neuron computes

```text
z = wᵀx + b
```

or

```text
z = w1*x1 + w2*x2 + ... + wn*xn + b
```

The weights determine how strongly the neuron responds to each input feature.

A layer evaluates many neurons at once:

```text
z = Wx + b
```

So matrix multiplication is simply an efficient way to compute many dot products simultaneously.

---

## 3. Shapes matter

Suppose one sample has 3 input features and the next layer has 4 neurons.

```text
x:  (3,)
W:  (4, 3)
b:  (4,)
z:  (4,)
```

For a batch of 100 samples stored row-wise:

```text
X:  (100, 3)
W:  (3, 4)
b:  (1, 4)
Z:  (100, 4)
```

A large part of neural-network debugging is simply getting tensor shapes right.

---

## 4. Why nonlinearity is necessary

If two layers are both linear,

```text
h = W1 x + b1
y = W2 h + b2
```

then

```text
y = W2(W1 x + b1) + b2
  = (W2W1)x + (W2b1 + b2)
```

So many stacked linear layers still collapse into one linear transformation.

That is why we need nonlinear activation functions.

For ReLU:

```text
ReLU(z) = max(0, z)
```

Now a network such as

```text
x → Linear → ReLU → Linear → ReLU → Linear
```

can represent nonlinear functions.

---

## 5. Forward propagation

A simple two-layer network is:

```text
z = W1 x + b1
a = ReLU(z)
ŷ = W2 a + b2
```

The flow from input to prediction is the **forward pass**.

For a regression example we used mean-squared error:

```text
L = (ŷ - y)^2
```

The loss answers:

> How bad is the current prediction?

---

## 6. Gradients and backpropagation

For

```text
L = (ŷ - y)^2
```

the first derivative is

```text
∂L/∂ŷ = 2(ŷ - y)
```

If

```text
ŷ = wᵀa + b
```

then

```text
∂ŷ/∂wi = ai
```

and by the chain rule,

```text
∂L/∂wi = (∂L/∂ŷ)(∂ŷ/∂wi)
```

Backpropagation is simply an efficient way to apply this chain rule through the entire computational graph.

For

```text
W1 → z → a → ŷ → L
```

the gradient follows the reverse path.

For ReLU:

```text
ReLU'(z) = 1 if z > 0
           0 if z < 0
```

This explains why an inactive ReLU receives zero gradient for that sample.

See: `01_single_sample_backprop.py`.

---

## 7. Vectorized backpropagation

For a batch:

```text
Z1 = XW1 + b1
A1 = ReLU(Z1)
Ŷ  = A1W2 + b2
```

With MSE,

```text
dŶ  = 2(Ŷ - Y) / N
dW2 = A1ᵀ dŶ
db2 = sum(dŶ over batch)
dA1 = dŶ W2ᵀ
dZ1 = dA1 * (Z1 > 0)
dW1 = Xᵀ dZ1
db1 = sum(dZ1 over batch)
```

The matrix products automatically accumulate gradient contributions from all samples.

---

## 8. XOR: the first nonlinear MLP

XOR is:

| Input | Target |
|---|---:|
| (0, 0) | 0 |
| (0, 1) | 1 |
| (1, 0) | 1 |
| (1, 1) | 0 |

A single linear classifier cannot solve XOR.

A hidden ReLU layer can.

See: `02_xor_mlp.py`.

That script explicitly implements:

```text
forward
→ loss
→ backward
→ update
```

without PyTorch or autograd.

---

## 9. Gradient descent, SGD and mini-batches

### Full-batch gradient descent

Each update uses the entire training set.

```text
g = average gradient over all N samples
```

This is stable but expensive for large datasets.

### Stochastic gradient descent

Each update uses one sample.

```text
θ ← θ - η ∇L_i
```

This is noisy but cheap.

### Mini-batch SGD

In practice, we usually use a batch of samples:

```text
batch size = 32, 64, 128, ...
```

This combines efficient matrix operations, manageable memory usage and useful gradient noise.

A simple NumPy batch iterator:

```python
def get_batches(X, y, batch_size):
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    X = X[indices]
    y = y[indices]

    for start in range(0, len(X), batch_size):
        end = start + batch_size
        yield X[start:end], y[start:end]
```

---

## 10. Momentum and Adam

Backpropagation answers:

> What is the gradient?

The optimizer answers:

> What should I do with that gradient?

### SGD

```text
θ(t+1) = θ(t) - η g(t)
```

### Momentum

Momentum remembers previous gradient directions:

```text
v(t) = β v(t-1) + (1-β) g(t)
θ(t+1) = θ(t) - η v(t)
```

It reduces oscillation and accelerates persistent directions.

### Adam

Adam keeps moving averages of both gradients and squared gradients:

```text
m(t) = β1 m(t-1) + (1-β1) g(t)
v(t) = β2 v(t-1) + (1-β2) g(t)^2
```

After bias correction:

```text
θ ← θ - η * m_hat / (sqrt(v_hat) + ε)
```

Adam therefore gives parameters adaptive effective step sizes.

See: `03_optimizers_from_scratch.py`.

---

## 11. Initialization

Weights should not all start at zero because hidden neurons would stay symmetric: same outputs, same gradients, same learned features.

Random initialization breaks symmetry, but the scale matters.

For

```text
z = Σ wi xi
```

a rough variance argument gives:

```text
Var(z) ≈ n Var(w) Var(x)
```

To preserve signal scale, we want approximately:

```text
Var(w) ≈ 1/n
```

This motivates Xavier-style initialization.

For ReLU, He initialization compensates for the fact that many negative values are clipped:

```text
Var(W) = 2 / n_in
```

In NumPy:

```python
W = np.random.randn(n_in, n_out) * np.sqrt(2 / n_in)
```

See: `04_initialization_experiment.py`.

---

## 12. Vanishing and exploding gradients

Backpropagation repeatedly multiplies local derivatives.

If typical factors are smaller than 1:

```text
0.5^20 ≈ 9.5e-7
```

the gradient can vanish.

If typical factors are larger than 1:

```text
1.5^20 ≈ 3325
```

the gradient can explode.

Sigmoid is especially vulnerable because:

```text
σ'(z) = σ(z)(1-σ(z)) ≤ 0.25
```

and in saturated regions the derivative approaches zero.

ReLU helps because its derivative is 1 on the positive side, although it can suffer from **dying ReLUs** when units stay inactive.

Initialization, activation choice and normalization all help stabilize forward activations and backward gradients.

---

## 13. Normalization

### Batch Normalization

BatchNorm standardizes activations using batch statistics:

```text
x_hat = (x - batch_mean) / sqrt(batch_var + ε)
y = γ x_hat + β
```

The learnable `γ` and `β` restore representational flexibility.

Typical pattern:

```text
Linear → BatchNorm → ReLU
```

BatchNorm usually uses running statistics at inference time.

### Layer Normalization

LayerNorm normalizes across features inside each sample rather than across the batch.

That makes it independent of other examples in the batch and particularly suitable for Transformer-style architectures.

---

## 14. Overfitting and regularization

Overfitting occurs when training performance keeps improving while performance on unseen data gets worse.

The objective is not to memorize the training set but to generalize.

### L2 regularization

Add a weight penalty:

```text
L_total = L_data + (λ/2) ||W||²
```

Then:

```text
∂L_total/∂W = ∂L_data/∂W + λW
```

So in code:

```python
dW += reg * W
```

### Weight decay and AdamW

For vanilla SGD, L2 regularization and weight decay are closely related.

For Adam, adding `λW` into the gradient also subjects that term to Adam's adaptive rescaling.

AdamW decouples parameter shrinkage from the adaptive gradient update.

### Dropout

During training, dropout randomly removes activations.

With keep probability `q`:

```python
mask = np.random.rand(*A.shape) < q
A_drop = A * mask / q
```

The division by `q` preserves the expected activation scale.

Dropout is normally disabled at inference time.

### Early stopping

Instead of always keeping the last epoch, save the checkpoint with the best validation performance.

---

## 15. Softmax and cross-entropy

For multiclass classification, the last linear layer produces **logits**.

Softmax converts logits into probabilities:

```text
p_i = exp(z_i) / Σ_j exp(z_j)
```

For numerical stability, subtract the row maximum before exponentiating:

```python
shifted = logits - np.max(logits, axis=1, keepdims=True)
```

Cross-entropy is:

```text
L = -(1/N) Σ log p(correct class)
```

With softmax + cross-entropy, the gradient with respect to logits simplifies to:

```text
dZ = (P - Y) / N
```

In code:

```python
dZ = probs.copy()
dZ[np.arange(N), y] -= 1
dZ /= N
```

---

## 16. Complete classifier: 2 → 16 → 16 → 2

We built:

```text
2 inputs
→ 16 ReLU units
→ 16 ReLU units
→ 2 class logits
```

The implementation includes:

- He initialization
- ReLU
- softmax
- cross-entropy
- L2 regularization
- vectorized backpropagation
- gradient-descent updates

See: `05_circle_classifier.py`.

---

## 17. Generalization: train/test separation

Training accuracy is not enough.

A correct workflow is:

1. split the data into training and test sets;
2. update parameters using only training data;
3. evaluate on samples never used for gradient updates.

This distinguishes **fitting** from **generalization**.

---

## 18. Linear classifier vs MLP

We used the nonlinear rule:

```text
class 1 if sqrt(x1^2 + x2^2) > 1
class 0 otherwise
```

The true decision boundary is a circle.

A linear classifier can only create a straight decision boundary, so it struggles.

In one fixed experiment:

| Model | Train accuracy | Test accuracy |
|---|---:|---:|
| Linear softmax classifier | 0.580 | 0.620 |
| MLP 2→16→16→2 | 0.980 | 0.980 |

The MLP does not explicitly know the circle equation. ReLU units create piecewise-linear regions that combine into an approximately circular nonlinear boundary.

See: `06_linear_vs_mlp.py`.

---

## 19. The mental model to keep

For any neural-network problem, think in this order:

1. **Task** — what am I predicting?
2. **Input/output shapes** — what enters and leaves the model?
3. **Architecture** — what function family is appropriate?
4. **Forward pass** — what is computed at each layer?
5. **Loss** — what does “wrong” mean?
6. **Backward pass** — how does the loss depend on each parameter?
7. **Optimizer** — how should gradients change the parameters?
8. **Generalization** — does it work on unseen data?

Framework code such as:

```python
prediction = model(X)
loss.backward()
optimizer.step()
```

is only an abstraction over the machinery implemented explicitly in this lesson.

Once this NumPy version makes sense, PyTorch becomes a convenience rather than a black box.

## Run the lesson

```bash
pip install -r requirements.txt

python lessons/01-neural-networks-from-scratch/01_single_sample_backprop.py
python lessons/01-neural-networks-from-scratch/02_xor_mlp.py
python lessons/01-neural-networks-from-scratch/03_optimizers_from_scratch.py
python lessons/01-neural-networks-from-scratch/04_initialization_experiment.py
python lessons/01-neural-networks-from-scratch/05_circle_classifier.py
python lessons/01-neural-networks-from-scratch/06_linear_vs_mlp.py
```

## Next

**Lesson 2 — Rebuilding the same ideas in PyTorch**

We will map the NumPy implementation to:

- `nn.Module`
- tensors
- autograd
- `loss.backward()`
- optimizers
- datasets and dataloaders
- training/evaluation modes
