# Deep Learning From Scratch

A structured, hands-on deep-learning learning series built from first principles.

The goal is to understand what neural networks are doing mathematically and computationally before relying on high-level frameworks.

## Series

### Lesson 1 — Neural Networks from Scratch with NumPy

This lesson covers:

- neurons, dot products, matrix multiplication and shape reasoning
- linear layers and why nonlinear activations are necessary
- ReLU
- forward propagation
- losses
- derivatives, gradients, chain rule and backpropagation
- single-sample backpropagation
- XOR with a NumPy MLP
- full-batch GD, SGD and mini-batches
- momentum and Adam
- initialization: Xavier and He
- vanishing/exploding gradients and dying ReLUs
- BatchNorm and LayerNorm intuition
- overfitting and generalization
- L2 regularization, weight decay, AdamW, dropout and early stopping
- softmax and cross-entropy
- a complete 2→16→16→2 classifier
- train/test evaluation
- linear classifier vs MLP on a nonlinear circular boundary

The code deliberately avoids PyTorch in Lesson 1.

## Run

```bash
pip install -r requirements.txt
python lessons/01-neural-networks-from-scratch/02_xor_mlp.py
```

## Planned next lesson

Lesson 2 will rebuild the same ideas in PyTorch and map the explicit NumPy implementation to:

- `nn.Module`
- tensors
- autograd
- `loss.backward()`
- optimizers
- `Dataset` / `DataLoader`
- `train()` / `eval()`
