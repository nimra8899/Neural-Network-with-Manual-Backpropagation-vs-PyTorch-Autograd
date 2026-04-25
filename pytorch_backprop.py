import torch

# Data
x = torch.tensor([[-2.0, 4.0],
                  [7.0, -2.0]])

y = torch.tensor([[5.0],
                  [-3.0]])

# Initialize weights with gradients
w1 = torch.tensor(0.1, requires_grad=True)
w2 = torch.tensor(0.2, requires_grad=True)
w3 = torch.tensor(0.15, requires_grad=True)
w4 = torch.tensor(0.7, requires_grad=True)
w5 = torch.tensor(0.21, requires_grad=True)
w6 = torch.tensor(-0.3, requires_grad=True)

lr = 0.1

# Split inputs
x1 = x[:, 0].view(-1, 1)
x2 = x[:, 1].view(-1, 1)

# Forward pass
x3 = w1 * x1 + w3 * x2
x4 = w2 * x1 + w4 * x2
y_hat = w5 * x3 + w6 * x4

# Loss
loss = torch.mean((y_hat - y) ** 2)

# Backward pass
loss.backward()

# Update weights
with torch.no_grad():
    w1 -= lr * w1.grad
    w2 -= lr * w2.grad
    w3 -= lr * w3.grad
    w4 -= lr * w4.grad
    w5 -= lr * w5.grad
    w6 -= lr * w6.grad

print("Updated Weights (PyTorch):")
print(w1.item(), w2.item(), w3.item(), w4.item(), w5.item(), w6.item())