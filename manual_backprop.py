import torch

# Data
x = torch.tensor([[-2.0, 4.0],
                  [7.0, -2.0]])

y = torch.tensor([[5.0],
                  [-3.0]])

# Initial Weights
w1 = 0.1
w2 = 0.2
w3 = 0.15
w4 = 0.7
w5 = 0.21
w6 = -0.3

lr = 0.1

# Initialize gradients
dw1 = dw2 = dw3 = dw4 = dw5 = dw6 = 0.0

# Loop over samples
for i in range(len(x)):
    x1 = x[i][0]
    x2 = x[i][1]
    yi = y[i][0]

    # Forward pass
    x3 = w1 * x1 + w3 * x2
    x4 = w2 * x1 + w4 * x2
    y_hat = w5 * x3 + w6 * x4

    # Loss gradient
    dL_dyhat = 2 * (y_hat - yi)

    # Gradients for output layer
    dw5 += dL_dyhat * x3
    dw6 += dL_dyhat * x4

    # Backprop to hidden layer
    dx3 = dL_dyhat * w5
    dx4 = dL_dyhat * w6

    dw1 += dx3 * x1
    dw3 += dx3 * x2
    dw2 += dx4 * x1
    dw4 += dx4 * x2

# Average gradients
n = len(x)
dw1 /= n
dw2 /= n
dw3 /= n
dw4 /= n
dw5 /= n
dw6 /= n

# Update weights
w1 -= lr * dw1
w2 -= lr * dw2
w3 -= lr * dw3
w4 -= lr * dw4
w5 -= lr * dw5
w6 -= lr * dw6

print("Updated Weights (Manual):")
print(w1, w2, w3, w4, w5, w6)