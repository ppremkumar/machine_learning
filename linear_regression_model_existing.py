import numpy as np
import torch

# MSE loss
def mse(t1, t2):
    diff = t1 - t2
    return torch.sum(diff * diff) / diff.numel()

def model(x):
    return x @ w.t() + b

# DATA FROM EXISTING TUTORIAL
# Input (temp, rainfall, humidity)
inputs = np.array([[73, 67, 43], 
                   [91, 88, 64], 
                   [87, 134, 58], 
                   [102, 43, 37], 
                   [69, 96, 70]], dtype='float32')
# Targets (apples, oranges)
targets = np.array([[56, 70], 
                    [81, 101], 
                    [119, 133], 
                    [22, 37], 
                    [103, 119]], dtype='float32')
# Convert inputs and targets to tensors
inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)
# Weights and biases
w = torch.randn(2, 3, requires_grad=True)
b = torch.randn(2, requires_grad=True)



# Generate predictions
preds = model(inputs)
print(f'PREDICTIONS: {preds}')
# Compare with targets
print(F'TARGETS: {targets}')
# Compute loss
loss = mse(preds, targets)
print('*********LOSS PRIOR TO TRAINING*********')
print(f'LOSS: {loss}')
# print('*'*35)
# Train for 5000 epochs
for i in range(5000):
    preds = model(inputs)
    loss = mse(preds, targets)
    loss.backward()
    with torch.no_grad():
        w -= w.grad * 1e-5
        b -= b.grad * 1e-5
        w.grad.zero_()
        b.grad.zero_()
# Calculate loss
preds = model(inputs)
loss = mse(preds, targets)
print('*********LOSS AFTER TRAINING*********')
print(f'LOSS: {loss}')
# print('*'*37)
preds = model(inputs)
loss = mse(preds, targets)
print('***************PREDICTIONS AND TARGETS***************')
print(preds)
print(targets)
# print('*'*50)
