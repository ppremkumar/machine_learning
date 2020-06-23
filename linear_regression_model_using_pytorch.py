import torch
import numpy as np

# Input (temp, rainfall, humidity)
inputs = np.array([[73, 67, 43], [91, 88, 64], [87, 134, 58], 
                   [102, 43, 37], [69, 96, 70], [73, 67, 43], 
                   [91, 88, 64], [87, 134, 58], [102, 43, 37], 
                   [69, 96, 70], [73, 67, 43], [91, 88, 64], 
                   [87, 134, 58], [102, 43, 37], [69, 96, 70]], 
                  dtype='float32')
# Targets (apples, oranges)
targets = np.array([[56, 70], [81, 101], [119, 133], 
                    [22, 37], [103, 119], [56, 70], 
                    [81, 101], [119, 133], [22, 37], 
                    [103, 119], [56, 70], [81, 101], 
                    [119, 133], [22, 37], [103, 119]], 
                   dtype='float32')

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

# CREATE TensorDataset THAT CONTAINS A TUPLE OF "inputs" AND "targets"
from torch.utils.data import TensorDataset
train_ds = TensorDataset(inputs, targets)

# CREATE DataLoader THAT CAN SHUFFLE "train_ds" AND SPLIT "train_ds" INTO
# PREDEFINED BATCH SIZE
BATCH_SIZE = 5
from torch.utils.data import DataLoader
train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True)


# CREATE WEIGHTS AND BIASES: nn.Linear
import torch.nn as nn
model = nn.Linear(3,2)
# "3": NUMBER OF INPUTS
# "2": NUMBER OF TARGETS
# TEST "model"
# print(model.weight)
# print(model.bias)
# print(list(model.parameters()))

# CREATE A LOSS FUNCTION
import torch.nn.functional as F
loss_fn = F.mse_loss

#TEST "loss_fn"
# FIRST CREATE PREDICTIONS
preds = model(inputs)
loss = loss_fn(preds, targets)
# print(f'{"*"*15}PRIOR TO TRAINING{"*"*15}')
# print(f'PREDICTIONS: {preds}')
# print(f'TARGETS: {targets}')
# print(f'LOSS: {loss}')


# CREATE AN OPTIMIZER TO ADJUST WEIGHTS AND BIASES WITH RESPECT TO GRADIENTS
opt = torch.optim.SGD(model.parameters(), lr=1e-5)


# NOW WE ARE READY TO TRAIN THE MODEL
# STEP 1: MAKE PREDICTIONS
# STEP 2: CALCULATE LOSS
# STEP 3: CALCULATE GRADIENTS WITH RESPECT TO WEIGHTS AND BIASES
# STEP 4: ADJUST WEIGHTS BY SUBTRACTING A SMALL QUANTITY PROPOTIONAL TO THE GRADIENT
# STEP 5: RESET GRADIENTS TO ZERO

from tqdm import tqdm

# MODEL TRAINING FUNCTION
def fit(num_epochs, loss_fn, model, train_dl, opt):
    for epoch in range(num_epochs):
        for xb, yb in train_dl:
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            opt.step()
            opt.zero_grad()
        
        # Print the progress
        if (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
