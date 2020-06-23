import numpy as np
import torch

# MSE loss
def mse(t1, t2):
    diff = t1 - t2
    return torch.sum(diff * diff) / diff.numel()

def model(x):
    return x @ w.t() + b

import pandas as pd
# DATA FROM W.H.O. ON LIFE EXPECTANCY
file = r'D:\01_automation\02_offline\MachineLearning\01_working\life_expectancy_df_for_calculation_filtered.xlsx'
df = pd.read_excel(file)
input_df = df[['schooling', 'alcohol', 'percentage_expenditure',
       'hepatitis_B', 'measles', 'BMI', 'polio', 'total_expenditure',
       'diphtheria', 'HIV_AIDS', 'GDP', 'population', 'thinness_1_19_years',
       'thinness_5_9_years', 'income_composition_of_resources']]
input_df = input_df.astype(np.float32)
target_df = df[['life_expectancy', 'adult_mortality', 'infant_deaths',
       'under_five_deaths']]
target_df = target_df.astype(np.float32)
inputs = torch.tensor(input_df.values)
targets = torch.tensor(target_df.values)
# Weights and biases
w = torch.randn(4, 15, requires_grad=True)
b = torch.randn(4, requires_grad=True)

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
