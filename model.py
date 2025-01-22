import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
 
# Model
model = nn.Sequential(nn.Linear(1, 1))
 
# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
 
# Data
xs = torch.tensor([[-1.0], [0.0], [1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
ys = torch.tensor([[-3.0], [-1.0], [1.0], [3.0], [5.0], [7.0]], dtype=torch.float32)
 
# Train
for _ in range(500):
  optimizer.zero_grad()
  outputs = model(xs)
  loss = criterion(outputs, ys)
  loss.backward()
  optimizer.step()
 
# Predict
with torch.no_grad():
  print(model(torch.tensor([[10.0]], dtype=torch.float32)))