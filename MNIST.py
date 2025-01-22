import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
 
# Load the dataset
transform = transforms.Compose([transforms.ToTensor()])
 
train_dataset = datasets.FashionMNIST(root='./data', train=True, 
    download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, 
    download=True, transform=transform)
 
train_loader = DataLoader(train_dataset, batch_size=64, 
 shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, 
 shuffle=False)
 
# Define the model
class FashionMNISTModel(nn.Module):
    def __init__(self):
    super(FashionMNISTModel, self).__init__()
    self.flatten = nn.Flatten()
    self.linear_relu_stack = nn.Sequential(
    nn.Linear(28*28, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
    nn.LogSoftmax(dim=1)
    )
 
    def forward(self, x):
    x = self.flatten(x)
    logits = self.linear_relu_stack(x)
    return logits
 
model = FashionMNISTModel()
 
# Define the loss function and optimizer
loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters())
 
# Train the model
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
    # Compute prediction and loss
    pred = model(X)
    loss = loss_fn(pred, y)
 
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
 
    if batch % 100 == 0:
    loss, current = loss.item(), batch * len(X)
    print(f"loss: {loss:>7f}  
    [{current:>5d}/{size:>5d}]")
 
# Training process
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_loader, model, loss_function, optimizer)
print("Done!")
 