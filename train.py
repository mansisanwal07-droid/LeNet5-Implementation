import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.lenet5 import LeNet5

# 1. Setup Data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

# 2. Initialize Model, Loss, and Optimizer
model = LeNet5()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. Training Loop
print("Starting Training...")
for epoch in range(3):
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if i % 300 == 0:
            print(f"Epoch [{epoch+1}/3], Step [{i}], Loss: {loss.item():.4f}")

# 4. Save the trained weights
torch.save(model.state_dict(), "lenet5_weights.pth")
print("Training Complete. Weights saved to lenet5_weights.pth")

