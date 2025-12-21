import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from dataset import train_loader, test_loader  # Import DataLoaders from dataset.py

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Model
class DeforestationCNN(nn.Module):
    def __init__(self):
        super(DeforestationCNN, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)  # 2 classes (Normal, Deforestation)

    def forward(self, x):
        return self.model(x)

# Initialize Model
model = DeforestationCNN().to(device)

# Define Loss Function & Optimizer
criterion = nn.CrossEntropyLoss()  # Suitable for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer with learning rate 0.001

# Training Loop
num_epochs = 10  # Set number of training epochs
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)  # Move data to GPU/CPU

        optimizer.zero_grad()  # Clear previous gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)  # Get predicted labels
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Train Accuracy: {train_acc:.2f}%")

# Save Model
torch.save(model.state_dict(), "deforestation_model.pth")
print("Training complete! Model saved as deforestation_model.pth")

# Testing Loop
model.eval()  # Set model to evaluation mode
correct = 0
total = 0

with torch.no_grad():  # No need to compute gradients for validation
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

test_acc = 100 * correct / total
print(f"Test Accuracy: {test_acc:.2f}%")
