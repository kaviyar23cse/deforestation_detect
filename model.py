import torch
import torch.nn as nn
import torchvision.models as models

# Define the CNN Model using ResNet18
class DeforestationCNN(nn.Module):
    def __init__(self):
        super(DeforestationCNN, self).__init__()
        # Load pre-trained ResNet18
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Modify the fully connected layer to match 2 output classes (normal, deforestation)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)

    def forward(self, x):
        return self.model(x)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the model and move to device
model = DeforestationCNN().to(device)

# Save the model architecture (optional)
if __name__ == "__main__":
    print(model)  # Print model summary
