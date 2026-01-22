import torch
import torchvision.transforms as transforms
from PIL import Image
from model import model  # Import trained model

# Load trained model
model.load_state_dict(torch.load("deforestation_model.pth"))
model.eval()

# Define transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Function to predict an image
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    class_names = ["Normal", "Deforestation"]
    return class_names[predicted.item()]

# Test with an image
test_image = ""# Change this to an actual image
prediction = predict_image(test_image)
print(f"Prediction result: {prediction}")
