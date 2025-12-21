import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError, ImageFile
from collections import Counter

# Fix truncated image errors
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images
    transforms.ToTensor(),          # Convert to tensor
])

# Custom Dataset Class
class DeforestationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Load images and labels
        for label, category in enumerate(["normal", "deforestation"]):
            category_path = os.path.join(root_dir, category)
            if not os.path.exists(category_path):
                print(f"Warning: {category_path} not found!")
                continue
            
            for img_name in os.listdir(category_path):
                img_path = os.path.join(category_path, img_name)
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        with Image.open(img_path) as img:
                            img.verify()  # Check if it's a valid image
                        self.image_paths.append(img_path)
                        self.labels.append(label)
                    except (OSError, UnidentifiedImageError):
                        print(f"Skipping corrupted image: {img_path}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert("RGB")  # Open image
            if self.transform:
                image = self.transform(image)
            return image, label
        except (OSError, UnidentifiedImageError):
            print(f"Skipping truncated/corrupted image: {img_path}")
            return self.__getitem__((idx + 1) % len(self))  # Get next valid image

# Load Dataset
train_dataset = DeforestationDataset("C:/Users/kaviy/Documents/deforest/train", transform=transform)
test_dataset = DeforestationDataset("C:/Users/kaviy/Documents/deforest/test", transform=transform)

# Check Class Distribution
train_labels = [label for _, label in train_dataset]
test_labels = [label for _, label in test_dataset]
print("Train Class Distribution:", Counter(train_labels))
print("Test Class Distribution:", Counter(test_labels))

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Check if dataset is loading properly
if __name__ == "__main__":
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}, Labels: {labels}")
        break
