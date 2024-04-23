import os
import torch
from torchvision import transforms
from PIL import Image
import random
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

# Define paths to the folders
fake_folder = "/content/dsl_final_project_ai_vs_real_art/detect ai artwork - resizing images for efficiency/ai vs real art/FAKE_resized"
real_folder = "/content/dsl_final_project_ai_vs_real_art/detect ai artwork - resizing images for efficiency/ai vs real art/REAL_resized"

# Define transformations to resize images
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Function to load images and assign labels
def load_images(folder, label):
    dataset = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg"):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path)
            img = transform(img)
            dataset.append((img, label))
    return dataset

# Load images and assign labels
fake_data = load_images(fake_folder, 0)
real_data = load_images(real_folder, 1)

# Combine the datasets
dataset = fake_data + real_data

# Shuffle the dataset
random.shuffle(dataset)

# Split into features and labels
features = torch.stack([data[0] for data in dataset])
labels = torch.tensor([data[1] for data in dataset])

# Print the shapes of features and labels
print("Features shape:", features.shape)
print("Labels shape:", labels.shape)

# Define a function to display images and labels
def show_images(dataset, num_samples=5):
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for i in range(num_samples):
        img, label = dataset[i]
        axes[i].imshow(img.permute(1, 2, 0))
        axes[i].set_title("Label: {}".format(label))
        axes[i].axis("off")
    plt.show()

# Display some samples
show_images(dataset)

# Define the size of training and testing sets
train_size = int(0.75 * len(dataset))
test_size = len(dataset) - train_size

# Split dataset into training and testing sets
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Define data loaders for training and testing sets
batch_size=32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Print the sizes of training and testing sets
print("Training set size:", len(train_dataset))
print("Testing set size:", len(test_dataset))

# MODEL 1
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 128 * 128, 128)
        self.fc2 = nn.Linear(128, 2)  # Output size 2 for binary classification (fake vs real)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 16 * 128 * 128)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

# MODEL 2
class SimpleCNN2(nn.Module):
    def __init__(self):
        super(SimpleCNN2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 64 * 64, 256)
        self.fc2 = nn.Linear(256, 2)  # Output size 2 for binary classification (fake vs real)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 64 * 64)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# MODEL 3
# Define the CNN architecture
class SimpleCNN3(nn.Module):
    def __init__(self):
        super(SimpleCNN3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 32 * 32, 256)
        self.fc2 = nn.Linear(256, 2)  # Output size 2 for binary classification (fake vs real)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 32 * 32)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
