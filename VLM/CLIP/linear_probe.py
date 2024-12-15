import os
import clip
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

# Argument parser 설정
parser = argparse.ArgumentParser(description="CLIP Linear Probing with PyTorch")
parser.add_argument("--dataset_root", type=str, required=True, help="Root directory for the CIFAR-100 dataset")
parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use (default: 0)")
args = parser.parse_args()

# Set the device based on GPU ID
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_id)
    device = f"cuda:{args.gpu_id}"
else:
    device = "cpu"

# Load the CLIP model
model, preprocess = clip.load('ViT-B/32', device)

# Load the CIFAR-100 dataset
root = args.dataset_root
train = CIFAR100(root, download=False, train=True, transform=preprocess)
test = CIFAR100(root, download=False, train=False, transform=preprocess)

def get_features(dataset, batch_size=50):
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size=batch_size)):
            features = model.encode_image(images.to(device))
            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

# Extract features for train and test sets
train_features, train_labels = get_features(train)
test_features, test_labels = get_features(test)

# Logistic Regression Model in PyTorch
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)

# Initialize the model
input_dim = train_features.shape[1]
num_classes = len(np.unique(train_labels))

# Convert labels to PyTorch tensors
train_labels = torch.tensor(train_labels, dtype=torch.long, device=device)
test_labels = torch.tensor(test_labels, dtype=torch.long, device=device)

# Convert features to PyTorch tensors
train_features = torch.tensor(train_features, dtype=torch.float32, device=device)
test_features = torch.tensor(test_features, dtype=torch.float32, device=device)

model = LogisticRegressionModel(input_dim, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(train_features)
    loss = criterion(outputs, train_labels)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# Evaluate the model
model.eval()
with torch.no_grad():
    predictions = torch.argmax(model(test_features), dim=1)
    accuracy = (predictions == test_labels).float().mean().item() * 100.

print(f"Accuracy = {accuracy:.3f}%")