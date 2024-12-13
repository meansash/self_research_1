# python zero_shot.py --dataset_root /home/meansash/private/selfResearch/VLM/CLIP/classification --gpu_id 0


import os
import clip
import torch
from torchvision.datasets import CIFAR100
import argparse

# Argument parser 설정
parser = argparse.ArgumentParser(description="CLIP Zero-shot Prediction")
parser.add_argument("--dataset_root", type=str, required=True, help="Root directory for the CIFAR-100 dataset")
parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use (default: 0)")
args = parser.parse_args()

# Set the device based on GPU ID
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_id)
    device = f"cuda:{args.gpu_id}"
else:
    device = "cpu"

# Load the model
model, preprocess = clip.load('ViT-B/32', device)

# Download the dataset
cifar100 = CIFAR100(root=args.dataset_root, download=True, train=False)

# Prepare the inputs
image, class_id = cifar100[3637]
image_input = preprocess(image).unsqueeze(0).to(device)
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)

# Calculate features
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)

# Pick the top 5 most similar labels for the image
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
values, indices = similarity[0].topk(5)

# Print the result
print("\nTop predictions:\n")
for value, index in zip(values, indices):
    print(f"{cifar100.classes[index]:>16s}: {100 * value.item():.2f}%")
