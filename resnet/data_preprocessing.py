from torch.utils.data import DataLoader
import torch
from torchvision import transforms
import torchvision.datasets as datasets

def load_data(dataset, batch_size, workers):
    normalize = transforms.Normalize(mean = [x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std = [x / 255.0 for x in [63.0, 62.1, 66.7]])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32,padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        normalize,
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    if dataset == 'cifar100':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('../data', train=True, download=True, transform=transform_train),
            batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True
        )
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('../data', train=False, download=True, transform=transform_val),
            batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True
        )
    elif dataset == 'cifar10':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('../data', train=True, download=True, transform=transform_train),
            batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True
        )
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('../data', train=False, download=True, transform=transform_val),
            batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True
        )

    return train_loader,val_loader