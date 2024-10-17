from vit import ViT
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm
import wandb

import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train Vision Transformer on CIFAR100')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training (default : 64)')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Learning rate (default: 3e-4)')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='Number of training epochs (default:32)')
    parser.add_argument('--image_size', type=int, default=32,
                        help='Input image size (default: 32)')
    parser.add_argument('--patch_size', type=int, default=4,
                        help='Patch size (default : 4)')
    parser.add_argument('--dim', type=int, default=512,
                        help='Dimensionality of the model (default : 512)')
    parser.add_argument('--depth', type=int, default=6,
                        help='Depth of the transformer (default : 6)')
    parser.add_argument('--heads', type=int, default=8,
                        help='Number of attention heads (default : 8)')
    parser.add_argument('--mlp_dim', type=int, default=1024,
                        help='Dimensionality of the MLP (default : 1024)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate (default : 0.1)')
    parser.add_argument('--emb_dropout', type=float, default=0.1,
                        help='Embedding dropout rate (default : 0.1)')
    parser.add_argument('--num_classes', type=int, default=100,
                        help='Number of class')
    parser.add_argument('--expname', default='TEST', type=str,
                        help='name of experiment')
    parser.add_argument('--wandb_project', type=str, default='ViT_CIFAR100',
                        help='wandb project name')

    return parser.parse_args()


def main():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.init(project=args.wandb_project, config=args, name=args.expname)


    transform_train = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=4)

    model = ViT(
        image_size=args.image_size,
        patch_size=args.patch_size,
        num_classes=args.num_classes,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
        pool='cls',
        channels=3,
        dim_head=64,
        dropout=args.dropout,
        emb_dropout=args.emb_dropout
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total Parameters : {total_params:,}')

    wandb.watch(model, log="all")


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    def train(model, loader, optimizer, criterion, device):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(loader):
            images, labels = images.to(device), labels.to(device),
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        wandb.log({"Train Loss":epoch_loss,"Train Accuracy":epoch_acc})
        return epoch_loss, epoch_acc

    def evaluate(model, loader, criterion, device):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(loader):
                images, labels = images.to(device), labels.to(device),
                outputs = model(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        wandb.log({"Validation Loss":epoch_loss,"Validation Accuracy":epoch_acc})
        return epoch_loss, epoch_acc

    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(1, args.num_epochs + 1):
        print(f'\nEpoch {epoch}/{args.num_epochs}')
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        print(f'Training loss : {train_loss:.4f}, Training acc : {train_acc:.2f}%')
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        print(f'Validation loss : {val_loss:.4f}, Validation acc : {val_acc:.2f}%')
        scheduler.step()

        if epoch % 10 == 0:
            torch.save(model.state_dict(), 'checkpoints/vit_cifar100.pth')

    wandb.finish()

if __name__ == '__main__':
    main()
