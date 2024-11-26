# original code : https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.v2 import RandAugment, Resize, RandomResizedCrop
from data_transform import mixup_data, cutmix_data

import os
import argparse
import wandb

from models import vgg,resnet
from models.vit_scratch import ViT
from torchvision.models import resnext50_32x4d, resnext101_32x8d, convnext
from models.vit_pretrained import get_vit_model
from utils import progress_bar

parser = argparse.ArgumentParser(description='CIFAR100 & ImageNet1k Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--datapath', type=str, help='data storage path')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
parser.add_argument('--model', type=str, help='model to use')
parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'imagenet1k'],
                    help='train & test dataset')
parser.add_argument('--augmentation', type=str, default='none', choices=['none', 'mixup', 'cutmix'],
                    help='Data augmentation')
parser.add_argument('--rand', action='store_true', help='Use RandAugment and RandomResizedCrop')
parser.add_argument('--loss', type=str, default='CE', choices=['CE', 'BCE'],
                    help='Loss function')
parser.add_argument('--warmup_epochs', type=int, default=20, help='warm-up epochs')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay')
parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adamw'],
                    help='optimizer')
parser.add_argument('--gpu', type=str, choices=['0', '1', '2', '3', '4', '5', '6', '7'],
                    help='Specify GPU IDs to use')
parser.add_argument('--wandb_project', type=str, default='CIFAR100_Project')
parser.add_argument('--wandb_run_name', type=str, default='TEST')

args = parser.parse_args()

# wandb initialization
wandb.init(
    project=args.wandb_project,
    name=args.wandb_run_name,
    config={
        'learning_rate': args.lr,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'model': args.model,
        'dataset': args.dataset,
        'extra_augmentation' : args.augmentation,
        'loss_function' : args.loss,
        'weight_decay' : args.weight_decay,
        'optimizer' : args.optimizer,
        'resume': args.resume,
    }
)

# model dictionary
model_dict = {
    'vgg11': lambda: vgg.VGG('VGG11'),
    'vgg13': lambda: vgg.VGG('VGG13'),
    'vgg16': lambda: vgg.VGG('VGG16'),
    'vgg19': lambda: vgg.VGG('VGG19'),
    'resnet18': lambda: resnet.resnet18(),
    'resnet34': lambda: resnet.resnet34(),
    'resnet50': lambda: resnet.resnet50(),
    'resnet101': lambda: resnet.resnet101(),
    'resnet152': lambda: resnet.resnet152(),
    'resnext50_32x4d': lambda: resnext50_32x4d(),
    'resnext101_32x8d': lambda: resnext101_32x8d(),
    'convnext_tiny': lambda: convnext.convnext_tiny(),
    'convnext_small': lambda: convnext.convnext_small(),
    'convnext_base': lambda: convnext.convnext_base(),
    'convnext_large': lambda: convnext.convnext_large(),
    'vit_b_16': lambda: get_vit_model('vit_b_16', num_classes=100),
    'vit_b_32': lambda: get_vit_model('vit_b_32', num_classes=100),
    'vit_l_16': lambda: get_vit_model('vit_l_16', num_classes=100),
    'vit_scratch' : lambda: ViT(image_size=224, patch_size=16, num_classes=100, dim=768, depth=12,
                                heads=12, mlp_dim=3072, pool='cls', channels=3, dim_head=64,
                                dropout=0.1, emb_dropout=0.1),
}

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0
start_epoch = 0

# Data
print('==> Preparing data...')
# With RandAugment
if args.rand:
    # ImageNet1K
    if 'imagenet' in args.dataset:
        transform_train = transforms.Compose([
            RandomResizedCrop(224),
            RandAugment(num_ops=2, magnitude=9),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # ImageNet Normalize
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.RandomErasing(),
        ])
        transform_test = transforms.Compose([
            Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    # CIFAR100
    else:
        transform_train = transforms.Compose([
            RandomResizedCrop(32),
            RandAugment(num_ops=2, magnitude=9),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # CIFAR100 Normalize
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.RandomErasing(),
        ])
        transform_test = transforms.Compose([
            Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
# Without RandAugment
else:
    # ImageNet1K
    if 'imagenet' in args.dataset:
        transform_train = transforms.Compose([
            RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        transform_test = transforms.Compose([
            Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    # Data augmentation for ViT Models
    elif 'vit' in args.model:
        transform_train = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # CIFAR Normalize
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

# Dataloader
# CIFAR100
if 'cifar' in args.dataset:
    trainset = torchvision.datasets.CIFAR100(
        root=args.datapath, train=True, download=False, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    testset = torchvision.datasets.CIFAR100(
        root=args.datapath, train=False, download=False, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
# ImageNet1K
else:
    trainset = torchvision.datasets.ImageNet(
        root=args.datapath, split='train', transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    testset = torchvision.datasets.ImageNet(
        root=args.datapath, split='val', transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )


# Learning rate adjustment function
# During the warm-up phase (first `warmup_epochs` epochs), the learning rate increases linearly
# After the warm-up phase, the learning rate follows a cosine decay schedule until `total_epochs`
def warmup_cosine_decay(epoch):
    if epoch < warmup_epochs:
        return epoch / warmup_epochs  # linear warm-up
    else:
        # cosine decay
        return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))


# Build model
print('==> Building model...')
if args.model in model_dict:
    model = model_dict[args.model]()
else:
    raise ValueError("Unknown model specified")

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters : {total_params}")
wandb.log({"total_params": total_params})

model = model.to(device)
if device == 'cuda':
    # Use DataParallel to distribute the model across multiple GPUs (if available)
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    model.load_state_dict(checkpoint['model'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# Select the loss function based on the input argument
if 'CE' in args.loss:
    criterion = nn.CrossEntropyLoss()
elif 'BCE' in args.loss:
    criterion = nn.BCELoss()
else:
    raise ValueError("Unknown loss function")

# Select the optimizer and corresponding learning rate scheduler based on the input argument
if 'sgd' in args.optimizer:
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
elif 'adamw' in args.optimizer:
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup_epochs = args.warmup_epochs
    total_epochs = args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_decay)
else:
    raise ValueError("Unknown optimizer")


# Training
def train(epoch):
    print('\nEpoch : %d' % epoch)
    model.train()
    train_loss = 0
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        if args.augmentation == 'mixup':
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=0.4)
            outputs = model(inputs)
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
        elif args.augmentation == 'cutmix':
            inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets, alpha=1.0)
            outputs = model(inputs)
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted_top1 = outputs.max(1)
        _, predicted_top5 = outputs.topk(5, 1, True, True)  # get top 5 predictions

        correct_top1 += predicted_top1.eq(targets).sum().item()
        correct_top5 += sum(targets[i] in predicted_top5[i] for i in range(targets.size(0)))

        total += targets.size(0)

        progress_bar(batch_idx, len(trainloader),
                     'Loss: %.3f | Top-1 Acc: %.3f%% (%d/%d) | Top-5 Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1),
                        100. * correct_top1 / total, correct_top1, total,
                        100. * correct_top5 / total, correct_top5, total))

        avg_train_loss = train_loss / len(trainloader)
        avg_train_top1_acc = 100. * correct_top1 / total
        avg_train_top5_acc = 100. * correct_top5 / total
        wandb.log({
            "train_loss": avg_train_loss,
            "train_top1_acc": avg_train_top1_acc,
            "train_top5_acc": avg_train_top5_acc
        })

# Evaluating
def test(epoch):
    global best_acc, best_top5_acc
    model.eval()
    test_loss = 0
    correct_top1 = 0
    correct_top5 = 0
    best_acc = 0
    best_top5_acc = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()

            # Top-1 and Top-5 accuracy calculation
            _, predicted_top1 = outputs.max(1)
            _, predicted_top5 = outputs.topk(5, 1, True, True)  # Get top 5 predictions

            # Update top-1 correct count
            correct_top1 += predicted_top1.eq(targets).sum().item()

            # Update top-5 correct count
            correct_top5 += sum(targets[i] in predicted_top5[i] for i in range(targets.size(0)))

            total += targets.size(0)

            # Modify progress bar message to include both Top-1 and Top-5 accuracy
            progress_bar(batch_idx, len(testloader),
                         'Loss: %.3f | Top-1 Acc: %.3f%% (%d/%d) | Top-5 Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1),
                            100. * correct_top1 / total, correct_top1, total,
                            100. * correct_top5 / total, correct_top5, total))

    # Calculate final accuracies
    avg_val_loss = test_loss / len(testloader)
    top1_acc = 100. * correct_top1 / total
    top5_acc = 100. * correct_top5 / total

    wandb.log({
        "val_loss": avg_val_loss,
        "val_top1_acc": top1_acc,
        "val_top5_acc": top5_acc
    })

    experiment_dir = f'./checkpoint/{args.wandb_run_name}'
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    # checkpoint save logic
    # save checkpoint if this is the best top-1 or top-5 accuracy
    save_checkpoint = False
    if top1_acc > best_acc:
        best_acc = top1_acc
        save_checkpoint = True

    if top5_acc > best_top5_acc:
        best_top5_acc = top5_acc
        save_checkpoint = True

    if save_checkpoint:
        print('Saving..')
        state = {
            'model': model.state_dict(),
            'top1_acc': top1_acc,
            'top5_acc': top5_acc,
            'epoch': epoch,
        }
        torch.save(state, f'{experiment_dir}/ckpt.pth')


for epoch in range(start_epoch, start_epoch + args.epochs):
    train(epoch)
    test(epoch)
    scheduler.step()
