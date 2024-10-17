import timm
import torch
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
import argparse
from tqdm import tqdm
import os
import wandb


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Vision Transformer on ImageNet1k/CIFAR100 using timm library')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for evaluation (default:64)')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Input image size (default : 224)')
    parser.add_argument('--num_classes',type=int, default=100,
                        help='Number of classes (default : 100)')
    parser.add_argument('--model_name', type=str, default='vit_base_patch16_224',
                        help='Model name from timm')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained model')
    parser.add_argument('--checkpoint_path', type=str, default='',
                        help='Path to model checkpoint (if not using pretrained model)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for evaluation (default : cuda)')
    parser.add_argument('--dataset', type=str, default='imagenet1k', choices=['imagenet1k','cifar100'],
                        help='Dataset to use for evaluation (default : ImageNet1k)')
    parser.add_argument('--datapath',type=str, default='./data',
                        help='Dataset path')
    parser.add_argument('--wandb_project', type=str, default='ViT_pretrained_eval',
                        help='wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default='ViT_pretrained',
                        help='wandb run name')

    return parser.parse_args()

def main():
    args = parse_args()
    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config={
        "batch_size" : args.batch_size,
        "image_size" : args.image_size,
        "num_classes" : args.num_classes,
        "model_name" : args.model_name,
        "dataset" : args.dataset,
    })
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                     std=[0.229,0.224,0.225])

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        normalize,
    ])

    # ImageNet21k dataloader
    if args.dataset == 'imagenet1k':
        test_set = torchvision.datasets.ImageNet(root=args.datapath, transform=transform, split='val')
        test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    #CIFAR-100 dataloader
    else:
        test_set = torchvision.datasets.CIFAR100(root=args.datapath, transform=transform, train=False, download=True)
        test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,num_workers=4)

    model = timm.create_model(args.model_name, pretrained=args.pretrained, num_classes=args.num_classes)
    if args.checkpoint_path:
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total Parameters: {total_params:,}')
    wandb.log({"Total Parameters": total_params})
    model.eval()

    top_1 = 0
    top_5 = 0
    total = 0

    with torch.no_grad():
        for idx, (images, labels) in enumerate(tqdm(test_loader)):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            # calculate top-1 acc
            _, pred1 = torch.max(outputs,1)
            total += labels.size(0)
            top_1 += (pred1 == labels).sum().item()

            # calculate top-5 acc
            _, pred5= outputs.topk(5,1,True,True)
            correct5 = pred5.eq(labels.view(-1,1).expand_as(pred5))
            top_5 += correct5.sum().item()

    top1_acc = top_1 / total * 100
    top5_acc = top_5 / total * 100
    wandb.log({"Top-1 Accuracy" : top1_acc, "Top-5 Accuracy" : top5_acc})
    print(f"Top-1 Acc : {top1_acc:.2f}%")
    print(f"Top-5 Acc : {top5_acc:.2f}%")
    wandb.finish()
if __name__ == "__main__":
    main()