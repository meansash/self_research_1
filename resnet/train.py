# original code : https://github.com/clovaai/CutMix-PyTorch/blob/master/train.py

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import datetime
import data_preprocessing
import resnet

import warnings
import wandb

warnings.filterwarnings("ignore")

parser=argparse.ArgumentParser(description="ResNet CIFAR-10, CIFAR-100")
parser.add_argument('-j','--workers',default=4,type=int, metavar='N',
                    help='number of data loading workers (default:4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b','--batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay','--wd',default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p',default=1,type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--depth', default=32, type=int,
                    help='depth of the network (default: 32)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='to use basic block for CIFAR datasets (default: bottleneck)')
parser.add_argument('--dataset', dest='dataset', default='cifar10', type=str,
                    help='dataset (options : cifar10, cifar100)')
parser.add_argument('--no-verbose', dest='verbose', action='store_false',
                    help='to print the status at every iteration')
parser.add_argument('--alpha', default=300, type=float,
                    help='number of new channel increases per depth (default: 300)')
parser.add_argument('--expname', default='TEST', type=str,
                    help='name of experiment')

args = parser.parse_args()

# Initialize wandb
wandb.init(project="self_research_resnet",name=args.expname)

# log hyperparameters in wandb
wandb.config.update({
    "epochs" : args.epochs,
    "batch_size" : args.batch_size,
    "learning_rate" : args.lr,
    "momentum" : args.momentum,
    "weight_decay" : args.weight_decay,
    "dataset" : args.dataset,
})

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

parser.set_defaults(bottleneck=True)
parser.set_defaults(verbose=True)

best_err1 = 100
best_err5 = 100

def main():
    global args, best_err1, best_err5

    if args.dataset == 'cifar10':
        numberofclass = 10
    elif args.dataset == 'cifar100':
        numberofclass = 100
    else:
        raise Exception('unknown dataset: {}'.format(args.dataset))
    train_loader, val_loader = data_preprocessing.load_data(args.dataset, args.batch_size, args.workers)
    model = resnet.ResNet50(numberofclass)
    model = torch.nn.DataParallel(model).cuda()

    print(model)
    print('the number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum = args.momentum, weight_decay=args.weight_decay, nesterov=True)
    cudnn.benchmark = True

    current_lr = args.lr
    start_time = time.time()
    for epoch in range(0, args.epochs):
        # write train metrics in wandb
        train_loss, train_error = train(train_loader, model, criterion, optimizer, epoch, start_time)
        wandb.log({"train_loss": train_loss, "train_error" : train_error})
        if (epoch + 1) % 1 == 0 or epoch == 0 or epoch == args.epochs -1:
            err1, err5, val_loss = validate(val_loader, model, criterion, epoch)

            # write validation metrics in wandb
            wandb.log({"val_top1_err" : err1, "val_top5_error" : err5, "val_loss" : val_loss})
            # remember best and save checkpoint
            is_best = err1 <= best_err1
            best_err1 = min(err1, best_err1)
            if is_best:
                best_err5 = err5
            print('current best accuracy (top-1 and 5 error) : ', best_err1, best_err5)
        current_lr = adjust_learning_rate(optimizer,epoch,current_lr,args.epochs)

    # write best results in wandb
    wandb.log({"best_top1_error" : best_err1, "best_top5_error" : best_err5})

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self,val,n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch, lr, total_epochs):
    '''

    :param optimizer: optimizer to adjust learning rate
    :param epoch: current epoch
    :param lr: current learning rate
    :param total_epochs:
    :return: When the test error is more than 10% greater than the train error,
    if there is no improvement for a patience epoch, reduce the learning rate by a factor 0.1
    '''
    if epoch == int(0.5 * total_epochs):
        lr *= 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f'Adjusted learning rate to {lr} at epoch {epoch} due to reaching 50% of total epochs')
    elif epoch == int(0.75 * total_epochs):
        lr *= 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f'Adjusted learning rate to {lr} at epoch {epoch} due to reaching 75% of total epochs')
    return lr

def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1,-1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0,keepdim=True)
        wrong_k = batch_size - correct_k
        res.append(wrong_k.mul_(100.0 / batch_size))

    return res

def train(train_loader, model, criterion, optimizer, epoch, start_time):
    '''

    :param train_loader: training data loader
    :param model: model to be trained
    :param criterion: loss function
    :param optimizer: optimizer
    :param epoch: current epoch number
    :param start_time: training start time
    :return:
    '''
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    end = time.time()
    current_LR = get_learning_rate(optimizer)
    print(f'Epoch [{epoch}] : Current Learning rate : {current_LR}')
    for i, (input, target) in enumerate(train_loader):
        # measuring data loading time
        data_time.update(time.time() - end)

        # move input data and target to GPU
        input = input.cuda()
        target = target.cuda()
        output = model(input)
        loss = criterion(output,target)

        err1, err5 = accuracy(output.data, target, topk=(1,5))
        # update loss and accuracy at current batch
        losses.update(loss.item(), input.size(0))
        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))
        # update weight
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if (epoch % 10 == 0 or epoch == args.epochs - 1) and args.verbose == True and i % (
                len(train_loader) // 10) == 0:
            print('Epoch: [{}][{}][{}][{}]\t'
                  'LR: {}\t'
                  'Time {:.3f}({:.3f})\t'
                  'Data {:.3f} ({:.3f})\t'
                  'Loss {:.4f} ({:.4f})\t'
                  'Top 1 err {:.4f} ({:.4f})\t'
                  'Top 5 err {:.4f} ({:.4f})'.format(
                epoch, args.epochs, i, len(train_loader), ", ".join(map(str, current_LR)), batch_time.val,
                batch_time.avg, data_time.val, data_time.avg, losses.val, losses.avg, top1.val, top1.avg, top5.val,
                top5.avg
            ))

    print(
        '* Epoch: [{}][{}]\t Top 1 err {:.3f} Top 5 err {:.3f}\t Train Loss {:.3f}'.format(epoch, args.epochs, top1.avg,
                                                                                           top5.avg, losses.avg))
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {datetime.timedelta(seconds=int(elapsed_time))}")

    return losses.avg, top1.avg

def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            output = model(input)
            loss = criterion(output, target)
            err1, err5 = accuracy(output.data, target, topk=(1,5))
            losses.update(loss.item(), input.size(0))
            top1.update(err1.item(), input.size(0))
            top5.update(err5.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if (i % (len(val_loader) // 10) == 0) and args.verbose == True:
                print('Test (on val set): [{}][{}][{}][{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Loss {:.4f} ({:.4f})\t'
                      'Top 1 err {:.4f} ({:.4f})\t'
                      'Top 5 err {:.4f} ({:.4f})'.format(
                    epoch, args.epochs, i, len(val_loader), batch_time.val, batch_time.avg, losses.val, losses.avg,
                    top1.val, top1.avg, top5.val, top5.avg
                ))
            print('* Epoch: [{}][{}]\t Top 1 err {:.3f} Top 5 err {:.3f}\t Test Loss {:.3f}'.format(epoch, args.epochs,
                                                                                                    top1.avg, top5.avg,
                                                                                                    losses.avg))

            return top1.avg, top5.avg, losses.avg

if __name__ == '__main__':
    main()