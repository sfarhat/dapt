# Linearly probing networks on downstream tasks
# Uses pre-trained models from Pytorch model hub

import argparse
import json
import os
import time
from tqdm import tqdm
import torch
import torch.nn as nn
from utils import datasets, utils

def validate(net, testloader, args):

    correct = 0
    total = 0

    net.eval()
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = correct / total

    return acc

def main(args):

    print(args.model)
    print(args.dataset)

    train_set, trainloader, _, _, _, testloader, _, num_classes, im_size = datasets.get_dataset(args)
    n_data = len(train_set)

    print(f"Train set size: {n_data}")

    net = utils.load_model(args, args.model, num_classes, 'pt')
    for param in net.model.parameters():
        param.requires_grad = False
    net.get_classifier().train()
    for param in net.get_classifier().parameters():
        param.requires_grad = True

    net.to(args.device)

    criterion = nn.CrossEntropyLoss().to(args.device)
    scheduler = None

    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(net.get_classifier().parameters())
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(net.get_classifier().parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        raise NotImplementedError(args.optimizer)

    if args.distributed:
        devices = list(range(0, torch.cuda.device_count()))        
        devices.remove(args.n_gpu)
        net = nn.DataParallel(net, device_ids = [args.n_gpu, *devices])

    best_acc = 0
    start_epoch = 0

    dir_name = f"{args.model}_lp_{args.optimizer}"
    save_path = os.path.join(args.cpt_path, args.dataset, dir_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        if args.cpt_start == 'last':
            last_cpt = torch.load(f"{save_path}/last_checkpoint.pt", map_location=args.device)
            best_cpt = torch.load(f"{save_path}/best_checkpoint.pt", map_location=args.device)

            best_acc = best_cpt['accuracy']
            net.model.load_state_dict(last_cpt['model_state_dict'])
            optimizer.load_state_dict(last_cpt['optimizer_state_dict'])
            scheduler.load_state_dict(last_cpt['scheduler_state_dict'])
            start_epoch = last_cpt['epoch']
        else:
            pass

    # Caches for embeddings so we only have to compute them once
    # Will incur a memory cost, but nothing crazy for reasonably sized datasets
    cached_embeddings = torch.zeros((n_data, net.feat_dim), device=args.device)
    print(f"Starting at epoch {start_epoch}")

    for epoch in tqdm(range(start_epoch, args.epochs)):

        start = time.time()
        net.train()

        for batch, data in enumerate(trainloader):
            inputs, labels, index = data
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            index = index.to(args.device)

            optimizer.zero_grad()

            # outputs = net(inputs)

            # On first epoch, cache the output of the teacher (embeddings and logits)
            # On future epochs, we can just fetch them from memory instead of another forward pass
            if epoch == 0:
                feat, logit_t = net(inputs, is_feat=True)
                cached_embeddings[index] = feat[-1]
            else:
                feat = cached_embeddings[index]
                logit_t = net.get_classifier()(feat)

            loss = criterion(logit_t, labels)
            loss.backward()
            optimizer.step()

        if scheduler:
            scheduler.step()

        if args.timing:
            torch.cuda.current_stream().synchronize()
            end = time.time()
            print(f"Time: {end - start}")

        else:
            acc = validate(net, testloader, args)

            print(f'Epoch: {epoch+1}, Test Accuracy: {acc}')

            if acc > best_acc:
                best_acc = acc
                utils.save_model(args, epoch, net.model, optimizer, scheduler, acc, save_path, cpt_type='best')

            utils.save_model(args, epoch, net.model, optimizer, scheduler, acc, save_path, cpt_type='last')

    if not args.timing:
        print(f"Reported Best Accuracy: {best_acc}")

        best_cpt = torch.load(f"{save_path}/best_checkpoint.pt")
        net.model.load_state_dict(best_cpt['model_state_dict'])
        acc = validate(net, testloader, args)

        print(f"Tested Best Accuracy: {acc}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Linear Probe Dataset')

    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar10', 'cifar100', 'mit_indoor', 'cub2011', 'dtd', 'caltech101'])

    parser.add_argument('--timing', action='store_true', help='config that times training, skips validation and saving')

    parser.add_argument('--model', type=str, default='resnet50', choices=['resnet50', 'resnet18', 'vit-b-16', 'mobilenetv2'])

    parser.add_argument('--projector', type=str, default='mocov2', choices=['simclrv2', 'simclr', 'mocov2', 'linear'], help='The architecture of the non-linear projector for contrastive learning')

    parser.add_argument('--seed', type=int, default=9)

    parser.add_argument('--cpt_start', type=str, choices=['last'])

    # optimization
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['sgd', 'adamw'])
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    parser.add_argument('--scheduler', default='cosine', choices=['cosine', 'step', 'none'])
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')

    parser.add_argument('--train_bs', type=int, default=64)
    parser.add_argument('--test_bs', type=int, default=32)

    parser.add_argument('--epochs', type=int, default=250, help='number of training epochs')

    parser.add_argument('--n_gpu', type=int, default=0, help='index of gpu if multiple available')

    args = parser.parse_args()

    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))

    torch.manual_seed(args.seed)

    if args.timing:
        args.epochs = 1
        args.logging = False
    d = json.load(open('./paths.json'))
    vars(args).update(d)

    args.device = torch.device(f"cuda:{args.n_gpu}" if torch.cuda.is_available() else "cpu")
    print(args.device)
    # args.distributed = torch.cuda.device_count() > 1
    args.distributed = False

    main(args)
