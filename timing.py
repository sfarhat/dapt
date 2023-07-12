import argparse
import json
import time
import torch
import torch.nn as nn
import utils.datasets as datasets

def main(args):

    print(args.model)
    print(args.dataset)
    print(args.finetune_mode)

    train_set, trainloader, val_set, valloader, test_set, testloader, \
    channel, num_classes, im_size= datasets.get_dataset(args)

    # All Pytorch pre-trained models assume input is at least 224x224
    if args.model == 'resnet50':
        from torchvision.models import resnet50
        net = resnet50(num_classes=num_classes)
        # Freezes weights of backbone
        if args.finetune_mode == 'head':
            for param in net.parameters():
                param.requires_grad = False
        # 18 and 34 have expansion = 1 since they start with a BasicBlock
        # 50 and above start with a Bottleneck which uses expansion = 4
        expansion = 4
        net.fc = torch.nn.Linear(512 * expansion, num_classes)

    elif args.model == 'mobilenetv2':
        from torchvision.models import mobilenet_v2
        net = mobilenet_v2(num_classes=num_classes)
        if args.finetune_mode == 'head':
            for param in net.parameters():
                param.requires_grad = False
        # This matches what library model has
        net.classifier = nn.Sequential(
            # This is hardcoded from the default params of the model
            # There is no way to access it if it's set, w/o overloading the init to create a var
            nn.Dropout(p=0.2),
            nn.Linear(net.last_channel, num_classes),
        )
    else:
        raise NotImplementedError(args.model)

    net.to(args.device)

    criterion = nn.CrossEntropyLoss().to(args.device)

    if args.finetune_mode == 'full':
        # We set a higher learning rate for the head
        adam_lr  = 1e-5
        if 'resnet' in args.model:

            params_1x = [param for name, param in net.named_parameters() if 'fc' not in str(name)]

            if args.optimizer == 'adamw':
                optimizer = torch.optim.AdamW([{'params':params_1x}, {'params': net.fc.parameters(), 'lr': adam_lr*10}], lr=adam_lr, weight_decay=args.weight_decay)
            elif args.optimizer == 'sgd':
                optimizer = torch.optim.SGD([{'params':params_1x}, {'params': net.fc.parameters(), 'lr': args.learning_rate*10}], lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
            else:
                raise NotImplementedError(args.optimizer)

        elif 'mobilenet' in args.model:

            params_1x = [param for name, param in net.named_parameters() if 'classifier' not in str(name)]

            if args.optimizer == 'adamw':
                optimizer = torch.optim.AdamW([{'params':params_1x}, {'params': net.classifier.parameters(), 'lr': adam_lr*10}], lr=adam_lr, weight_decay=args.weight_decay)
            elif args.optimizer == 'sgd':
                optimizer = torch.optim.SGD([{'params':params_1x}, {'params': net.classifier.parameters(), 'lr': args.learning_rate*10}], lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
            else:
                raise NotImplementedError(args.optimizer)
        else:
            raise NotImplementedError(args.model)

    else:
        if args.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(net.parameters())
        elif args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        else:
            raise NotImplementedError(args.optimizer)

    if args.distributed:
        devices = list(range(0, torch.cuda.device_count()))        
        devices.remove(args.n_gpu)
        net = nn.DataParallel(net, device_ids = [args.n_gpu, *devices])

    net.train()

    forward_times = []
    back_times = []

    for batch, data in enumerate(trainloader):
        inputs, labels = data
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        optimizer.zero_grad()
        start = time.time()
        outputs = net(inputs)
        end = time.time()
        forward_times.append(end - start)
        loss = criterion(outputs, labels)
        start = time.time()
        loss.backward()
        end = time.time()
        back_times.append(end - start)
        optimizer.step()

    forward_avg = sum(forward_times) / len(forward_times)
    back_avg = sum(back_times) / len(back_times)

    print(f"Forward avg time: {forward_avg}")
    print(f"Backward avg time: {back_avg}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train Dataset')

    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'mit_indoor', 'cub2011'])

    parser.add_argument('--model', type=str, default='resnet50', choices=['resnet50', 'resnet34', 'resnet18', 'mobilenetv2', 'crd_mobilenetv2'])
    parser.add_argument('--finetune_mode', type=str, default='full', choices=['full', 'head'])

    parser.add_argument('--seed', type=int, default=9)

    # optimization
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adamw'])
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')

    parser.add_argument('--train_bs', type=int, default=64)
    parser.add_argument('--test_bs', type=int, default=32)

    parser.add_argument('--n_gpu', type=int, default=0, help='index of gpu if multiple available')

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    d = json.load(open('./paths.json'))
    vars(args).update(d)

    args.device = torch.device(f"cuda:{args.n_gpu}" if torch.cuda.is_available() else "cpu")
    print(args.device)
    #args.distributed = torch.cuda.device_count() > 1
    args.distributed = False

    main(args)
