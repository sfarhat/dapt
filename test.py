import argparse
import json
import os
import torch
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
    print(args.model_init)
    print(args.model_opt)

    train_set, trainloader, _, _, _, testloader, _, num_classes, im_size = datasets.get_dataset(args)

    cpt_path = os.path.join(args.cpt_path, args.dataset, f"{args.model}_{args.model_init}_{args.model_opt}")
    cpt = torch.load(f"{cpt_path}/best_checkpoint.pt", map_location=args.device)

    net = utils.load_model(args, args.model, num_classes, args.model_init, args.model_opt)

    net.to(args.device)

    #net.load_state_dict(cpt["model_state_dict"])
    print(f"Best accuracy: {cpt['accuracy']}")

    #last_cpt = torch.load(f"{cpt_path}/last_checkpoint.pt", map_location=args.device)
    #print(f"Last accuracy: {last_cpt['accuracy']}")

    print("Validating Teacher")
    teacher_acc = validate(net, testloader, args)
    print(f"Teacher Accuracy: {teacher_acc}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Testing Models')

    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar10', 'cifar100', 'mit_indoor', 'cub2011', 'dtd', 'caltech101'])

    # teacher
    parser.add_argument('--model', type=str, default='resnet50', choices=['resnet50', 'resnet18', 'mobilenetv2', 'resnet50_moco', 'resnet50_moco_au',
                                                                          'vit-b-16'])
    parser.add_argument('--model_init', type=str, default='fr', choices=['fr', 'lp'])
    parser.add_argument('--model_opt', type=str, default='adamw', choices=['sgd', 'adamw'])

    parser.add_argument('--projector', type=str, default='mocov2', choices=['simclrv2', 'simclr', 'mocov2', 'linear'], help='The architecture of the non-linear projector for contrastive learning')

    parser.add_argument('--train_bs', type=int, default=64)
    parser.add_argument('--test_bs', type=int, default=32)

    parser.add_argument('--seed', type=int, default=9)

    parser.add_argument('--data_path', type=str)
    parser.add_argument('--cpt_path', type=str)

    parser.add_argument('--logging', action='store_true')
    parser.add_argument('--no-logging', dest='logging', action='store_false')
    parser.set_defaults(logging=True)

    parser.add_argument('--n_gpu', type=int, default=0, help='index of gpu if multiple available')

    args = parser.parse_args()

    d = json.load(open('./paths.json'))
    vars(args).update(d)

    args.device = torch.device(f"cuda:{args.n_gpu}" if torch.cuda.is_available() else "cpu")
    print(args.device)
    #args.distributed = torch.cuda.device_count() > 1
    args.distributed = False

    torch.manual_seed(args.seed)

    main(args)
