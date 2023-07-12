import argparse
import json
import os
import time
from kd_losses import align_uniform, kd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from utils.augment import DiffAugment, ParamDiffAug
from utils import datasets, utils

def validate(model, testloader, args):

    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total

    return acc

def main(args):

    if args.aug:
        aug_strategy = 'flip_color_crop_rotate_scale_cutout'

        args.dsa_strategy = aug_strategy
        dsa_param = ParamDiffAug()
        dsa_param.aug_mode = args.aug_mode

    if args.logging:
        wandb.init(project="Foundation Assistant (Generative)",
                    config=args)

        for key in wandb.config._items:
            setattr(args, key, wandb.config._items[key])

    syn_train = datasets.SyntheticDataset(args.syn_data_path, args.dataset, args.synset_size)
    print(f'==> Synthetic Training data loaded.. Size: {len(syn_train)}')

    train_set, trainloader, val_set, valloader, test_set, testloader, channel, num_classes, im_size = datasets.get_dataset(args)
    print(f"Train set size: {len(train_set)}")

    combined_ds = ConcatDataset([train_set, syn_train])
    trainloader = DataLoader(combined_ds, batch_size=args.train_bs, shuffle=True)

    foundation = utils.load_model(args, args.teacher_model, num_classes, args.teacher_init, args.teacher_optim)
    foundation.model.eval()
    for param in foundation.model.parameters():
        param.requires_grad = False
    foundation.projector.train()
    for param in foundation.projector.parameters():
        param.requires_grad = True

    model = utils.load_model(args, args.student_model, num_classes)
    model.model.train()
    model.projector.train()

    params_in_path = f"{args.distill}_optimizer_{args.optimizer}_lambd1_{args.lambd1}_lambd2_{args.lambd2}_lambd3_{args.lambd3}_ \
                    _projector{args.projector}"
    save_path = os.path.join(args.cpt_path, args.dataset, f"{args.synset_size}_generated_{args.student_model}_transferred_from_ \
                                {args.teacher_init}_{args.teacher_model}_{args.teacher_optim}_{params_in_path}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    module_list = nn.ModuleList([model.model, model.projector])
    trainable_list = nn.ModuleList([model.model, model.projector])
    if 'moco' not in args.teacher_model:
        module_list.append(foundation.projector)
        trainable_list.append(foundation.projector)

    criterion_label = torch.nn.CrossEntropyLoss().to(args.device)
    if args.distill == 'align_uniform':
        criterion_kd = align_uniform.AlignUniform(model.get_projector(), foundation.get_projector(), args.align_alpha, 
                                                  args.unif_t, args.align_w, args.unif_w, logging=args.logging)
    else:
        criterion_kd = nn.MSELoss()

    module_list.append(foundation.model)

    module_list.to(args.device)
    criterion_label.to(args.device)
    criterion_kd.to(args.device)

    if args.distributed:
        devices = list(range(0, torch.cuda.device_count()))        
        devices.remove(args.n_gpu)
        foundation = nn.DataParallel(foundation, device_ids = [args.n_gpu, *devices])
        model = nn.DataParallel(model, device_ids = [args.n_gpu, *devices])

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(trainable_list.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(trainable_list.parameters())
    else:
        raise NotImplementedError(args.optimizer)

    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_decay_epochs, gamma=args.lr_decay_rate)
    elif args.scheduler == 'none':
        scheduler = None
    else:
        raise NotImplementedError(args.scheduler)

    # Validating Teacher Acc
    print("Validating Teacher")
    teacher_acc = validate(foundation, testloader, args)
    print(f"Teacher Accuracy: {teacher_acc}")

    best_acc = 0
    for epoch in tqdm(range(args.epochs)):

        start = time.time()

        # TRAINING
        # (moved out of function so that DataParallel will work)

        # set modules as train()
        for module in module_list:
            module.train()

        # set teacher as eval()
        foundation.model.eval()

        for step, data in enumerate(trainloader, start=epoch * len(trainloader)):

            inputs, labels, index = data

            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            index = index.to(args.device)

            if args.aug:
                inputs = DiffAugment(inputs, args.dsa_strategy, param=dsa_param)

            feat_s, logit_s = model(inputs, is_feat=True)

            with torch.no_grad():
                feat_t, logit_t = foundation(inputs, is_feat=True)
                feat_t = [f.detach() for f in feat_t]
            f_t = feat_t[-1]

            f_s = feat_s[-1]

            if args.distill == 'kd':
                loss_embed = torch.tensor(0)
            else:
                loss_embed = criterion_kd(f_s, f_t)

            loss_label_s = criterion_label(logit_s, labels)

            loss_kd = kd.DistillKL(args.kd_T)(logit_s, logit_t)

            loss = args.lambd1 * loss_label_s + args.lambd2 * loss_embed + args.lambd3 * loss_kd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.logging:

                _, predicted = torch.max(logit_s.data, 1)
                acc = (predicted == labels).sum().item() / labels.size(0)

                wandb.log({"Train Accuracy": acc,
                        "Label Loss": loss_label_s.item(),
                        "Embedding Loss": loss_embed.item(),
                        "KD Loss": loss_kd.item()})

        if scheduler:
            scheduler.step()
            wandb.log({"Learning Rate": scheduler.get_last_lr()[0]})

        if args.timing:
            torch.cuda.current_stream().synchronize()
            end = time.time()
            print(f"Time: {end - start}")

        else:
            acc = validate(model, testloader, args)

            if args.logging:
                wandb.log({"Test Accuracy": acc,
                        "Iteration": epoch})

            if acc > best_acc:
                best_acc = acc
                utils.save_model(args, epoch, model, optimizer, scheduler, acc, save_path, cpt_type='best')
                if args.logging:
                    wandb.log({"Best Accuracy": best_acc})

            utils.save_model(args, epoch, model, optimizer, scheduler, acc, save_path, cpt_type='last')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Assisting Learning w/ Foundations')

    parser.add_argument('--dataset', type=str, default='mit_indoor', choices=['mit_indoor', 'dtd', 'caltech101'])
    parser.add_argument('--synset_size', type=str, default='1x', choices=['1x', '2x'])

    parser.add_argument('--timing', action='store_true', help='config that times training, skips validation and saving')

    # augmentation
    parser.add_argument('--aug', action='store_true', help='augment the training images')
    parser.add_argument('--aug_mode', default='S', choices=['S', 'M'], help='use single (S) or multiple (M) augmentations for each image')

    # teacher
    parser.add_argument('--teacher_model', type=str, default='resnet50', choices=['resnet50', 'vit-b-16'])
    parser.add_argument('--teacher_init', type=str, default='lp', choices=['fr', 'lp'])
    parser.add_argument('--teacher_optim', type=str, default='adamw', choices=['sgd', 'adamw'])

    # student
    parser.add_argument('--student_model', type=str, default='mobilenetv2', choices=['mobilenetv2', 'resnet18'])

    # distillation
    parser.add_argument('--distill', type=str, default='align_uniform', choices=['align_uniform', 'kd'])

    parser.add_argument('--train_bs', type=int, default=64)
    parser.add_argument('--test_bs', type=int, default=64)

    parser.add_argument('--epochs', type=int, default=250, help='number of training epochs')

    parser.add_argument('--seed', type=int, default=9)

    # optimization
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adamw'])
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    parser.add_argument('--scheduler', type=str, default='none', choices=['step', 'cosine', 'none'])
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')

    # Alignment dimension
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')

    # Alignment and Uniformity for Contrastive Learning
    parser.add_argument('--align_w', type=float, default=1, help='Alignment loss weight')
    parser.add_argument('--unif_w', type=float, default=1, help='Uniformity loss weight')
    parser.add_argument('--align_alpha', type=float, default=2, help='alpha in alignment loss')
    parser.add_argument('--unif_t', type=float, default=2, help='t in uniformity loss')

    # Projector for contrastive learning
    parser.add_argument('--projector', type=str, default='mocov2', choices=['simclrv2', 'simclr', 'mocov2', 'linear'], help='The architecture of the non-linear projector for contrastive learning')

    # weight tradeoff
    # lambd1 for label loss
    parser.add_argument('--lambd1', type=float, default=1)
    # lambd2 for embedding loss
    parser.add_argument('--lambd2', type=float, default=.8)
    # lambd3 for KL between student and teacher logits
    parser.add_argument('--lambd3', type=float, default=1)

    parser.add_argument('--data_path', type=str)
    parser.add_argument('--syn_data_path', type=str)
    parser.add_argument('--cpt_path', type=str)

    parser.add_argument('--logging', action='store_true')
    parser.add_argument('--no-logging', dest='logging', action='store_false')
    parser.set_defaults(logging=True)

    parser.add_argument('--n_gpu', type=int, default=0, help='index of gpu if multiple available')

    args = parser.parse_args()

    if 'mobile' in args.student_model:
        args.learning_rate = 0.01

    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))

    if args.timing:
        args.epochs = 1
        args.logging = False

    d = json.load(open('./paths.json'))
    vars(args).update(d)

    args.device = torch.device(f"cuda:{args.n_gpu}" if torch.cuda.is_available() else "cpu")
    print(args.device)
    args.distributed = torch.cuda.device_count() > 1
    args.distributed = False

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True # keep True if all the input have same size.

    if args.logging:
        import wandb

    if args.distill == 'kd':
        args.lambd1 = 1
        args.lambd2 = 0
        args.lambd3 = 1
    elif args.distill == 'align_uniform':
        args.lambd1 = 1
        args.lambd2 = 1
        args.lambd3 = 1

    main(args)
