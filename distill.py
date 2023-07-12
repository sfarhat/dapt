import argparse
import json
import os
import time
from kd_losses import align_uniform, crd, kd
from models.projector import Embed
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import datasets, utils
from utils.augment import DiffAugment, ParamDiffAug

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

    if args.aug and (args.aug_mode == 'S' or args.aug_mode == 'M'):
        aug_strategy = 'flip_color_crop_rotate_scale_cutout'

        args.dsa_strategy = aug_strategy
        dsa_param = ParamDiffAug()
        dsa_param.aug_mode = args.aug_mode

    if args.logging:
        wandb.init(project="DPT", 
                    config=args)

        for key in wandb.config._items:
            setattr(args, key, wandb.config._items[key])

    train_set, trainloader, _, _, _, testloader, _, num_classes, im_size = datasets.get_dataset(args)
    n_data = len(train_set)
    print(f"Train set size: {n_data}")

    foundation = utils.load_model(args, args.teacher_model, num_classes, args.teacher_init, args.teacher_optim)
    foundation.model.eval()
    for param in foundation.model.parameters():
        param.requires_grad = False
    foundation.projector.train()
    for param in foundation.projector.parameters():
        param.requires_grad = True

    model = utils.load_model(args, args.student_model, num_classes)
    model.model.train()

    # This is only necesary if we wish to use the teacher's classifier
    t_dim = foundation.feat_dim
    s_dim = model.feat_dim
    mid_dim = (s_dim + t_dim) // 2
    if args.connector == 'mlp':
        connector = nn.Sequential(nn.Linear(s_dim, t_dim),
                                    nn.ReLU(inplace=True))
        connector.train()
    elif args.connector == 'mlp2':
        connector = nn.Sequential(nn.Linear(s_dim, mid_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(mid_dim, t_dim),
                                    nn.ReLU(inplace=True))
        connector.train()
    else:
        connector = None

    if connector:
        # Projector should start from teacher's dimension
        model.projector = Embed(in_dim=foundation.feat_dim, hidden_dim=foundation.feat_dim, out_dim=128, projector=args.projector)
    model.projector.train()

    #if args.logging:
    #    wandb.watch(model, log='all', log_freq=1)

    params_in_path = f"{args.distill}_optimizer_{args.optimizer}_lambd1_{args.lambd1}_lambd2_{args.lambd2}_lambd3_{args.lambd3}_ \
                    lambd4_{args.lambd4}_projector{args.projector}"
    save_path = os.path.join(args.cpt_path, args.dataset, f"{args.student_model}_transferred_from_ \
                            {args.teacher_model}_{params_in_path}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    module_list = nn.ModuleList([model.model, model.projector])
    trainable_list = nn.ModuleList([model.model, model.projector])
    if 'moco' not in args.teacher_model:
        module_list.append(foundation.projector)
        trainable_list.append(foundation.projector)
    if connector:
        module_list.append(connector)
        trainable_list.append(connector)

    criterion_label = torch.nn.CrossEntropyLoss().to(args.device)
    if args.distill == 'align_uniform':
        criterion_kd = align_uniform.AlignUniform(model.get_projector(), foundation.get_projector(), args.align_alpha, 
                                                  args.unif_t, args.align_w, args.unif_w, logging=args.logging)
    elif args.distill == 'crd':
        criterion_kd = crd.CRD(model.get_projector(), foundation.get_projector(), args.feat_dim, args.nce_k, args.nce_t, args.nce_m, 
                               n_data)
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

    print("Validating Teacher")
    teacher_acc = validate(foundation, testloader, args)
    print(f"Teacher Accuracy: {teacher_acc}")

    best_acc = 0

    # Caches for teacher embeddings and logits so we only have to compute them once
    # Will incur a memory cost, but nothing crazy for reasonably sized datasets
    # cached_embeddings = torch.zeros((n_data, foundation.feat_dim), device=args.device)
    # cached_logits = torch.zeros((n_data, num_classes), device=args.device)

    for epoch in tqdm(range(args.epochs)):

        start = time.time()

        # TRAINING
        # (moved out of function so that DataParallel will work)

        # set modules as train()
        for module in trainable_list:
            module.train()

        # set teacher as eval()
        foundation.model.eval()

        for step, data in enumerate(trainloader, start=epoch * len(trainloader)):

            if args.distill == 'crd':
                inputs, labels, index, contrast_idx = data
                contrast_idx = contrast_idx.to(args.device)
            else:
                inputs, labels, index = data

            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            index = index.to(args.device)

            if args.aug and (args.aug_mode == 'S' or args.aug_mode == 'M'):
                inputs = DiffAugment(inputs, args.dsa_strategy, param=dsa_param)
                # This is simpler(?) augmentation
                # img = augment(img, dc_aug_param, device=device)

            feat_s, logit_s = model(inputs, is_feat=True)

            # On first epoch, cache the output of the teacher (embeddings and logits)
            # On future epochs, we can just fetch them from memory instead of another forward pass
            # if epoch == 0:
            with torch.no_grad():
                feat_t, logit_t = foundation(inputs, is_feat=True)
                feat_t = [f.detach() for f in feat_t]
            f_t = feat_t[-1]
            #     cached_embeddings[index] = f_t
            #     cached_logits[index] = logit_t
            # else:
            #     f_t = cached_embeddings[index]
            #     logit_t = cached_logits[index]

            f_s = feat_s[-1]
            if connector:
                # Using teacher's classifier
                f_s = connector(f_s)
                logit_s_t = foundation.get_classifier()(f_s)
                loss_srrl = nn.MSELoss()(logit_s_t, logit_t)
            else:
                loss_srrl = torch.tensor(0)

            if args.distill == 'crd':
                loss_embed = criterion_kd(f_s, f_t, index, contrast_idx)
            elif args.distill == 'kd':
                loss_embed = torch.tensor(0)
            else:
                loss_embed = criterion_kd(f_s, f_t)

            loss_label_s = criterion_label(logit_s, labels)

            loss_kd = kd.DistillKL(args.kd_T)(logit_s, logit_t)


            loss = args.lambd1 * loss_label_s + args.lambd2 * loss_embed + args.lambd3 * loss_kd + args.lambd4 * loss_srrl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.logging:

                _, predicted = torch.max(logit_s.data, 1)
                acc = (predicted == labels).sum().item() / labels.size(0)

                # grad_norm = 0
                # for p in model.parameters():
                #     grad_norm += torch.sum(p.grad**2)
                # grad_norm = grad_norm ** (1/2)

                wandb.log({"Train Accuracy": acc,
                        "Label Loss": loss_label_s.item(),
                        "Embedding Loss": loss_embed.item(),
                        "KD Loss": loss_kd.item(),
                        "SRRL Loss": loss_srrl.item()})
                        # "Gradient Magnitude": grad_norm.item()})

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

    parser.add_argument('--timing', action='store_true', help='config that times training, skips validation and saving')

    parser.add_argument('--dataset', type=str, default='dtd', choices=['cifar100', 'mit_indoor', 'cub2011', 'imagenet',
                                                                            'dtd', 'caltech101', 'cifar10'])

    # augmentation
    parser.add_argument('--aug', action='store_true', help='augment the training images')
    parser.add_argument('--aug_mode', default='S', choices=['S', 'M'], help='use single (S) or multiple (M) augmentations for each image')

    # teacher
    parser.add_argument('--teacher_model', type=str, default='resnet50', choices=['resnet50_moco', 'resnet50_moco_au', 'resnet50', 'vit-b-16'])
    parser.add_argument('--teacher_init', type=str, default='lp', choices=['ft', 'fr', 'lp', 'pt'])
    parser.add_argument('--teacher_optim', type=str, default='adamw', choices=['sgd', 'adamw'])

    # student
    parser.add_argument('--student_model', type=str, default='mobilenetv2', choices=['mobilenetv2', 'resnet18'])

    # distillation
    parser.add_argument('--distill', type=str, default='align_uniform', choices=['crd', 'align_uniform', 'srrl', 'kd'])

    parser.add_argument('--train_bs', type=int, default=64)
    parser.add_argument('--test_bs', type=int, default=32)

    parser.add_argument('--epochs', type=int, default=250, help='number of training epochs')

    parser.add_argument('--seed', type=int, default=9)

    # optimization
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['sgd', 'adamw'])
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

    # NCE distillation
    parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])
    parser.add_argument('--nce_k', default=4096, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')

    # Alignment and Uniformity for Contrastive Learning
    parser.add_argument('--align_w', type=float, default=1, help='Alignment loss weight')
    parser.add_argument('--unif_w', type=float, default=1, help='Uniformity loss weight')
    parser.add_argument('--align_alpha', type=float, default=2, help='alpha in alignment loss')
    parser.add_argument('--unif_t', type=float, default=2, help='t in uniformity loss')

    # Projector for contrastive learning
    parser.add_argument('--projector', type=str, default='mocov2', choices=['simclrv2', 'simclr', 'mocov2', 'linear'], help='The architecture of the non-linear projector for contrastive learning')

    # Connector for SRRL
    parser.add_argument('--connector', type=str, default='mlp2', choices=['mlp2', 'mlp'])

    # weight tradeoff
    # lambd1 for label loss
    parser.add_argument('--lambd1', type=float, default=1)
    # lambd2 for embedding loss
    parser.add_argument('--lambd2', type=float, default=.8)
    # lambd3 for KL between student and teacher logits
    parser.add_argument('--lambd3', type=float, default=1)
    # lambd4 for SRRL loss between student and teacher logits
    parser.add_argument('--lambd4', type=float, default=1)

    parser.add_argument('--data_path', type=str)
    parser.add_argument('--cpt_path', type=str)

    parser.add_argument('--logging', action='store_true')
    parser.add_argument('--no-logging', dest='logging', action='store_false')
    parser.set_defaults(logging=True)

    parser.add_argument('--n_gpu', type=int, default=0, help='index of gpu if multiple available')

    args = parser.parse_args()

    # CRD paper does this for MobileNet
    if 'mobile' in args.student_model or 'shuffle' in args.student_model:
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
    # args.distributed = torch.cuda.device_count() > 1
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
        args.lambd4 = 0
    elif args.distill == 'srrl':
        args.lambd1 = 1
        args.lambd2 = 1
        args.lambd3 = 0
        args.lambd4 = 1
    elif args.distill in ('align_uniform', 'crd'):
        args.lambd1 = 1
        args.lambd2 = 1
        args.lambd3 = 1
        args.lambd4 = 0

    if args.lambd4 == 0:
        args.connector = None

    main(args)
