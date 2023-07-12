import os
import torch
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights, ResNet18_Weights, MobileNet_V2_Weights

from models import mobilenetv2, resnet, vit
from models.projector import _Normalize, Embed

def load_model(args, model_name, num_classes, model_init=None, model_optim=None):

    # If we wish to load a pre-trained model, we specify model_init and model_optim

    if 'moco' in model_name:

        # Code adapted from https://github.com/ssnl/moco_align_uniform/blob/align_uniform/hubconf.py

        # This is an ResNet50 backbone that has an extra linear (2048 -> 2048) -> ReLU -> linear (2048 -> 128) -> normalize
        # It has an encoder_q backbone and the fc projector
        # This matches the form of the moco checkpoint where the embedding mlp is absorbed into the 'fc' layer

        fnames = {'resnet50_moco': 'moco_v2_200ep_pretrain.pth.tar',
                'resnet50_moco_au': 'imagenet_align_uniform.pth.tar'}

        teacher_path = os.path.join(args.cpt_path, 'imagenet', fnames[model_name])

        model = resnet.ResNet50(num_classes=128)
        dim_mlp = model.feat_dim
        model.model.fc = torch.nn.Sequential(
            torch.nn.Linear(dim_mlp, dim_mlp),
            torch.nn.ReLU(),
            model.model.fc,
            _Normalize(),
        )

        ckpt = torch.load(teacher_path)

        state_dict = {}
        for k, v in ckpt['state_dict'].items():
            if k.startswith('module.encoder_q.'):
                state_dict[k.split('.', 2)[-1]] = v

        model.model.load_state_dict(state_dict)

        # Added to fit in with our projector/classifier naming scheme
        model.projector = model.model.fc
        model.set_classifier(num_classes)

        return model

    if model_init == 'pt':
        # If pre-trained, then stick a dummy classifier
        # When linearly probing, we use this
        # When avoiding teacher head when assisting, we use this
        if model_name == 'resnet50':
            model = resnet.ResNet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        elif model_name == 'vit-b-16':
            model = vit.ViT_B_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        elif model_name == 'resnet18':
            model = resnet.ResNet18(weights=ResNet18_Weights.DEFAULT)
        elif model_name == 'mobilenetv2':
            model = mobilenetv2.MobileV2(weights=MobileNet_V2_Weights.DEFAULT)
        else:
            raise NotImplementedError(f"{model_name}: {model_init}")
        model.set_classifier(num_classes)
    else:
        if model_name == 'resnet50':
            model = resnet.ResNet50(num_classes=num_classes)
        elif model_name == 'vit-b-16':
            model = vit.ViT_B_16(num_classes=num_classes)
        elif model_name == 'mobilenetv2':
            model = mobilenetv2.MobileV2(num_classes=num_classes)
        elif model_name == 'resnet18':
            model = resnet.ResNet18(num_classes=num_classes)
        else:
            raise NotImplementedError(model_name)

        if model_init == 'lp' or model_init == 'ft' or model_init == 'fr':
            # Loads a model that was trained for a task, so classifier is correct
            pretrained_path = os.path.join(args.cpt_path, args.dataset, f"{model_name}_{model_init}_{model_optim}")
            foundation_cpt = torch.load(f"{pretrained_path}/best_checkpoint.pt", map_location=args.device)
            model.model.load_state_dict(foundation_cpt["model_state_dict"])

    model.projector = Embed(in_dim=model.feat_dim, hidden_dim=model.feat_dim, out_dim=128, projector=args.projector)
    return model

def save_model(args, epoch, model, optimizer, scheduler, acc, save_path, cpt_type='last'):

    cpt_path = f"{save_path}/{cpt_type}_checkpoint.pt"

    if args.distributed:
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    if scheduler is None:
        sch_state_dict = None
    else:
        sch_state_dict = scheduler.state_dict()

    torch.save({
        'epoch': epoch,
        'model_state_dict': state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': sch_state_dict,
        'accuracy': acc,
    }, cpt_path)
