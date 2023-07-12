from collections import OrderedDict
from torchvision.models.vision_transformer import vit_b_16
import torch
import torch.nn as nn

class ViT_B_16(nn.Module):

    def __init__(self, weights=None, progress=True, **kwargs):

        super().__init__()

        self.model = vit_b_16(weights=weights, progress=progress, **kwargs)

        self.feat_dim = self.get_classifier()[0].in_features
        self.projector = None

    def get_classifier(self):
        return self.model.heads

    def set_classifier(self, num_classes):

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if self.model.representation_size is None:
            heads_layers["head"] = nn.Linear(self.feat_dim, num_classes)
        else:
            heads_layers["pre_logits"] = nn.Linear(self.feat_dim, self.model.representation_size)
            heads_layers["act"] = nn.Tanh()
            heads_layers["head"] = nn.Linear(self.model.representation_size, num_classes)

        self.model.heads = nn.Sequential(heads_layers)

    def get_projector(self):
        return self.projector

    def forward(self, x, is_feat=False):

        # Reshape and permute the input tensor
        x = self.model._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.model.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]
        f0 = x

        x = self.model.heads(x)

        if is_feat:
            return [f0], x
        return x
