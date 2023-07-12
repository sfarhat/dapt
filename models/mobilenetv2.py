from torchvision.models.mobilenetv2 import mobilenet_v2
import torch
import torch.nn as nn

class MobileV2(nn.Module):

    def __init__(self, weights=None, progress=True, **kwargs):
        super().__init__()

        self.model = mobilenet_v2(weights=weights, progress=progress, **kwargs)

        self.feat_dim = self.get_classifier()[1].in_features
        self.projector = None

    def get_classifier(self):
        return self.model.classifier

    def set_classifier(self, num_classes):
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.feat_dim, num_classes)
        )

    def get_projector(self):
        return self.projector

    def forward(self, x, is_feat=False):
        # A modified forward that also spits out the penultimate features

        x = self.model.features(x)

        f0 = x
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        f1 = x
        x = self.model.classifier(x)

        if is_feat:
            return [f0, f1], x
        else:
            return x
