from torchvision.models.resnet import resnet50, resnet18
import torch
import torch.nn as nn

class ResNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = None
        self.projector = None

    def get_classifier(self):
        return self.model.fc

    def set_classifier(self, num_classes):
        self.model.fc = nn.Linear(self.feat_dim, num_classes)
    
    def get_projector(self):
        return self.projector
        
    def forward(self, x, is_feat=False):

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        f0 = x
        x = self.model.fc(x)

        if is_feat:
            return [f0], x

        return x

class ResNet50(ResNet):

    def __init__(self, weights=None, progress=True, **kwargs):
        super().__init__()

        self.model = resnet50(weights=weights, progress=progress, **kwargs)
        self.feat_dim = self.get_classifier().in_features

class ResNet18(ResNet):

    def __init__(self, weights=None, progress=True, **kwargs):
        super().__init__()

        self.model = resnet18(weights=weights, progress=progress, **kwargs)
        self.feat_dim = self.get_classifier().in_features
