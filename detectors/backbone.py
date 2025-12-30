import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool

class ResNetFPNBackbone(nn.Module):
    """
    ResNet backbone with FPN.
    Returns:
        OrderedDict of feature maps [feat0, feat1, feat2, feat3]
        corresponding to decreasing spatial resolution.
    """
    def __init__(self, backbone_name='resnet50', pretrained=True, trainable_layers=3):
        super(ResNetFPNBackbone, self).__init__()
        
        # Load backbone
        backbone = models.resnet50(pretrained=pretrained) if backbone_name == 'resnet50' else models.resnet18(pretrained=pretrained)
        
        # Select layers to extract features from
        # C2, C3, C4, C5
        self.body = nn.ModuleDict({
            'layer1': backbone.layer1, # stride 4
            'layer2': backbone.layer2, # stride 8
            'layer3': backbone.layer3, # stride 16
            'layer4': backbone.layer4  # stride 32
        })
        
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )
        
        # FPN
        in_channels_list = [256, 512, 1024, 2048] if backbone_name == 'resnet50' else [64, 128, 256, 512]
        self.out_channels = 256
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=self.out_channels,
            extra_blocks=LastLevelMaxPool()
        )
        
    def forward(self, x):
        # Forward pass through ResNet
        x = self.stem(x)
        
        c2 = self.body['layer1'](x)
        c3 = self.body['layer2'](c2)
        c4 = self.body['layer3'](c3)
        c5 = self.body['layer4'](c4)
        
        # FPN
        features = {
            '0': c2,
            '1': c3,
            '2': c4,
            '3': c5
        }
        
        out_features = self.fpn(features)
        return out_features
