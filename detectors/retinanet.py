import torch
import torch.nn as nn
import math

class ClassificationHead(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes):
        super(ClassificationHead, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.conv = nn.ModuleList([nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1) for _ in range(4)])
        self.cls_logits = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        # Initialization
        for layer in self.conv:
            torch.nn.init.normal_(layer.weight, std=0.01)
            torch.nn.init.constant_(layer.bias, 0)
        
        torch.nn.init.normal_(self.cls_logits.weight, std=0.01)
        # Initialize bias to favor background (focal loss trick)
        prob = 0.01
        bias_value = -math.log((1 - prob) / prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, x):
        for layer in self.conv:
            x = self.relu(layer(x))
        return self.cls_logits(x)

class RegressionHead(nn.Module):
    def __init__(self, in_channels, num_anchors):
        super(RegressionHead, self).__init__()
        self.conv = nn.ModuleList([nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1) for _ in range(4)])
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        for layer in self.conv:
            torch.nn.init.normal_(layer.weight, std=0.01)
            torch.nn.init.constant_(layer.bias, 0)
        torch.nn.init.normal_(self.bbox_pred.weight, std=0.01)
        torch.nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x):
        for layer in self.conv:
            x = self.relu(layer(x))
        return self.bbox_pred(x)

class RetinaNet(nn.Module):
    def __init__(self, backbone, num_classes=1, num_anchors=9):
        super(RetinaNet, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        
        in_channels = backbone.out_channels # 256 for FPN
        
        self.cls_head = ClassificationHead(in_channels, num_anchors, num_classes)
        self.reg_head = RegressionHead(in_channels, num_anchors)
        
    def forward(self, x):
        # x: [B, 3, H, W] or [B, T, 3, H, W]?
        # Standard RetinaNet works on batches of frames
        features = self.backbone(x)
        # features is Dict: '0', '1', '2', '3', 'pool'
        
        cls_logits = []
        bbox_preds = []
        
        for feat_name, feat_map in features.items():
            cls_logits.append(self.cls_head(feat_map))
            bbox_preds.append(self.reg_head(feat_map))
            
        return cls_logits, bbox_preds, features 
        # Returning features is KEY for the temporal module
