import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import sys
import os
import math
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from detectors.backbone import ResNetFPNBackbone
from detectors.retinanet import RetinaNet
from detectors.temporal_model import TemporalRetinaNet
from detectors.loss import FocalLoss
from data.sequence_generator import SyntheticSequenceGenerator

# --- Anchor Generator ---
class AnchorGenerator:
    """
    Generates anchors for RetinaNet.
    Simplified version for demo: 3 scales * 3 aspect ratios = 9 anchors per location.
    """
    def __init__(self, sizes=[32, 64, 128, 256, 512], ratios=[0.5, 1.0, 2.0], scales=[1.0, 1.26, 1.58]):
        self.sizes = sizes
        self.ratios = ratios
        self.scales = scales
        
    def generate_anchors(self, feature_map_sizes, image_size):
        """
        feature_map_sizes: list of (H, W) for each level
        image_size: (H, W)
        """
        anchors_all = []
        
        # Stride info
        strides = [image_size[0] / f[0] for f in feature_map_sizes]
        
        for level, (grid_h, grid_w) in enumerate(feature_map_sizes):
            base_size = self.sizes[level]
            stride = strides[level]
            
            # Generate base anchors for this level
            level_anchors = []
            for y in range(grid_h):
                for x in range(grid_w):
                    c_x = (x + 0.5) * stride
                    c_y = (y + 0.5) * stride
                    
                    for r in self.ratios:
                        for s in self.scales:
                            w = base_size * s * math.sqrt(r)
                            h = base_size * s / math.sqrt(r)
                            level_anchors.append([c_x - w/2, c_y - h/2, c_x + w/2, c_y + h/2])
                            
            anchors_all.append(torch.tensor(level_anchors))
            
        return torch.cat(anchors_all, dim=0)

# --- Matcher ---
def match_anchors_to_targets(anchors, targets, iou_threshold=0.5):
    """
    targets: [N, 4]
    anchors: [M, 4]
    Returns:
        matched_gt_boxes: [M, 4]
        matched_labels: [M] (1=fg, 0=bg, -1=ignore)
    """
    if len(targets) == 0:
        return torch.zeros_like(anchors), torch.zeros(len(anchors), dtype=torch.long)
        
    # IoU Matrix
    lt = torch.max(anchors[:, None, :2], targets[:, :2])
    rb = torch.min(anchors[:, None, 2:], targets[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    
    area_a = (anchors[:, 2] - anchors[:, 0]) * (anchors[:, 3] - anchors[:, 1])
    area_t = (targets[:, 2] - targets[:, 0]) * (targets[:, 3] - targets[:, 1])
    union = area_a[:, None] + area_t - inter
    
    iou = inter / (union + 1e-6) # [M, N]
    
    # Best match for each anchor
    vals, inds = iou.max(dim=1)
    
    matched_gt_boxes = targets[inds]
    matched_labels = torch.zeros(len(anchors), dtype=torch.long)
    matched_labels[vals > iou_threshold] = 1 # Examples
    matched_labels[vals < 0.4] = 0 # Background
    matched_labels[(vals >= 0.4) & (vals <= iou_threshold)] = -1 # Ignore
    
    return matched_gt_boxes, matched_labels

# --- Dummy Dataset for Demo ---
class RandomSequenceDataset(Dataset):
    def __init__(self, length=100):
        self.length = length
        self.gen = SyntheticSequenceGenerator(sequence_length=5)
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        # Generate random image
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        # Random box
        x = np.random.randint(0, 200)
        y = np.random.randint(0, 200)
        w = np.random.randint(20, 50)
        h = np.random.randint(20, 50)
        boxes = np.array([[x, y, x+w, y+h]])
        
        frames, box_seq = self.gen.generate(img, boxes)
        
        # Normalize
        frames_t = torch.stack([torch.from_numpy(f).float().permute(2,0,1)/255.0 for f in frames])
        
        # Boxes need strictly (x1,y1,x2,y2) tensor list
        boxes_list = [torch.tensor(b, dtype=torch.float32) for b in box_seq]
        
        return frames_t, boxes_list

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    
    anchor_gen = AnchorGenerator(sizes=[16, 32, 64, 128, 256]) # Scaled down for 256x256
    focal_loss = FocalLoss()
    reg_loss_fn = nn.SmoothL1Loss(reduction='sum')
    
    for i, (images, targets) in enumerate(loader):
        # images: [B, T, 3, H, W]
        B, T, C, H, W = images.shape
        images = images.to(device)
        
        optimizer.zero_grad()
        
        # Forward Pass
        cls_logits, bbox_preds = model.forward_sequence(images)
        # Returns flattened outputs: list[Level] -> [B*T, A*C, H', W']
        
        # Flatten anchors generation
        # Get feature map sizes from model or calculate dynamically
        # For simplicity, assume standard downsampling 4, 8, 16, 32...
        feat_sizes = [(math.ceil(H / (2**(l+2))), math.ceil(W / (2**(l+2)))) for l in range(len(cls_logits))]
        anchors = anchor_gen.generate_anchors(feat_sizes, (H, W)).to(device)
        
        # Reshape outputs to [B*T, TotalAnchors, Features]
        # We need to concat all levels
        
        all_cls = []
        all_reg = []
        for l in range(len(cls_logits)):
            # [B*T, A*C, H, W] -> [B*T, H, W, A*C] -> [B*T, H*W*A, C]
            c = cls_logits[l].permute(0, 2, 3, 1).contiguous().view(B*T, -1, 1) # 1 class
            r = bbox_preds[l].permute(0, 2, 3, 1).contiguous().view(B*T, -1, 4)
            all_cls.append(c)
            all_reg.append(r)
            
        all_cls = torch.cat(all_cls, dim=1) # [B*T, TotalAnchors, 1]
        all_reg = torch.cat(all_reg, dim=1)
        
        # Calculate Loss per frame
        loss = 0
        # flatten targets B list of T lists -> B*T list
        flat_targets = []
        for b in range(B):
            flat_targets.extend(targets[b]) # targets[b] is a list of T tensors
            
        valid_samples = 0
        for idx in range(B*T):
            gt_boxes = flat_targets[idx].to(device)
            # Only if gt exists
            if len(gt_boxes) == 0:
                continue
                
            matched_boxes, labels = match_anchors_to_targets(anchors, gt_boxes)
            
            # Select relevant samples
            pos_mask = labels == 1
            valid_mask = labels != -1
            
            # Classification Loss
            cls_pred_frame = all_cls[idx].squeeze(-1) # [TotalAnchors]
            loss += focal_loss(cls_pred_frame[valid_mask], labels[valid_mask].float())
            
            # Regression Loss (only on positives)
            if pos_mask.sum() > 0:
                reg_pred = all_reg[idx][pos_mask]
                reg_target = matched_boxes[pos_mask] 
                # Note: should encode target to deltas, skipping for brevity
                loss += reg_loss_fn(reg_pred, reg_target) # Simplified
                
            valid_samples += 1
            
        if valid_samples > 0:
            loss = loss / valid_samples
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if i % 5 == 0:
            print(f"Batch {i}, Loss: {loss.item() if valid_samples > 0 else 0:.4f}")
            
    return total_loss / len(loader)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Init
    backbone = ResNetFPNBackbone(backbone_name='resnet18', pretrained=False)
    retinanet = RetinaNet(backbone)
    model = TemporalRetinaNet(retinanet).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    # 2. Data
    # Use synthetic random data for functional verification
    dataset = RandomSequenceDataset(length=20)
    # Collate fn needed for list of lists
    def collate_fn(batch):
        images = torch.stack([b[0] for b in batch])
        targets = [b[1] for b in batch]
        return images, targets
        
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    
    print("Starting Training Loop...")
    for epoch in range(1):
        loss = train_epoch(model, loader, optimizer, device)
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
    print("Training Verification Complete.")

if __name__ == "__main__":
    main()
