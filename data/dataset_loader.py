import os
import torch
import cv2
import glob
import numpy as np
from torch.utils.data import Dataset
from .sequence_generator import SyntheticSequenceGenerator

class CVCClinicDBDataset(Dataset):
    """
    Loader for CVC-ClinicDB dataset.
    Expected structure:
    root/
      Original/
        1.png, 2.png...
      Ground Truth/
        1.png, 2.png... (masks)
    """
    def __init__(self, root_dir, augmentations=None, use_synthetic_sequence=False, sequence_length=5):
        self.root_dir = root_dir
        self.augmentations = augmentations
        self.use_synthetic_sequence = use_synthetic_sequence
        self.sequence_gen = SyntheticSequenceGenerator(sequence_length=sequence_length)
        
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, "Original", "*.png")))
        self.mask_paths = sorted(glob.glob(os.path.join(root_dir, "Ground Truth", "*.png")))
        
        print(f"Found {len(self.image_paths)} images in {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def mask_to_boxes(self, mask):
        # Mask is HxW, assumed binary or near binary
        # We need to find connected components
        # CVC masks are 255 for polyp, 0 for bg
        if mask.max() < 1: # Handle empty masks
             return np.zeros((0, 4))
             
        mask = (mask > 128).astype(np.uint8)
        num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(mask)
        
        boxes = []
        for i in range(1, num_labels): # 0 is background
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            boxes.append([x, y, x+w, y+h])
            
        return np.array(boxes)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        boxes = self.mask_to_boxes(mask)

        if self.use_synthetic_sequence:
            frames, box_seq = self.sequence_gen.generate(image, boxes)
            # Normalize and Totensor
            frames_t = [torch.from_numpy(f).float().permute(2,0,1) / 255.0 for f in frames]
            return torch.stack(frames_t), box_seq # Return sequence

        # Single frame mode
        if self.augmentations:
            # Albumentations implementation would go here
            pass
            
        image_t = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        
        # Target dict for standard detection training
        target = {}
        target['boxes'] = torch.tensor(boxes, dtype=torch.float32)
        target['labels'] = torch.ones((len(boxes),), dtype=torch.int64) # Class 1 = Polyp
        
        return image_t, target
