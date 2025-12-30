import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_attention_map(feature_maps, image_size=(512, 512)):
    """
    Generates a mock 'attention map' by averaging the activations 
    of the highest semantic level feature map (e.g., P5).
    
    args:
        feature_maps: Tensor [1, C, H_feat, W_feat]
        image_size: (H, W)
    """
    # Detach and move to cpu
    activations = feature_maps.detach().cpu()
    
    # Simple strategy: Mean activation across channels
    # [1, C, H, W] -> [H, W]
    heatmap = torch.mean(activations, dim=1).squeeze(0).numpy()
    
    # Normalize to 0-1
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (heatmap.max() + 1e-8)
    
    # Resize to original image size
    heatmap = cv2.resize(heatmap, (image_size[1], image_size[0]))
    
    # Colormap
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    return heatmap

def overlay_heatmap(image, heatmap, alpha=0.5):
    """
    Overlays heatmap on original image.
    image: uint8 [H, W, 3]
    """
    return cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
