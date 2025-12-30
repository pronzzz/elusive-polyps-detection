import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

def xywh_to_xyxy(boxes):
    """Convert [x, y, w, h] to [x1, y1, x2, y2]."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = x1 + boxes[:, 2]
    y2 = y1 + boxes[:, 3]
    return torch.stack([x1, y1, x2, y2], dim=1)

def xyxy_to_xywh(boxes):
    """Convert [x1, y1, x2, y2] to [x, y, w, h]."""
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    x = boxes[:, 0]
    y = boxes[:, 1]
    return torch.stack([x, y, w, h], dim=1)

def plot_image_with_boxes(image, boxes, scores=None, threshold=0.5):
    """
    Plots image with bounding boxes.
    image: HWC numpy array (0-255 or 0-1)
    boxes: [N, 4] (x1, y1, x2, y2)
    scores: [N] confidence scores (optional)
    """
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
        # Denormalize if needed, assuming simple 0-1 range for visualization if roughly valid
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
    
    image = image.copy()
    
    for i, box in enumerate(boxes):
        if scores is not None and scores[i] < threshold:
            continue
            
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        if scores is not None:
            text = f"{scores[i]:.2f}"
            cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.show()
