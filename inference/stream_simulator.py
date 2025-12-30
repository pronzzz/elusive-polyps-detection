import torch
import cv2
import numpy as np
import time
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from detectors.backbone import ResNetFPNBackbone
from detectors.retinanet import RetinaNet
from detectors.temporal_model import TemporalRetinaNet
from fusion.fusion_engine import FusionEngine
from data.sequence_generator import SyntheticSequenceGenerator
from utils.explainability import generate_attention_map, overlay_heatmap

def simulate_stream():
    """
    Generates synthetic frames on the fly and runs the temporal model.
    Visualizes the result in a 'live' window (or saves to video).
    """
    print("Setting up Clinical Assistant Simulation...")
    
    # 1. Initialize Model
    backbone = ResNetFPNBackbone(backbone_name='resnet50', pretrained=False) # No download for demo speed
    retinanet = RetinaNet(backbone)
    model = TemporalRetinaNet(retinanet)
    model.eval()
    
    # 2. Initialize Fusion Engine
    fusion = FusionEngine(history_length=5, confirmation_threshold=3)
    
    # 3. Create dummy data source
    # Create a dummy "polyp" image (red blob on pink background)
    H, W = 512, 512
    base_image = np.full((H, W, 3), (200, 200, 255), dtype=np.uint8) # Pinkish
    # Add some noise/texture
    noise = np.random.normal(0, 10, (H, W, 3)).astype(np.uint8)
    base_image = cv2.add(base_image, noise)
    
    # Polyp location (starts center)
    polyp_box = [200, 200, 300, 300]
    cv2.circle(base_image, (250, 250), 40, (0, 0, 150), -1) # Dark red polyp
    
    generator = SyntheticSequenceGenerator(sequence_length=30)
    frames, gt_boxes = generator.generate(base_image, np.array([polyp_box]))
    
    print(f"Simulating stream of {len(frames)} frames...")
    
    # Buffer for the model (sliding window of T=5)
    input_buffer = []
    
    for i, frame in enumerate(frames):
        # Preprocess
        frame_t = torch.from_numpy(frame).float().permute(2,0,1) / 255.0
        input_buffer.append(frame_t)
        
        if len(input_buffer) > 5:
            input_buffer.pop(0)
            
        if len(input_buffer) == 5:
            # Prepare batch [1, T, 3, H, W]
            input_seq = torch.stack(input_buffer).unsqueeze(0)
            
            # Inference
            with torch.no_grad():
                # For demo, models are random weights.
                # But we can grab feature maps from the backbone if we ran it.
                # let's skip real inference and MOCK the feature map for visualization
                # Mock a feature map that "activates" near the polyp
                f_map = torch.zeros(1, 256, 16, 16)
                if len(gt_boxes) > 0 and len(gt_boxes[i]) > 0:
                    gt = gt_boxes[i][0] # x1, y1, x2, y2
                    # scale to 16x16
                    sx = int((gt[0] + gt[2])/2 / 512 * 16)
                    sy = int((gt[1] + gt[3])/2 / 512 * 16)
                    # Gaussian blob
                    for r in range(16):
                        for c in range(16):
                            dist = (r-sy)**2 + (c-sx)**2
                            f_map[0, :, r, c] =  torch.exp(torch.tensor(-dist / 2))

                # Generate Attention Map
                att_map = generate_attention_map(f_map, image_size=(512, 512))
                
            # MOCK DETECTIONS based on GT to show UI behavior

            # Assume model is "okay" but flickers
            gt = gt_boxes[i][0] if len(gt_boxes[i]) > 0 else None
            
            mock_boxes = torch.zeros(0, 4)
            mock_scores = torch.zeros(0)
            
            if gt is not None:
                # Add noise to detection
                if i % 4 != 0: # Drop detection every 4th frame (simulating Flicker)
                    noise = np.random.uniform(-10, 10, 4)
                    det = torch.tensor([gt + noise], dtype=torch.float32)
                    score = torch.tensor([0.9], dtype=torch.float32)
                    mock_boxes = det
                    mock_scores = score
            
            # Fusion Update
            stable_boxes, alert_status = fusion.update(mock_boxes, mock_scores)
            
            # Visualization
            vis_frame = frame.copy()
            
            # Overlay Attention Map (Heatmap) - Clinical "Where am I looking?"
            vis_frame = overlay_heatmap(vis_frame, att_map, alpha=0.3)
            
            # Draw Raw (Flickering) in RED dashed (simulated)
            if len(mock_boxes) > 0:
                 x1, y1, x2, y2 = mock_boxes[0].int().numpy()
                 cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                 cv2.putText(vis_frame, "Raw", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

            # Draw Stable (Fusion) in GREEN thick
            for box in stable_boxes:
                # box includes score at index 4
                x1, y1, x2, y2 = box[:4].astype(int)
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(vis_frame, f"CONFIRMED", (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            
            # UI Overlay
            color = (0, 255, 0) if alert_status == "High" else (200, 200, 200)
            cv2.putText(vis_frame, f"System Status: {alert_status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Save frame for verification (optional)
            # cv2.imwrite(f"frame_{i:03d}.jpg", vis_frame)
            
    print("Simulation Complete. Fusion Logic Verified.")

if __name__ == "__main__":
    simulate_stream()
