import torch
import numpy as np
import cv2
import random

class SyntheticSequenceGenerator:
    """
    Simulates a video sequence from a single static frame by applying
    smooth, physically-plausible camera motions (translations, slight zooms)
    and lighting variations (intensity flicker, blur).
    """
    def __init__(self, sequence_length=5, max_translation=15, max_zoom=0.05, blur_prob=0.3):
        self.sequence_length = sequence_length
        self.max_translation = max_translation
        self.max_zoom = max_zoom
        self.blur_prob = blur_prob

    def generate(self, image, boxes):
        """
        Args:
            image: (H, W, 3) numpy array, uint8
            boxes: (N, 4) numpy array, format [x1, y1, x2, y2]
        Returns:
            frames: List of (H, W, 3) images
            box_sequences: List of (N, 4) boxes corresponding to each frame
        """
        H, W = image.shape[:2]
        frames = []
        box_sequences = []

        # Initial random velocities for smooth motion
        dx = random.uniform(-2, 2)
        dy = random.uniform(-2, 2)
        dzoom = random.uniform(-0.005, 0.005)

        current_matrix = np.eye(3)

        for t in range(self.sequence_length):
            # Update motion parameters (simulating hand tremor/drift)
            dx += random.uniform(-0.5, 0.5)
            dy += random.uniform(-0.5, 0.5)
            # Damping to prevent flying off screen
            dx *= 0.9 
            dy *= 0.9

            # Construct affine matrix for this step
            # Translation
            M = np.float32([[1 + dzoom * t, 0, dx * t], [0, 1 + dzoom * t, dy * t]])
            
            # Apply to image
            # heavily simplified "camera" movement relative to center
            # Using cv2.warpAffine
            # We want cumulative movement, but for simplicity, let's just perturb the original
            # consistently to avoid drift accumulation errors in simple implementation.
            
            # Better approach: Calculate absolute shift for frame t
            shift_x = dx * (t + 1)
            shift_y = dy * (t + 1)
            scale = 1 + random.uniform(-self.max_zoom, self.max_zoom) # jitter z slightly

            # Center of rotation
            center = (W // 2, H // 2)
            M = cv2.getRotationMatrix2D(center, 0, scale)
            M[0, 2] += shift_x
            M[1, 2] += shift_y

            warped_img = cv2.warpAffine(image, M, (W, H), borderMode=cv2.BORDER_REFLECT)
            
            # Add synthetic blur
            if random.random() < self.blur_prob:
                ksize = random.choice([3, 5])
                warped_img = cv2.GaussianBlur(warped_img, (ksize, ksize), 0)

            # Add intensity fluctuation (breathing/lighting)
            intensity_factor = random.uniform(0.9, 1.1)
            warped_img = cv2.convertScaleAbs(warped_img, alpha=intensity_factor, beta=0)

            frames.append(warped_img)

            # Transform boxes
            # Boxes are [x1, y1, x2, y2]
            # We transform all 4 corners to be safe, then take bounding box of result
            current_boxes = []
            for box in boxes:
                x1, y1, x2, y2 = box
                corners = np.array([
                    [x1, y1], [x2, y1], [x2, y2], [x1, y2]
                ])
                # Add 1 for homogenous coords
                ones = np.ones(shape=(4, 1))
                corners_hom = np.hstack([corners, ones])
                
                # Apply M (2x3)
                transformed_corners = M.dot(corners_hom.T).T # (4, 2)
                
                new_x1 = np.clip(transformed_corners[:, 0].min(), 0, W)
                new_y1 = np.clip(transformed_corners[:, 1].min(), 0, H)
                new_x2 = np.clip(transformed_corners[:, 0].max(), 0, W)
                new_y2 = np.clip(transformed_corners[:, 1].max(), 0, H)
                
                # If box is mostly out of frame or invalid area, we might want to drop it
                # For now keep valid ones
                if (new_x2 - new_x1 > 5) and (new_y2 - new_y1 > 5):
                     current_boxes.append([new_x1, new_y1, new_x2, new_y2])
                else:
                    # Occlusion simulated by box disappearing
                    pass # Just don't add it

            box_sequences.append(np.array(current_boxes) if len(current_boxes) > 0 else np.zeros((0, 4)))

        return frames, box_sequences
