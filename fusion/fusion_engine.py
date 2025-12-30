import torch
import numpy as np

class FusionEngine:
    """
    Manages detection history to provide stable alerts.
    Filters out transient flickers and reinforces consistent detections.
    """
    def __init__(self, history_length=5, confirmation_threshold=3, iou_threshold=0.5):
        self.history_length = history_length
        self.confirmation_threshold = confirmation_threshold
        self.iou_threshold = iou_threshold
        
        # List of detections per frame: each element is list of [x1, y1, x2, y2, score]
        self.track_history = [] 
        
    def update(self, boxes, scores):
        """
        Args:
            boxes: Tensor [N, 4]
            scores: Tensor [N]
        Returns:
            stable_boxes: List of boxes that are clinically relevant
            alert_level: 'High' if persistent polyp, 'Low' if unsure
        """
        # 1. Filter weak single-frame detections
        mask = scores > 0.3
        current_frame_dets = []
        if mask.any():
            valid_boxes = boxes[mask]
            valid_scores = scores[mask]
            for b, s in zip(valid_boxes, valid_scores):
                current_frame_dets.append(np.append(b.cpu().numpy(), s.cpu().numpy()))
                
        self.track_history.append(current_frame_dets)
        if len(self.track_history) > self.history_length:
            self.track_history.pop(0)
            
        return self._compute_stable_alerts()
        
    def _compute_stable_alerts(self):
        """
        Checks if current detections have predecessors in history.
        """
        if not self.track_history[-1]:
            return [], "None"
            
        current_objects = self.track_history[-1]
        stable_objects = []
        
        for obj in current_objects:
            obj_box = obj[:4]
            confidence_chain = 1 # Start with current
            
            # Check backwards
            # Very simple IoU tracking
            for prev_frame_objs in reversed(self.track_history[:-1]):
                match_found = False
                for prev_obj in prev_frame_objs:
                    iou = self._iou(obj_box, prev_obj[:4])
                    if iou > self.iou_threshold:
                        confidence_chain += 1
                        match_found = True
                        break # Assume one match needed
                if not match_found:
                    pass # Chain broken? or just missing one frame?
                    
            if confidence_chain >= self.confirmation_threshold:
                stable_objects.append(obj)
                
        if stable_objects:
             return stable_objects, "High"
        elif current_objects:
             return current_objects, "Low (Flicker)" # Show but maybe distinct color
        else:
             return [], "None"

    def _iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        
        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)
