import torch
import torch.nn as nn

class TemporalFeatureAggregator(nn.Module):
    """
    Simulates a ConvLSTM-like behavior to aggregate features over time.
    For simplicity and speed, we implement a GRU-based or simple RNN-based aggregation
    on the feature maps before sending them to the head.
    """
    def __init__(self, in_channels, hidden_channels=256):
        super(TemporalFeatureAggregator, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        
        # ConvGRU update gate
        self.conv_z = nn.Conv2d(in_channels + hidden_channels, hidden_channels, 3, padding=1)
        # ConvGRU reset gate
        self.conv_r = nn.Conv2d(in_channels + hidden_channels, hidden_channels, 3, padding=1)
        # ConvGRU candidate memory
        self.conv_h = nn.Conv2d(in_channels + hidden_channels, hidden_channels, 3, padding=1)
        
    def forward(self, x, h_prev=None):
        """
        x: Current frame feature map [B, C, H, W]
        h_prev: Previous hidden state [B, C, H, W]
        """
        if h_prev is None:
            h_prev = torch.zeros_like(x)
            
        combined = torch.cat([x, h_prev], dim=1)
        
        z = torch.sigmoid(self.conv_z(combined))
        r = torch.sigmoid(self.conv_r(combined))
        
        combined_r = torch.cat([x, r * h_prev], dim=1)
        h_tilde = torch.tanh(self.conv_h(combined_r))
        
        h_next = (1 - z) * h_prev + z * h_tilde
        return h_next

class TemporalRetinaNet(nn.Module):
    """
    Wraps RetinaNet components.
    Processes video sequences.
    """
    def __init__(self, retinanet_model):
        super(TemporalRetinaNet, self).__init__()
        self.backbone = retinanet_model.backbone
        self.cls_head = retinanet_model.cls_head
        self.reg_head = retinanet_model.reg_head
        
        # One aggregator per FPN level?
        # For efficiency, let's just do it on the highest resolution or top 2.
        # But to be correct, we should do it on all levels used for detection.
        self.aggregators = nn.ModuleDict()
        # FPN levels usually '0', '1', '2', '3', 'pool'
        for key in ['0', '1', '2', '3', 'pool']:
             self.aggregators[key] = TemporalFeatureAggregator(256, 256)

    def forward_sequence(self, x_seq):
        """
        x_seq: [B, T, 3, H, W]
        """
        B, T, C, H, W = x_seq.shape
        
        # Flatten for backbone: [B*T, 3, H, W]
        x_flat = x_seq.view(B * T, C, H, W)
        
        # Extract Spatial Features
        # features_flat is Dict mapping level -> [B*T, 256, H', W']
        features_flat = self.backbone(x_flat)
        
        # Reshape back to [B, T, ...] and apply temporal aggregation
        temporal_features = {}
        
        # Initialize hidden states
        hidden_states = {}
        for key, tens in features_flat.items():
            _, _, h_dim, w_dim = tens.shape
            hidden_states[key] = torch.zeros(B, 256, h_dim, w_dim).to(x_seq.device)
            # Placeholder for outputs
            temporal_features[key] = []

        # Loop through time
        for t in range(T):
            for key in features_flat.keys():
                # Extract frame t features: [B, 256, H, W]
                # features_flat[key] is [B*T, ...]. frame t is index b*T + t... complex indexing
                # Easier to reshape first
                feat_t = features_flat[key].view(B, T, 256, *features_flat[key].shape[2:])[:, t]
                
                if key in self.aggregators:
                    h_next = self.aggregators[key](feat_t, hidden_states[key])
                    hidden_states[key] = h_next
                    temporal_features[key].append(h_next)
                else:
                    temporal_features[key].append(feat_t)

        # Stack temporal features back: level -> [B*T, 256, H, W]
        final_features_flat = {}
        for key, val_list in temporal_features.items():
            # Stack along T [B, T, C, H, W] -> Flatten [B*T, C, H, W]
            stacked = torch.stack(val_list, dim=1)
            B_s, T_s, C_s, H_s, W_s = stacked.shape
            final_features_flat[key] = stacked.view(B_s * T_s, C_s, H_s, W_s)
            
        # Detect
        cls_logits = []
        bbox_preds = []
        
        for key in ['0', '1', '2', '3', 'pool']:
            if key in final_features_flat:
                f_map = final_features_flat[key]
                cls_logits.append(self.cls_head(f_map))
                bbox_preds.append(self.reg_head(f_map))
                
        # Returns raw logits/coords for the whole sequence (flattened B*T)
        return cls_logits, bbox_preds
