# User Guide: Elusive Polyps Detection System

This guide provides detailed instructions on how to use, train, and configure the system.

## 1. Dataset Setup

The system is designed to work with **CVC-ClinicDB** (static frames) or **LDPolypVideo** (video).

### Option A: Synthetic Video Training (Recommended for Initial Testing)
If you only have static images (like CVC-ClinicDB), the system uses `data.sequence_generator.SyntheticSequenceGenerator` to create video clips on the fly.
1. Download CVC-ClinicDB.
2. Structure:
   ```
   /path/to/db/
      Original/
         1.png
         ...
      Ground Truth/
         1.png
         ...
   ```
3. Update `training/train.py` to point to this directory instead of `RandomSequenceDataset`.

### Option B: Real Video Data
If using LDPolypVideo:
1. Implement a specific `VideoDataset` class in `data/dataset_loader.py` that reads the video files directly.
2. Ensure it yields tensors of shape `[Sequence_Length, 3, H, W]`.

## 2. Configuration Parameters

### Fusion Engine (`fusion/fusion_engine.py`)
- **`history_length`** (Default: 5): How many frames the system "remembers". Increasing this makes alerts more stable but slightly delays them (latency).
- **`confirmation_threshold`** (Default: 3): How many times an object must be seen in the history window to trigger a "High Confidence" alert. 
    - *Lower* = Faster detection, more false positives.
    - *Higher* = Fewer false positives, potential to miss brief glimpses.

### Temporal Model (`detectors/temporal_model.py`)
- **`TemporalFeatureAggregator`**: Using a ConvGRU. You can switch to ConvLSTM if you need more long-term dependency modeling, though GRU is faster for real-time.

## 3. Training Process

The training script `training/train.py` performs the following:
1. **Backbone Loading**: Loads ResNet50 (or ResNet18 for speed).
2. **Anchor Generation**: Creates 9 anchors per spatial location.
3. **Matching**: Matches prediction anchors to Ground Truth boxes using IoU.
4. **Loss Calculation**:
   - **Classification**: Focal Loss (handles the huge number of background anchors).
   - **Regression**: Smooth L1 Loss (only for matched positive anchors).

To run training:
```bash
python training/train.py
```

## 4. Explainability

Feature maps are visualized in `inference/stream_simulator.py`. 
- **Blue/Red Heatmap**: Indicates areas of high activation in the last layer of the FPN.
- **Clinical Use**: If the model alerts but the box is ambiguous, the heatmap shows where the "attention" is focused. If it's focused on a reflection or bubble, the clinician can safely ignore it.
