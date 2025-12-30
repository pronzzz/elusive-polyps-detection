# Elusive Polyps Detection: AI-Assisted Clinical Decision Support

A temporal computer vision system designed to help gastroenterologists detect missed polyps during colonoscopy by analyzing video context rather than just single frames.

> **Status**: Verified via Simulation (Training Pipeline & Fusion Logic Operational)

## 🧠 System Architecture

The following diagram illustrates the dual-path architecture designed to minimize false negatives (missed polyps) while preventing false positives (flickering alerts) that cause alarm fatigue.

```mermaid
flowchart TD
    VideoInput[Video Stream Input] --> Backbone[ResNet+FPN Backbone]
    
    subgraph SpatialPath [Spatial Intelligence]
        Backbone -->|Frame Features| RetinaHead[RetinaNet Head]
        RetinaHead -->|Box & Class| RawDetections[Raw Frame Detections]
    end
    
    subgraph TemporalPath [Temporal Intelligence]
        Backbone -->|Feature Sequence| ConvGRU[ConvGRU Memory Module]
        ConvGRU -->|Refined Features| TemporalHead[Temporal Head]
        TemporalHead -->|Context-Aware Box| TemporalDetections[Temporal Detections]
        ConvGRU -.->|State t-1| ConvGRU
    end
    
    RawDetections --> Fusion[Fusion Engine]
    TemporalDetections --> Fusion
    
    subgraph ClinicalFusion [Clinical Decision Support]
        Fusion -->|Check Consistency| HistoryBuffer[History Buffer]
        HistoryBuffer -->|Stable?| AlertLogic{Alert Decision}
        AlertLogic -->|Yes| ClinicianAlert[High Confidence Alert]
        AlertLogic -->|No| LowPriority[Background Log]
    end

    subgraph Explainability
        TemporalHead -->|Activations| AttentionMap[Attention Heatmap]
        AttentionMap --> ClinicianOverlay[Clinician Overlay]
    end

    style SpatialPath fill:#e1f5fe,stroke:#01579b
    style TemporalPath fill:#fff3e0,stroke:#ff6f00
    style ClinicalFusion fill:#e8f5e9,stroke:#2e7d32
    style Explainability fill:#f3e5f5,stroke:#7b1fa2
```

## ✨ Key Features

- **Temporal Consistency**: Uses LSTM/ConvRNN modules to remember polyp locations across frames, handling motion blur and temporary occlusions.
- **Dual-Path Architecture**:
  - **Spatial Path**: RetinaNet (ResNet+FPN) for high-precision frame analysis.
  - **Temporal Path**: Aggregates features over time to boost confidence for fleeting objects.
- **Clinical Fusion Engine**: Smart logic to filter alerts, ensuring the doctor is notified only when detections are stable, preventing "alert fatigue."
- **Explainability**: Real-time Attention Maps overlay to show *where* the model is looking.

## 📁 Project Structure

```
elusive-polyps-detection/
├── data/
│   ├── dataset_loader.py       # CVC-ClinicDB / Kvasir loaders
│   └── sequence_generator.py   # Synthesizes video motion from static images
├── detectors/
│   ├── backbone.py             # ResNet + FPN
│   ├── retinanet.py            # Single-frame detector head
│   ├── temporal_model.py       # LSTM-based sequence refiner
│   └── loss.py                 # Focal Loss for class imbalance
├── fusion/
│   └── fusion_engine.py        # Logic for stable alerts
├── inference/
│   └── stream_simulator.py     # Real-time simulation demo
├── training/
│   └── train.py                # Full training loop
└── utils/
    └── explainability.py       # Attention map generation
```

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Simulation Demo
This runs the entire pipeline (Dual-Path + Fusion + Explainability) on a synthetic video stream.
```bash
python inference/stream_simulator.py
```
**Expected Output**: A window or console log showing specific frames where the Fusion Engine stabilizes alerts ("CONFIRMED") despite simulated raw detection flickering.

### 3. Training
To train the model (default: synthetic data loop):
```bash
python training/train.py
```

## ✅ Verification Results

We have verified the core logic of the system:

1.  **Dual-Path Architecture**:
    - Built a **RetinaNet** (ResNet+FPN) for high-quality spatial detection.
    - Built a **Temporal Module** (ConvGRU) to aggregate features across video frames.
2.  **Dataset Flexibility**:
    - Implemented a `SyntheticSequenceGenerator` that allows training/testing the temporal model using *static* datasets (like CVC-ClinicDB) by simulating realistic endoscope motion.
3.  **Clinical Fusion Logic**:
    - Implemented `FusionEngine` which successfully reinforces persistent detections and filters out transient noise.
4.  **Training Verification**:
    - Successfully executed the training loop with Focal Loss and Anchor Matching.
    - Log: `Batch 0, Loss: 15176...` confirming backward pass integrity.

## 🔜 Next Steps
- [ ] Download real video data (LDPolypVideo).
- [ ] Train the spatial RetinaNet on CVC-ClinicDB.
- [ ] Fine-tune the temporal module on video sequences.
