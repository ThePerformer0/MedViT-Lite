# MedViT-Lite

A modular PyTorch prototype of a spatio-temporal transformer pipeline for ultrasound video diagnosis, designed around computational efficiency and trustworthy predictions (attention-based explanations and uncertainty estimation).

> **Status: research prototype.** The architecture and the end-to-end inference pipeline are implemented. The model has **not been trained or evaluated yet**, so no performance figures are reported. See [Status and limitations](#status-and-limitations).

Developed as a capstone project.

## Motivation

Ultrasound exam videos contain many near-identical frames and a large black background. Running a full transformer on every frame and every pixel is wasteful, especially on edge devices. MedViT-Lite explores a pipeline that:

1. skips redundant frames,
2. crops the useful region before the transformer sees it,
3. aggregates per-frame features over time,
4. returns both a prediction and an uncertainty estimate,
5. exposes attention scores so the model's focus can be visualized.

## Pipeline

```
video frames
   │
   ▼
[1] FrameSelector      SSIM vs. last kept frame → skip near-duplicates
   │
   ▼
[2] SpatialSelector    threshold + largest contour → crop the active ultrasound area
   │
   ▼
[3] PatchEmbedding + MedViTBlock    ViT-style encoder (self-attention, returns attention scores)
   │   (mean-pooled features, one vector per kept frame)
   ▼
[4] TemporalMemory     GRU over the sequence of frame features
   │
   ▼
[5] DiagnosticHead     MLP + Monte Carlo Dropout → class probabilities + uncertainty (std)
   │
   ▼
[6] AttentionVisualizer   heatmap from averaged attention scores
```

## Repository layout

| Path | Content |
|---|---|
| `models/frame_selector.py` | Temporal sparsification with SSIM (default threshold 0.96 in `main.py`) |
| `models/spatial_selector.py` | ROI extraction with OpenCV (thresholding, largest contour, padding) |
| `models/transformer_block.py` | Patch embedding, multi-head self-attention returning attention scores, patch merging, transformer block |
| `models/temporal_module.py` | GRU-based temporal aggregation |
| `models/diagnostic_module.py` | Classification head with Monte Carlo Dropout (10 stochastic passes by default) |
| `models/explicability.py` | Attention heatmap generation |
| `main.py` | `MedViTLiteSystem`, the end-to-end inference pipeline |
| `Dockerfile`, `requirements.txt` | Reproducible environment (Python 3.9, PyTorch, OpenCV, scikit-image) |

## Usage

```bash
pip install -r requirements.txt
```

```python
from main import MedViTLiteSystem

system = MedViTLiteSystem()
# frames: list of BGR images (numpy arrays), e.g. read with cv2.VideoCapture
result = system.process_video_stream(frames)

result["prediction"]    # mean class probabilities over MC Dropout passes
result["uncertainty"]   # standard deviation across passes
result["last_attn_map"] # attention scores of the last processed frame
result["last_roi"]      # cropped region of the last processed frame
```

With Docker:

```bash
docker build -t medvit-lite .
docker run --rm medvit-lite
```

The default container command only initializes the system. Weights are randomly initialized until a training script is added, so outputs are **not** meaningful predictions yet.

## Status and limitations

Stated plainly, so you know exactly what this repository does and does not contain:

- **No training and no evaluation.** There is no training loop, no dataset loader and no reported metric.
- **Single resolution.** `PatchMerging` is implemented but is not yet used in the pipeline, which currently runs one transformer block at a single scale. The "hierarchical" design is therefore planned, not active.
- **Explainability is attention-based.** The heatmap averages self-attention scores. It is a simple visualization, not Grad-CAM++ and not a validated clinical explanation.
- **Uncertainty is approximate.** Monte Carlo Dropout is a common approximation of Bayesian inference. The uncertainty values are uncalibrated and unvalidated.
- **Efficiency is not measured.** The frame and region selection are designed to reduce computation, but latency, memory and FLOPs have not been benchmarked.
- **Skipped frames are dropped,** not replaced by cached features.
- **Binary head.** The diagnostic head is configured for 2 classes.
- **Not a medical device.** This is research code and must not be used for clinical decisions.

## Author

Feke Jimmy Wilson, Master 2 in Computer Science, National Advanced School of Engineering of Yaoundé.
