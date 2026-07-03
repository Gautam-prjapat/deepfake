# 👁️ DeepSight — ISTVT Video Deepfake Detector

DeepSight is a Streamlit web app that analyzes videos for signs of deepfake manipulation using **ISTVT (Interpretable Spatial-Temporal Video Transformer)**, a custom PyTorch model that combines an Xception CNN backbone with decomposed spatial and temporal attention blocks.

**🔗 Live demo:** [deepfake-huycynwhmndahtyxc4dczd.streamlit.app](https://deepfake-huycynwhmndahtyxc4dczd.streamlit.app/)

---

## ✨ Features

- **Upload & analyze** `.mp4`, `.avi`, or `.mov` videos directly in the browser
- **Autonomous streaming inference** — automatically samples up to 520 frames from the video, processed in memory-safe chunks of 16
- **Top-10% pooling** — the final anomaly score is the average of the most suspicious 10% of sampled frame-windows, reducing noise from a few false-positive frames
- **Calibrated verdict scoring** — an exponential curve sharpens confidence on the "authentic" side of the threshold
- **Attention heatmap visualization** — overlays a spatial activation map on the most suspicious frame when a deepfake is detected
- **Automatically downloads model weights** from Kaggle on first run (cached afterward)

---

## 🧠 How It Works

### 1. Model Architecture (`model_utils.py`)

- **Backbone:** an `xception` model (via `timm`) extracts spatial features from each frame
- **SelfSubtract:** computes frame-to-frame residuals to emphasize temporal changes across the sequence
- **DecomposedAttentionBlock:** applies temporal self-attention (across frames) and spatial self-attention (across patches) separately, then fuses them through an MLP
- **ISTVT:** stacks several `DecomposedAttentionBlock`s on top of the backbone features and pools the result into a single real/fake logit per 4-frame window

### 2. Streaming Inference Pipeline (`predict_video`)

1. The video is sampled at an even stride up to a maximum of **520 frames** (rounded to a multiple of 4).
2. Frames are read and processed in **chunks of 16** to keep memory usage low.
3. Each chunk is split into 4-frame windows and passed through the model, producing a probability per window.
4. The **top 10%** most suspicious window probabilities are averaged to produce the final video-level score — this "Top-K% pooling" approach avoids diluting a real deepfake segment with long stretches of authentic footage.
5. The window with the single highest anomaly score is used to generate an **attention heatmap** overlay via a forward hook on the model's projection layer.

### 3. Verdict Calibration (`app.py`)

- **Score > 0.5:** flagged as **Deepfake**, confidence scales linearly with the raw probability.
- **Score ≤ 0.5:** flagged as **Authentic**, confidence is boosted using an exponential curve (`100 − 50 × (probability / 0.5)³`) to compensate for the naturally pessimistic baseline introduced by Top-10% pooling.

---

## 📂 Project Structure

```
.
├── app.py              # Streamlit UI, model caching, video upload & verdict display
├── model_utils.py       # ISTVT architecture, streaming inference engine, heatmap visualization
├── requirements.txt     # Python dependencies
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/xenoz27/DeepFake.git
cd DeepFake
pip install -r requirements.txt
```

### Requirements

```
streamlit
kagglehub
torch
torchvision
opencv-python-headless
numpy
Pillow
timm
```

---

## 🚀 Usage

Run the app locally:

```bash
streamlit run app.py
```

On first launch, the app will automatically download the pretrained ISTVT weights (`istvt_master_weights.pth`) from Kaggle via `kagglehub` and cache the model in memory. Then:

1. Open the app in your browser (default: `http://localhost:8501`)
2. Upload a video (`.mp4`, `.avi`, or `.mov`)
3. Click **"Run Autonomous ISTVT Analysis"**
4. View the verdict, confidence score, and attention heatmap

---

## 📊 Dataset & Weights

Model weights are hosted on Kaggle:
🔗 [kaggle.com/datasets/gam888i/istvt-pth](https://www.kaggle.com/datasets/gam888i/istvt-pth)

They are downloaded automatically at runtime — no manual setup required.

---

## 🛠️ Tech Stack

| Component | Library |
|---|---|
| Web UI | Streamlit |
| Deep Learning | PyTorch, timm (Xception backbone) |
| Video/Image Processing | OpenCV, Pillow, torchvision |
| Model Weights Hosting | Kaggle (via kagglehub) |

---

## ⚠️ Disclaimer

This tool is intended for research and educational purposes. Deepfake detection scores are probabilistic estimates and should not be treated as definitive proof of video authenticity or manipulation.
