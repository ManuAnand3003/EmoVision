# 🧠 EmoVision — AI Emotion Intelligence System

> Real-time face emotion detection from live camera and uploaded images.  
> Built with FastAPI · DeepFace · EfficientNet · WebSockets · Vanilla JS

---

## ✨ Features

| Feature | Details |
|---|---|
| 📁 **Image Upload** | Drag & drop any JPG/PNG, detects all faces, shows annotated result |
| 📷 **Live Camera** | Real-time WebSocket stream, ~8fps send / instant inference |
| 🎭 **7 Emotions** | angry, disgust, fear, happy, sad, surprise, neutral |
| 👥 **Multi-face** | Detects and tracks multiple faces simultaneously with unique color per face |
| 📊 **Analytics** | Emotion bars, session history doughnut chart, live timeline chart |
| 📸 **Snapshot** | Save annotated camera frames as JPEG |
| 🏋️ **Custom Training** | Train your own EfficientNet-B0 on FER2013 + RAF-DB |
| 🚀 **ONNX Export** | Deploy custom model for 5-10x faster CPU inference |

---

## 🚀 Quick Start (5 minutes)

### 1. Clone & setup environment

```bash
git clone <your-repo>
cd emovision

python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

> ⚠️ First install downloads TensorFlow (~500MB) + model weights (~600MB).  
> This only happens once — subsequent starts are instant.

### 2. Run

```bash
python main.py
```

Open your browser at **http://localhost:8000**

---

## 🏗️ Project Architecture

```
emovision/
│
├── main.py                     # FastAPI app, REST + WebSocket endpoints
├── engine/
│   ├── __init__.py
│   └── pipeline.py             # ML inference engine (DeepFace + FER fallback)
├── static/
│   └── index.html              # Full SPA — zero build step, pure HTML/CSS/JS
├── training/
│   └── train.py                # Custom EfficientNet-B0 training on FER2013/RAF-DB
├── models/                     # Saved model checkpoints (created after training)
├── Dockerfile
└── requirements.txt
```

### API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Serves the frontend SPA |
| `POST` | `/api/analyze` | Analyze uploaded image → JSON results + annotated base64 image |
| `WS` | `/ws/stream` | WebSocket stream for real-time camera frames |
| `GET` | `/api/health` | Backend health + model info |

---

## 🧠 ML Engine

### Primary: DeepFace + RetinaFace

DeepFace is the main backend. It uses:
- **RetinaFace** for face detection — state-of-the-art, handles extreme angles and low light
- **DeepEmotion** network for classification — trained on large-scale datasets

### Fallback: FER + MTCNN

If DeepFace is not installed, falls back to:
- **MTCNN** for face detection
- **Mini-Xception** CNN trained on FER2013

### Fast Mode (Live Stream)
For real-time performance, live stream uses `opencv` detector backend instead of `retinaface`.  
Switch in `pipeline.py → _analyze_deepface(fast_mode=True)`.

---

## 🏋️ Train Your Own Model

Training a custom EfficientNet-B0 gives you full control over accuracy and which datasets to use.

### Step 1: Get the data

**FER2013** (primary, free on Kaggle):
```bash
# Set up Kaggle API first: https://www.kaggle.com/docs/api
kaggle datasets download -d msambare/fer2013
unzip fer2013.zip -d data/fer2013/
```

**RAF-DB** (better quality, optional):
```
Register at: http://www.whdeng.cn/RAF/model1.html
Place in: data/rafdb/
```

### Step 2: Train

```bash
# FER2013 only
python training/train.py --data data/fer2013 --epochs 40 --output models/

# FER2013 + RAF-DB combined (better accuracy)
python training/train.py --data data/fer2013 --rafdb data/rafdb --epochs 50
```

### Step 3: Use your model

The training script automatically:
1. Saves best checkpoint → `models/best_model.pth`
2. Exports ONNX → `models/emotion_model.onnx` (for fast inference)
3. Saves confusion matrix → `models/confusion_matrix.png`
4. Saves training history → `models/training_history.json`

---

## 🐳 Deploy with Docker

```bash
docker build -t emovision .
docker run -p 8000:8000 emovision
```

---

## 🌐 Deploy Free (Share with Anyone)

### Option A: Render.com (recommended)
1. Push to GitHub
2. Create account at render.com
3. New Web Service → Connect repo → `python main.py`
4. Get a public HTTPS URL

### Option B: Railway.app
```bash
npm i -g @railway/cli
railway init && railway up
```

---

## 📊 Dataset Information

| Dataset | Size | Quality | License |
|---|---|---|---|
| FER2013 | 35,887 images, 48×48px | Medium (label noise) | Open |
| RAF-DB | 29,672 images, aligned | High | Research only |
| AffectNet | 450,000 images | Very high | Research only |

**Known limitation:** FER2013 has heavy class imbalance (disgust has <5% of samples). The training script uses weighted sampling and loss weighting to compensate.

---

## 🔬 How Detection Works

```
Frame Input (any resolution)
   │
   ▼
RetinaFace / MTCNN
   ├─ Detects bounding boxes for all faces
   └─ Returns: [x, y, w, h] per face
   │
   ▼
For each face:
   ├─ Crop + resize to model input size (224×224 or 64×64)
   ├─ Normalize pixel values to [-1, 1]
   └─ Run through emotion CNN
   │
   ▼
Softmax output → 7 probability scores (sum = 1.0)
   │
   ▼
Response: { dominant: "happy", confidence: 0.91, emotions: {...} }
```

---

## 📈 Performance

| Metric | Value |
|---|---|
| FER2013 test accuracy | ~66-68% (human: ~65%) |
| Inference time (CPU) | 80-200ms per frame |
| Live stream FPS | 5-8 FPS (CPU) / 25+ FPS (GPU) |

To improve accuracy: train on RAF-DB, use AffectNet, or fine-tune on your own data.

---

## 💡 Extend & Improve

- [ ] **Age + Gender detection** — DeepFace supports this with one extra action
- [ ] **Emotion smoothing** — average last 5 frames to reduce jitter
- [ ] **Export session report** — PDF with emotion timeline + charts
- [ ] **Authentication** — add a login page for personal sessions
- [ ] **Mobile app** — wrap in React Native with expo-camera
- [ ] **Grad-CAM heatmap** — visualize where the model looks
- [ ] **Voice emotion** — add speech emotion recognition alongside face

---

## 🎯 Key Concepts to Understand

| Concept | What it means |
|---|---|
| **Softmax** | Converts raw logits → probabilities that sum to 1.0 |
| **Transfer learning** | Using ImageNet weights as starting point, fine-tuning on emotions |
| **Weighted sampling** | Oversample rare classes (disgust) to fix class imbalance |
| **Label smoothing** | Slightly soft targets (0.9 instead of 1.0) to prevent overconfidence |
| **WebSocket** | Full-duplex persistent connection — better than HTTP polling for real-time |
| **ONNX** | Open format for ML models — faster inference, language-agnostic |
