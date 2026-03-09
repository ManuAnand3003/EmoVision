# 🎯 EmoVision — Fine-Tuning for Indian Faces

Improving accuracy on Indian faces is straightforward. The base model was trained mostly on
Western faces from FER2013, so fine-tuning on Indian face data makes a significant difference.

---

## Step 1 — Collect Images

You need **50–200 images per emotion** (more = better, but 50 works).

### Option A: Use your own photos
Organize selfies, family photos, etc. with consent:
```
my_dataset/
  happy/     ← smiling photos
  angry/     ← frowning/irritated expressions
  sad/       ← sad expressions
  neutral/   ← neutral/resting face
  surprise/  ← surprised expressions
  fear/      ← fearful expressions
  disgust/   ← disgusted expressions
```

### Option B: Download from Kaggle (free, diverse)
```bash
# RAF-DB has better demographic diversity than FER2013
kaggle datasets download -d shuvoalok/raf-db
unzip raf-db.zip -d my_dataset_raw/
# Then sort into emotion subfolders using the label files
```

### Option C: AffectNet subset
Register at: http://mohammadmahoor.com/affectnet/
Contains ~450K images with better ethnic diversity.

### Tips for Indian face images:
- Natural lighting works fine — no studio required
- Include variety: different ages, genders, lighting conditions
- Even 30-40 images per emotion will show improvement
- Prioritize emotions that currently perform worst (usually `disgust`, `fear`, `sad`)

---

## Step 2 — Run Fine-Tuning

```bash
# Activate your venv first
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Mac/Linux

# Install training deps if missing
pip install torch torchvision scikit-learn matplotlib seaborn tqdm

# Run fine-tuning (20 epochs, ~5-15 mins on CPU)
python finetune_local.py --data my_dataset --output models/ --epochs 20
```

### Continue from existing checkpoint:
```bash
python finetune_local.py --data my_dataset --output models/ --epochs 15 \
  --checkpoint models/finetuned_model.pth
```

### Output files:
| File | What it is |
|---|---|
| `models/finetuned_model.pth` | Fine-tuned PyTorch weights |
| `models/finetuned_model.onnx` | Fast ONNX version for inference |
| `models/finetune_confusion.png` | Confusion matrix |
| `models/finetune_curves.png` | Training/validation loss curves |

---

## Step 3 — Use the Fine-Tuned Model

Edit `engine/pipeline.py` to load your custom ONNX model.

Add this class after the `EMOTION_META` dict:

```python
import onnxruntime as ort
import cv2

class OnnxEmotionModel:
    """Lightweight ONNX inference — replaces DeepFace for emotion classification."""
    
    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        print(f"[EmoVision] Loaded ONNX model: {model_path}")

    def predict(self, face_bgr: np.ndarray) -> dict:
        # Preprocess
        img = cv2.resize(face_bgr, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        img  = (img - mean) / std
        img  = img.transpose(2, 0, 1)[np.newaxis]  # HWC → NCHW

        logits = self.session.run(None, {self.input_name: img})[0][0]
        exp    = np.exp(logits - logits.max())
        probs  = exp / exp.sum()

        emotions_list = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        return {e: round(float(p), 4) for e, p in zip(emotions_list, probs)}
```

Then in `EmotionPipeline.__init__`, add:
```python
# Load custom fine-tuned model if available
from pathlib import Path
onnx_path = Path("models/finetuned_model.onnx")
if onnx_path.exists():
    self.custom_model = OnnxEmotionModel(str(onnx_path))
    print("[EmoVision] Custom fine-tuned model loaded — using for emotion classification")
else:
    self.custom_model = None
```

And in `_analyze_deepface`, replace the emotion extraction with:
```python
# Use custom model for emotion classification if available
if self.custom_model and region:
    x, y, w, h = region.get("x",0), region.get("y",0), region.get("w",0), region.get("h",0)
    face_crop = img_bgr[max(0,y):y+h, max(0,x):x+w]
    if face_crop.size > 0:
        emotions = self.custom_model.predict(face_crop)
    else:
        emotions = {k.lower(): round(float(v)/total, 4) for k, v in raw_emotions.items()}
else:
    emotions = {k.lower(): round(float(v)/total, 4) for k, v in raw_emotions.items()}
```

---

## Why accuracy is lower on Indian faces

| Cause | Details |
|---|---|
| **Training data bias** | FER2013 is ~80% Western/East Asian faces |
| **Skin tone normalization** | Some models normalize lighting in ways that hurt darker tones |
| **Expression style** | Subtle cultural differences in how emotions are expressed facially |
| **Dataset size** | Indian faces are underrepresented even in "diverse" datasets |

Fine-tuning with even 50 images per emotion typically improves accuracy by **10-20 percentage points** on the target demographic.

---

## Suggested Improvements Beyond Fine-Tuning

1. **Temporal smoothing** — Average predictions across last 5 frames to reduce jitter:
   ```python
   from collections import deque
   emotion_buffer = deque(maxlen=5)
   emotion_buffer.append(current_emotions)
   smoothed = {e: np.mean([f[e] for f in emotion_buffer]) for e in emotions}
   ```

2. **Confidence threshold** — Only display prediction if confidence > 55%:
   ```python
   if confidence < 0.55:
       dominant_emotion = "uncertain"
   ```

3. **Face alignment** — Use facial landmarks to align face before classification (improves accuracy ~3-5%)

4. **Ensemble** — Average predictions from DeepFace + your fine-tuned model for better robustness
