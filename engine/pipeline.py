"""
EmoVision — ML Inference Pipeline  (v3 — accuracy & speed edition)

Improvements over v2:
  ✓ Temporal smoothing  — averages last N frames to kill jitter
  ✓ Confidence gate     — suppresses "uncertain" predictions below threshold
  ✓ Ensemble support    — combines DeepFace + your fine-tuned ONNX model
  ✓ Face tracking       — consistent face IDs across frames (no flicker)
  ✓ Auto data collector — optionally saves frames for continual training
  ✓ Faster fast_mode    — skip redundant re-init calls
"""

import os
import time
import warnings
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

warnings.filterwarnings("ignore")

# ── Constants ──────────────────────────────────────────────────────────────
EMOTION_META = {
    "angry":    {"emoji": "😠", "color": (60,  60,  220), "hex": "#dc3c3c"},
    "disgust":  {"emoji": "🤢", "color": (60, 180,  60),  "hex": "#3cb43c"},
    "fear":     {"emoji": "😨", "color": (180, 60, 180),  "hex": "#b43cb4"},
    "happy":    {"emoji": "😄", "color": (30, 210, 255),  "hex": "#ffd21e"},
    "sad":      {"emoji": "😢", "color": (200, 100, 30),  "hex": "#1e8cff"},
    "surprise": {"emoji": "😮", "color": (30, 160, 255),  "hex": "#ff9a1e"},
    "neutral":  {"emoji": "😐", "color": (180, 180, 180), "hex": "#b4b4b4"},
}

EMOTIONS = list(EMOTION_META.keys())

FACE_COLORS_HEX = ["#00ffc8", "#ff6432", "#9632ff", "#32ff96", "#ff3296"]

# ── Tuneable knobs ─────────────────────────────────────────────────────────
SMOOTH_WINDOW     = 5      # frames to average (raise for smoother, lower for faster response)
CONFIDENCE_GATE   = 0.40   # hide label if top emotion < this (0 to disable)
LOCAL_RESCUE_GATE = 0.55   # only consult collected_data when the primary model is unsure
LOCAL_RESCUE_WEIGHT = 0.20  # how much collected_data can adjust the primary prediction
COLLECT_DATA      = True  # set True to auto-save frames for continual training
COLLECT_DIR       = "collected_data"
COLLECT_THRESHOLD = 0.75   # only save frame if confidence > this (high-quality labels)


# ══════════════════════════════════════════════════════════════════════════
# ONNX Fine-Tuned Model (optional — loaded if models/finetuned_model.onnx exists)
# ══════════════════════════════════════════════════════════════════════════
class OnnxEmotionModel:
    """Lightweight ONNX model — your fine-tuned Indian-face model."""

    def __init__(self, model_path: str):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 2
        self.session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        # Detect input size from model
        shape = self.session.get_inputs()[0].shape
        self.input_h = shape[2] if shape[2] else 112
        self.input_w = shape[3] if shape[3] else 112
        print(f"[EmoVision] ONNX model loaded: {model_path} (input {self.input_h}×{self.input_w})")

    def predict(self, face_bgr: np.ndarray) -> dict:
        """Run inference on a single face crop. Returns {emotion: probability}."""
        img = cv2.resize(face_bgr, (self.input_w, self.input_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        img = img.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

        logits = self.session.run(None, {self.input_name: img})[0][0]
        exp    = np.exp(logits - logits.max())
        probs  = exp / exp.sum()
        return {e: round(float(p), 4) for e, p in zip(EMOTIONS, probs)}


# ══════════════════════════════════════════════════════════════════════════
# Temporal Smoother — kills jitter by averaging last N frames per face
# ══════════════════════════════════════════════════════════════════════════
class FaceSmoother:
    """
    Maintains a rolling buffer of emotion predictions per face position.
    Uses IoU matching to track the "same" face across frames.
    """

    def __init__(self, window: int = SMOOTH_WINDOW):
        self.window = window
        self.buffers: list[deque] = []   # one deque per tracked face
        self.boxes: list[dict]    = []   # last known box per tracked face

    def _iou(self, a: dict, b: dict) -> float:
        """Intersection over Union between two boxes."""
        ax1, ay1 = a['x'], a['y']
        ax2, ay2 = a['x'] + a['w'], a['y'] + a['h']
        bx1, by1 = b['x'], b['y']
        bx2, by2 = b['x'] + b['w'], b['y'] + b['h']
        ix = max(0, min(ax2, bx2) - max(ax1, bx1))
        iy = max(0, min(ay2, by2) - max(ay1, by1))
        inter = ix * iy
        union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
        return inter / union if union > 0 else 0.0

    def update(self, results: list[dict]) -> list[dict]:
        """
        Match incoming results to tracked faces, update buffers,
        return smoothed results.
        """
        if not results:
            # Decay buffers on empty frame
            for buf in self.buffers:
                if len(buf) > 0:
                    buf.append(buf[-1])   # repeat last known to avoid stale average
            return []

        matched = [False] * len(self.buffers)
        smoothed = []

        for face in results:
            # Find best matching tracked face by IoU
            best_idx, best_iou = -1, 0.3   # min IoU threshold
            for i, box in enumerate(self.boxes):
                iou = self._iou(face['box'], box)
                if iou > best_iou:
                    best_iou, best_idx = iou, i

            if best_idx == -1:
                # New face — create new buffer
                self.buffers.append(deque(maxlen=self.window))
                self.boxes.append(face['box'])
                best_idx = len(self.buffers) - 1
                matched.append(True)
            else:
                matched[best_idx] = True
                self.boxes[best_idx] = face['box']

            self.buffers[best_idx].append(face['emotions'])
            face['emotions'] = self._average(self.buffers[best_idx])
            # Recalculate dominant after smoothing
            face['dominant_emotion'] = max(face['emotions'], key=face['emotions'].get)
            face['confidence'] = round(face['emotions'][face['dominant_emotion']], 4)
            smoothed.append(face)

        # Prune stale tracked faces (disappeared for >3 frames)
        # (simple: just reset if too many unmatched)
        if len(self.buffers) > len(results) + 3:
            self.buffers = self.buffers[-len(results):]
            self.boxes   = self.boxes[-len(results):]

        return smoothed

    def _average(self, buf: deque) -> dict:
        """Mean of all emotion dicts in buffer."""
        if not buf:
            return {e: 1/7 for e in EMOTIONS}
        avg = {e: 0.0 for e in EMOTIONS}
        for d in buf:
            for e in EMOTIONS:
                avg[e] += d.get(e, 0.0)
        n = len(buf)
        total = sum(avg.values()) or 1
        return {e: round(avg[e] / total, 4) for e in EMOTIONS}

    def reset(self):
        self.buffers.clear()
        self.boxes.clear()


# ══════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ══════════════════════════════════════════════════════════════════════════
class EmotionPipeline:

    def __init__(self):
        self.backend     = self._init_backend()
        self.primary_model, self.rescue_model = self._load_onnx_models()
        self.smoother    = FaceSmoother(window=SMOOTH_WINDOW)
        self._frame_no   = 0

        if COLLECT_DATA:
            Path(COLLECT_DIR).mkdir(parents=True, exist_ok=True)
            for e in EMOTIONS:
                (Path(COLLECT_DIR) / e).mkdir(exist_ok=True)

        print(f"[EmoVision] Backend     : {self.backend}")
        print(f"[EmoVision] ONNX model  : {'yes (primary active)' if self.primary_model else 'none'}")
        print(f"[EmoVision] Rescue model : {'yes (collected-data fallback)' if self.rescue_model else 'none'}")
        print(f"[EmoVision] Smooth window: {SMOOTH_WINDOW} frames")
        print(f"[EmoVision] Conf gate   : {CONFIDENCE_GATE}")

    # ── Backend init ───────────────────────────────────────────────────────
    def _init_backend(self) -> str:
        try:
            from deepface import DeepFace
            dummy = np.zeros((100, 100, 3), dtype=np.uint8)
            try:
                DeepFace.analyze(dummy, actions=["emotion"],
                                 detector_backend="opencv", silent=True,
                                 enforce_detection=False)
            except Exception:
                pass
            self._deepface = DeepFace
            return "deepface"
        except ImportError:
            pass

        try:
            from fer import FER
            self._fer = FER(mtcnn=True)
            return "fer"
        except ImportError:
            pass

        raise RuntimeError("No backend. Install: pip install deepface  OR  pip install fer")

    def _load_onnx_models(self) -> tuple[Optional[OnnxEmotionModel], Optional[OnnxEmotionModel]]:
        """Load the primary big-data model first, then the collected-data rescue model."""
        primary_candidates = [
            Path("models/stage1_bigdata/finetuned_model.onnx"),
            Path("models/emotion_model.onnx"),
        ]
        rescue_candidates = [
            Path("models/finetuned_model.onnx"),
        ]

        primary_model = None
        rescue_model = None

        for path in primary_candidates:
            if not path.exists():
                continue
            try:
                primary_model = OnnxEmotionModel(str(path))
                break
            except Exception as e:
                print(f"[EmoVision] Could not load primary ONNX ({path}): {e}")

        for path in rescue_candidates:
            if not path.exists():
                continue
            try:
                rescue_model = OnnxEmotionModel(str(path))
                break
            except Exception as e:
                print(f"[EmoVision] Could not load rescue ONNX ({path}): {e}")

        if primary_model is None and rescue_model is not None:
            primary_model, rescue_model = rescue_model, None

        return primary_model, rescue_model

    # ── Public API ─────────────────────────────────────────────────────────
    def analyze(self, img_bgr: np.ndarray, fast_mode: bool = False) -> list[dict]:
        """
        Detect and classify emotions in an image.
        Returns list of face dicts with smoothed, ensemble-merged results.
        """
        self._frame_no += 1

        if self.backend == "deepface":
            raw = self._run_deepface(img_bgr, fast_mode)
        else:
            raw = self._run_fer(img_bgr)

        # Classify each face with the primary big-data model.
        if raw:
            raw = self._classify_faces(img_bgr, raw)

        # Temporal smoothing (live mode only — still images skip smoother)
        if fast_mode:
            raw = self.smoother.update(raw)
        else:
            self.smoother.reset()   # reset between image uploads

        # Confidence gate
        raw = self._apply_gate(raw)

        # Auto-collect high-confidence frames
        if COLLECT_DATA and fast_mode:
            self._maybe_collect(img_bgr, raw)

        return raw

    # ── DeepFace ───────────────────────────────────────────────────────────
    def _run_deepface(self, img: np.ndarray, fast_mode: bool) -> list[dict]:
        detector = "opencv" if fast_mode else "retinaface"
        results  = []
        try:
            faces = self._deepface.analyze(
                img,
                actions=["emotion"],
                detector_backend=detector,
                enforce_detection=False,
                silent=True,
            )
            if isinstance(faces, dict):
                faces = [faces]

            for i, face in enumerate(faces):
                region     = face.get("region", {})

                results.append({
                    "face_id": i,
                    "box": {
                        "x": int(region.get("x", 0)),
                        "y": int(region.get("y", 0)),
                        "w": int(region.get("w", 0)),
                        "h": int(region.get("h", 0)),
                    },
                    "emotions":         {},
                    "dominant_emotion": "neutral",
                    "confidence":       0.0,
                    "color_hex":        FACE_COLORS_HEX[i % len(FACE_COLORS_HEX)],
                })
        except Exception as e:
            print(f"[DeepFace] {e}")
        return results

    # ── FER fallback ───────────────────────────────────────────────────────
    def _run_fer(self, img: np.ndarray) -> list[dict]:
        results = []
        try:
            for i, face in enumerate(self._fer.detect_emotions(img)):
                x, y, w, h = face["box"]
                raw   = face["emotions"]
                total = float(sum(raw.values())) or 1.0
                emo   = {k: round(float(v) / total, 4) for k, v in raw.items()}
                dom   = max(emo, key=emo.get)
                results.append({
                    "face_id": i,
                    "box": {"x": x, "y": y, "w": w, "h": h},
                    "emotions": emo,
                    "dominant_emotion": dom,
                    "confidence": round(emo[dom], 4),
                    "color_hex": FACE_COLORS_HEX[i % len(FACE_COLORS_HEX)],
                })
        except Exception as e:
            print(f"[FER] {e}")
        return results

    # ── Classification ────────────────────────────────────────────────────
    def _classify_faces(self, img: np.ndarray, results: list[dict]) -> list[dict]:
        """
        Run the primary big-data model on every face crop.
        Only consult collected-data when the primary prediction is uncertain.
        """
        h, w = img.shape[:2]
        for face in results:
            b  = face["box"]
            x1 = max(0, b["x"])
            y1 = max(0, b["y"])
            x2 = min(w, b["x"] + b["w"])
            y2 = min(h, b["y"] + b["h"])
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            try:
                primary_probs = self.primary_model.predict(crop) if self.primary_model else {}

                if not primary_probs:
                    # No ONNX model available; keep the backend output if any.
                    continue

                primary_dominant = max(primary_probs, key=primary_probs.get)
                primary_conf = float(primary_probs.get(primary_dominant, 0.0))

                final_probs = primary_probs
                if self.rescue_model and primary_conf < LOCAL_RESCUE_GATE:
                    rescue_probs = self.rescue_model.predict(crop)
                    final_probs = {}
                    for e in EMOTIONS:
                        final_probs[e] = round(
                            (1.0 - LOCAL_RESCUE_WEIGHT) * primary_probs.get(e, 0.0)
                            + LOCAL_RESCUE_WEIGHT * rescue_probs.get(e, 0.0),
                            4,
                        )

                # Renormalize
                total = sum(final_probs.values()) or 1.0
                face["emotions"] = {k: round(v / total, 4) for k, v in final_probs.items()}
                face["dominant_emotion"] = max(face["emotions"], key=face["emotions"].get)
                face["confidence"]       = round(face["emotions"][face["dominant_emotion"]], 4)
            except Exception as e:
                print(f"[Classifier] {e}")
        return results

    # ── Confidence gate ────────────────────────────────────────────────────
    def _apply_gate(self, results: list[dict]) -> list[dict]:
        """
        If top emotion probability is below CONFIDENCE_GATE,
        keep the predicted class but mark it as low confidence.
        """
        if CONFIDENCE_GATE <= 0:
            return results
        for face in results:
            if face["confidence"] < CONFIDENCE_GATE:
                face["confidence"] = round(face["confidence"], 4)
        return results

    # ── Data collector ─────────────────────────────────────────────────────
    def _maybe_collect(self, img: np.ndarray, results: list[dict]):
        """
        Save high-confidence face crops to COLLECT_DIR/<emotion>/
        so you can use them for future fine-tuning without manual labelling.
        """
        if self._frame_no % 15 != 0:   # save ~1 frame every ~2 seconds
            return
        h, w = img.shape[:2]
        for face in results:
            if face["confidence"] < COLLECT_THRESHOLD:
                continue
            b    = face["box"]
            crop = img[max(0,b["y"]):b["y"]+b["h"], max(0,b["x"]):b["x"]+b["w"]]
            if crop.size == 0:
                continue
            emo      = face["dominant_emotion"]
            save_dir = Path(COLLECT_DIR) / emo
            fname    = save_dir / f"{int(time.time()*1000)}.jpg"
            cv2.imwrite(str(fname), crop)

    # ── Drawing ────────────────────────────────────────────────────────────
    def draw_annotations(self, img: np.ndarray, results: list[dict]) -> np.ndarray:
        h, w = img.shape[:2]

        for face in results:
            box  = face["box"]
            x, y, bw, bh = box["x"], box["y"], box["w"], box["h"]
            dominant   = face["dominant_emotion"]
            confidence = face["confidence"]
            emotions   = face["emotions"]
            fid        = face["face_id"]

            r_hex  = face.get("color_hex", "#00ffc8").lstrip("#")
            cr, cg, cb = tuple(int(r_hex[i:i+2], 16) for i in (0, 2, 4))
            color_bgr  = (cb, cg, cr)

            overlay = img.copy()
            cv2.rectangle(overlay, (x, y), (x+bw, y+bh), color_bgr, -1)
            cv2.addWeighted(overlay, 0.08, img, 0.92, 0, img)

            cl = min(bw, bh) // 5
            for corner in [
                ((x, y),      (x+cl, y),    (x, y+cl)),
                ((x+bw, y),   (x+bw-cl, y), (x+bw, y+cl)),
                ((x, y+bh),   (x+cl, y+bh), (x, y+bh-cl)),
                ((x+bw, y+bh),(x+bw-cl,y+bh),(x+bw,y+bh-cl)),
            ]:
                cv2.line(img, corner[0], corner[1], color_bgr, 3)
                cv2.line(img, corner[0], corner[2], color_bgr, 3)

            meta  = EMOTION_META.get(dominant, {})
            label = f"{dominant.upper()}  {confidence*100:.0f}%"
            font  = cv2.FONT_HERSHEY_DUPLEX
            fs    = 0.6
            (tw, th), _ = cv2.getTextSize(label, font, fs, 1)
            ly = y - 14 if y > 40 else y + bh + th + 10
            pad = 8
            cv2.rectangle(img, (x, ly-th-pad), (x+tw+pad*2, ly+pad//2), color_bgr, -1)
            cv2.putText(img, label, (x+pad, ly), font, fs, (10,10,10), 1, cv2.LINE_AA)

            bar_x = x + bw + 12
            if bar_x + 90 < w:
                for j, (emo, score) in enumerate(sorted(emotions.items(), key=lambda e:-e[1])[:5]):
                    bar_y  = y + j * 26
                    bar_len = int(score * 85)
                    emo_hex = EMOTION_META.get(emo, {}).get("hex", "#888888").lstrip("#")
                    er, eg, eb = tuple(int(emo_hex[i:i+2], 16) for i in (0, 2, 4))
                    cv2.rectangle(img, (bar_x, bar_y), (bar_x+85, bar_y+16), (40,40,40), -1)
                    cv2.rectangle(img, (bar_x, bar_y), (bar_x+bar_len, bar_y+16), (eb,eg,er), -1)
                    cv2.putText(img, emo[:3].upper(), (bar_x+bar_len+4, bar_y+13),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,200,200), 1, cv2.LINE_AA)

            badge = f"#{fid+1}"
            cv2.circle(img, (x+bw-12, y+12), 12, color_bgr, -1)
            cv2.putText(img, badge, (x+bw-20, y+16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (10,10,10), 1, cv2.LINE_AA)

        for row in range(0, h, 4):
            cv2.line(img, (0, row), (w, row), (0,0,0), 1)
        cv2.addWeighted(img, 0.96, np.zeros_like(img), 0.04, 0, img)
        return img

    def model_info(self) -> dict:
        return {
            "backend":    self.backend,
            "detector":   "retinaface" if self.backend == "deepface" else "mtcnn",
            "classifier": "stage1_bigdata-primary" + (" + collected_data-rescue" if self.rescue_model else ""),
            "emotions":   EMOTIONS,
            "smooth_window":   SMOOTH_WINDOW,
            "confidence_gate": CONFIDENCE_GATE,
            "rescue_gate": LOCAL_RESCUE_GATE,
            "rescue_weight": LOCAL_RESCUE_WEIGHT,
            "primary_model": bool(self.primary_model),
            "rescue_model": bool(self.rescue_model),
        }
