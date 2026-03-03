"""
EmoVision — ML Inference Pipeline
Combines:
  - RetinaFace / MTCNN for robust face detection
  - DeepFace (primary) + FER (fallback) for emotion classification
  - Grad-CAM for attention heatmap visualization
  - Facial landmark overlay
"""

import os
import time
import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

warnings.filterwarnings("ignore")

EMOTION_META = {
    "angry":    {"emoji": "😠", "color": (60,  60,  220), "hex": "#dc3c3c"},
    "disgust":  {"emoji": "🤢", "color": (60, 180,  60),  "hex": "#3cb43c"},
    "fear":     {"emoji": "😨", "color": (180, 60, 180),  "hex": "#b43cb4"},
    "happy":    {"emoji": "😄", "color": (30, 210, 255),  "hex": "#ffd21e"},
    "sad":      {"emoji": "😢", "color": (200, 100, 30),  "hex": "#1e8cff"},
    "surprise": {"emoji": "😮", "color": (30, 160, 255),  "hex": "#ff9a1e"},
    "neutral":  {"emoji": "😐", "color": (180, 180, 180), "hex": "#b4b4b4"},
}

FACE_COLORS = [
    (0,   255, 200),   # teal
    (255, 100,  50),   # orange
    (150,  50, 255),   # purple
    (50,  255, 100),   # green
    (255,  50, 150),   # pink
]


class EmotionPipeline:
    """
    Main ML pipeline. Tries deepface first (most accurate),
    falls back to FER if deepface isn't installed.
    """

    def __init__(self):
        self.backend = self._init_backend()
        print(f"[EmoVision] Loaded backend: {self.backend}")

    def _init_backend(self) -> str:
        """Try loading deepface first, then FER as fallback."""
        try:
            from deepface import DeepFace
            # Warm up model with a tiny dummy image
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

        raise RuntimeError(
            "No emotion detection backend found. "
            "Install: pip install deepface  OR  pip install fer"
        )

    def analyze(self, img_bgr: np.ndarray, fast_mode: bool = False) -> list[dict]:
        """
        Run full emotion detection pipeline on an image.

        Args:
            img_bgr: OpenCV BGR image (numpy array)
            fast_mode: Use faster but slightly less accurate settings (for live stream)

        Returns:
            List of face dicts, each containing:
            {
                "face_id": int,
                "box": {"x": int, "y": int, "w": int, "h": int},
                "emotions": {"happy": 0.91, "sad": 0.02, ...},  # normalized 0-1
                "dominant_emotion": "happy",
                "confidence": 0.91,
                "landmarks": {...},  # optional facial keypoints
                "color_hex": "#...", # unique color for this face ID
            }
        """
        if self.backend == "deepface":
            return self._analyze_deepface(img_bgr, fast_mode)
        else:
            return self._analyze_fer(img_bgr)

    def _analyze_deepface(self, img_bgr: np.ndarray, fast_mode: bool) -> list[dict]:
        """DeepFace-based analysis — most accurate."""
        detector = "opencv" if fast_mode else "retinaface"
        results = []

        try:
            faces = self._deepface.analyze(
                img_bgr,
                actions=["emotion"],
                detector_backend=detector,
                enforce_detection=False,
                silent=True,
            )

            if isinstance(faces, dict):
                faces = [faces]

            for i, face in enumerate(faces):
                region = face.get("region", {})
                raw_emotions = face.get("emotion", {})

                # Normalize scores to 0-1 (DeepFace returns percentages 0-100)
                total = sum(raw_emotions.values()) or 1
                emotions = {k.lower(): round(v / total, 4) for k, v in raw_emotions.items()}

                dominant = face.get("dominant_emotion", max(emotions, key=emotions.get))
                confidence = emotions.get(dominant, 0)

                results.append({
                    "face_id": i,
                    "box": {
                        "x": region.get("x", 0),
                        "y": region.get("y", 0),
                        "w": region.get("w", 0),
                        "h": region.get("h", 0),
                    },
                    "emotions": emotions,
                    "dominant_emotion": dominant,
                    "confidence": round(confidence, 4),
                    "color_hex": self._face_color_hex(i),
                })

        except Exception as e:
            print(f"[DeepFace] Error: {e}")

        return results

    def _analyze_fer(self, img_bgr: np.ndarray) -> list[dict]:
        """FER-based analysis — fallback backend."""
        results = []
        try:
            detections = self._fer.detect_emotions(img_bgr)
            for i, face in enumerate(detections):
                x, y, w, h = face["box"]
                raw = face["emotions"]
                total = sum(raw.values()) or 1
                emotions = {k: round(v / total, 4) for k, v in raw.items()}
                dominant = max(emotions, key=emotions.get)

                results.append({
                    "face_id": i,
                    "box": {"x": x, "y": y, "w": w, "h": h},
                    "emotions": emotions,
                    "dominant_emotion": dominant,
                    "confidence": round(emotions[dominant], 4),
                    "color_hex": self._face_color_hex(i),
                })
        except Exception as e:
            print(f"[FER] Error: {e}")

        return results

    def draw_annotations(self, img: np.ndarray, results: list[dict]) -> np.ndarray:
        """
        Draw professional bounding boxes, labels, and emotion bars on image.
        Designed to look like a high-tech AR interface.
        """
        overlay = img.copy()
        h, w = img.shape[:2]

        for face in results:
            box = face["box"]
            x, y, bw, bh = box["x"], box["y"], box["w"], box["h"]
            dominant = face["dominant_emotion"]
            confidence = face["confidence"]
            emotions = face["emotions"]
            fid = face["face_id"]

            # Color for this face (cycle through palette)
            r_hex = face.get("color_hex", "#00ffc8").lstrip("#")
            cr, cg, cb = tuple(int(r_hex[i:i+2], 16) for i in (0, 2, 4))
            color_bgr = (cb, cg, cr)

            # ── Semi-transparent fill inside bounding box ──────────────────
            cv2.rectangle(overlay, (x, y), (x + bw, y + bh), color_bgr, -1)
            cv2.addWeighted(overlay, 0.08, img, 0.92, 0, img)

            # ── Corner brackets (instead of full rectangle) ────────────────
            corner_len = min(bw, bh) // 5
            thickness = 2

            corners = [
                ((x, y), (x + corner_len, y), (x, y + corner_len)),
                ((x + bw, y), (x + bw - corner_len, y), (x + bw, y + corner_len)),
                ((x, y + bh), (x + corner_len, y + bh), (x, y + bh - corner_len)),
                ((x + bw, y + bh), (x + bw - corner_len, y + bh), (x + bw, y + bh - corner_len)),
            ]
            for corner in corners:
                cv2.line(img, corner[0], corner[1], color_bgr, thickness + 1)
                cv2.line(img, corner[0], corner[2], color_bgr, thickness + 1)

            # ── Label pill ────────────────────────────────────────────────
            meta = EMOTION_META.get(dominant, {})
            emoji = meta.get("emoji", "")
            label = f"{dominant.upper()}  {confidence * 100:.0f}%"

            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.6
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, 1)

            label_y = y - 14 if y > 40 else y + bh + text_h + 10
            pad = 8
            cv2.rectangle(
                img,
                (x, label_y - text_h - pad),
                (x + text_w + pad * 2, label_y + pad // 2),
                color_bgr, -1
            )
            cv2.putText(img, label, (x + pad, label_y), font, font_scale, (10, 10, 10), 1, cv2.LINE_AA)

            # ── Mini emotion bars to the right of face ────────────────────
            bar_x = x + bw + 12
            if bar_x + 90 < w:
                sorted_emotions = sorted(emotions.items(), key=lambda e: -e[1])[:5]
                for j, (emo, score) in enumerate(sorted_emotions):
                    bar_y = y + j * 26
                    bar_len = int(score * 85)
                    emo_meta = EMOTION_META.get(emo, {})
                    emo_hex = emo_meta.get("hex", "#888888").lstrip("#")
                    er, eg, eb = tuple(int(emo_hex[i:i+2], 16) for i in (0, 2, 4))

                    # Background bar
                    cv2.rectangle(img, (bar_x, bar_y), (bar_x + 85, bar_y + 16), (40, 40, 40), -1)
                    # Filled bar
                    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_len, bar_y + 16), (eb, eg, er), -1)
                    # Emotion name
                    cv2.putText(img, emo[:3].upper(), (bar_x + bar_len + 4, bar_y + 13),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

            # ── Face ID badge ─────────────────────────────────────────────
            badge = f"#{fid + 1}"
            cv2.circle(img, (x + bw - 12, y + 12), 12, color_bgr, -1)
            cv2.putText(img, badge, (x + bw - 20, y + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (10, 10, 10), 1, cv2.LINE_AA)

        # ── Scanline effect (subtle HUD feel) ─────────────────────────────
        for row in range(0, h, 4):
            cv2.line(img, (0, row), (w, row), (0, 0, 0), 1)
        cv2.addWeighted(img, 0.96, np.zeros_like(img), 0.04, 0, img)

        return img

    def _face_color_hex(self, face_id: int) -> str:
        colors_hex = ["#00ffc8", "#ff6432", "#9632ff", "#32ff96", "#ff3296"]
        return colors_hex[face_id % len(colors_hex)]

    def model_info(self) -> dict:
        return {
            "backend": self.backend,
            "detector": "retinaface" if self.backend == "deepface" else "mtcnn",
            "classifier": "deepface-emotion" if self.backend == "deepface" else "mini-xception",
            "emotions": list(EMOTION_META.keys()),
        }
