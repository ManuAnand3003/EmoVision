"""
EmoVision — AI Face Emotion Detection System
Backend: FastAPI + WebSockets + DeepFace + Custom Model
"""

import base64
import io
import json
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from engine.pipeline import EmotionPipeline

# ─── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="EmoVision API",
    description="Real-time face emotion detection powered by deep learning",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (our frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ─── Load ML Pipeline (once at startup) ───────────────────────────────────────
pipeline = EmotionPipeline()


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main frontend SPA."""
    html_path = Path("static/index.html")
    return HTMLResponse(content=html_path.read_text(), status_code=200)


@app.post("/api/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """
    Analyze an uploaded image for facial emotions.
    Returns: list of detected faces with emotion scores, bounding boxes,
             landmarks, dominant emotion, and annotated image as base64.
    """
    try:
        # Read uploaded image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Could not decode image. Please upload a valid JPG/PNG."},
            )

        start = time.perf_counter()
        results = pipeline.analyze(img_bgr)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)

        # Draw annotations on a copy of the image
        annotated = pipeline.draw_annotations(img_bgr.copy(), results)

        # Encode annotated image as base64 for frontend
        _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
        annotated_b64 = base64.b64encode(buffer).decode("utf-8")

        return {
            "success": True,
            "inference_ms": elapsed_ms,
            "face_count": len(results),
            "faces": results,
            "annotated_image": f"data:image/jpeg;base64,{annotated_b64}",
        }

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time camera emotion detection.
    Client sends base64 JPEG frames, server responds with JSON results.

    Protocol:
      Client → Server: base64 encoded JPEG string
      Server → Client: JSON { faces: [...], inference_ms: float, fps: float }
    """
    await websocket.accept()
    frame_times = []

    try:
        while True:
            # Receive frame from browser
            data = await websocket.receive_text()

            frame_start = time.perf_counter()

            # Decode base64 → OpenCV image
            if data.startswith("data:image"):
                data = data.split(",", 1)[1]

            img_bytes = base64.b64decode(data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            # Run inference
            results = pipeline.analyze(frame, fast_mode=True)
            elapsed_ms = round((time.perf_counter() - frame_start) * 1000, 1)

            # Rolling FPS calculation
            frame_times.append(time.perf_counter())
            frame_times = [t for t in frame_times if time.perf_counter() - t < 2.0]
            fps = round(len(frame_times) / 2.0, 1)

            await websocket.send_json({
                "faces": results,
                "inference_ms": elapsed_ms,
                "fps": fps,
                "face_count": len(results),
            })

    except WebSocketDisconnect:
        print("Client disconnected from WebSocket stream.")
    except Exception as e:
        traceback.print_exc()
        await websocket.send_json({"error": str(e)})


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model": pipeline.model_info(),
        "version": "2.0.0",
    }


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
