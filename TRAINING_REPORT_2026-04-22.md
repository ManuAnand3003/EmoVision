# EmoVision Retraining Report (2026-04-22)

## Scope Completed

- Reconfigured environment to GPU-enabled PyTorch (`torch 2.11.0+cu128`)
- Stage-1 retraining on big dataset (`my_dataset`)
- Stage-2 fine-tuning on collected data (`collected_data`) using stage-1 checkpoint
- ONNX exports for stage-1 and stage-2 models
- Inference and API smoke tests
- Documentation updates
- Privacy hardening (`collected_data/` now git-ignored)

## Environment

- OS: Windows
- Python: 3.12 (venv)
- GPU: NVIDIA GeForce RTX 4080 Laptop GPU
- CUDA available in PyTorch: `True`

## Dataset Snapshot

### my_dataset (big-data stage)
- angry: 4953
- disgust: 547
- fear: 5121
- happy: 8988
- neutral: 6198
- sad: 6077
- surprise: 4002
- total: 35886

### collected_data (fine-tune stage)
- angry: 22
- disgust: 5
- fear: 0
- happy: 9
- neutral: 10
- sad: 9
- surprise: 12
- total: 67

## Training Runs

## Stage 1 (Big Dataset)
Command:

```bash
python finetune_local.py --data my_dataset --output models/stage1_bigdata --epochs 6 --batch 64 --lr 1e-4 --workers 0
```

Result summary:
- best val accuracy: 0.6283 (62.8%)
- val set size: 5383
- saved checkpoint: `models/stage1_bigdata/finetuned_model.pth`

## Stage 2 (Collected Data)
Command:

```bash
python finetune_local.py --data collected_data --output models --checkpoint models/stage1_bigdata/finetuned_model.pth --epochs 20 --batch 8 --lr 2e-5 --workers 0
```

Result summary:
- best val accuracy: 0.7273 (72.7%)
- val set size: 11
- saved checkpoint: `models/finetuned_model.pth`

Important caveat:
- Stage-2 validation is very small and class-missing (`fear` has 0 samples). The 72.7% is useful as a local directional signal, not a stable benchmark.

## Artifacts Produced

- `models/stage1_bigdata/finetuned_model.pth`
- `models/stage1_bigdata/finetuned_model.onnx`
- `models/stage1_bigdata/finetune_confusion.png`
- `models/stage1_bigdata/finetune_curves.png`
- `models/finetuned_model.pth`
- `models/finetuned_model.onnx`
- `models/finetune_confusion.png`
- `models/finetune_curves.png`
- `models/finetune_history.json`

## Runtime Validation

## Inference Pipeline Smoke Test
- Loaded pipeline successfully
- Backend: `deepface`
- ONNX ensemble active: `True`
- Face detection on sample image: 1 face detected

## API Smoke Test
Using FastAPI TestClient:
- `GET /api/health`: 200
- `POST /api/analyze`: 200
- sample response included valid `face_count` and `inference_ms`

## Issues Found and Fixed

1. ONNX export initially failed (`ModuleNotFoundError: onnxscript`)
   - Fix: installed `onnxscript` and added it to `requirements.txt`

2. Fine-tuning script inefficiency (double forward pass per batch)
   - Fix: reuse model outputs for loss and accuracy

3. Classification report could break when classes are missing
   - Fix: explicit labels list for 7-class report + confusion matrix

4. Checkpoint loading order bug for stage-2 adaptation
   - Fix: replace classifier head before loading checkpoint

5. Unstable scheduler for tiny dataset
   - Fix: replaced `OneCycleLR` with `CosineAnnealingLR`

6. Potential privacy leak via git tracking of local collected faces
   - Fix: added `collected_data/` and nested model artifacts to `.gitignore`

## Time Estimate

Approximate runtime on this machine (RTX 4080 laptop GPU):
- Environment prep / dependency fix: 10-20 min (first time)
- Stage 1 (6 epochs, 35.9k images): ~10-25 min
- Stage 2 (20 epochs, 67 images): ~1-3 min
- Validation + docs update: ~10-20 min

Expected full rerun time: ~25-70 minutes depending on cache state, disk speed, and first-time package installs.

## Suggestions and Upgrade Ideas

1. Data quality first:
- Add real `fear` samples to `collected_data`; current count is zero.
- Increase `disgust` and `happy` samples to reduce class skew.

2. Better evaluation:
- Create a fixed holdout set (100+ images/class) not used in training.
- Track macro-F1 and per-class recall, not only accuracy.

3. Stronger adaptation strategy:
- Keep stage-1 as foundation and run short low-LR stage-2 sessions weekly.
- Save date-versioned checkpoints (for rollback and comparison).

4. Training robustness:
- Add early stopping + best-checkpoint by macro-F1.
- Add class-balanced batch sampler with minimum per-class batch presence.

5. Production inference:
- Benchmark ONNX Runtime with CUDA provider for low-latency inference.
- Add confidence calibration and optional "uncertain" output class.

6. Privacy and governance:
- Keep `collected_data/` local only.
- Add optional script to blur/export audit-safe samples for sharing metrics without raw faces.
