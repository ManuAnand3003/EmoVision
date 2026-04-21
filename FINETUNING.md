# EmoVision Fine-Tuning Guide

This project now uses a two-stage training flow:

1. Stage 1: train on large dataset (`my_dataset`)
2. Stage 2: fine-tune on local collected data (`collected_data`)

Both stages run through `finetune_local.py` and support GPU automatically.

## 1) Environment Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

GPU check:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 2) Dataset Layout

Expected structure for both datasets:

```text
<dataset_root>/
  angry/
  disgust/
  fear/
  happy/
  neutral/
  sad/
  surprise/
```

Current project datasets:

- Big dataset: `my_dataset/` (~35k images)
- Local adaptation dataset: `collected_data/` (small and imbalanced)

## 3) Stage 1 - Big Dataset Training

```bash
python finetune_local.py \
  --data my_dataset \
  --output models/stage1_bigdata \
  --epochs 6 \
  --batch 64 \
  --lr 1e-4 \
  --workers 0
```

Outputs:

- `models/stage1_bigdata/finetuned_model.pth`
- `models/stage1_bigdata/finetuned_model.onnx`
- `models/stage1_bigdata/finetune_confusion.png`
- `models/stage1_bigdata/finetune_curves.png`

## 4) Stage 2 - Fine-Tune on Collected Data

```bash
python finetune_local.py \
  --data collected_data \
  --output models \
  --checkpoint models/stage1_bigdata/finetuned_model.pth \
  --epochs 20 \
  --batch 8 \
  --lr 2e-5 \
  --workers 0
```

Outputs:

- `models/finetuned_model.pth`
- `models/finetuned_model.onnx`
- `models/finetune_confusion.png`
- `models/finetune_curves.png`
- `models/finetune_history.json`

## 5) Inference Integration

No manual code changes needed.

`engine/pipeline.py` automatically loads these in order:

1. `models/finetuned_model.onnx`
2. `models/emotion_model.onnx`

If ONNX is found, it is used in an ensemble with DeepFace.

## 6) Smoke Test

```bash
python -c "from fastapi.testclient import TestClient; import main; c=TestClient(main.app); print(c.get('/api/health').status_code)"
```

## 7) Troubleshooting

- If training hangs on Windows, use `--workers 0`.
- If ONNX export fails, confirm `onnxscript` is installed.
- If collected-data metrics are unstable, add more balanced samples, especially `fear` and `disgust`.
- Tiny validation sets can give noisy accuracy. Treat stage-2 accuracy as directional, not final.

## 8) Privacy and Git

`collected_data/` is now ignored in `.gitignore` so local face data is not committed.
