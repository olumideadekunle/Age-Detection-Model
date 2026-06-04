# Age Detection Model

Detects **age range** and **gender** from images or live webcam using OpenCV's pre-trained Caffe deep learning models. No dataset download required — model weights are downloaded automatically on first run.

## Setup

```bash
pip install -r requirements.txt
```

## Run

**On an image:**
```bash
python age_detection.py --image photo.jpg
```

**On webcam (live):**
```bash
python age_detection.py --webcam
```

**Custom output path:**
```bash
python age_detection.py --image photo.jpg --output result.jpg
```

## How It Works

1. Face detection — OpenCV DNN + ResNet SSD model
2. Age prediction — Caffe model classifies into 8 age buckets: `(0-2) (4-6) (8-12) (15-20) (25-32) (38-43) (48-53) (60-100)`
3. Gender prediction — Caffe model classifies as Male / Female

Model weights (~30 MB) are downloaded automatically into `models/` on first run.

## Project Structure

```
Age-Detection-Model/
├── age_detection.py   ← main script
├── requirements.txt
├── README.md
└── models/            ← auto-downloaded on first run
```
