# Age Detection Model

A **CNN (Convolutional Neural Network)** trained on the **UTKFace dataset** to predict a person's age from facial images. Uses both regression (exact age) and classification (age group) with batch normalization and dropout.

## Dataset

Download UTKFace dataset and place images in `data/UTKFace/`:
👉 https://susanqq.github.io/UTKFace/

```
Age-Detection-Model/
└── data/
    └── UTKFace/
        ├── 1_0_0_20161219140623097.jpg
        ├── 2_1_0_20161219140624209.jpg
        └── ...
```

## Setup

```bash
pip install -r requirements.txt
```

## Run

**Train the CNN:**
```bash
python train.py
```

**Predict on a single image:**
```bash
python predict.py --image photo.jpg
```

**Predict on a folder of images:**
```bash
python predict.py --folder sample_images/
```

**Generate sample test images:**
```bash
python generate_sample_images.py
python predict.py --folder sample_images/
```

## CNN Architecture

```
Input (64x64x3)
    → Conv2D(32) + BatchNorm + MaxPool
    → Conv2D(64) + BatchNorm + MaxPool
    → Conv2D(128) + BatchNorm + MaxPool
    → Conv2D(256) + BatchNorm + GlobalAvgPool
    → Dense(256) + Dropout(0.4)
    → Dense(128) + Dropout(0.3)
    ├── age_output   → Dense(1)          [Regression — exact age]
    └── group_output → Dense(5, softmax) [Classification — age group]
```

## Age Groups

| Group | Age Range |
|-------|-----------|
| Child | 0 - 12 |
| Teen | 13 - 19 |
| Young Adult | 20 - 35 |
| Middle Age | 36 - 55 |
| Senior | 56+ |

## Output Files

| File | Description |
|------|-------------|
| `outputs/best_model.keras` | Best saved CNN model |
| `output.csv` | Sample predictions with true vs predicted age |
| `model_report.txt` | MAE, accuracy and training summary |
| `plots/training_curves.png` | MAE and accuracy curves per epoch |
| `sample_output.csv` | Preview of expected results |

## Project Structure

```
Age-Detection-Model/
├── train.py                   ← train CNN on UTKFace
├── predict.py                 ← predict on new images
├── generate_sample_images.py  ← create test images for demo
├── requirements.txt
├── README.md
├── sample_output.csv
├── data/UTKFace/              ← download dataset here
├── outputs/                   ← saved model
└── plots/                     ← training curves
```

## Expected Results

| Metric | Score |
|--------|-------|
| Age MAE | ~6-8 years |
| Age Group Accuracy | ~70-80% |
