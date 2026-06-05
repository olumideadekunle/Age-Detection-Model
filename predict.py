"""
Run age prediction on a single image or folder using the trained CNN model.

Usage:
    python predict.py --image photo.jpg
    python predict.py --folder sample_images/
"""

import os
import argparse
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf

MODEL_PATH = "outputs/best_model.keras"
IMG_SIZE   = (64, 64)
AGE_GROUPS = ["Child (0-12)", "Middle (36-55)", "Senior (56+)", "Teen (13-19)", "Young (20-35)"]

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Trained model not found.\n"
        "Run train.py first to train and save the model."
    )

model = tf.keras.models.load_model(MODEL_PATH)

def preprocess(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype("float32") / 255.0

def predict_image(image_path: str):
    img = preprocess(image_path)
    if img is None:
        print(f"Could not read: {image_path}")
        return None
    preds      = model.predict(np.expand_dims(img, 0), verbose=0)
    pred_age   = int(preds[0][0][0])
    pred_group = AGE_GROUPS[np.argmax(preds[1][0])]
    return pred_age, pred_group

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Age Detection Predictor")
    parser.add_argument("--image",  type=str, help="Path to a single image")
    parser.add_argument("--folder", type=str, help="Path to a folder of images")
    args = parser.parse_args()

    rows = []

    if args.image:
        result = predict_image(args.image)
        if result:
            age, group = result
            print(f"\nImage  : {args.image}")
            print(f"Age    : {age} years")
            print(f"Group  : {group}")
            rows.append({"Image": args.image, "PredictedAge": age, "AgeGroup": group})

    elif args.folder:
        supported = (".jpg", ".jpeg", ".png", ".bmp")
        files = [f for f in os.listdir(args.folder) if f.lower().endswith(supported)]
        print(f"Found {len(files)} images in {args.folder}\n")
        for fname in files:
            path   = os.path.join(args.folder, fname)
            result = predict_image(path)
            if result:
                age, group = result
                print(f"  {fname:30s} → Age: {age:3d}  Group: {group}")
                rows.append({"Image": fname, "PredictedAge": age, "AgeGroup": group})
    else:
        print("Usage:")
        print("  python predict.py --image photo.jpg")
        print("  python predict.py --folder sample_images/")

    if rows:
        pd.DataFrame(rows).to_csv("predict_output.csv", index=False)
        print(f"\nResults saved to predict_output.csv")
