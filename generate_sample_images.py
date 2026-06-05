"""
Generates simple synthetic face-like test images in sample_images/.
These are placeholder images to test the batch pipeline without needing real photos.
For best results, replace these with real face photos.

Usage: python generate_sample_images.py
"""

import cv2
import numpy as np
import os

os.makedirs("sample_images", exist_ok=True)

def make_face_image(filename, skin_color, age_label, gender_label):
    img = np.ones((300, 300, 3), dtype=np.uint8) * 200

    # Face oval
    cv2.ellipse(img, (150, 140), (85, 105), 0, 0, 360, skin_color, -1)

    # Eyes
    cv2.circle(img, (115, 120), 12, (50, 50, 50), -1)
    cv2.circle(img, (185, 120), 12, (50, 50, 50), -1)
    cv2.circle(img, (118, 118), 5, (255, 255, 255), -1)
    cv2.circle(img, (188, 118), 5, (255, 255, 255), -1)

    # Nose
    cv2.line(img, (150, 135), (140, 165), (100, 80, 60), 2)
    cv2.line(img, (150, 135), (160, 165), (100, 80, 60), 2)

    # Mouth
    cv2.ellipse(img, (150, 185), (30, 15), 0, 0, 180, (120, 60, 60), 2)

    # Label
    cv2.putText(img, f"{gender_label}, {age_label}", (30, 280),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    path = os.path.join("sample_images", filename)
    cv2.imwrite(path, img)
    print(f"  Created {path}")

samples = [
    ("person1.jpg", (180, 140, 110), "25-32", "Male"),
    ("person2.jpg", (210, 170, 140), "15-20", "Female"),
    ("person3.jpg", (160, 120, 90),  "38-43", "Male"),
    ("person4.jpg", (220, 185, 155), "48-53", "Female"),
    ("person5.jpg", (170, 130, 100), "60+",   "Male"),
]

print("Generating sample images in sample_images/...\n")
for args in samples:
    make_face_image(*args)

print(f"\nDone! {len(samples)} sample images created.")
print("Run batch detection with:")
print("  python age_detection.py --batch sample_images/")
