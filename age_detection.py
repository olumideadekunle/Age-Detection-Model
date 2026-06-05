import cv2
import numpy as np
import argparse
import urllib.request
import os

# ── Pre-trained model files (Caffe models by Gil Levi & Tal Hassner) ──────────
MODEL_FILES = {
    "face_proto":  "deploy_face.prototxt",
    "face_model":  "res10_300x300_ssd_iter_140000.caffemodel",
    "age_proto":   "deploy_age.prototxt",
    "age_model":   "age_net.caffemodel",
    "gender_proto": "deploy_gender.prototxt",
    "gender_model": "gender_net.caffemodel",
}

URLS = {
    "deploy_face.prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel": "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
    "deploy_age.prototxt": "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/models/age_net.prototxt",
    "age_net.caffemodel": "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/age_net.caffemodel",
    "deploy_gender.prototxt": "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/models/gender_net.prototxt",
    "gender_net.caffemodel": "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/gender_net.caffemodel",
}

AGE_BUCKETS   = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
GENDER_LIST   = ["Male", "Female"]
MODEL_MEAN    = (78.4263377603, 87.7689143744, 114.895847746)
FACE_CONF_THR = 0.7


def download_models():
    """Download model weights if not already present."""
    os.makedirs("models", exist_ok=True)
    for filename, url in URLS.items():
        dest = os.path.join("models", filename)
        if not os.path.exists(dest):
            print(f"Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, dest)
                print(f"  Saved to {dest}")
            except Exception as e:
                print(f"  Failed to download {filename}: {e}")
                print(f"  Please download manually from:\n  {url}\n  and place in models/")


def load_networks():
    face_net   = cv2.dnn.readNet(os.path.join("models", MODEL_FILES["face_proto"]),
                                  os.path.join("models", MODEL_FILES["face_model"]))
    age_net    = cv2.dnn.readNet(os.path.join("models", MODEL_FILES["age_proto"]),
                                  os.path.join("models", MODEL_FILES["age_model"]))
    gender_net = cv2.dnn.readNet(os.path.join("models", MODEL_FILES["gender_proto"]),
                                  os.path.join("models", MODEL_FILES["gender_model"]))
    return face_net, age_net, gender_net


def detect_faces(face_net, frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), MODEL_MEAN, swapRB=False)
    face_net.setInput(blob)
    detections = face_net.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > FACE_CONF_THR:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            # Add padding around face
            pad = 20
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
            faces.append((x1, y1, x2, y2, float(confidence)))
    return faces


def predict_age_gender(age_net, gender_net, face_img):
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN, swapRB=False)

    gender_net.setInput(blob)
    gender = GENDER_LIST[gender_net.forward()[0].argmax()]

    age_net.setInput(blob)
    age = AGE_BUCKETS[age_net.forward()[0].argmax()]

    return age, gender


def process_image(image_path, output_path="output.jpg"):
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    face_net, age_net, gender_net = load_networks()
    faces = detect_faces(face_net, frame)

    if not faces:
        print("No faces detected in the image.")
        return

    print(f"Detected {len(faces)} face(s):\n")
    for i, (x1, y1, x2, y2, conf) in enumerate(faces):
        face_img = frame[y1:y2, x1:x2]
        age, gender = predict_age_gender(age_net, gender_net, face_img)

        label = f"{gender}, {age}"
        print(f"  Face {i+1}: {label}  (confidence: {conf:.2f})")

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(frame, (x1, y1 - 30), (x2, y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    cv2.imwrite(output_path, frame)
    print(f"\nResult saved to: {output_path}")

    # Export results to CSV
    import csv
    csv_path = "output.csv"
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["ImageName", "FaceID", "DetectedGender", "AgeRange", "Confidence"])
        for i, (x1, y1, x2, y2, conf) in enumerate(faces):
            face_img = frame[y1:y2, x1:x2]
            age, gender = predict_age_gender(age_net, gender_net, face_img)
            writer.writerow([os.path.basename(image_path), i + 1, gender, age, round(conf, 4)])
    print(f"Results saved to {csv_path}")


def process_webcam():
    face_net, age_net, gender_net = load_networks()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("Webcam started. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detect_faces(face_net, frame)
        for (x1, y1, x2, y2, conf) in faces:
            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0:
                continue
            age, gender = predict_age_gender(age_net, gender_net, face_img)
            label = f"{gender}, {age}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y1 - 30), (x2, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        cv2.imshow("Age & Gender Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Age & Gender Detection")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--webcam", action="store_true", help="Use webcam")
    parser.add_argument("--output", type=str, default="output.jpg", help="Output image path")
    args = parser.parse_args()

    download_models()

    if args.image:
        process_image(args.image, args.output)
    elif args.webcam:
        process_webcam()
    else:
        print("Usage:")
        print("  Image:  python age_detection.py --image photo.jpg")
        print("  Webcam: python age_detection.py --webcam")
