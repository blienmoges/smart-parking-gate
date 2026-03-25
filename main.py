# Import required libraries
import json
import cv2
from ultralytics import YOLO
import numpy as np
import re
import os
import sqlite3
from datetime import datetime
from paddleocr import PaddleOCR
from fastapi import FastAPI, UploadFile, File

app = FastAPI()

@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
   
    pass
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# -------- CONFIG --------
IMAGE_PATH = "data/carImage12.jpg"   # <-- your image
WEIGHTS_PATH = "weights/best.pt"
CONF_THRES = 0             # lower = more detections for debugging
IMG_SIZE = 960                     # larger helps small plates

# Expand YOLO box so crop includes the full plate
PAD_L = 0.20   # 5% left padding
PAD_R = 0.55   # 55% right padding  <-- IMPORTANT (your box was cutting off "29AA")
PAD_T = 0.12   # 12% top padding
PAD_B = 0.12   # 12% bottom padding
# ------------------------

os.makedirs("json", exist_ok=True)

# Load image (NOT VideoCapture)
frame = cv2.imread(IMAGE_PATH)
if frame is None:
    raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

# If image has alpha channel (RGBA), convert to BGR
if len(frame.shape) == 3 and frame.shape[2] == 4:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

# Initialize YOLOv10 model
model = YOLO(WEIGHTS_PATH)

# OCR language:
# - Use "ch" for Chinese plates (黑E99999 etc.)
# - Use "en" for Latin plates (94529AA etc.)
# Change this if needed:
OCR_LANG = "en"
ocr = PaddleOCR(use_angle_cls=True, use_gpu=False, lang=OCR_LANG)


def paddle_ocr(img, x1, y1, x2, y2):
    h, w = img.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return ""

    crop = img[y1:y2, x1:x2]

    # Save crop for debugging (optional)
    # cv2.imwrite("json/debug_crop.jpg", crop)

    # OPTIONAL: remove left symbols only if you want digits/letters only.
    # For Ethiopian plate, keeping everything can still work, but removing left symbols helps.
    ch, cw = crop.shape[:2]
    crop = crop[:, int(cw * 0.18):]   # less aggressive than 0.25 so we don't lose useful parts

    # Upscale + contrast
    crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    crop = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    result = ocr.ocr(crop, det=False, rec=True, cls=False)
    if not result:
        return ""

    # ---- normalize possible nested result formats into candidates ----
    candidates = []
    item = result
    while isinstance(item, list) and len(item) == 1 and isinstance(item[0], list):
        item = item[0]

    if isinstance(item, list) and len(item) == 2 and isinstance(item[0], str):
        candidates = [item]
    elif isinstance(item, list):
        for cand in item:
            if isinstance(cand, (list, tuple)) and len(cand) >= 2 and isinstance(cand[0], str):
                candidates.append([cand[0], cand[1]])

    if not candidates:
        return "", 0.0

    # Pick best: prefer longer strings, then higher score
    best_text = ""
    best_score = -1.0

    for txt, score in candidates:
        if txt is None:
            continue

        cleaned = re.sub(r"\W", "", str(txt))
        cleaned = cleaned.replace("O", "0").replace("???", "").replace("粤", "")
        if not cleaned:
            continue

        s = 0.0
        try:
            if score is not None:
                s = float(score)
        except:
            s = 0.0

        # Prefer longer than 4 chars (plate-like)
        if len(cleaned) < 5:
            continue

        if (len(cleaned) > len(best_text)) or (len(cleaned) == len(best_text) and s > best_score):
            best_text = cleaned
            best_score = s

    # Fallback: accept best short result if nothing else
    if not best_text:
        for txt, score in candidates:
            cleaned = re.sub(r"\W", "", str(txt)).replace("O", "0")
            if len(cleaned) > len(best_text):
                best_text = cleaned

    return best_text, best_score


def save_to_database(license_plates, start_time, end_time):
    conn = sqlite3.connect("licensePlatesDatabase.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS LicensePlates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_time TEXT,
            end_time TEXT,
            license_plate TEXT
        )
    """)

    for plate in license_plates:
        cursor.execute("""
            INSERT INTO LicensePlates(start_time, end_time, license_plate)
            VALUES (?, ?, ?)
        """, (start_time.isoformat(), end_time.isoformat(), plate))

    conn.commit()
    conn.close()


def save_json(license_plates, startTime, endTime):
    interval_data = {
        "Start Time": startTime.isoformat(),
        "End Time": endTime.isoformat(),
        "License Plate": list(license_plates)
    }

    interval_file_path = "json/output_" + datetime.now().strftime("%Y%m%d%H%M%S") + ".json"
    with open(interval_file_path, "w") as f:
        json.dump(interval_data, f, indent=2)

    cumulative_file_path = "json/LicensePlateData.json"
    if os.path.exists(cumulative_file_path):
        with open(cumulative_file_path, "r") as f:
            existing_data = json.load(f)
    else:
        existing_data = []

    existing_data.append(interval_data)

    with open(cumulative_file_path, "w") as f:
        json.dump(existing_data, f, indent=2)

    save_to_database(license_plates, startTime, endTime)


# ---- PROCESS IMAGE ONCE ----
startTime = datetime.now()
license_plates = set()

results = model.predict(frame, conf=CONF_THRES, imgsz=IMG_SIZE)
det_count = sum(len(r.boxes) for r in results)
print("Detections:", det_count)
# print("Detections:", sum(len(r.boxes) for r in results))
if det_count == 0:
    print("YOLO failed → running OCR on full image")

    crop = frame.copy()
    crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    crop = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    result = ocr.ocr(crop, det=True, rec=True, cls=False)

    if result:
        print("OCR result:", result)
H, W = frame.shape[:2]

for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # --- expand YOLO box so crop includes full plate ---
        bw = x2 - x1
        bh = y2 - y1
        x1p = max(0, int(x1 - PAD_L * bw))
        y1p = max(0, int(y1 - PAD_T * bh))
        x2p = min(W, int(x2 + PAD_R * bw))
        y2p = min(H, int(y2 + PAD_B * bh))

        # Draw expanded box (this is the crop used for OCR)
        cv2.rectangle(frame, (x1p, y1p), (x2p, y2p), (255, 0, 0), 2)

        label, ocr_conf = paddle_ocr(frame, x1p, y1p, x2p, y2p)
        if label:
            print(f"PLATE: {label} | OCR confidence: {ocr_conf:.2f}")
            license_plates.add(label)

            cv2.putText(frame, label, (x1p, max(0, y1p - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

endTime = datetime.now()
save_json(license_plates, startTime, endTime)

annotated_path = "json/annotated_" + datetime.now().strftime("%Y%m%d%H%M%S") + ".jpg"
cv2.imwrite(annotated_path, frame)
print("Saved annotated image:", annotated_path)

# cv2.imshow("Image", frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()     