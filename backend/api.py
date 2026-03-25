from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import re
import os
import json
import sqlite3
import base64
from datetime import datetime
from ultralytics import YOLO
from paddleocr import PaddleOCR

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = FastAPI(
    title="SmartParking Gate",
    description="Automated License Plate Recognition (ALPR) System for Smart Parking Management",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- CONFIG --------
MY_SECRET_KEY = "SmartParking_2026_Secure"
WEIGHTS_PATH = "weights/best.pt"
CONF_THRES = 0.1
IMG_SIZE = 960

PAD_L = 0.15
PAD_R = 0.65
PAD_T = 0.12
PAD_B = 0.12

OCR_LANG = "en"
DB_PATH = "licensePlatesDatabase.db"
JSON_DIR = "json"

os.makedirs(JSON_DIR, exist_ok=True)

# -------- INITIALIZE MODELS --------
model = YOLO(WEIGHTS_PATH)
ocr = PaddleOCR(use_angle_cls=True, use_gpu=False, lang=OCR_LANG)


def init_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS LicensePlates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_time TEXT,
            end_time TEXT,
            license_plate TEXT
        )
    """)

    cursor.execute("PRAGMA table_info(LicensePlates)")
    columns = [row[1] for row in cursor.fetchall()]

    if "confidence" not in columns:
        cursor.execute("ALTER TABLE LicensePlates ADD COLUMN confidence REAL")

    if "method" not in columns:
        cursor.execute("ALTER TABLE LicensePlates ADD COLUMN method TEXT")

    # Table for owner lookup
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS RegisteredUsers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT UNIQUE,
            owner_name TEXT
        )
    """)

    conn.commit()
    conn.close()


init_database()

# -------- HELPER FOR OWNER/STATUS --------
def get_user_info(plate_text):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT owner_name FROM RegisteredUsers WHERE plate_number = ?", (plate_text,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return row[0], "Access Granted"
    return "Unknown Vehicle", "Access Denied"


def clean_text(text: str) -> str:
    cleaned = re.sub(r"\W", "", str(text))
    cleaned = cleaned.replace("O", "0").replace("???", "").replace("粤", "")
    return cleaned.strip()


def is_plate_like(text: str) -> bool:
    if len(text) < 5:
        return False
    has_letter = any(c.isalpha() for c in text)
    has_digit = any(c.isdigit() for c in text)
    return has_letter and has_digit


def score_plate_candidate(text: str, confidence: float) -> tuple:
    mixed = 1 if is_plate_like(text) else 0
    return (mixed, len(text), confidence)


def paddle_ocr_crop(img, x1, y1, x2, y2):
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return "", 0.0

    crop = img[y1:y2, x1:x2]
    _, cw = crop.shape[:2]
    crop = crop[:, int(cw * 0.18):]

    crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    crop = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    result = ocr.ocr(crop, det=False, rec=True, cls=False)
    if not result:
        return "", 0.0

    item = result
    while isinstance(item, list) and len(item) == 1 and isinstance(item[0], list):
        item = item[0]

    candidates = []
    if isinstance(item, list) and len(item) == 2 and isinstance(item[0], str):
        candidates = [item]
    elif isinstance(item, list):
        for cand in item:
            if isinstance(cand, (list, tuple)) and len(cand) >= 2:
                candidates.append([cand[0], cand[1]])

    best_text, best_score = "", -1.0
    for txt, score in candidates:
        cleaned = clean_text(txt)
        
        # We still keep the length check to avoid "noise"
        if len(cleaned) < 3: 
            continue
            
        try: 
            s = float(score)
        except: 
            s = 0.0
            
        # PURE CONFIDENCE LOGIC: Pick the highest decimal number
        if s > best_score:
            best_text, best_score = cleaned, s

    return best_text, best_score


def save_to_database(detections, start_time, end_time):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    for item in detections:
        cursor.execute("""
            INSERT INTO LicensePlates(start_time, end_time, license_plate, confidence, method)
            VALUES (?, ?, ?, ?, ?)
        """, (start_time.isoformat(), end_time.isoformat(), item["plate_number"], item.get("confidence_score", 0.0), item.get("method", "unknown")))
    conn.commit()
    conn.close()


def save_json(detections, start_time, end_time):
    interval_data = {"Start Time": start_time.isoformat(), "End Time": end_time.isoformat(), "Detections": detections}
    interval_file_path = os.path.join(JSON_DIR, "output_" + datetime.now().strftime("%Y%m%d%H%M%S") + ".json")
    with open(interval_file_path, "w", encoding="utf-8") as f:
        json.dump(interval_data, f, indent=2)

    cumulative_file_path = os.path.join(JSON_DIR, "LicensePlateData.json")
    if os.path.exists(cumulative_file_path):
        with open(cumulative_file_path, "r", encoding="utf-8") as f: existing_data = json.load(f)
    else: existing_data = []
    existing_data.append(interval_data)
    with open(cumulative_file_path, "w", encoding="utf-8") as f: json.dump(existing_data, f, indent=2)
    save_to_database(detections, start_time, end_time)


@app.get("/")
def root():
    return {"message": "SmartParking Gate API is running"}


@app.post("/predict")
async def predict_plate(file: UploadFile = File(...), authorization: str = Header(None)):
    # AUTHORIZATION CHECK
    if authorization != MY_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized Hardware Key")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"filename": file.filename, "detections": [], "count": 0, "error": "Invalid image file"}

    if len(frame.shape) == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    display_frame = frame.copy()
    h, w = frame.shape[:2]
    start_time = datetime.now()
    final_results, seen_plates = [], set()

    # YOUR ORIGINAL YOLO LOGIC
    results = model.predict(frame, conf=CONF_THRES, imgsz=IMG_SIZE, verbose=True)
    det_count = sum(len(r.boxes) for r in results)

    # FALLBACK: FULL IMAGE OCR (YOUR ORIGINAL LOGIC)
    if det_count == 0:
        crop = frame.copy()
        crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        crop = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        full_ocr_result = ocr.ocr(crop, det=True, rec=True, cls=False)
        fallback_candidates = []

        if full_ocr_result and full_ocr_result[0]:
            for line in full_ocr_result[0]:
                text, conf = line[1][0], line[1][1]
                cleaned = clean_text(text)
                if len(cleaned) >= 5:
                    owner, status = get_user_info(cleaned) # LOOKUP
                    fallback_candidates.append({
                        "plate_number": cleaned,
                        "owner": owner,    # REQUESTED FIELD
                        "status": status,  # REQUESTED FIELD
                        "confidence_score": round(float(conf), 4),
                        "method": "full_image_ocr"
                    })

            fallback_candidates.sort(key=lambda x: score_plate_candidate(x["plate_number"], x["confidence_score"]), reverse=True)
            for idx, item in enumerate(fallback_candidates):
                if item["plate_number"] not in seen_plates:
                    seen_plates.add(item["plate_number"])
                    final_results.append(item)
                    cv2.putText(display_frame, item["plate_number"], (20, 45 + idx * 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    # STANDARD YOLO + OCR PATH (YOUR ORIGINAL LOGIC)
    else:
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bw, bh = x2 - x1, y2 - y1
                x1p, y1p = max(0, int(x1 - PAD_L * bw)), max(0, int(y1 - PAD_T * bh))
                x2p, y2p = min(w, int(x2 + PAD_R * bw)), min(h, int(y2 + PAD_B * bh))

                cv2.rectangle(display_frame, (x1p, y1p), (x2p, y2p), (255, 0, 0), 2)
                label, ocr_conf = paddle_ocr_crop(frame, x1p, y1p, x2p, y2p)
                
                if label and label not in seen_plates:
                    owner, status = get_user_info(label) # LOOKUP
                    seen_plates.add(label)
                    final_results.append({
                        "plate_number": label,
                        "owner": owner,    # REQUESTED FIELD
                        "status": status,  # REQUESTED FIELD
                        "confidence_score": round(float(ocr_conf), 4),
                        "method": "yolo_crop"
                    })
                    cv2.putText(display_frame, label, (x1p, max(0, y1p - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    end_time = datetime.now()
    save_json(final_results, start_time, end_time)

    _, buffer = cv2.imencode(".jpg", display_frame)
    encoded_img = base64.b64encode(buffer).decode("utf-8")

    return {
        "filename": file.filename,
        "detections": final_results,
        "count": len(final_results),
        "annotated_image": f"data:image/jpeg;base64,{encoded_img}"
    }

@app.get("/plates")
def get_plates():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, start_time, end_time, license_plate, confidence, method FROM LicensePlates ORDER BY id DESC LIMIT 100")
    rows = cursor.fetchall()
    conn.close()
    return {"data": [{"id": r[0], "timestamp": r[2], "plate_number": r[3], "confidence": r[4], "method": r[5]} for r in rows]}