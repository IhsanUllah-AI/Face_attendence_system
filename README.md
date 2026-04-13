# 🎭 Facial Recognition Attendance System

A production-ready, end-to-end **Facial Recognition Attendance System** built with:

| Component | Technology |
|---|---|
| Face Detection | **MTCNN** (Multi-Task Cascaded CNN) |
| Face Recognition | **FaceNet** (InceptionResnetV1 – VGGFace2) |
| Similarity Search | **FAISS** (Inner Product / Cosine Similarity) |
| Backend API | **FastAPI** + Uvicorn |
| Image Processing | **OpenCV** |
| Database | **SQLite** via SQLAlchemy |

---

## 📁 Project Structure

```
Facial Recognition Attendance System/
│
├── Dataset/                          ← Training images (one folder per person)
│   ├── messi/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   ├── neymar/
│   └── ronaldo/
│
├── storage/                          ← Auto-created at runtime
│   ├── embeddings/
│   │   ├── face_index.faiss         ← FAISS vector index
│   │   └── metadata.json            ← Label metadata
│   ├── unknown_faces/               ← Saved images of unrecognised people
│   ├── logs/
│   └── attendance.db                ← SQLite attendance database
│
├── app/
│   ├── __init__.py
│   ├── main.py                      ← FastAPI app factory
│   ├── config.py                    ← All configurable parameters
│   │
│   ├── core/
│   │   ├── detector.py              ← MTCNN face detection
│   │   ├── embedder.py              ← FaceNet embedding generation
│   │   ├── recognizer.py            ← FAISS cosine-similarity search
│   │   ├── enrollment.py            ← Dataset processing pipeline
│   │   └── attendance_manager.py   ← Attendance & unknown-face logic
│   │
│   ├── api/
│   │   ├── enroll.py               ← POST /enroll
│   │   ├── recognize.py            ← POST /recognize
│   │   └── attendance.py           ← GET  /attendance
│   │
│   ├── models/
│   │   └── schemas.py              ← Pydantic request/response models
│   │
│   └── database/
│       └── db.py                   ← SQLAlchemy engine + ORM models
│
├── realtime.py                      ← Standalone webcam recognition script
├── enroll_cli.py                    ← CLI enrollment (no server needed)
├── run.py                           ← Uvicorn launcher
└── requirements.txt
```

---

## ⚡ Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

> **GPU Note:** Replace `faiss-cpu` with `faiss-gpu` and install the CUDA-compatible `torch` + `torchvision` builds for GPU acceleration.

### 2. Prepare Your Dataset

Your dataset is already structured correctly:

```
Dataset/
├── person_name_1/
│   ├── img1.jpg
│   └── img2.jpg
└── person_name_2/
    └── img1.jpg
```

### 3. Enroll (Build FAISS Index)

**Option A – CLI (recommended for first setup):**

```bash
python enroll_cli.py
```

**Option B – via API (after starting the server):**

```bash
curl -X POST http://localhost:8000/enroll
```

### 4. Start the API Server

```bash
python run.py
```

Server runs at: **http://localhost:8000**
Interactive docs: **http://localhost:8000/docs**

### 5. Real-Time Webcam Recognition

```bash
python realtime.py
```

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `R` | Reload FAISS index (hot-reload after new enrollment) |

---

## 🔌 API Endpoints

### `POST /enroll`
Process the entire Dataset/ folder and build the FAISS recognition index.

**Response:**
```json
{
  "success": true,
  "total_persons": 3,
  "total_images": 14,
  "persons": ["messi", "neymar", "ronaldo"],
  "message": "Successfully enrolled 3 person(s) with 14 face embedding(s)."
}
```

---

### `POST /recognize`
Upload an image to identify all faces in it.

```bash
curl -X POST http://localhost:8000/recognize \
  -F "file=@photo.jpg"
```

**Response:**
```json
{
  "success": true,
  "faces_detected": 2,
  "results": [
    {
      "name": "messi",
      "confidence": 0.8742,
      "is_unknown": false,
      "bbox": [120, 45, 280, 215]
    },
    {
      "name": "Unknown",
      "confidence": 0.3102,
      "is_unknown": true,
      "bbox": [350, 60, 490, 210]
    }
  ],
  "message": "Processed 2 face(s) successfully."
}
```

---

### `GET /attendance`
Fetch attendance records with optional filters.

```bash
# All records
curl http://localhost:8000/attendance

# Filter by date
curl "http://localhost:8000/attendance?date=2026-03-28"

# Filter by name
curl "http://localhost:8000/attendance?name=messi"

# Combined filter
curl "http://localhost:8000/attendance?date=2026-03-28&name=messi&limit=50"
```

---

## ⚙️ Configuration

All settings are in `app/config.py`:

| Parameter | Default | Description |
|---|---|---|
| `COSINE_SIMILARITY_THRESHOLD` | `0.65` | Min similarity to recognise a face |
| `FRAME_SKIP` | `2` | Process every Nth frame (performance) |
| `MIN_FACE_SIZE` | `40` | Min face size in px for MTCNN |
| `ATTENDANCE_COOLDOWN_SECONDS` | `60` | Min seconds between attendance marks |
| `WEBCAM_INDEX` | `0` | OpenCV webcam device index |
| `EMBEDDING_DIM` | `512` | FaceNet output dimension |

---

## 🧠 Technical Architecture

```
Image / Webcam Frame
        │
        ▼
  ┌─────────────┐
  │   MTCNN     │  ← Face detection & 160×160 crop
  └──────┬──────┘
         │ Face Tensor(s)
         ▼
  ┌─────────────┐
  │  FaceNet    │  ← L2-normalised 512-d embedding
  └──────┬──────┘
         │ Embedding Vector(s)
         ▼
  ┌─────────────┐
  │    FAISS    │  ← Inner Product search (≡ Cosine Similarity)
  │  IndexFlatIP│
  └──────┬──────┘
         │ (name, confidence)
         ▼
  ┌─────────────────────────┐
  │  AttendanceManager      │
  │  - Mark known faces     │
  │  - Log unknown faces    │
  │  - Save unknown images  │
  └─────────────────────────┘
```

### Why Cosine Similarity via FAISS Inner Product?

FaceNet embeddings are **L2-normalised** (‖v‖₂ = 1). Therefore:

```
cosine_similarity(a, b) = a · b   (when ‖a‖ = ‖b‖ = 1)
```

Using `IndexFlatIP` (Inner Product) gives **exact cosine similarity** with fast FAISS performance — no approximation needed for typical attendance-system scale.

---

## 🔐 Security Features

- **Unknown face detection**: Faces below the similarity threshold are labelled "Unknown".
- **Image saving**: Unknown face crops are saved to `storage/unknown_faces/` with timestamped filenames.
- **DB logging**: Every unknown detection is logged to `unknown_faces` table.
- **Alert hook**: `AttendanceManager._trigger_alert()` is an extensible hook (currently prints a warning — extend to send email/SMS/push notifications).

---

## 📦 Dependencies

```
torch            ≥ 2.0.0
torchvision      ≥ 0.15.0
facenet-pytorch  ≥ 2.5.3
opencv-python    ≥ 4.8.0
Pillow           ≥ 9.5.0
faiss-cpu        ≥ 1.7.4
fastapi          ≥ 0.110.0
uvicorn[std]     ≥ 0.29.0
python-multipart ≥ 0.0.9
sqlalchemy       ≥ 2.0.0
numpy            ≥ 1.24.0
pydantic         ≥ 2.0.0
```
