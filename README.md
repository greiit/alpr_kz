# alpr_kz — Automatic License Plate Recognition for Kazakhstan

A REST microservice that recognizes Kazakhstani vehicle license plates from images.
Built with **OpenCV**, **PaddleOCR**, **FastAPI** and **Docker**; runs on a plain CPU (no GPU required).

```
Image upload → FastAPI → OpenCV preprocessing → PaddleOCR → Regex post-processing → JSON
```

## Features

- **Preprocessing (OpenCV):** grayscale, `bilateralFilter` denoising, CLAHE contrast equalization, Otsu binarization, perspective alignment by plate contour.
- **OCR (PaddleOCR 2.7.x):** the model is loaded once at server startup, not on every request.
- **Post-processing:** cleans the OCR output and fixes typical confusions (`O↔0`, `I↔1`, `B↔8`, ...) separately for digit and letter positions, then validates against the Kazakhstan format.
- **API:** async `POST /recognize`, `GET /health`, auto-generated Swagger UI at `/docs`.
- **Docker:** multi-stage build on `python:3.10-slim`, non-root user, healthcheck.

## Plate format

`3 digits + 3 letters + 2-digit region code`, for example `777AAA02`.

Allowed letters: `A B C E H K M O P T X Y`
Regex: `^\d{3}[ABCEHKMOPTXY]{3}\d{2}$`

## Project structure

```
alpr_kz/
├── app/
│   ├── __init__.py
│   ├── preprocessing.py   # OpenCV pipeline + perspective transform
│   ├── ocr_engine.py      # PaddleOCR wrapper + plate text cleaning
│   └── main.py            # FastAPI application
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md
```

## Requirements

- Python **3.10**
- Internet access on first start (PaddleOCR downloads its models, ~15–20 MB)
- Docker (optional)

## Quick start (local)

### Windows (PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Linux / macOS

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open Swagger UI: http://localhost:8000/docs

> Tip: avoid `--reload` when testing OCR — it reloads the model on every file change.

## Docker

```bash
docker build -t alpr-kz .
docker run -p 8000:8000 alpr-kz
```

The container needs internet access on its first start to download the OCR models.
Health status is available at `GET /health` and used by the Docker `HEALTHCHECK`.

## API

### `POST /recognize`

Accepts `multipart/form-data` with a `file` field (JPEG, PNG, BMP or WebP).

```bash
curl -X POST "http://localhost:8000/recognize" -F "file=@plate.jpg"
```

PowerShell:

```powershell
curl.exe -X POST "http://localhost:8000/recognize" -F "file=@plate.jpg"
```

**Response (200):**

```json
{
  "status": "success",
  "plate_number": "777AAA02",
  "raw_text": "777 AAA 02",
  "is_valid_format": true,
  "confidence": 0.96,
  "processing_time_ms": 35.4
}
```

| `status` | Meaning |
| --- | --- |
| `success` | Text found and it matches the Kazakhstan plate format |
| `low_confidence_format` | Text found but it does not match the format after cleaning |
| `not_found` | No text detected on the image (`plate_number` is `null`) |

**Error codes:**

| Code | Reason |
| --- | --- |
| 400 | Empty file or the image cannot be decoded |
| 415 | Unsupported content type |
| 500 | Error inside the recognition pipeline |
| 503 | OCR model is not loaded yet |

### `GET /health`

```json
{ "status": "ok", "model_loaded": true }
```

## Configuration

| Variable | Description |
| --- | --- |
| `ALPR_MODELS_DIR` | Folder where PaddleOCR models are stored (subfolders `det`, `rec`, `cls`). Optional. |

On Windows, if your user profile path contains non-ASCII characters (e.g. Cyrillic),
the models are automatically stored in `C:\paddle_models` (see Troubleshooting).

## Troubleshooting

**`No matching distribution found for paddlepaddle==2.6.1`**
This version is not published for Windows. The project uses `paddlepaddle==2.6.2`.

**`json.exception.parse_error.101 ... empty input` / traceback mentions `paddlex`**
PaddleOCR 3.x is installed. This project is written for 2.x. Reinstall:

```powershell
pip uninstall -y paddleocr paddlepaddle paddlex
pip install numpy==1.26.4 paddlepaddle==2.6.2 paddleocr==2.7.3
```

**`Cannot open file ...\.paddleocr\...\inference.pdmodel`**
Usually caused by non-ASCII characters in the Windows path. The code already redirects
models to `C:\paddle_models`. You can also set the folder manually:

```powershell
$env:ALPR_MODELS_DIR = "C:\paddle_models"
```

If the problem persists, delete `C:\paddle_models` to force a fresh download,
or move the whole project and venv to an ASCII-only path such as `C:\projects\alpr_kz`.

**`numpy` errors (`np.int`, binary incompatibility)**
`pip install numpy==1.26.4`

**`protobuf` error (`Descriptors cannot be created directly`)**
`pip install protobuf==3.20.2`

**`ImportError` from `cv2` / OpenCV conflicts**

```powershell
pip uninstall -y opencv-python opencv-contrib-python
pip install opencv-python-headless==4.9.0.80
```

**`libGL.so.1` / `libgomp.so.1` missing (Linux)**
Install `libgl1`, `libglib2.0-0` and `libgomp1`. The provided Dockerfile already installs the OpenCV-related libraries.

## Limitations and next steps

- The pipeline expects an image where the plate is large and clearly visible (a cropped plate or a close-up). For full vehicle photos, add a dedicated plate detector (e.g. YOLO) before OCR.
- Only the `3 digits + 3 letters + 2 digits` format is validated; other plate types (two-line, special, older formats) are not covered by the regex.
- Accuracy drops in night-time and adverse-weather conditions; training or fine-tuning on a local dataset would help.
- Possible improvements: batch requests, metrics (Prometheus), automated tests for the post-processing rules, models baked into the Docker image.