"""
main.py
-------
FastAPI-сервис для распознавания автомобильных номеров РК (ALPR).

Запуск локально:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Эндпоинт:
    POST /recognize  — принимает изображение (multipart/form-data),
                        возвращает распознанный номер.
"""

from __future__ import annotations

import time
import logging

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.preprocessing import preprocess_image
from app.ocr_engine import OCREngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("alpr.main")

app = FastAPI(
    title="ALPR KZ — Automatic License Plate Recognition",
    description="Сервис распознавания автомобильных номеров Республики Казахстан",
    version="1.0.0",
)

# Модель OCR загружается один раз при старте процесса (тяжёлая операция),
# а не при каждом запросе — иначе задержка на запрос будет огромной.
ocr_engine: OCREngine | None = None

# Допустимые форматы изображений на входе
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/webp"}


class RecognizeResponse(BaseModel):
    """Схема успешного ответа эндпоинта /recognize."""
    status: str
    plate_number: str | None
    raw_text: str
    is_valid_format: bool
    confidence: float
    processing_time_ms: float


@app.on_event("startup")
def load_models() -> None:
    """
    Инициализация тяжёлых моделей при старте приложения.
    Вынесено из тела эндпоинта, чтобы не платить временем загрузки
    весов PaddleOCR на каждый HTTP-запрос.
    """
    global ocr_engine
    logger.info("Загрузка модели PaddleOCR...")
    ocr_engine = OCREngine(lang="en", use_gpu=False)
    logger.info("Модель PaddleOCR загружена.")


def read_image_from_upload(file_bytes: bytes) -> np.ndarray:
    """
    Декодирует байты загруженного файла в numpy-массив (формат OpenCV, BGR).
    Бросает HTTPException(400), если файл повреждён или не является
    корректным изображением.
    """
    np_array = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Не удалось декодировать изображение. Проверьте формат файла.",
        )
    return image


@app.get("/health")
def health_check() -> dict:
    """Простой healthcheck для контейнерных оркестраторов (Docker/K8s)."""
    return {"status": "ok", "model_loaded": ocr_engine is not None}


@app.post("/recognize", response_model=RecognizeResponse)
async def recognize_plate(file: UploadFile = File(...)) -> JSONResponse:
    """
    Основной эндпоинт распознавания номера.

    Пайплайн:
        1. валидация входного файла
        2. декодирование изображения
        3. предобработка (grayscale, denoise, CLAHE, Otsu, perspective)
        4. OCR + нормализация под формат номера РК
        5. сборка JSON-ответа с замером времени обработки
    """
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Неподдерживаемый тип файла: {file.content_type}. "
                   f"Ожидается одно из: {sorted(ALLOWED_CONTENT_TYPES)}",
        )

    if ocr_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Модель OCR ещё не загружена, повторите запрос позже.",
        )

    start_time = time.perf_counter()

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Пустой файл.")

    image = read_image_from_upload(file_bytes)

    try:
        # Предобработка (perspective + grayscale + denoise + CLAHE + Otsu)
        preprocessed = preprocess_image(image, do_perspective=True)

        # PaddleOCR ожидает 3-канальное изображение, поэтому конвертируем
        # бинарный результат предобработки обратно в BGR перед подачей.
        ocr_input = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)

        result = ocr_engine.recognize(ocr_input)
    except Exception as exc:  # noqa: BLE001 — верхнеуровневый guard для API
        logger.exception("Ошибка в пайплайне распознавания")
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {exc}") from exc

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

    if result.plate_number is None:
        response_body = RecognizeResponse(
            status="not_found",
            plate_number=None,
            raw_text=result.raw_text,
            is_valid_format=False,
            confidence=result.confidence,
            processing_time_ms=elapsed_ms,
        )
        return JSONResponse(status_code=200, content=response_body.model_dump())

    response_body = RecognizeResponse(
        status="success" if result.is_valid_format else "low_confidence_format",
        plate_number=result.plate_number,
        raw_text=result.raw_text,
        is_valid_format=result.is_valid_format,
        confidence=result.confidence,
        processing_time_ms=elapsed_ms,
    )
    return JSONResponse(status_code=200, content=response_body.model_dump())
