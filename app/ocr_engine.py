"""
ocr_engine.py
-------------
Модуль распознавания текста на номерном знаке (PaddleOCR) и
постобработки результата под стандарт номеров Республики Казахстан.

Формат номера РК (актуальный, "квадратный" стиль):
    3 цифры + 3 буквы + 2 цифры региона, например: 777AAA02
Используются только буквы, совпадающие по начертанию с латиницей/
кириллицей (A, B, E, K, M, H, O, P, C, T, Y, X и т.д.), поэтому
разрешённый алфавит сужен под этот стандарт.
"""

from __future__ import annotations

import os
import re
import logging
from dataclasses import dataclass

import numpy as np
from paddleocr import PaddleOCR

logger = logging.getLogger("alpr.ocr_engine")

# Буквы, разрешённые в номерах РК (соответствуют кириллице/латинице
# с одинаковым начертанием — так исторически устроен стандарт РК)
ALLOWED_LETTERS = "ABCEHKMOPTXY"

# Регулярное выражение стандарта РК: 3 цифры + 3 буквы + 2 цифры региона
KZ_PLATE_REGEX = re.compile(rf"^\d{{3}}[{ALLOWED_LETTERS}]{{3}}\d{{2}}$")

# Таблица частых ошибок OCR (символ -> на что заменить),
# используется отдельно для "цифровых" и "буквенных" позиций номера,
# так как одна и та же клякса OCR может быть то "0", то "O".
DIGIT_CONFUSIONS = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1",
                     "Z": "2", "S": "5", "B": "8", "G": "6"}
LETTER_CONFUSIONS = {"0": "O", "1": "I", "5": "S", "8": "B", "6": "G"}


def resolve_models_dir() -> str | None:
    """
    Определяет папку для хранения моделей PaddleOCR.

    Проблема: C++-ядро Paddle на Windows не может открывать файлы по путям
    с не-ASCII символами (например, C:\\Users\\Апай\\.paddleocr).
    Поэтому:
      1. если задана переменная окружения ALPR_MODELS_DIR — берём её;
      2. если домашняя папка содержит не-ASCII символы (Windows) —
         используем C:\\paddle_models;
      3. иначе возвращаем None (PaddleOCR использует папку по умолчанию).
    """
    env_dir = os.getenv("ALPR_MODELS_DIR")
    if env_dir:
        return env_dir

    home = os.path.expanduser("~")
    if os.name == "nt" and not home.isascii():
        drive = os.path.splitdrive(home)[0] or "C:"
        return os.path.join(drive + os.sep, "paddle_models")
    return None


@dataclass
class OCRResult:
    """Результат распознавания одного номерного знака."""
    raw_text: str
    plate_number: str | None
    confidence: float
    is_valid_format: bool


class OCREngine:
    """
    Обёртка над PaddleOCR, инкапсулирующая инициализацию модели
    и логику постобработки текста под казахстанский стандарт номеров.

    Модель инициализируется один раз при старте приложения (см. main.py),
    чтобы не тратить время на загрузку весов на каждый запрос.
    """

    def __init__(self, lang: str = "en", use_gpu: bool = False):
        # use_angle_cls=True — доп. классификатор поворота текста на 180°,
        # полезен, если номер снят "вверх ногами".
        models_dir = resolve_models_dir()
        extra = {}
        if models_dir:
            # Модели будут скачаны/читаются из ASCII-папки
            logger.info("Папка моделей PaddleOCR: %s", models_dir)
            extra = {
                "det_model_dir": os.path.join(models_dir, "det"),
                "rec_model_dir": os.path.join(models_dir, "rec"),
                "cls_model_dir": os.path.join(models_dir, "cls"),
            }

        self._ocr = PaddleOCR(
            use_angle_cls=True,
            lang=lang,
            use_gpu=use_gpu,
            show_log=False,
            **extra,
        )

    def _run_paddle(self, image: np.ndarray) -> list:
        """Низкоуровневый вызов PaddleOCR, возвращает сырой результат."""
        result = self._ocr.ocr(image, cls=True)
        # PaddleOCR оборачивает результат в список по числу страниц/изображений
        if not result or result[0] is None:
            return []
        return result[0]

    @staticmethod
    def _select_best_line(lines: list) -> tuple[str, float]:
        """
        Из всех найденных на изображении текстовых строк выбирает
        наиболее вероятную кандидатуру на номерной знак: строку с
        наибольшим числом буквенно-цифровых символов и хорошей
        уверенностью модели.
        """
        best_text, best_conf = "", 0.0
        best_score = -1.0

        for line in lines:
            # line имеет вид: [box, (text, confidence)]
            text, conf = line[1]
            cleaned_len = len(re.sub(r"[^A-Za-z0-9]", "", text))
            score = cleaned_len * conf  # эвристика "длина * уверенность"
            if score > best_score:
                best_score = score
                best_text, best_conf = text, conf

        return best_text, best_conf

    @staticmethod
    def clean_plate_text(raw_text: str) -> tuple[str | None, bool]:
        """
        Приводит сырой текст OCR к формату номера РК:
          1. убирает пробелы, дефисы, приводит к верхнему регистру
          2. если длина совпадает с форматом (8 символов), корректирует
             типичные ошибки OCR отдельно на цифровых и буквенных позициях
          3. проверяет итоговую строку регулярным выражением

        Возвращает (нормализованный_номер, флаг_валидности_формата).
        Если формат не удаётся восстановить, возвращает (raw_upper, False).
        """
        text = re.sub(r"[^A-Za-z0-9]", "", raw_text).upper()

        if len(text) != 8:
            # Не соответствует длине "3 цифры + 3 буквы + 2 цифры"
            return (text or None), False

        chars = list(text)
        # Позиции 0-2 и 6-7 должны быть цифрами
        for i in (0, 1, 2, 6, 7):
            if chars[i] in DIGIT_CONFUSIONS:
                chars[i] = DIGIT_CONFUSIONS[chars[i]]
        # Позиции 3-5 должны быть буквами
        for i in (3, 4, 5):
            if chars[i] in LETTER_CONFUSIONS:
                chars[i] = LETTER_CONFUSIONS[chars[i]]

        candidate = "".join(chars)
        is_valid = bool(KZ_PLATE_REGEX.match(candidate))
        return candidate, is_valid

    def recognize(self, image: np.ndarray) -> OCRResult:
        """
        Полный цикл распознавания: сырой OCR -> выбор лучшей строки ->
        нормализация под формат РК.
        """
        lines = self._run_paddle(image)

        if not lines:
            logger.warning("PaddleOCR не нашёл текст на изображении")
            return OCRResult(raw_text="", plate_number=None,
                              confidence=0.0, is_valid_format=False)

        raw_text, confidence = self._select_best_line(lines)
        plate_number, is_valid = self.clean_plate_text(raw_text)

        return OCRResult(
            raw_text=raw_text,
            plate_number=plate_number,
            confidence=round(float(confidence), 4),
            is_valid_format=is_valid,
        )
