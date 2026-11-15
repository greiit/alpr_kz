"""
preprocessing.py
-----------------
Модуль предобработки изображений для пайплайна распознавания
автомобильных номеров РК (ALPR).

Все функции принимают/возвращают numpy.ndarray (формат, с которым
работает OpenCV), что позволяет свободно комбинировать их в конвейер.
"""

from __future__ import annotations

import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Переводит цветное изображение (BGR) в оттенки серого.

    Если изображение уже одноканальное — возвращает его без изменений,
    чтобы функцию можно было безопасно вызывать в любом месте пайплайна.
    """
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def denoise(image: np.ndarray,
            d: int = 9,
            sigma_color: int = 75,
            sigma_space: int = 75) -> np.ndarray:
    """
    Удаляет шум с сохранением резких границ символов номера
    с помощью билатерального фильтра (bilateralFilter).

    d            — диаметр окрестности пикселя, участвующей в фильтрации.
    sigma_color  — чем больше, тем сильнее смешиваются цвета в окрестности.
    sigma_space  — чем больше, тем дальше друг от друга пиксели
                   могут влиять друг на друга (при близких цветах).
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def enhance_contrast(gray_image: np.ndarray,
                      clip_limit: float = 3.0,
                      tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Выравнивает контраст с помощью CLAHE (Contrast Limited Adaptive
    Histogram Equalization). Особенно полезно для номеров, снятых
    при неравномерном освещении / бликах / тени.

    clip_limit     — ограничение усиления контраста (защита от шума).
    tile_grid_size — размер сетки, на которую делится изображение
                     для локального выравнивания гистограммы.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray_image)


def binarize_otsu(gray_image: np.ndarray, invert: bool = False) -> np.ndarray:
    """
    Адаптивная бинаризация методом Otsu — автоматически подбирает
    оптимальный порог разделения "текст / фон" для символов номера.

    invert=True возвращает изображение, где символы белые на чёрном
    фоне (иногда удобнее для последующего поиска контуров).
    """
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, binary = cv2.threshold(
        gray_image, 0, 255, flag + cv2.THRESH_OTSU
    )
    return binary


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Упорядочивает 4 точки контура в порядке:
    [верх-лево, верх-право, низ-право, низ-лево].
    Это необходимо для корректного perspective transform,
    так как cv2.findContours не гарантирует порядок вершин.
    """
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # верх-лево: минимальная сумма x+y
    rect[2] = pts[np.argmax(s)]   # низ-право: максимальная сумма x+y

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # верх-право: минимальная разница x-y
    rect[3] = pts[np.argmax(diff)]  # низ-лево: максимальная разница x-y

    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Выполняет перспективное преобразование (warp) области,
    заданной 4 точками, приводя её к виду "анфас" —
    как будто камера смотрит на номер строго перпендикулярно.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    # Целевые координаты прямоугольника без перспективных искажений
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
    return warped


def find_plate_contour(image: np.ndarray) -> np.ndarray | None:
    """
    Ищет на изображении наиболее вероятный контур номерного знака
    (прямоугольник / четырёхугольник) и возвращает его 4 точки.

    Возвращает None, если подходящий контур не найден — в этом случае
    вызывающий код должен использовать исходное изображение целиком
    (например, если на вход уже подан кроп номера с детектора).
    """
    gray = to_grayscale(image)
    gray = denoise(gray)
    edged = cv2.Canny(gray, 30, 200)

    contours, _ = cv2.findContours(
        edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    # Сортируем контуры по площади и рассматриваем 10 самых крупных —
    # номер обычно один из наиболее заметных прямоугольников на кропе.
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approx) == 4:
            area = cv2.contourArea(approx)
            # Отсекаем слишком маленькие контуры (шум)
            if area > 0.1 * image.shape[0] * image.shape[1] * 0.05:
                return approx.reshape(4, 2).astype("float32")

    return None


def align_perspective(image: np.ndarray) -> np.ndarray:
    """
    Пытается найти прямоугольный контур номера и выровнять перспективу.
    Если подходящий контур не найден — возвращает исходное изображение
    без изменений (fail-safe поведение для устойчивости пайплайна).
    """
    pts = find_plate_contour(image)
    if pts is None:
        return image
    return four_point_transform(image, pts)


def preprocess_image(image: np.ndarray, do_perspective: bool = True) -> np.ndarray:
    """
    Полный конвейер предобработки изображения перед подачей в OCR:

        1. (опционально) выравнивание перспективы по контуру номера
        2. перевод в градации серого
        3. удаление шума (bilateralFilter)
        4. выравнивание контраста (CLAHE)
        5. адаптивная бинаризация (Otsu)

    Возвращает бинарное изображение, готовое для OCR-движка.
    """
    working_image = align_perspective(image) if do_perspective else image

    gray = to_grayscale(working_image)
    denoised = denoise(gray)
    contrasted = enhance_contrast(denoised)
    binary = binarize_otsu(contrasted)

    return binary
