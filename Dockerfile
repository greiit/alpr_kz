# =========================================================
# Stage 1: builder — собираем зависимости в отдельный слой,
# чтобы финальный образ не тащил компиляторы и кэш pip.
# =========================================================
FROM python:3.10-slim AS builder

WORKDIR /build

# Системные библиотеки, нужные для сборки некоторых Python-пакетов
# (например, лёгкие C-расширения внутри paddlepaddle/opencv деп.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Устанавливаем зависимости в изолированный префикс /install,
# который затем целиком копируется в финальный образ.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt


# =========================================================
# Stage 2: runtime — минимальный финальный образ
# =========================================================
FROM python:3.10-slim AS runtime

# Системные библиотеки, необходимые OpenCV/PaddleOCR в рантайме
# (без них падает импорт cv2: "libGL.so.1: cannot open shared object file")
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Копируем уже собранные зависимости из стадии builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Копируем код приложения
COPY app ./app

# Непривилегированный пользователь — хорошая практика безопасности
RUN useradd --create-home --shell /bin/bash alpr_user
USER alpr_user

EXPOSE 8000

# Healthcheck для оркестраторов (docker-compose / k8s)
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
