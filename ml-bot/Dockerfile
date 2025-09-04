FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


# FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# # Установка системных зависимостей
# RUN apt-get update && apt-get install -y \
#     python3.10 \
#     python3-pip \
#     python3.10-venv \
#     git \
#     wget \
#     curl \
#     && rm -rf /var/lib/apt/lists/*

# # Создание рабочей директории
# WORKDIR /app

# # Установка Python зависимостей
# COPY requirements.txt .

# # Установка pip и пакетов
# RUN pip install --no-cache-dir --upgrade pip && \
#     pip install --no-cache-dir -r requirements.txt

# # # Переменные окружения
# # ENV PYTHONPATH=/app
# # ENV PYTHONUNBUFFERED=1

# # Рабочая директория
# WORKDIR /app

# CMD ["uvicorn", "main:app"]