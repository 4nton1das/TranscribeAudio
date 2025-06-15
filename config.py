# config.py

import os
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

# Настройки Flask
UPLOAD_FOLDER = 'uploads/'
MAX_CONTENT_LENGTH = 2 * 1024 * 1024 * 1024  # 2 GB
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'mp4', 'avi', 'mov'}

# Настройки Whisper
WHISPER_MODEL_DIR = 'D:\\Program Files\\whisper'
DEFAULT_MODEL = "base"
GIGACHAT_MODEL = "GigaChat:latest"
GIGACHAT_API_KEY = os.getenv("GIGACHAT_API_KEY")

# Информация о моделях
MODEL_INFO = {
    "tiny": {"params": "39M", "vram": "~1GB", "speed": "~10x", "quality": "Низкая"},
    "base": {"params": "74M", "vram": "~1GB", "speed": "~7x", "quality": "Базовая"},
    "small": {"params": "244M", "vram": "~2GB", "speed": "~4x", "quality": "Средняя"},
    "medium": {"params": "769M", "vram": "~5GB", "speed": "~2x", "quality": "Хорошая"},
    "large": {"params": "1550M", "vram": "~10GB", "speed": "1x", "quality": "Высшая"},
    "turbo": {"params": "809M", "vram": "~6GB", "speed": "~8x", "quality": "Оптимизированная"}
}

LANGUAGES = {
    "ru": "Русский",
    "en": "Английский",
    "de": "Немецкий",
    "fr": "Французский",
    "es": "Испанский",
    "ja": "Японский",
    "ko": "Корейский"
}

# Доступные задачи
TASKS = {
    "transcribe": "Только транскрипция",
    "correct": "Коррекция и форматирование",
    "summarize": "Суммаризация",
    "translate": "Перевод"
}

# Настройки TTS
TTS_OUTPUT_DIR = 'tts_output/'
