# asr.py

import whisper
import torch


class ASRProcessor:
    def __init__(self, model_dir):
        self.model = None
        self.current_model = None
        self.model_dir = model_dir

    def load_model(self, model_name="base"):
        """Загружает или перезагружает модель Whisper"""
        if self.model is not None and self.current_model == model_name:
            return self.model

        # Очищаем память перед загрузкой новой модели
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.model = whisper.load_model(
            model_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
            download_root=self.model_dir
        )
        self.current_model = model_name
        return self.model

    def transcribe_audio(self, filepath, language=None):
        """Транскрибирует аудио/видео файл"""
        return self.model.transcribe(
            filepath,
            language=language,
            fp16=torch.cuda.is_available()
        )
