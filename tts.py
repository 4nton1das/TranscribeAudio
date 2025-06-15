# tts.py
import torch
import os
import uuid
from datetime import datetime
from scipy.io.wavfile import write
from config import TTS_OUTPUT_DIR


class TTSProcessor:
    def __init__(self):
        self.languages = {
            'ru': {
                'model_id': 'v4_ru',
                'sample_rate': 48000,
                'speakers': {
                    'aidar': 'Мужской (Aidar)',
                    'baya': 'Женский (Baya)',
                    'kseniya': 'Женский (Kseniya)',
                    'xenia': 'Женский (Xenia)'
                }
            },
            'en': {
                'model_id': 'v3_en',
                'sample_rate': 48000,
                'speakers': {
                    'en_0': 'Женский (EN 0)',
                    'en_10': 'Женский (EN 1)',
                    'en_2': 'Мужской (EN 2)',
                    'en_7': 'Мужской (EN 3)'
                }
            }
        }
        self.models = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_model(self, language):
        """Загружает модель TTS для указанного языка"""
        if language not in self.languages:
            raise ValueError(f"Unsupported language: {language}")

        if language in self.models:
            return self.models[language]

        print(f"Loading TTS model for {language}...")
        lang_config = self.languages[language]
        model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-models',
            model='silero_tts',
            language=language,
            speaker=lang_config['model_id']
        )
        model.to(self.device)
        self.models[language] = model
        print(f"TTS model for {language} loaded")
        return model

    def get_languages(self):
        """Возвращает доступные языки и спикеров"""
        return {
            lang: {
                'name': lang.upper(),
                'speakers': list(config['speakers'].keys()),
                'speaker_names': config['speakers']
            }
            for lang, config in self.languages.items()
        }

    def generate_speech(self, text, language='ru', speaker='aidar'):
        """Генерирует речь из текста и возвращает путь к файлу"""
        if language not in self.languages:
            raise ValueError(f"Unsupported language: {language}")

        lang_config = self.languages[language]

        # Проверяем доступность спикера для языка
        if speaker not in lang_config['speakers']:
            speaker = list(lang_config['speakers'].keys())[0]

        # Загружаем модель для языка
        model = self.load_model(language)

        # Генерация аудио
        audio = model.apply_tts(
            text=text,
            speaker=speaker,
            sample_rate=lang_config['sample_rate']
        )

        # Создание уникального имени файла
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"tts_{language}_{timestamp}_{uuid.uuid4().hex[:6]}.wav"
        filepath = os.path.join(TTS_OUTPUT_DIR, filename)

        # Сохранение файла
        os.makedirs(TTS_OUTPUT_DIR, exist_ok=True)
        write(filepath, lang_config['sample_rate'], audio.numpy())

        return filename, filepath
