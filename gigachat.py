# gigachat.py
import os
import requests
import json
from dotenv import load_dotenv
from prompts import get_system_prompt
import time

# Загрузка переменных окружения
load_dotenv()


class GigaChatProcessor:
    def __init__(self):
        self.api_key = os.getenv("GIGACHAT_API_KEY")
        if not self.api_key:
            raise ValueError("GIGACHAT_API_KEY не найден в .env файле")

        self.auth_url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
        self.api_url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
        self.access_token = None
        self.token_expires = 0

    def _get_access_token(self):
        """Получает временный access token"""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json',
            'RqUID': '6f0b1291-c7f3-43c6-bb2e-9f3efb2dc98e'
        }
        data = {'scope': 'GIGACHAT_API_PERS'}

        response = requests.post(
            self.auth_url,
            headers=headers,
            data=data,
            verify=False,
            timeout=10
        )

        if response.status_code != 200:
            raise ConnectionError(f"Ошибка аутентификации: {response.status_code} - {response.text}")

        token_data = response.json()
        return token_data['access_token'], token_data['expires_at']

    def _ensure_valid_token(self):
        """Проверяет и обновляет токен при необходимости"""
        if not self.access_token or time.time() > self.token_expires - 60:
            self.access_token, expires_at = self._get_access_token()
            self.token_expires = expires_at

    def process_text(self, text, task="correct", target_language=None):
        """Обрабатывает текст с помощью GigaChat API"""
        self._ensure_valid_token()

        # Получаем системный промпт для задачи
        system_prompt = get_system_prompt(task, target_language)

        # Формируем сообщения
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": text})

        # Параметры запроса
        payload = {
            "model": "GigaChat:latest",
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 2000
        }

        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }

        response = requests.post(
            self.api_url,
            headers=headers,
            json=payload,
            timeout=30,
            verify=False
        )

        # Проверка статуса ответа
        if response.status_code != 200:
            raise ConnectionError(f"Ошибка API: {response.status_code} - {response.text}")

        result = response.json()

        # Проверка наличия ожидаемых данных в ответе
        if 'choices' not in result or len(result['choices']) == 0:
            raise ValueError("Некорректный ответ API: отсутствуют choices")

        if 'message' not in result['choices'][0] or 'content' not in result['choices'][0]['message']:
            raise ValueError("Некорректный ответ API: отсутствует content в сообщении")

        return result['choices'][0]['message']['content']
