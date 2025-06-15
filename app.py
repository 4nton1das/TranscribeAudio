# app.py
from flask import Flask, render_template, request, jsonify, send_from_directory
import torch
import os
import time
from werkzeug.utils import secure_filename
import warnings
from config import (UPLOAD_FOLDER, MAX_CONTENT_LENGTH, ALLOWED_EXTENSIONS, MODEL_INFO, LANGUAGES, WHISPER_MODEL_DIR,
                    DEFAULT_MODEL, TASKS, TTS_OUTPUT_DIR)
from asr import ASRProcessor
from gigachat import GigaChatProcessor
from tts import TTSProcessor

warnings.filterwarnings("ignore", category=FutureWarning, module="whisper")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Инициализация процессоров
asr_processor = ASRProcessor(model_dir=WHISPER_MODEL_DIR)
asr_processor.load_model(DEFAULT_MODEL)
gigachat_processor = GigaChatProcessor()
tts_processor = TTSProcessor()


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html',
                           gpu_available=torch.cuda.is_available(),
                           current_model=asr_processor.current_model,
                           model_info=MODEL_INFO,
                           languages=LANGUAGES,
                           tasks=TASKS)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "Файл не найден"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Не выбран файл"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Неподдерживаемый формат файла"}), 400

    try:
        # Получаем параметры из формы
        selected_model = request.form.get('model', DEFAULT_MODEL)
        language = request.form.get('language', None)
        task = request.form.get('task', 'transcribe')
        target_language = request.form.get('target_language', None)

        # Загружаем модель Whisper
        asr_processor.load_model(selected_model)

        # Сохраняем файл
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Транскрибация
        asr_start_time = time.time()
        result = asr_processor.transcribe_audio(filepath, language=language)
        asr_time = round(time.time() - asr_start_time, 2)

        transcript_text = result["text"].strip()
        detected_language = result.get("language", "unknown")

        # Обработка LLM (если задача не обычная транскрипция)
        llm_time = 0
        processed_text = None

        if task != 'transcribe':
            llm_start_time = time.time()
            try:
                processed_text = gigachat_processor.process_text(
                    transcript_text,
                    task=task,
                    target_language=target_language
                )
                llm_time = round(time.time() - llm_start_time, 2)
            except Exception as e:
                processed_text = f"Ошибка обработки LLM: {str(e)}"
                llm_time = 0

        # Удаляем временный файл
        os.remove(filepath)

        return jsonify({
            "transcript": transcript_text,
            "processed_text": processed_text,
            "filename": filename,
            "asr_time": asr_time,
            "llm_time": llm_time,
            "model_used": selected_model,
            "detected_language": detected_language,
            "task": task
        })

    except Exception as e:
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({"error": str(e)}), 500


@app.route('/tts_config')
def tts_config():
    return jsonify(tts_processor.get_languages())


@app.route('/tts', methods=['POST'])
def text_to_speech():
    data = request.json
    text = data.get('text', '').strip()
    language = data.get('language', 'ru')
    speaker = data.get('speaker', 'aidar')

    if not text:
        return jsonify({"error": "Текст для озвучивания не предоставлен"}), 400

    try:
        filename, _ = tts_processor.generate_speech(text, language, speaker)
        return jsonify({
            "audio_url": f"/tts_download/{filename}",
            "filename": filename
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/tts_download/<filename>')
def download_tts(filename):
    return send_from_directory(TTS_OUTPUT_DIR, filename, as_attachment=True)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
