from flask import Flask, request, render_template, jsonify,Blueprint
from googletrans import Translator

trans_bp = Blueprint("trans", __name__)
translator = Translator()

@trans_bp.route('/')
def index():
    return render_template('smart_translator.html')

@trans_bp.route('/translate', methods=['POST'])
def translate():
    data = request.json
    text = data.get("text", "")
    target_lang = data.get("target", "en")  # По умолчанию перевод на английский

    if not text:
        return jsonify({"error": "Text is required"}), 400

    translated_text = translator.translate(text, dest=target_lang).text
    return jsonify({"translated_text": translated_text})

