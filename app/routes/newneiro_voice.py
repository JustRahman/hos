from flask import Flask, render_template, request, jsonify, Blueprint
import g4f

i_voice_bp = Blueprint("i_voice", __name__)


@i_voice_bp.route('/')
def serve_html():
    return render_template('newneiro_voice.html')

@i_voice_bp.route('/api/get_response', methods=['POST'])
def get_response():
    data = request.json
    prompt = data.get("prompt", "")

    if not prompt:
        return jsonify({"response": "Введите текст для ответа."}), 400

    if prompt.lower() == "кто ты?":
        return jsonify({"response": "Я NewNeiro, ваш голосовой помощник!"})

    try:
        response = g4f.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"response": f"Произошла ошибка: {str(e)}"}), 500


