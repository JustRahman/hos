from flask import Blueprint, render_template, request, jsonify
import g4f

voice_bp = Blueprint("voice", __name__)

@voice_bp.route('/')
def index():
    return render_template('newneiro.html')  # Имя файла HTML

@voice_bp.route('/api/get_response', methods=['POST'])
def get_response():
    data = request.json
    prompt = data.get("prompt", "")

    if not prompt:
        return jsonify({"error": "Введите запрос"}), 400

    if prompt.lower() == "кто ты?":
        return jsonify({"response": "Я NEWNEIRO, ваш помощник!"})

    try:
        response = g4f.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": f"Произошла ошибка: {str(e)}"}), 500



