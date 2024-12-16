from flask import Flask,render_template
import logging
logging.basicConfig(level=logging.DEBUG)


def create_app():
    app = Flask(__name__)
    logging.debug('Creating the app...')
    app.config['SECRET_KEY'] = '7d8e37d4f657b2c0e4d109e5b3d54c7123472fd1f0c8923456789abcd1234efg'
    app.config['UPLOAD_FOLDER'] = 'uploads'

    # Register blueprints for modular routes
    from app.routes.newneiro_voice import i_voice_bp
    from app.routes.newneiro import voice_bp
    from app.routes.smart_cam import video_bp
    from app.routes.smart_translator import trans_bp
    

    app.register_blueprint(i_voice_bp, url_prefix="/i_voice")
    app.register_blueprint(voice_bp, url_prefix="/voice")
    app.register_blueprint(video_bp, url_prefix="/video")
    app.register_blueprint(trans_bp, url_prefix="/trans")

    @app.route("/")
    def index():
        return render_template("index.html")

    return app
