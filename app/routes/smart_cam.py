from flask import Flask, render_template, request, redirect, url_for, Response,Blueprint,current_app
import cv2
import numpy as np
from deepface import DeepFace
import json
import os
from werkzeug.utils import secure_filename

video_bp = Blueprint("video", __name__)

# Configurations for file upload
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
fullbody_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
upperbody_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')
lowerbody_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lowerbody.xml')



BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
MODELS_DIR = os.path.join(BASE_DIR, '../models')       
faceProto = os.path.join(MODELS_DIR, 'opencv_face_detector.pbtxt')
faceModel = os.path.join(MODELS_DIR, 'opencv_face_detector_uint8.pb')
ageProto = os.path.join(MODELS_DIR, 'age_deploy.prototxt')
ageModel = os.path.join(MODELS_DIR, 'age_net.caffemodel')
genderProto = os.path.join(MODELS_DIR, 'gender_deploy.prototxt')
genderModel = os.path.join(MODELS_DIR, 'gender_net.caffemodel')

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

camera = cv2.VideoCapture(0)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def highlight_face(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, faceBoxes

def detect_and_annotate(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces and annotate
    resultImg, faceBoxes = highlight_face(faceNet, frame)
    if faceBoxes:
        for faceBox in faceBoxes:
            face = frame[max(0, faceBox[1] - 20):min(faceBox[3] + 20, frame.shape[0] - 1),
                         max(0, faceBox[0] - 20):min(faceBox[2] + 20, frame.shape[1] - 1)]
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]

            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]

            label = f"{gender}, {age}"
            cv2.putText(resultImg, label, (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Detect eyes
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in eyes:
        cv2.rectangle(resultImg, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(resultImg, 'Eye', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Detect smiles
    smiles = smile_cascade.detectMultiScale(gray, scaleFactor=1.8, minNeighbors=20)
    for (x, y, w, h) in smiles:
        cv2.rectangle(resultImg, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(resultImg, 'Smile', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Detect full body
    fullbodies = fullbody_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
    for (x, y, w, h) in fullbodies:
        cv2.rectangle(resultImg, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(resultImg, 'Full Body', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Detect upper body
    upperbodies = upperbody_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
    for (x, y, w, h) in upperbodies:
        cv2.rectangle(resultImg, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.putText(resultImg, 'Upper Body', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Detect lower body
    lowerbodies = lowerbody_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
    for (x, y, w, h) in lowerbodies:
        cv2.rectangle(resultImg, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(resultImg, 'Lower Body', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    return resultImg

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = detect_and_annotate(frame)
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def face_analyze(img_path):
    try:
        result_dict = DeepFace.analyze(img_path=img_path, actions=['age', 'gender', 'race', 'emotion'])
        
        with open('face_analyze.json', 'w') as file:
            json.dump(result_dict, file, indent=4, ensure_ascii=False)
        
        print(f'[+] Age: {result_dict.get("age")}')
        print(f'[+] Gender: {result_dict.get("gender")}')
        print('[+] Race:')
        
        for k, v in result_dict.get('race').items():
            print(f'{k} - {round(v, 2)}%')
            
        print('[+] Emotions:')
        
        for k, v in result_dict.get('emotion').items():
            print(f'{k} - {round(v, 2)}%')
            
    except Exception as _ex:
        return _ex

@video_bp.route('/')
def index():
    return render_template('smart_cam.html')

@video_bp.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@video_bp.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            # Use `current_app` to access the Flask app's config
            upload_folder = current_app.config['UPLOAD_FOLDER']
            file_path = os.path.join(upload_folder, filename)

            # Save the uploaded file
            file.save(file_path)
            
            # Analyze the uploaded image
            result = face_analyze(file_path)

            # Render the result in the template
            return render_template('result.html', result=result)
    
    # Render the upload page for GET requests
    return render_template('upload.html')

