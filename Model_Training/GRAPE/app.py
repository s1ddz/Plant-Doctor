from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import joblib
import os
from PIL import Image
from io import BytesIO
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, MobileNetV2
import cv2

app = Flask(__name__)
CORS(app)

IMG_SIZE = (224, 224)

# Configuration for plants using SVM and Keras models (excluding grape here)
PLANT_CONFIGS = {
    'potato': {
        'model_path': "/home/smurfy/Desktop/Plant-Doctor/Model_Training/POTATO/MODELS/MobileNetV2/SVM/MobileNetV2_SVM.joblib",
        'model_type': 'svm',
        'class_names': [
            "Bacterial Wilt", "Early Blight", "Healthy", "Late Blight", "Leaf Roll Virus",
            "Mosaic Virus", "Nematode", "Pests", "Phytophthora"
        ]
    },
    'bean': {
        'model_path': "/home/smurfy/Desktop/Plant-Doctor/Model_Training/BEAN/MODELS/MobileNetV2/SVM/MobileNetV2_SVM.joblib",
        'model_type': 'svm',
        'class_names': [
            "Angular Leaf Spot", "Bean Rust", "Healthy"
        ]
    }
    # Note: grape handled separately below
}

TOMATO_MODEL_PATH = "/home/smurfy/Desktop/Plant-Doctor/Model_Training/TOMATO/MODELS/NEW.h5"
TOMATO_CLASSES = [
    "Bacterial Spot", "Early Blight", "Healthy", "Late Blight", "Leaf Mold",
    "Septoria Leaf Spot", "Spider Mites", "Target Spot",
    "Tomato Mosaic Virus", "Tomato Yellow Leaf Curl Virus"
]

# Load MobileNetV2 feature extractor for SVM and grape CNN features
feature_extractor = MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights='imagenet',
    pooling='avg'
)
feature_extractor.trainable = False

models = {}

# Load SVM or Keras models for potato and bean
for plant, config in PLANT_CONFIGS.items():
    if os.path.exists(config['model_path']):
        if config['model_type'] == 'keras':
            models[plant] = tf.keras.models.load_model(config['model_path'])
        else:
            models[plant] = joblib.load(config['model_path'])

# Load tomato keras model
if os.path.exists(TOMATO_MODEL_PATH):
    tomato_model = tf.keras.models.load_model(TOMATO_MODEL_PATH)
else:
    tomato_model = None

# Load grape keras CNN model and add to models dict
GRAPE_MODEL_PATH = "/home/smurfy/Desktop/Plant-Doctor/Model_Training/GRAPE/MODELS/MobileNetV2/CNN/MobileNetV2_CNN.h5"
GRAPE_CLASSES = ['Black Rot', 'ESCA', 'healthy', 'Leaf Blight']

if os.path.exists(GRAPE_MODEL_PATH):
    grape_model = tf.keras.models.load_model(GRAPE_MODEL_PATH)
    models['grape'] = grape_model
else:
    grape_model = None

def prepare_image(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image)
    img_array = preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

def prepare_tomato_image(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = np.array(image)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

@app.route("/api/<plant>/predict", methods=["POST"])
def predict_plant(plant):
    if plant not in PLANT_CONFIGS:
        return jsonify({'error': f'Plant {plant} not supported'}), 400

    if plant not in models:
        return jsonify({'error': f'Model for {plant} not available'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        config = PLANT_CONFIGS[plant]
        model_type = config['model_type']
        class_names = config['class_names']
        model = models[plant]

        if model_type == 'svm':
            img = prepare_image(file.read())
            features = feature_extractor(img, training=False).numpy()
            prediction = model.predict(features)
            predicted_class = class_names[prediction[0]]

        elif model_type == 'keras':
            img = prepare_image(file.read())
            prediction = model.predict(img)[0]
            predicted_class = class_names[int(np.argmax(prediction))]

        else:
            return jsonify({'error': f'Unknown model type: {model_type}'}), 500

        return jsonify({
            'plant': plant,
            'prediction': predicted_class
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route("/api/tomato/predict", methods=["POST"])
def predict_tomato():
    if tomato_model is None:
        return jsonify({'error': 'Tomato model not available'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        img = prepare_tomato_image(file.read())
        prediction = tomato_model.predict(img)[0]
        predicted_class = TOMATO_CLASSES[int(np.argmax(prediction))]

        return jsonify({
            'plant': 'tomato',
            'prediction': predicted_class
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


import traceback

@app.route("/api/grape/predict", methods=["POST"])
def predict_grape():
    try:
        if 'grape' not in models or models['grape'] is None:
            return jsonify({'error': 'Grape model not available'}), 500

        if 'file' not in request.files:
            return jsonify({'error': 'No file part in request'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        image_bytes = file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image = image.resize(IMG_SIZE)  # 224x224

        img_array = np.array(image)
        img_array = preprocess_input(img_array)  # MobileNetV2 preprocessing
        img_array = np.expand_dims(img_array, axis=0)  # (1,224,224,3)

        features = feature_extractor(img_array, training=False).numpy()  # (1,1280)
        prediction = models['grape'].predict(features)[0]
        predicted_class = GRAPE_CLASSES[int(np.argmax(prediction))]

        return jsonify({
            'plant': 'grape',
            'prediction': predicted_class
        }), 200

    except Exception as e:
        print("Error in /api/grape/predict:", e)
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route("/api/health", methods=["GET"])
def health_check():
    available = list(models.keys())
    if tomato_model:
        available.append('tomato')
    if grape_model:
        available.append('grape')

    return jsonify({
        'status': 'healthy',
        'available_plants': available
    }), 200


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
