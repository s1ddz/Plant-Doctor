import streamlit as st
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

# Set page config
st.set_page_config(page_title="Plant Disease Classifier", layout="centered")

# Class labels
CLASS_NAMES = ['angular_leaf_spot', 'bean_rust', 'healthy']

# Load model
@st.cache_resource
def load_model():
    return joblib.load("/home/smurfy/Desktop/Plant_Disease_Detection/MAJOR_PROJECT/MODELS/MobileNetV2/SVM/MobileNetV2_SVM.joblib")  # Adjust filename if needed

# Load MobileNetV2 as feature extractor
@st.cache_resource
def load_feature_extractor():
    model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))
    model.trainable = False
    return model

# Extract features from uploaded image
def extract_features(img, model):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features

# Streamlit UI
st.title("Plant Disease Classification")
st.write("Upload a bean leaf image to predict whether it's healthy or diseased.")

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    with st.spinner(" Extracting features and predicting..."):
        # Load models
        clf = load_model()
        feature_extractor = load_feature_extractor()

        # Extract features and predict
        features = extract_features(img, feature_extractor)
        pred = clf.predict(features)[0]
        confidence = clf.predict_proba(features)[0][pred] if hasattr(clf, "predict_probablities") else None

    # Output
    st.success(f" Prediction: **{CLASS_NAMES[pred]}**")
    if confidence is not None:
        st.write(f" Confidence: {confidence:.2f}")
