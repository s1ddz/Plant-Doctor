import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2 as cv

# Set page configuration
st.set_page_config(page_title="Tomato Leaf Disease Classifier", layout="centered")

# Define class names
CLASS_NAMES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___healthy",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
]

# Load the trained Keras model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("/home/smurfy/Desktop/Plant_Disease_Detection/MAJOR_PROJECT/TOMATO/MODELS/NEW.h5")

model = load_model()

# UI Header
st.title("Tomato Plant Disease Classification")
st.write("Upload an image of a tomato leaf (64x64 or larger), and the model will predict the disease.")

# File uploader
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image_pil = Image.open(uploaded_file).convert("RGB")
    st.image(image_pil, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Processing image and predicting..."):
        # Convert and preprocess
        img = np.array(image_pil)
        img = cv.resize(img, (64, 64))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        # Predict
        prediction = model.predict(img)[0]
        predicted_class = np.argmax(prediction)
        confidence = prediction[predicted_class]

    # Show result
    st.success(f"Prediction: **{CLASS_NAMES[predicted_class]}**")
    st.write(f"Confidence: **{confidence*100:.2f}%**")
