import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Set page title
st.title("Garbage Classification - Model 2 (SGD)")

# Upload Keras model file
model_file = st.file_uploader("Upload your Keras Model (.h5)", type=["h5"])

if model_file is not None:
    # Load model from the uploaded .h5 file
    model = tf.keras.models.load_model(model_file)
    st.success("Model loaded successfully!")

    # Upload image for prediction
    uploaded_image = st.file_uploader("Upload an image to classify", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert('L')  # Convert to grayscale
        image = image.resize((224, 224))
        img_array = np.array(image).astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=(0, -1))  # Shape: (1, 224, 224, 1)

        # Make prediction
        predictions = model.predict(img_array)
        class_id = np.argmax(predictions)

        # Update this to match your actual class names
        class_names = ['battery', 'biological', 'cardboard', 'clothes', 'glass', 
                       'metal', 'paper', 'plastic', 'shoes']

        st.write("### Prediction:")
        st.write(f"**Class:** {class_names[class_id]}")
        st.write(f"**Confidence:** {np.max(predictions):.2f}")
