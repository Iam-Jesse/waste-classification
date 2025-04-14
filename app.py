import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from tempfile import NamedTemporaryFile

# Set page title
st.title("Garbage Classification - Model 2 (SGD)")

# Upload Keras model file
model_file = st.file_uploader("Upload your Keras Model (.h5)", type=["h5"])

if model_file is not None:
    # Save uploaded file to a temporary file on disk (required for .h5 loading)
    with NamedTemporaryFile(delete=False, suffix=".h5") as temp_model_file:
        temp_model_file.write(model_file.read())
        temp_model_file.flush()
        model = tf.keras.models.load_model(temp_model_file.name)

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

        # Class names (update if needed)
        class_names = ['battery', 'biological', 'cardboard', 'clothes', 'glass', 
                       'metal', 'paper', 'plastic', 'shoes']

        st.write("### Prediction:")
        st.write(f"**Class:** {class_names[class_id]}")
        st.write(f"**Confidence:** {np.max(predictions):.2f}")
