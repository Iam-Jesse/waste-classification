import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from PIL import Image
import numpy as np
import io

# --- Build model architecture (same as during training) ---
def build_model():
    inputs = Input(shape=(224, 224, 1), name="input_layer")

    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D(2, 2)(x)

    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(2, 2)(x)

    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(2, 2)(x)

    x = layers.Conv2D(256, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(2, 2)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(9, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model

# --- Define class names ---
class_names = [
    "battery", "biological", "cardboard", "clothes", "glass",
    "metal", "paper", "plastic", "shoes", "trash"
]

# --- Streamlit UI ---
st.title("🗑️ Garbage Classifier")
st.write("Upload your trained `.h5` model and an image of waste to get its predicted category.")

# Upload the model file
model_file = st.file_uploader("📦 Upload your `.h5` model file", type="h5")

if model_file is not None:
    # Load the model
    with st.spinner("Loading model..."):
        model = build_model()
        model.load_weights(io.BytesIO(model_file.read()))
        st.success("Model loaded successfully!")

    # Upload an image to classify
    uploaded_file = st.file_uploader("🖼️ Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # Load and display the image
        image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess: resize, convert to array, scale, and add batch dimension
        img = image.resize((224, 224))
        img_array = np.array(img).reshape(224, 224, 1) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array)
        pred_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions)

        # Display results
        st.markdown(f"### 🔍 Prediction: **{pred_class}**")
        st.markdown(f"Confidence: `{confidence:.2f}`")
else:
    st.warning("Please upload a `.h5` model file to get started.")
