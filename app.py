import streamlit as st
import numpy as np
import pickle
import os
from PIL import Image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import tensorflow as tf

from use import (
    create_model,
    predict_caption,
    tokenizer,
    max_length,
    idx_to_word
)

# --- Load pre-trained model ---
model_path = os.path.join(".", "kaggle", "working", "best_model.h5")
model = create_model(len(tokenizer.word_index) + 1, max_length)
model.load_weights(model_path)

# --- Load VGG16 model ---
vgg_model = VGG16()
vgg_model = tf.keras.Model(vgg_model.input, vgg_model.layers[-2].output)

# --- Image feature extraction ---


def extract_features(img_path):
    img = Image.open(img_path).resize((224, 224)).convert("RGB")
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    feature = vgg_model.predict(img_array, verbose=0)
    return feature


# --- Streamlit App ---
st.title("Image Caption Generator")
st.write("Upload an image and generate a caption using a pre-trained model.")

uploaded_file = st.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Generating caption..."):
        # Save to temp path to reuse existing pipeline
        temp_path = "temp_image.jpg"
        image.save(temp_path)

        # Extract VGG features
        features = extract_features(temp_path)

        # Predict caption
        caption = predict_caption(model, features, tokenizer, max_length)

        # Clean result
        cleaned = caption.replace("startseq", "").replace("endseq", "").strip()
        st.markdown("### Caption:")
        st.success(cleaned)
