import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

st.set_page_config(page_title="Image Denoising", layout="centered")

st.markdown(
    """
    <h2 style='text-align:center;'>Image Denoising using UNet (TensorFlow)</h2>
    <p style='text-align:center;'>Upload citra buram atau penuh noise, dan model akan memulihkannya secara otomatis.</p>
    """,
    unsafe_allow_html=True
)

def resize_image(image, base=16):
    w, h = image.size
    new_w = (w // base) * base
    new_h = (h // base) * base
    return image.resize((new_w, new_h))

@st.cache_resource
def load_denoising_model():
    # Ubah nama model di sini sesuai file yang mau dipakai
    model_path = os.path.join(os.getcwd(), "best_denoising_model.h5")
    model = load_model(model_path, compile=False)
    return model

model = load_denoising_model()

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  # grayscale
    image = resize_image(image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocessing
    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))  # (1, H, W, 1)

    # Inference
    output = model.predict(img_array)
    output = np.clip(output[0, :, :, 0], 0, 1)

    # Convert back to image
    output_image = Image.fromarray((output * 255).astype(np.uint8))
    st.image(output_image, caption="Denoised Image", use_column_width=True)
