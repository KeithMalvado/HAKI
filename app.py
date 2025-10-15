import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

st.set_page_config(page_title="Image Denoising", layout="centered")

st.markdown(
    """
    <h2 style='text-align:center;'>🧼 Image Denoising using UNet (TensorFlow)</h2>
    <p style='text-align:center;'>Upload citra buram atau penuh noise, dan model akan memulihkannya secara otomatis.</p>
    """,
    unsafe_allow_html=True
)

def resize_image(image, base=16):
    w, h = image.size
    new_w = (w // base) * base
    new_h = (h // base) * base
    return image.resize((new_w, new_h))

@st.cache_resource(show_spinner=True)
def load_denoising_model(model_name="best_denoising_model.h5"):
    model_path = os.path.join(os.path.dirname(__file__), model_name)
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop()
    try:
        model = load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

# Model selector (optional)
model_choice = st.selectbox(
    "Pilih model:",
    ("best_denoising_model.h5", "denoising_unet_fine_tuned.h5")
)
model = load_denoising_model(model_choice)

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  # grayscale
    image = resize_image(image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))  # (1, H, W, 1)

    with st.spinner("⏳ Denoising in progress..."):
        try:
            output = model.predict(img_array)
            output = np.clip(output[0, :, :, 0], 0, 1)
            output_image = Image.fromarray((output * 255).astype(np.uint8))
            st.image(output_image, caption="Denoised Image", use_column_width=True)
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
