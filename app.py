import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

st.set_page_config(page_title="Image Denoising", layout="centered")

st.markdown(
    """
    <h2 style='text-align:center;'>🧼 Restorasi Citra Berbasis U-Net</h2>
    <p style='text-align:center;'>Upload citra buram atau penuh noise, dan model akan memulihkannya secara otomatis.</p>
    <p style='text-align:center;'>Model yang digunakan adalah <b>U-Net</b>, sebuah arsitektur deep learning yang dirancang untuk tugas segmentasi dan restorasi citra</p>
    """,
    unsafe_allow_html=True
)

def resize_image(image, base=16):
    w, h = image.size
    new_w = (w // base) * base
    new_h = (h // base) * base
    return image.resize((new_w, new_h))

@st.cache_resource(show_spinner=True)
def load_denoising_model(model_name="denoising_unet_fine_tuned.h5"):
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
# model_choice = st.selectbox(
#     "Pilih model:",
#     ("best_denoising_model.h5", "denoising_unet_fine_tuned.h5")
# )



model = load_denoising_model()

uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

# 🔹 Tambahan deskripsi di bawah upload
st.markdown(
    """
    <small>
    <p style='text-align:justify;'>
    Silakan unggah gambar yang memiliki noise atau buram untuk diproses oleh sistem. 
    Setelah diunggah, model <b>U-Net</b> akan secara otomatis melakukan proses <i>denoising</i> 
    untuk memulihkan detail dan kejernihan citra.  
    Pastikan ukuran gambar tidak terlalu besar agar proses berjalan lebih cepat.
    </p>
    </small>
    """,
    unsafe_allow_html=True
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # 🔧 Pastikan gambar dalam format RGB
    image = image.convert("RGB")

    # 🔧 Resize ke 256x256 sesuai model
    image = image.resize((256, 256))

    # 🔧 Ubah ke numpy array dan normalisasi
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # tambah dimensi batch

    # Prediksi
    denoised = model.predict(img_array)

    # Konversi hasil ke gambar
    denoised_image = Image.fromarray((denoised[0] * 255).astype(np.uint8))
    # st.image([image, denoised_image], caption=["Original", "Denoised"], width=300, padding)
    st.markdown("<h4 style='text-align:center;'>Hasil Restorasi Citra</h4>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1], gap="large")  # atur jarak antar gambar
    with col1:
        st.image(image, caption="🖼️ Citra Asli", use_column_width=True)
    with col2:
        st.image(denoised_image, caption="✨ Citra Hasil Denoising", use_column_width=True)






