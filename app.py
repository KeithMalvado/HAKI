import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
from model_def import SRCNN, ResNetSR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_models():
    srcnn = SRCNN().to(device)
    resnet = ResNetSR().to(device)
    srcnn.load_state_dict(torch.load("model_srcnn.pt", map_location=device))
    resnet.load_state_dict(torch.load("model_resnetsr.pt", map_location=device))
    srcnn.eval()
    resnet.eval()
    return srcnn, resnet

srcnn_model, resnet_model = load_models()
transform = transforms.Compose([transforms.ToTensor()])

def restore_image(model, image):
    lr = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        sr = model(lr).cpu().squeeze().permute(1, 2, 0).numpy()
    sr = np.clip(sr, 0, 1)
    return Image.fromarray((sr * 255).astype(np.uint8))

st.set_page_config(page_title="Image Restoration", page_icon="🧠", layout="centered")

st.markdown(
    """
    <h2 style='text-align:center;'>🧠 Image Restoration using CNN & ResNetSR</h2>
    <p style='text-align:center;'>Upload citra buram atau redup untuk direstorasi menggunakan model Deep Learning.</p>
    """,
    unsafe_allow_html=True
)

uploaded = st.file_uploader("📤 Upload gambar", type=["jpg", "jpeg", "png"])
model_choice = st.selectbox("🧩 Pilih model restorasi", ["SRCNN", "ResNetSR"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Citra Asli", use_container_width=True)

    if st.button("🔧 Pulihkan Gambar"):
        model = srcnn_model if model_choice == "SRCNN" else resnet_model
        with st.spinner("🧠 Model sedang memulihkan citra... harap tunggu..."):
            result = restore_image(model, img)
        st.image(result, caption=f"Hasil Restorasi ({model_choice})", use_container_width=True)
        st.success("✅ Proses restorasi selesai!")
