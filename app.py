import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
from model_def import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = UNet().to(device)
    model.load_state_dict(torch.load("model_unet_dnoise.pt", map_location=device))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([transforms.ToTensor()])

def denoise_image(model, image):
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor).cpu().squeeze(0)
    output_img = output.permute(1, 2, 0).numpy()
    output_img = np.clip(output_img, 0, 1)
    return Image.fromarray((output_img * 255).astype(np.uint8))

st.set_page_config(page_title="Image Denoising", layout="centered")

st.markdown(
    """
    <h2 style='text-align:center;'>Image Denoising using UNet</h2>
    <p style='text-align:center;'>Upload citra buram atau penuh noise, dan model akan memulihkannya secara otomatis.</p>
    """,
    unsafe_allow_html=True
)

uploaded = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Citra Asli (Noisy)", use_container_width=True)
    if st.button("Pulihkan Gambar"):
        with st.spinner("Model sedang memulihkan citra..."):
            result = denoise_image(model, img)
        st.image(result, caption="Citra Setelah Denoising", use_container_width=True)


