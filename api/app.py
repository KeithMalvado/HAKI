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
    srcnn.eval(); resnet.eval()
    return srcnn, resnet

srcnn_model, resnet_model = load_models()
transform = transforms.Compose([transforms.ToTensor()])

def restore_image(model, image):
    lr = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        sr = model(lr).cpu().squeeze().permute(1,2,0).numpy()
    sr = np.clip(sr, 0, 1)
    return Image.fromarray((sr * 255).astype(np.uint8))

st.title("🧠 Image Restoration using CNN and ResNet50")
st.write("Upload an image (blurred or low-resolution), choose model, and view restored result below.")

uploaded = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])
model_choice = st.selectbox("Choose model", ["SRCNN", "ResNetSR"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Original Image", use_container_width=True)

    if st.button("Restore Image"):
        model = srcnn_model if model_choice == "SRCNN" else resnet_model
        with st.spinner("Restoring image... please wait"):
            result = restore_image(model, img)
        st.image(result, caption=f"Restored Image ({model_choice})", use_container_width=True)
        st.success("Restoration completed!")
