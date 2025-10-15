import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import os

st.set_page_config(page_title="Image Denoising", layout="centered")

st.markdown(
    """
    <h2 style='text-align:center;'>Image Denoising using UNet</h2>
    <p style='text-align:center;'>Upload citra buram atau penuh noise, dan model akan memulihkannya secara otomatis.</p>
    """,
    unsafe_allow_html=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[64,128,256,512]):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        for feature in features:
            self.downs.append(nn.Sequential(
                nn.Conv2d(in_channels, feature, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(feature, feature, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ))
            in_channels = feature
        for feature in reversed(features):
            self.ups.append(nn.Sequential(
                nn.Conv2d(feature*2, feature, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(feature, feature, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ))
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features[-1], features[-1]*2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[-1]*2, features[-1]*2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(len(self.ups)):
            x = F.interpolate(x, size=skip_connections[idx].shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat((skip_connections[idx], x), dim=1)
            x = self.ups[idx](x)
        return self.final_conv(x)

@st.cache_resource
def load_model():
    model = UNet()
    model_path = os.path.join(os.getcwd(), "model_unet_denoise.pt")
    if not os.path.exists(model_path):
        st.error("Model file not found! Upload 'model_unet_denoise.pt' ke repo.")
        st.stop()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model

def resize_image(image, base=16):
    w, h = image.size
    new_w = (w // base) * base
    new_h = (h // base) * base
    return image.resize((new_w, new_h))

model = load_model()

uploaded_file = st.file_uploader("Upload an image", type=["png","jpg","jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image = resize_image(image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    input_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output_tensor = model(input_tensor)
    output_image = transforms.ToPILImage()(output_tensor.squeeze().cpu())
    st.image(output_image, caption="Denoised Image", use_column_width=True)
