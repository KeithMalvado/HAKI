import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

st.set_page_config(page_title="Image Denoising", layout="centered")

st.markdown(
    """
    <h2 style='text-align:center;'>Image Denoising using Deep UNet</h2>
    <p style='text-align:center;'>Upload citra buram atau penuh noise, dan model akan memulihkannya secara otomatis.</p>
    """,
    unsafe_allow_html=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.down1 = DoubleConv(1,64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64,128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128,256)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(256,512)
        self.pool4 = nn.MaxPool2d(2)
        self.bridge = DoubleConv(512,1024)

        self.up1 = nn.ConvTranspose2d(1024,512,2,stride=2)
        self.upconv1 = DoubleConv(1024,512)
        self.up2 = nn.ConvTranspose2d(512,256,2,stride=2)
        self.upconv2 = DoubleConv(512,256)
        self.up3 = nn.ConvTranspose2d(256,128,2,stride=2)
        self.upconv3 = DoubleConv(256,128)
        self.up4 = nn.ConvTranspose2d(128,64,2,stride=2)
        self.upconv4 = DoubleConv(128,64)

        self.final = nn.Conv2d(64,1,1)
        self.tanh = nn.Tanh()

    def forward(self,x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        d3 = self.down3(p2)
        p3 = self.pool3(d3)
        d4 = self.down4(p3)
        p4 = self.pool4(d4)
        b = self.bridge(p4)

        up1 = self.up1(b)
        up1 = torch.cat([up1,d4],dim=1)
        up1 = self.upconv1(up1)
        up2 = self.up2(up1)
        up2 = torch.cat([up2,d3],dim=1)
        up2 = self.upconv2(up2)
        up3 = self.up3(up2)
        up3 = torch.cat([up3,d2],dim=1)
        up3 = self.upconv3(up3)
        up4 = self.up4(up3)
        up4 = torch.cat([up4,d1],dim=1)
        up4 = self.upconv4(up4)

        out = self.final(up4)
        out = self.tanh(out)
        return (x+out).clamp(0,1)

def resize_image(image, base=16):
    w,h = image.size
    new_w = (w//base)*base
    new_h = (h//base)*base
    return image.resize((new_w,new_h))

@st.cache_resource
def load_model():
    model = UNet().to(device)
    model_path = os.path.join(os.getcwd(),"model_unet_denoise.pt")
    model.load_state_dict(torch.load(model_path,map_location=device))
    model.eval()
    return model

model = load_model()

uploaded_file = st.file_uploader("Upload an image", type=["png","jpg","jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    image = resize_image(image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    input_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output_tensor = model(input_tensor)
    output_image = transforms.ToPILImage()(output_tensor.squeeze().cpu())
    st.image(output_image, caption="Denoised Image", use_column_width=True)
