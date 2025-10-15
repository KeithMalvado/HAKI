import torch
import streamlit as st

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = torch.load("model_unet_dnoise.pt", map_location=device)
    model.eval()
    return model

model = load_model()
