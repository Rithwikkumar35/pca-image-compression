import streamlit as st
import numpy as np
from PIL import Image
import pickle
import os

from training.utils import apply_pca, psnr

st.set_page_config(page_title="PCA Compression", layout="wide")

# Load CSS safely
css_path = os.path.join(os.path.dirname(__file__), "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("🚀 Intelligent Image Compression using PCA + ML")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)

    # Slider (IMPORTANT RANGE FIX)
    k = st.slider("Select Compression Level (K)", 20, 150, 50)

    # Apply PCA
    compressed, variance = apply_pca(img, k)
    score = psnr(img, compressed)

    # Load ML model safely
    model_path = os.path.join(os.path.dirname(__file__), "../model/model.pkl")
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        predicted_k = int(model.predict([[variance, score]])[0])
        predicted_k = max(20, min(150, predicted_k))
    else:
        predicted_k = k

    # Layout
    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Original Image", use_column_width=True)

    with col2:
        st.image(compressed, caption="Compressed Image", use_column_width=True)

    # Metrics
    st.markdown("### 📊 Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("PSNR", f"{score:.2f}")
    col2.metric("Variance", f"{variance:.2f}")
    col3.metric("Recommended K (AI)", predicted_k)

    # Download
    result = Image.fromarray(compressed)
    result.save("output.png")

    with open("output.png", "rb") as f:
        st.download_button("📥 Download Compressed Image", f, "compressed.png")