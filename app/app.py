import streamlit as st
import numpy as np
from PIL import Image
import pickle
import os
import sys
import matplotlib.pyplot as plt

# ---------- FIX IMPORT PATH ----------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.image_utils import apply_pca, calculate_psnr

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="PCA Compression", layout="wide")

# ---------- LOAD CSS ----------
css_path = os.path.join(os.path.dirname(__file__), "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    return pickle.load(open("model/model.pkl", "rb"))

model = load_model()

# ---------- TITLE ----------
st.markdown("<h1 class='title'>🚀 Intelligent Image Compression using PCA + AI</h1>", unsafe_allow_html=True)

# ---------- FILE UPLOAD ----------
uploaded = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded:

    # ---------- LOAD IMAGE ----------
    image = Image.open(uploaded).convert("RGB")
    img = np.array(image)

    # ---------- SLIDER ----------
    k = st.slider("🎚️ Compression Level (K)", 5, 50, 25)

    # ---------- FAST PROCESS (CACHED) ----------
    @st.cache_data
    def process(img, k):
        compressed = apply_pca(img, k)
        psnr = calculate_psnr(img, compressed)
        variance = np.var(img - compressed)
        return compressed, psnr, variance

    compressed, score, variance = process(img, k)

    # ---------- AI PREDICTION ----------
    try:
        predicted_k = model.predict([[variance, score]])[0]
        predicted_k = int(max(5, min(50, predicted_k)))
    except:
        predicted_k = k  # fallback

    # ---------- DISPLAY IMAGES ----------
    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="📷 Original Image", use_column_width=True)

    with col2:
        st.image(compressed, caption="🧠 Compressed Image", use_column_width=True)

    # ---------- METRICS ----------
    st.markdown("### 📊 Metrics")

    m1, m2, m3 = st.columns(3)

    m1.metric("PSNR", f"{score:.2f}")
    m2.metric("Variance", f"{variance:.2f}")
    m3.metric("Recommended K (AI)", predicted_k)

    # ---------- REAL FILE COMPRESSION ----------
    result = Image.fromarray(compressed)

    output_path = "compressed.jpg"
    result.save(output_path, "JPEG", quality=60, optimize=True)

    # ---------- FILE SIZE ----------
    original_size = len(uploaded.getvalue()) / 1024
    compressed_size = os.path.getsize(output_path) / 1024

    st.markdown("### 📦 File Size Comparison")

    s1, s2 = st.columns(2)
    s1.metric("Original Size (KB)", f"{original_size:.2f}")
    s2.metric("Compressed Size (KB)", f"{compressed_size:.2f}")

    # ---------- DOWNLOAD ----------
    with open(output_path, "rb") as f:
        st.download_button("📥 Download Compressed Image", f, "compressed.jpg")

    # ---------- GRAPH ----------
    st.markdown("### 📈 Compression Analysis")

    ks = list(range(5, 50, 5))
    psnr_values = []

    for val in ks:
        comp = apply_pca(img, val)
        psnr_values.append(calculate_psnr(img, comp))

    fig = plt.figure()
    plt.plot(ks, psnr_values)
    plt.xlabel("K (Compression Level)")
    plt.ylabel("PSNR (Quality)")
    plt.title("Compression vs Quality")

    st.pyplot(fig)