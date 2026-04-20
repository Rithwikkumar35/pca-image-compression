import streamlit as st
import numpy as np
from PIL import Image
import os
import sys
import pickle
import matplotlib.pyplot as plt

# ---------- FIX IMPORT ----------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.image_utils import apply_pca, calculate_psnr

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="PCA Compression",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# ---------- SIDEBAR ----------
st.sidebar.title("⚙️ Controls")

uploaded = st.sidebar.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
k = st.sidebar.slider("Compression Level (K)", 5, 50, 25)

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.write("AI-based PCA Image Compression System")

# ---------- HEADER ----------
st.markdown("<h1 class='title'>🚀 Intelligent Image Compression</h1>", unsafe_allow_html=True)

if uploaded:

    # ---------- LOAD IMAGE ----------
    image = Image.open(uploaded).convert("RGB")

    # 🔥 Resize for performance
    image = image.resize((256, 256))
    img = np.array(image)

    # ---------- PROCESS ----------
    @st.cache_data
    def process(img, k):
        compressed = apply_pca(img, k)
        psnr = calculate_psnr(img, compressed)
        variance = np.var(img - compressed)
        return compressed, psnr, variance

    compressed, score, variance = process(img, k)

    # ---------- AI ----------
    try:
        predicted_k = model.predict([[variance, score]])[0]
        predicted_k = int(max(5, min(50, predicted_k)))
    except:
        predicted_k = k

    # ---------- IMAGE SECTION ----------
    st.markdown("## 🖼️ Image Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Original Image")

    with col2:
        st.image(compressed, caption="Compressed Image")

    # ---------- METRICS ----------
    st.markdown("## 📊 Performance Metrics")

    m1, m2, m3 = st.columns(3)

    m1.markdown(f'<div class="card"><div class="label">PSNR</div><div class="metric">{score:.2f}</div></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="card"><div class="label">Variance</div><div class="metric">{variance:.2f}</div></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="card"><div class="label">AI K</div><div class="metric">{predicted_k}</div></div>', unsafe_allow_html=True)

    # ---------- FILE SIZE ----------
    result = Image.fromarray(compressed)
    output_path = "compressed.jpg"
    result.save(output_path, "JPEG", quality=60, optimize=True)

    original_size = len(uploaded.getvalue()) / 1024
    compressed_size = os.path.getsize(output_path) / 1024

    st.markdown("## 📦 File Size")

    s1, s2 = st.columns(2)
    s1.metric("Original (KB)", f"{original_size:.2f}")
    s2.metric("Compressed (KB)", f"{compressed_size:.2f}")

    # ---------- DOWNLOAD ----------
    with open(output_path, "rb") as f:
        st.download_button("📥 Download Compressed Image", f, "compressed.jpg")

    # ---------- GRAPH ----------
    st.markdown("## 📈 Compression Analysis")

    ks = [5, 15, 25, 35, 45]

    @st.cache_data
    def generate_graph(img):
        values = []
        for val in ks:
            comp = apply_pca(img, val)
            values.append(calculate_psnr(img, comp))
        return values

    psnr_values = generate_graph(img)

    fig, ax = plt.subplots()
    ax.plot(ks, psnr_values, marker='o')
    ax.set_xlabel("K")
    ax.set_ylabel("PSNR")
    ax.set_title("Compression vs Quality")

    st.pyplot(fig)

else:
    st.info("👈 Upload an image from the sidebar to start")