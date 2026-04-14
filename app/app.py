import streamlit as st
import numpy as np
from PIL import Image
import pickle
import os
import sys
import matplotlib.pyplot as plt

# ---------- FIX IMPORT ----------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.image_utils import apply_pca, calculate_psnr

st.set_page_config(page_title="PCA Compression", layout="wide")

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    return pickle.load(open("model/model.pkl", "rb"))

model = load_model()

# ---------- TITLE ----------
st.title("🚀 Intelligent Image Compression")

# ---------- UPLOAD ----------
uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")

    # 🔥 RESIZE FOR SPEED (IMPORTANT)
    image = image.resize((256, 256))

    img = np.array(image)

    k = st.slider("Compression Level (K)", 5, 50, 25)

    # ---------- CACHE PROCESS ----------
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

    # ---------- DISPLAY ----------
    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Original")

    with col2:
        st.image(compressed, caption="Compressed")

    # ---------- METRICS ----------
    st.metric("PSNR", f"{score:.2f}")
    st.metric("Variance", f"{variance:.2f}")
    st.metric("AI K", predicted_k)

    # ---------- REAL COMPRESSION ----------
    result = Image.fromarray(compressed)

    output_path = "compressed.jpg"
    result.save(output_path, "JPEG", quality=60, optimize=True)

    # ---------- DOWNLOAD ----------
    with open(output_path, "rb") as f:
        st.download_button("Download", f, "compressed.jpg")

    # ---------- FAST GRAPH ----------
    st.subheader("Compression Analysis")

    ks = [5, 15, 25, 35, 45]   # 🔥 reduced points (FAST)

    @st.cache_data
    def generate_graph(img):
        psnr_values = []
        for val in ks:
            comp = apply_pca(img, val)
            psnr_values.append(calculate_psnr(img, comp))
        return psnr_values

    psnr_values = generate_graph(img)

    fig = plt.figure()
    plt.plot(ks, psnr_values)
    plt.xlabel("K")
    plt.ylabel("PSNR")
    plt.title("Compression vs Quality")

    st.pyplot(fig)