import streamlit as st
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import pickle
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="PCA Compression", layout="wide")

# Load CSS
with open("app/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load model
model = pickle.load(open("model/model.pkl", "rb"))

# PSNR
def psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

# PCA grayscale
def compress_gray(image, k):
    pca = PCA(n_components=int(k))
    transformed = pca.fit_transform(image)
    return np.clip(pca.inverse_transform(transformed), 0, 255)

# PCA color
def compress_color(image, k):
    r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
    return np.stack([
        compress_gray(r, k),
        compress_gray(g, k),
        compress_gray(b, k)
    ], axis=2)

# Cached graph
@st.cache_data
def compute_graph(img_np):
    k_values = [5, 15, 25]
    psnr_values = []

    for val in k_values:
        if len(img_np.shape) == 2:
            temp = compress_gray(img_np, val)
        else:
            temp = compress_color(img_np, val)

        psnr_values.append(psnr(img_np, temp))

    return k_values, psnr_values


# 🔥 HEADER
st.markdown('<div class="title">🚀 Intelligent Image Compression using PCA + ML</div>', unsafe_allow_html=True)

st.write("Upload an image and explore compression with AI-powered recommendations.")

# 🔥 SIDEBAR
st.sidebar.header("⚙️ Controls")
k = st.sidebar.slider("Compression Level (K)", 5, 25, 10)

file = st.file_uploader("📤 Upload Image", type=["jpg", "png", "jpeg"])

if file:
    img = Image.open(file)
    img_np = np.array(img)

    # Compression
    if len(img_np.shape) == 2:
        compressed = compress_gray(img_np, k)
    else:
        compressed = compress_color(img_np, k)

    # Metrics
    variance = np.var(compressed)
    score = psnr(img_np, compressed)

    predicted_k = model.predict([[variance, score]])[0]
    predicted_k = int(max(5, min(25, predicted_k)))

    # 🔥 IMAGE DISPLAY
    col1, col2 = st.columns(2)

    with col1:
        st.image(img_np, caption="🖼️ Original Image", use_column_width=True)

    with col2:
        st.image(compressed.astype('uint8'), caption="📉 Compressed Image", use_column_width=True)

    # 🔥 METRICS CARDS
    m1, m2, m3 = st.columns(3)

    with m1:
        st.markdown(f'<div class="card"><div class="metric">📊 PSNR<br>{score:.2f}</div></div>', unsafe_allow_html=True)

    with m2:
        st.markdown(f'<div class="card"><div class="metric">📈 Variance<br>{variance:.2f}</div></div>', unsafe_allow_html=True)

    with m3:
        st.markdown(f'<div class="card"><div class="metric">🤖 AI K<br>{predicted_k}</div></div>', unsafe_allow_html=True)

    # 🔥 GRAPH
    st.subheader("📊 Compression Analysis")

    with st.spinner("Generating graph..."):
        k_values, psnr_values = compute_graph(img_np)

    fig, ax = plt.subplots()
    ax.plot(k_values, psnr_values, marker='o')
    ax.set_xlabel("K")
    ax.set_ylabel("PSNR")
    ax.set_title("Compression vs Quality")

    st.pyplot(fig)

    # 🔥 DOWNLOAD
    result = Image.fromarray(compressed.astype('uint8'))
    result.save("output.png")

    with open("output.png", "rb") as f:
        st.download_button("📥 Download Compressed Image", f, "compressed.png")