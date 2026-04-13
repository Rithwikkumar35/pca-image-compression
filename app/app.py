import streamlit as st
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import pickle
import matplotlib.pyplot as plt

# Load model
model = pickle.load(open("model/model.pkl", "rb"))

def psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

# PCA grayscale
def compress_gray(image, k):
    pca = PCA(n_components=int(k))
    transformed = pca.fit_transform(image)
    reconstructed = pca.inverse_transform(transformed)
    return np.clip(reconstructed, 0, 255)

# PCA color
def compress_color(image, k):
    r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]

    r = compress_gray(r, k)
    g = compress_gray(g, k)
    b = compress_gray(b, k)

    return np.stack([r, g, b], axis=2)

# UI
st.title("🚀 Intelligent Image Compression using PCA + ML")
st.write("Supports grayscale & color images with ML-based optimal compression.")

file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if file:
    img = Image.open(file)
    img_np = np.array(img)

    k = st.sidebar.slider("Compression Level (K)", 5, 25, 10)

    # Detect grayscale or color
    if len(img_np.shape) == 2:
        compressed = compress_gray(img_np, k)
    else:
        compressed = compress_color(img_np, k)

    # Metrics
    variance = np.var(compressed)
    score = psnr(img_np, compressed)

    predicted_k = model.predict([[variance, score]])[0]
    predicted_k = int(max(5, min(25, predicted_k)))

    # Layout
    col1, col2 = st.columns(2)

    with col1:
        st.image(img_np, caption="Original Image")

    with col2:
        st.image(compressed.astype('uint8'), caption="Compressed Image")

    # Metrics display
    st.write(f"📊 PSNR: {score:.2f}")
    st.write(f"📈 Variance: {variance:.2f}")
    st.write(f"🤖 Recommended K (AI): {predicted_k}")

    # Compression Ratio
    original_size = img_np.size
    compressed_size = compressed.size
    ratio = original_size / compressed_size
    st.write(f"📦 Compression Ratio: {ratio:.2f}")

    # 🔥 GRAPH (NEW)
    st.subheader("📊 PSNR vs K Graph")

    k_values = [5, 10, 15, 20, 25]
    psnr_values = []

    for val in k_values:
        if len(img_np.shape) == 2:
            temp = compress_gray(img_np, val)
        else:
            temp = compress_color(img_np, val)

        psnr_values.append(psnr(img_np, temp))

    fig, ax = plt.subplots()
    ax.plot(k_values, psnr_values)
    ax.set_xlabel("K")
    ax.set_ylabel("PSNR")
    ax.set_title("Compression vs Quality")

    st.pyplot(fig)

    # Download
    result = Image.fromarray(compressed.astype('uint8'))
    result.save("output.png")

    with open("output.png", "rb") as f:
        st.download_button("📥 Download Compressed Image", f, "compressed.png")