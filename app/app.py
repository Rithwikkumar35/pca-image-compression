import streamlit as st
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import pickle

# Load trained model
model = pickle.load(open("../model/model.pkl", "rb"))

# PSNR function
def psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

# PCA compression
def compress(image, k):
    pca = PCA(n_components=int(k))
    transformed = pca.fit_transform(image)
    reconstructed = pca.inverse_transform(transformed)

    # 🔥 FIX 1: Clip values to valid image range
    reconstructed = np.clip(reconstructed, 0, 255)

    return reconstructed

# UI
st.title("🚀 Intelligent Image Compression using PCA + ML")
st.write("This system compresses images using PCA and predicts optimal compression using a trained ML model.")

file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if file:
    img = Image.open(file).convert("L")
    img = np.array(img)

    # Slider
    k = st.slider("Select Compression Level (K)", 5, 25, 10)

    # Compress
    compressed = compress(img, k)

    # Metrics
    variance = np.var(compressed)
    score = psnr(img, compressed)

    # 🔥 FIX 2: Clamp ML output
    predicted_k = model.predict([[variance, score]])[0]
    predicted_k = int(max(5, min(25, predicted_k)))

    # Layout
    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Original Image")

    with col2:
        # 🔥 FIX 3: Convert to uint8 before display
        st.image(compressed.astype('uint8'), caption="Compressed Image")

    # Show metrics
    st.write(f"📊 PSNR: {score:.2f}")
    st.write(f"📈 Variance: {variance:.2f}")
    st.write(f"🤖 Recommended K (AI): {predicted_k}")

    # Download button
    result = Image.fromarray(compressed.astype('uint8'))
    result.save("output.png")

    with open("output.png", "rb") as f:
        st.download_button("📥 Download Compressed Image", f, "compressed.png")