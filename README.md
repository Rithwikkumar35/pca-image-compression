# 🚀 Intelligent Image Compression using PCA + ML

This project is an end-to-end machine learning application that performs image compression using Principal Component Analysis (PCA) and predicts the optimal compression level using a trained ML model.

🌐 **Live Demo:** https://pca-image-compression-autwn9ng38h5brhychvur7.streamlit.app

---

## 📌 Features

* 📷 Upload any image (JPG, PNG)
* 🎚 Adjust compression level (K) using slider
* 🧠 AI predicts optimal compression level
* 📊 Displays:

  * PSNR (quality metric)
  * Variance
* 🖼 Shows original vs compressed image
* 📥 Download compressed image
* 🌍 Deployed online using Streamlit Cloud

---

## 🧠 How It Works

### 1. PCA Compression

* Converts image into lower dimensions
* Reduces redundant information
* Reconstructs compressed image

### 2. Feature Extraction

* Variance of compressed image
* PSNR (Peak Signal-to-Noise Ratio)

### 3. Machine Learning Model

* Trained on multiple K values
* Learns relationship between:

  * Compression level (K)
  * Image quality metrics
* Predicts best K for new images

---

## 🏗️ Project Structure

pca-image-compression/
│
├── app/
│   └── app.py                # Streamlit web app
│
├── model/
│   └── model.pkl            # Trained ML model
│
├── training/
│   ├── dataset.py           # Load dataset
│   ├── train.py             # Generate training data
│   ├── model.py             # Train ML model
│   ├── utils.py             # PCA + PSNR functions
│
├── requirements.txt
└── README.md

---

## ⚙️ Tech Stack

* Python
* Streamlit
* NumPy
* Pandas
* Scikit-learn
* Pillow

---

## 📊 Key Concepts Used

* Principal Component Analysis (PCA)
* Dimensionality Reduction
* Image Reconstruction
* Regression Model
* PSNR (Image Quality Metric)

---

## 🚀 How to Run Locally

git clone https://github.com/Rithwikkumar35/pca-image-compression.git
cd pca-image-compression/app
pip install -r ../requirements.txt
streamlit run app.py

---

## 🌐 Deployment

This project is deployed using Streamlit Community Cloud.

---

## 🎯 Future Improvements

* Use Deep Learning (Autoencoders)
* Support color images (RGB)
* Batch image compression
* Improve UI/UX design

---

## 👨‍💻 Author

Rithwik Kumar Kodari

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
