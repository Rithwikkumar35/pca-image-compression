Intelligent Image Compression using PCA & Machine Learning

This project is a web-based application that performs image compression using **Principal Component Analysis (PCA)** and enhances it with a **machine learning model** to recommend the optimal compression level.

The system focuses on balancing **image quality** and **file size**, while providing an intuitive and interactive user interface.

---

## 🌐 Live Application
👉 https://pca-image-compression-autwn9ng38h5brhychvur7.streamlit.app

---

## 📌 Key Features

- 📷 Upload images in JPG, PNG, or JPEG format  
- 🎚 Adjust compression level dynamically using a slider  
- 🧠 AI-based prediction of optimal compression level (K)  
- 📊 Real-time metrics:
  - PSNR (Peak Signal-to-Noise Ratio)
  - Variance (error distribution)
- 🖼 Side-by-side comparison of original and compressed images  
- 📉 Visualization of compression vs quality using graphs  
- 📥 Export compressed images in multiple formats:
  - JPEG
  - PNG
  - WEBP  
- ⚡ Optimized for performance using caching and resizing  

---

## 🧠 System Overview

The project is built around two core components:

### 1. PCA-Based Compression
- Each image is converted into numerical data
- PCA reduces dimensionality by retaining important features
- The image is reconstructed using selected principal components

### 2. Machine Learning Model
- A regression model is trained on:
  - Compression level (K)
  - PSNR values
  - Variance
- The model predicts the **best K value** for new images automatically

---

## ⚙️ How It Works

1. User uploads an image  
2. Image is resized for faster processing  
3. PCA is applied based on selected K value  
4. Image is reconstructed  
5. Quality metrics (PSNR & Variance) are calculated  
6. ML model predicts optimal K  
7. Results are displayed with visual comparison and graphs  
8. User can download the compressed image  

---

## 📊 Metrics Used

- **PSNR (Peak Signal-to-Noise Ratio)**  
  Measures image quality after compression  
  Higher value = better quality  

- **Variance**  
  Indicates how much information is lost during compression  

---

## 🏗 Project Structure


pca-image-compression/
│
├── app/
│ ├── app.py # Main Streamlit application
│ └── style.css # Custom UI styling
│
├── utils/
│ └── image_utils.py # PCA and PSNR functions
│
├── training/
│ ├── dataset.py # Dataset handling
│ ├── train.py # Training data generation
│ ├── model.py # ML model training
│
├── model/
│ └── model.pkl # Trained ML model
│
├── requirements.txt
└── README.md


---

## 🛠 Tech Stack

- **Python**
- **NumPy** – numerical operations  
- **Scikit-learn** – PCA and ML model  
- **Pillow** – image processing  
- **Matplotlib** – data visualization  
- **Streamlit** – web application framework  

---

## 🚀 Running Locally

```bash
git clone https://github.com/Rithwikkumar35/pca-image-compression.git
cd pca-image-compression

pip install -r requirements.txt
streamlit run app/app.py
📈 Performance Optimization
Image resizing for faster PCA computation
Caching to avoid recomputation
Reduced graph computations for smooth UI interaction
🎯 Future Improvements
Implement deep learning-based compression (Autoencoders)
Support full-resolution image processing
Add batch image compression
Improve model accuracy with more training data
Build API backend for integration
👨‍💻 Author

Rithwik Kumar Kodari

