from dataset import load_mnist
from utils import apply_pca, psnr
import pandas as pd

print("STARTING TRAINING...")

images = load_mnist()
print("Loaded images:", len(images))

results = []

# FIXED (IMPORTANT)
k_values = [5, 10, 15, 20, 25]

for img in images:
    for k in k_values:
        reconstructed, variance = apply_pca(img, k)
        score = psnr(img, reconstructed)

        results.append([k, variance, score])

df = pd.DataFrame(results, columns=["K", "Variance", "PSNR"])
df.to_csv("training_data.csv", index=False)

print("Training data created")