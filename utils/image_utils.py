import numpy as np
from sklearn.decomposition import PCA
import math

def apply_pca_channel(channel, k):
    pca = PCA(n_components=k)
    transformed = pca.fit_transform(channel)
    restored = pca.inverse_transform(transformed)
    return restored

def apply_pca(img, k):
    if len(img.shape) == 2:
        compressed = apply_pca_channel(img, k)
    else:
        channels = []
        for i in range(3):
            channel = apply_pca_channel(img[:, :, i], k)
            channels.append(channel)
        compressed = np.stack(channels, axis=2)

    compressed = np.clip(compressed, 0, 255)
    return compressed.astype(np.uint8)

def calculate_psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return 100
    return 20 * math.log10(255.0 / math.sqrt(mse))
