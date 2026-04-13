import numpy as np
from sklearn.decomposition import PCA

def apply_pca(image, k):
    image = image.astype(float) / 255.0

    # Grayscale image
    if len(image.shape) == 2:
        h, w = image.shape
        k = min(k, w)

        pca = PCA(n_components=k)
        transformed = pca.fit_transform(image)
        reconstructed = pca.inverse_transform(transformed)

    # Color image (RGB)
    else:
        h, w, c = image.shape
        reconstructed = np.zeros_like(image)

        for i in range(c):
            channel = image[:, :, i]
            k_channel = min(k, channel.shape[1])

            pca = PCA(n_components=k_channel)
            transformed = pca.fit_transform(channel)
            reconstructed[:, :, i] = pca.inverse_transform(transformed)

    # Clip + convert back
    reconstructed = np.clip(reconstructed, 0, 1)
    reconstructed = (reconstructed * 255).astype(np.uint8)

    variance = np.var(reconstructed)

    return reconstructed, variance


def psnr(original, compressed):
    original = original.astype(float)
    compressed = compressed.astype(float)

    mse = np.mean((original - compressed) ** 2)

    if mse == 0:
        return 100

    return 20 * np.log10(255.0 / np.sqrt(mse))