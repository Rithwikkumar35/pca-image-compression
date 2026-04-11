import numpy as np
from sklearn.decomposition import PCA

def psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

def apply_pca(image, k):
    pca = PCA(n_components=k)
    transformed = pca.fit_transform(image)
    reconstructed = pca.inverse_transform(transformed)

    variance = sum(pca.explained_variance_ratio_)
    return reconstructed, variance
