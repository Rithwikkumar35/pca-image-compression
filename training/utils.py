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
    reconstructed = np.clip(reconstructed, 0, 255)

    variance = sum(pca.explained_variance_ratio_)
    return reconstructed, variance


# 🔥 NEW: COLOR IMAGE SUPPORT
def apply_pca_color(image, k):
    r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]

    r_rec, _ = apply_pca(r, k)
    g_rec, _ = apply_pca(g, k)
    b_rec, _ = apply_pca(b, k)

    reconstructed = np.stack([r_rec, g_rec, b_rec], axis=2)
    return reconstructed