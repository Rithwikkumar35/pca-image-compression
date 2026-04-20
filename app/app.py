import streamlit as st
import numpy as np
from PIL import Image
import os, sys, pickle, io
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.image_utils import apply_pca, calculate_psnr

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PCA Compression Studio",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
css_path = os.path.join(os.path.dirname(__file__), "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── MODEL ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return pickle.load(open("model/model.pkl", "rb"))

model = load_model()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div class='sidebar-title'>PCA Compress</div>", unsafe_allow_html=True)
    st.markdown("---")

    uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    st.caption("Supports JPG · PNG · JPEG")

    st.markdown("#### Compression Level")
    preset = st.radio("Preset", ["High (K=45)", "Balanced (K=25)", "Fast (K=10)"], index=1, horizontal=True, label_visibility="collapsed")
    preset_map = {"High (K=45)": 45, "Balanced (K=25)": 25, "Fast (K=10)": 10}
    default_k = preset_map[preset]

    k = st.slider("K Components", 5, 50, default_k, help="Higher K = better quality, larger file")

    st.markdown("---")
    st.markdown("#### Model Status")
    st.success("Model loaded and ready")
    st.markdown("<div class='model-info'>AI will suggest an optimal K based on image statistics.</div>", unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='app-header'>
    <div class='app-title'>Intelligent Image Compression</div>
    <div class='app-subtitle'>Principal Component Analysis · AI-Assisted · Real-time Metrics</div>
</div>
""", unsafe_allow_html=True)

# ── MAIN CONTENT ──────────────────────────────────────────────────────────────
if not uploaded:
    st.markdown("""
    <div class='empty-state'>
        <div class='empty-icon'>🖼️</div>
        <div class='empty-title'>No image uploaded</div>
        <div class='empty-sub'>Upload a JPG, PNG, or JPEG from the sidebar to begin</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── PROCESS ───────────────────────────────────────────────────────────────────
image = Image.open(uploaded).convert("RGB").resize((256, 256))
img = np.array(image)

@st.cache_data
def process(img_bytes, k):
    img_arr = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((256, 256)))
    compressed = apply_pca(img_arr, k)
    psnr      = calculate_psnr(img_arr, compressed)
    variance  = float(np.var(img_arr - compressed))
    ratio     = 1 - (k * 3 * 256) / (256 * 256 * 3)
    return compressed, psnr, variance, max(0.0, ratio)

compressed, score, variance, ratio = process(uploaded.getvalue(), k)

try:
    predicted_k = int(max(5, min(50, model.predict([[variance, score]])[0])))
except Exception:
    predicted_k = k

# Save compressed file
result_img = Image.fromarray(compressed)
output_path = "compressed.jpg"
result_img.save(output_path, "JPEG", quality=60, optimize=True)

original_kb   = len(uploaded.getvalue()) / 1024
compressed_kb = os.path.getsize(output_path) / 1024
size_saved    = max(0.0, (1 - compressed_kb / original_kb) * 100)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab_overview, tab_analysis, tab_export, tab_help = st.tabs(["Overview", "Analysis", "Export", "How to Use"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    # AI suggestion banner
    delta_k   = predicted_k - k
    direction = "increase" if delta_k > 0 else "decrease" if delta_k < 0 else "keep"
    if direction != "keep":
        tip = f"Model suggests K = **{predicted_k}** ({"+" if delta_k > 0 else ""}{delta_k} from current). " \
              f"{'Increasing K could recover fine detail.' if delta_k > 0 else 'Decreasing K reduces file size with minimal quality loss.'}"
    else:
        tip = f"Current K = **{k}** is already optimal for this image."

    st.info(f"🤖 **AI Recommendation** — {tip}")

    # Image comparison
    st.markdown("#### Image Comparison")
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.image(img, caption=f"Original  ·  {original_kb:.1f} KB", use_container_width=True)
    with col2:
        st.image(compressed, caption=f"Compressed  ·  {compressed_kb:.1f} KB  ·  K={k}", use_container_width=True)

    # Metrics row
    st.markdown("#### Performance Metrics")
    m1, m2, m3, m4 = st.columns(4)

    quality_label = "Excellent" if score >= 38 else "Good" if score >= 32 else "Fair" if score >= 28 else "Low"
    m1.markdown(f"""<div class="metric-card">
        <div class="metric-label">PSNR</div>
        <div class="metric-val">{score:.1f} <span class="metric-unit">dB</span></div>
        <div class="metric-sub">{quality_label} quality</div>
    </div>""", unsafe_allow_html=True)

    m2.markdown(f"""<div class="metric-card">
        <div class="metric-label">Pixel Variance</div>
        <div class="metric-val">{variance:.1f} <span class="metric-unit">σ²</span></div>
        <div class="metric-sub">Error spread</div>
    </div>""", unsafe_allow_html=True)

    m3.markdown(f"""<div class="metric-card metric-card--green">
        <div class="metric-label">Size Reduction</div>
        <div class="metric-val">{size_saved:.1f} <span class="metric-unit">%</span></div>
        <div class="metric-sub">{original_kb:.0f} → {compressed_kb:.0f} KB</div>
    </div>""", unsafe_allow_html=True)

    m4.markdown(f"""<div class="metric-card metric-card--blue">
        <div class="metric-label">AI Suggested K</div>
        <div class="metric-val">{predicted_k}</div>
        <div class="metric-sub">{'Optimal ✓' if predicted_k == k else f'Try K={predicted_k}'}</div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab_analysis:
    st.markdown("#### PSNR vs K Components")

    @st.cache_data
    def sweep_k(img_bytes):
        img_arr = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((256, 256)))
        ks, psnrs, sizes = [], [], []
        for val in range(5, 51, 5):
            comp = apply_pca(img_arr, val)
            ks.append(val)
            psnrs.append(calculate_psnr(img_arr, comp))
            buf = io.BytesIO()
            Image.fromarray(comp).save(buf, "JPEG", quality=60)
            sizes.append(buf.tell() / 1024)
        return ks, psnrs, sizes

    ks, psnr_values, size_values = sweep_k(uploaded.getvalue())

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    fig.patch.set_facecolor("#0e1117")

    for ax in axes:
        ax.set_facecolor("#111827")
        ax.tick_params(colors="#9ca3af", labelsize=9)
        ax.spines[:].set_color("#1f2937")
        ax.grid(color="#1f2937", linewidth=0.8, linestyle="--")

    # PSNR curve
    axes[0].plot(ks, psnr_values, color="#1f77b4", linewidth=2, marker="o", markersize=5)
    axes[0].axvline(x=k, color="#ff4b4b", linewidth=1.5, linestyle="--", alpha=0.9, label=f"Current K={k}")
    axes[0].axvline(x=predicted_k, color="#10b981", linewidth=1.5, linestyle=":", alpha=0.9, label=f"AI K={predicted_k}")
    axes[0].set_xlabel("K Components", color="#9ca3af", fontsize=9)
    axes[0].set_ylabel("PSNR (dB)", color="#9ca3af", fontsize=9)
    axes[0].set_title("Quality Curve", color="#e5e7eb", fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=8, facecolor="#111827", edgecolor="#374151", labelcolor="#9ca3af")

    # Size curve
    axes[1].fill_between(ks, size_values, alpha=0.3, color="#f59e0b")
    axes[1].plot(ks, size_values, color="#f59e0b", linewidth=2, marker="s", markersize=5)
    axes[1].axvline(x=k, color="#ff4b4b", linewidth=1.5, linestyle="--", alpha=0.9)
    axes[1].set_xlabel("K Components", color="#9ca3af", fontsize=9)
    axes[1].set_ylabel("File Size (KB)", color="#9ca3af", fontsize=9)
    axes[1].set_title("File Size Curve", color="#e5e7eb", fontsize=11, fontweight="bold")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("#### Pixel Error Distribution")
    error = (img.astype(np.float32) - compressed.astype(np.float32)).flatten()

    fig2, ax2 = plt.subplots(figsize=(10, 2.8))
    fig2.patch.set_facecolor("#0e1117")
    ax2.set_facecolor("#111827")
    ax2.tick_params(colors="#9ca3af", labelsize=9)
    ax2.spines[:].set_color("#1f2937")
    ax2.grid(color="#1f2937", linewidth=0.8, linestyle="--", axis="y")

    ax2.hist(error, bins=80, color="#1f77b4", alpha=0.85, edgecolor="#111827", linewidth=0.4)
    ax2.axvline(0, color="#ff4b4b", linewidth=1.5, linestyle="--", label="Zero error")
    ax2.set_xlabel("Pixel Error Value", color="#9ca3af", fontsize=9)
    ax2.set_ylabel("Count", color="#9ca3af", fontsize=9)
    ax2.set_title("Reconstruction Error Histogram — tighter = better", color="#e5e7eb", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8, facecolor="#111827", edgecolor="#374151", labelcolor="#9ca3af")

    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — EXPORT
# ══════════════════════════════════════════════════════════════════════════════
with tab_export:
    st.markdown("#### Export Settings")

    eq1, eq2 = st.columns(2)
    with eq1:
        export_format = st.selectbox("Format", ["JPEG", "PNG", "WEBP"])
        jpeg_quality  = st.slider("JPEG Quality", 10, 95, 60, disabled=(export_format != "JPEG"))
    with eq2:
        st.markdown("#### Summary")
        st.markdown(f"""
        | Property | Value |
        |---|---|
        | K used | {k} |
        | AI suggested K | {predicted_k} |
        | PSNR | {score:.2f} dB |
        | Variance | {variance:.2f} |
        | Original size | {original_kb:.1f} KB |
        | Compressed size | {compressed_kb:.1f} KB |
        | Space saved | {size_saved:.1f}% |
        """)

    # Re-save with chosen format/quality
    export_buf = io.BytesIO()
    fmt = export_format.upper()
    save_kwargs = {"quality": jpeg_quality, "optimize": True} if fmt == "JPEG" else {}
    result_img.save(export_buf, fmt, **save_kwargs)
    export_buf.seek(0)

    st.download_button(
        label=f"⬇ Download Compressed Image ({export_format})",
        data=export_buf,
        file_name=f"compressed.{export_format.lower()}",
        mime=f"image/{export_format.lower()}",
        use_container_width=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — HOW TO USE
# ══════════════════════════════════════════════════════════════════════════════
with tab_help:
    st.markdown("""
<div class="help-hero">
    <div class="help-hero-title">How to Use PCA Compression Studio</div>
    <div class="help-hero-sub">A step-by-step guide to compressing images with AI-assisted PCA</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("### Quick Start")
    s1, s2, s3, s4 = st.columns(4)
    for col, num, icon, title, body in [
        (s1, "1", "⬆️", "Upload",  "Click the sidebar uploader or drag & drop a JPG / PNG / JPEG image."),
        (s2, "2", "🎚️", "Set K",   "Use the K slider or pick a Preset. Higher K = better quality, bigger file."),
        (s3, "3", "📊", "Analyse", "Check PSNR, Variance, and the AI-suggested K on the Overview tab."),
        (s4, "4", "⬇️", "Export",  "Go to Export, pick your format and quality, then download."),
    ]:
        col.markdown(f"""<div class="step-card">
            <div class="step-num">{num}</div>
            <div class="step-icon">{icon}</div>
            <div class="step-title">{title}</div>
            <div class="step-body">{body}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    c_left, c_right = st.columns(2, gap="large")

    with c_left:
        st.markdown("### Understanding K")
        st.markdown("""
**K** is the number of principal components kept after PCA decomposition.
Each colour channel (R, G, B) is independently projected onto K eigenvectors.

| K range | Effect |
|---|---|
| 5 – 15 | Aggressive compression · noticeable blur |
| 16 – 30 | Balanced — good for most images |
| 31 – 45 | Near-lossless · larger file |
| 46 – 50 | Minimal gain over original |

**Rule of thumb:** Start with the *Balanced* preset (K = 25), check the PSNR,
then nudge K up if fine detail matters or down if file size is the priority.
        """)

        st.markdown("### Reading PSNR")
        st.markdown("""
PSNR (Peak Signal-to-Noise Ratio) measures reconstruction quality in decibels.

| PSNR | Quality label |
|---|---|
| ≥ 38 dB | Excellent — virtually indistinguishable |
| 32 – 37 dB | Good — minor artefacts |
| 28 – 31 dB | Fair — visible softening |
| < 28 dB | Low — significant loss |

Higher is always better. Typical JPEG compression targets 30–35 dB.
        """)

    with c_right:
        st.markdown("### AI Suggested K")
        st.markdown("""
The trained model analyses two statistics from your image:

- **Variance** of the pixel-level error map
- **PSNR** of the current compression

It predicts the K value that best balances quality and size for this image.
The suggestion appears as a green dashed line on the Analysis graphs and
as the **AI K** metric card on the Overview tab.

> The model is a regression model stored in `model/model.pkl`, trained on
> (variance, PSNR → optimal K) pairs.
        """)

        st.markdown("### Presets Explained")
        st.markdown("""
| Preset | K | Best for |
|---|---|---|
| **High** | 45 | Archival, print, medical |
| **Balanced** | 25 | Web images, previews |
| **Fast** | 10 | Thumbnails, quick tests |

Presets set the slider default — you can still fine-tune K afterwards.
        """)

        st.markdown("### Error Histogram (Analysis tab)")
        st.markdown("""
The histogram shows per-pixel reconstruction errors `(original − compressed)`.

- **Narrow spike at 0** → very accurate reconstruction
- **Wide, flat curve** → significant information loss

Use it alongside PSNR to spot channels that degrade more than others.
        """)

    st.markdown("---")
    st.markdown("### Project Structure")
    st.code("""pca-compression/
├── app/
│   ├── app.py          ← Main Streamlit app (this file)
│   └── style.css       ← Custom dark-theme styles
├── utils/
│   └── image_utils.py  ← apply_pca(), calculate_psnr()
├── model/
│   └── model.pkl       ← Trained regression model
└── requirements.txt""", language="text")

    st.markdown("### Running Locally")
    st.code("""# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the app
streamlit run app/app.py""", language="bash")

    st.markdown("### `image_utils.py` — Expected API")
    st.code('''import numpy as np

def apply_pca(img: np.ndarray, k: int) -> np.ndarray:
    """
    img  : uint8 array of shape (H, W, 3)
    k    : number of principal components to keep
    returns: reconstructed uint8 array, same shape
    """
    ...

def calculate_psnr(original: np.ndarray, compressed: np.ndarray) -> float:
    """
    Returns PSNR in dB (float). Higher is better.
    """
    ...''', language="python")

    st.markdown("### FAQ")
    with st.expander("Why is my image resized to 256 × 256?"):
        st.write(
            "PCA on large images is computationally expensive. 256 × 256 keeps "
            "processing under a second while demonstrating all compression effects. "
            "Remove the `.resize()` call in app.py if you need full-resolution output."
        )
    with st.expander("The AI K is very different from my slider — why?"):
        st.write(
            "The model generalises from its training distribution. If your image is "
            "unusually textured or very uniform, the predicted K may seem off. Treat it "
            "as a starting suggestion — verify with the PSNR curve on the Analysis tab."
        )
    with st.expander("Which export format should I choose?"):
        st.write(
            "JPEG — smallest file, slight colour artefacts. Best for photos.\n\n"
            "PNG — lossless container around PCA-reconstructed pixels. Larger than JPEG but no extra encode loss.\n\n"
            "WEBP — modern format with good compression and transparency support. Recommended for web use."
        )
    with st.expander("How do I retrain the model?"):
        st.write(
            "Collect images, run apply_pca at each K from 5 to 50, record (variance, psnr) as features "
            "and the 'best K' as the label (e.g. smallest K where PSNR >= 34 dB). "
            "Train any sklearn regressor and pickle it to model/model.pkl."
        )