import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="VisionAI | Cat vs Dog",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Professional UI styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    /* Global */
    .stApp {
        background: #f4f7fb;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #1f2937;
    }

    [data-testid="stSidebar"] * {
        color: #e5e7eb;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Hero */
    .hero {
        background: linear-gradient(135deg, #111827 0%, #1f2937 55%, #334155 100%);
        border-radius: 22px;
        padding: 30px 34px;
        margin-bottom: 24px;
        box-shadow: 0 12px 35px rgba(15, 23, 42, 0.16);
    }

    .hero-kicker {
        color: #93c5fd;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .hero-title {
        color: white;
        font-size: 38px;
        font-weight: 800;
        line-height: 1.15;
        margin: 0;
    }

    .hero-subtitle {
        color: #cbd5e1;
        font-size: 16px;
        margin-top: 10px;
        margin-bottom: 0;
    }

    /* Cards */
    .card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 22px;
        margin-bottom: 18px;
        box-shadow: 0 5px 18px rgba(15, 23, 42, 0.06);
    }

    .card-title {
        color: #111827;
        font-size: 20px;
        font-weight: 750;
        margin-bottom: 3px;
    }

    .card-subtitle {
        color: #6b7280;
        font-size: 13px;
        margin-bottom: 16px;
    }

    /* Metric cards */
    .metric {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 15px;
        padding: 17px 18px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.045);
    }

    .metric-label {
        color: #6b7280;
        font-size: 12px;
        font-weight: 650;
        text-transform: uppercase;
        letter-spacing: .7px;
    }

    .metric-value {
        color: #111827;
        font-size: 21px;
        font-weight: 800;
        margin-top: 5px;
    }

    /* Result */
    .result {
        text-align: center;
        border-radius: 18px;
        padding: 26px 18px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
    }

    .result-label {
        color: #64748b;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    .result-value {
        color: #111827;
        font-size: 42px;
        font-weight: 850;
        margin: 7px 0;
    }

    .result-confidence {
        color: #475569;
        font-size: 16px;
    }

    .status-pill {
        display: inline-block;
        background: #ecfdf5;
        color: #047857;
        border: 1px solid #a7f3d0;
        border-radius: 999px;
        padding: 5px 11px;
        font-size: 12px;
        font-weight: 700;
    }

    .empty-state {
        text-align: center;
        padding: 58px 20px;
        border: 1.5px dashed #cbd5e1;
        border-radius: 16px;
        background: #f8fafc;
    }

    .empty-icon {
        font-size: 42px;
        margin-bottom: 8px;
    }

    .empty-title {
        color: #334155;
        font-size: 17px;
        font-weight: 750;
    }

    .empty-text {
        color: #64748b;
        font-size: 13px;
    }

    /* Sidebar */
    .sidebar-brand {
        font-size: 22px;
        font-weight: 800;
        color: white;
        margin-bottom: 3px;
    }

    .sidebar-caption {
        color: #94a3b8;
        font-size: 12px;
        margin-bottom: 22px;
    }

    .info-row {
        padding: 9px 0;
        border-bottom: 1px solid #374151;
        font-size: 13px;
    }

    .info-row:last-child {
        border-bottom: none;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 11px;
        font-weight: 700;
        min-height: 44px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 12px;
        padding: 28px 0 8px;
    }

    /* Remove excess top spacing */
    div[data-testid="stVerticalBlock"] > div:has(> div.stMarkdown) {
        scroll-margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Model
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/cat_dog_cnn.keras")


try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model = None
    model_loaded = False
    st.error(
        "Model could not be loaded. Make sure "
        "'model/cat_dog_cnn.keras' exists."
    )
    with st.expander("Technical details"):
        st.code(str(e))

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-brand">🐾 VisionAI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-caption">Image classification dashboard</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### Model Overview")
    st.markdown(
        """
        <div class="info-row"><b>Architecture</b><br>CNN</div>
        <div class="info-row"><b>Framework</b><br>TensorFlow / Keras</div>
        <div class="info-row"><b>Classes</b><br>🐱 Cat &nbsp; • &nbsp; 🐶 Dog</div>
        <div class="info-row"><b>Input Size</b><br>180 × 180 × 3</div>
        <div class="info-row"><b>Task</b><br>Binary image classification</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Workflow")
    st.markdown(
        """
        **01** Upload multiple images  
        **02** Preview images  
        **03** Run batch CNN prediction  
        **04** Review all results
        """
    )

    st.divider()

    if model_loaded:
        st.markdown('<span class="status-pill">● Model Online</span>', unsafe_allow_html=True)
    else:
        st.error("Model Offline")

# ---------------------------------------------------------
# Hero
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">AI Image Intelligence</div>
        <div class="hero-title">🐱 Cat vs Dog Classifier</div>
        <div class="hero-subtitle">
            A clean, production-style dashboard for CNN-powered image classification.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# KPI row
# ---------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(
        '<div class="metric"><div class="metric-label">AI Model</div>'
        '<div class="metric-value">CNN</div></div>',
        unsafe_allow_html=True,
    )

with m2:
    st.markdown(
        '<div class="metric"><div class="metric-label">Framework</div>'
        '<div class="metric-value">TensorFlow</div></div>',
        unsafe_allow_html=True,
    )

with m3:
    st.markdown(
        '<div class="metric"><div class="metric-label">Classes</div>'
        '<div class="metric-value">2</div></div>',
        unsafe_allow_html=True,
    )

with m4:
    st.markdown(
        '<div class="metric"><div class="metric-label">Input</div>'
        '<div class="metric-value">180 × 180</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Main dashboard
# ---------------------------------------------------------
left_column, right_column = st.columns([1, 1], gap="large")

with left_column:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📤 Upload Image</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="card-subtitle">Upload a JPG, JPEG, or PNG image for analysis.</div>',
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Choose images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="You can select multiple cat/dog images at once.",
    )

    if not uploaded_files:
        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-icon">🖼️</div>
                <div class="empty-title">No image selected</div>
                <div class="empty-text">Upload a cat or dog image to begin.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_files:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="card-title">🖼️ Image Preview</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="card-subtitle">{len(uploaded_files)} image(s) selected</div>',
            unsafe_allow_html=True,
        )

        preview_cols = st.columns(min(len(uploaded_files), 4))
        for i, uploaded_file in enumerate(uploaded_files):
            image_preview = Image.open(uploaded_file).convert("RGB")
            with preview_cols[i % len(preview_cols)]:
                st.image(
                    image_preview,
                    caption=uploaded_file.name,
                    width="stretch",
                )

        st.markdown("</div>", unsafe_allow_html=True)

with right_column:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🔍 Batch Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="card-subtitle">Analyze multiple images in one run and review every result.</div>',
        unsafe_allow_html=True,
    )

    if not uploaded_files:
        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-icon">🤖</div>
                <div class="empty-title">Ready for batch analysis</div>
                <div class="empty-text">
                    Upload one or more images and run the model to classify them.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        predict_button = st.button(
            f"🔍  Analyze {len(uploaded_files)} Image(s)",
            type="primary",
            width="stretch",
        )

        if predict_button:
            if not model_loaded:
                st.error(
                    "Model is not available. Save the trained model as "
                    "'model/cat_dog_cnn.keras'."
                )
            else:
                results = []

                progress = st.progress(0)
                status = st.empty()

                for index, uploaded_file in enumerate(uploaded_files):
                    status.text(
                        f"Analyzing {index + 1} of {len(uploaded_files)}: "
                        f"{uploaded_file.name}"
                    )

                    image = Image.open(uploaded_file).convert("RGB")
                    image_resized = image.resize((180, 180))
                    image_array = np.array(image_resized)

                    # Notebook training images use BGR channel order.
                    image_array = image_array[:, :, ::-1]
                    image_array = image_array.astype("float32") / 255.0
                    image_array = np.expand_dims(image_array, axis=0)

                    prediction = model.predict(image_array, verbose=0)
                    probability = float(np.asarray(prediction).squeeze())

                    # Notebook labels: Cat = 0, Dog = 1
                    if probability >= 0.5:
                        result = "🐶 DOG"
                        confidence = probability
                    else:
                        result = "🐱 CAT"
                        confidence = 1.0 - probability

                    results.append(
                        {
                            "file": uploaded_file.name,
                            "image": image,
                            "result": result,
                            "confidence": confidence * 100.0,
                        }
                    )

                    progress.progress((index + 1) / len(uploaded_files))

                status.empty()
                progress.empty()

                st.success(
                    f"Completed analysis of {len(results)} image(s)."
                )

                # Summary metrics
                cats = sum(1 for r in results if r["result"] == "🐱 CAT")
                dogs = sum(1 for r in results if r["result"] == "🐶 DOG")
                avg_confidence = sum(
                    r["confidence"] for r in results
                ) / len(results)

                s1, s2, s3 = st.columns(3)
                with s1:
                    st.metric("🐱 Cats", cats)
                with s2:
                    st.metric("🐶 Dogs", dogs)
                with s3:
                    st.metric("Average Confidence", f"{avg_confidence:.2f}%")

                st.markdown("### Results")

                # Result cards in a compact grid
                result_cols = st.columns(2)

                for index, item in enumerate(results):
                    with result_cols[index % 2]:
                        st.markdown(
                            f"""
                            <div class="result">
                                <div class="result-label">
                                    {item["file"]}
                                </div>
                                <div class="result-value">
                                    {item["result"]}
                                </div>
                                <div class="result-confidence">
                                    Confidence:
                                    <b>{item["confidence"]:.2f}%</b>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        st.progress(
                            min(max(item["confidence"] / 100.0, 0.0), 1.0)
                        )

                        if item["confidence"] >= 90:
                            st.caption("🟢 High confidence")
                        elif item["confidence"] >= 70:
                            st.caption("🟡 Moderate confidence")
                        else:
                            st.caption("🔴 Low confidence")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Reset + footer
# ---------------------------------------------------------
st.divider()

reset_col, _ = st.columns([1, 4])
with reset_col:
    if st.button("🔄 Reset Dashboard", width="stretch"):
        st.rerun()

st.markdown(
    """
    <div class="footer">
        <b>VisionAI — Cat vs Dog Classifier</b><br>
        Powered by Convolutional Neural Network · TensorFlow · Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
