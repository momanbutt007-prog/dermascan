"""
Streamlit web app for the skin lesion classifier.

Run locally with:
    streamlit run streamlit_app.py
"""

import os
import streamlit as st
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "best.pt")

CLASS_INFO = {
    "actinic keratosis": "Pre-cancerous, scaly patch caused by sun damage.",
    "atopic dermatitis": "Chronic inflammatory skin condition (eczema).",
    "benign keratosis": "Non-cancerous skin growth, common with age.",
    "dermatofibroma": "Benign fibrous skin nodule.",
    "melanocytic nevus": "Common mole — benign melanocyte cluster.",
    "melanoma": "Malignant skin cancer arising from melanocytes.",
    "squamous cell carcinoma": "Malignant skin cancer of squamous cells.",
    "tinea ringworm candidiasis": "Fungal skin infection.",
    "vascular lesion": "Benign blood vessel abnormality in the skin.",
}


def inject_css():
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 15% 20%, rgba(15,118,110,0.15) 0%, transparent 45%),
                radial-gradient(circle at 85% 10%, rgba(245,158,11,0.12) 0%, transparent 40%),
                radial-gradient(circle at 50% 90%, rgba(15,118,110,0.10) 0%, transparent 50%),
                linear-gradient(160deg, #ECFDF5 0%, #F0FDFA 40%, #FEFCE8 100%);
            background-attachment: fixed;
        }
        .main-header {
            background: linear-gradient(135deg, #0F766E 0%, #134E4A 100%);
            padding: 2rem 2rem 1.5rem 2rem;
            border-radius: 16px;
            margin-bottom: 1.5rem;
            color: white;
            box-shadow: 0 8px 24px rgba(15,118,110,0.25);
        }
        .main-header h1 {
            margin: 0;
            font-size: 2rem;
            font-weight: 700;
        }
        .main-header p {
            margin: 0.4rem 0 0 0;
            opacity: 0.9;
            font-size: 0.95rem;
        }
        .disclaimer-box {
            background: rgba(255, 251, 235, 0.9);
            backdrop-filter: blur(6px);
            border-left: 4px solid #F59E0B;
            padding: 0.9rem 1.1rem;
            border-radius: 8px;
            font-size: 0.88rem;
            color: #78350F;
            margin-bottom: 1.5rem;
        }
        .result-card {
            background: rgba(255, 255, 255, 0.85);
            backdrop-filter: blur(8px);
            border: 1px solid #E5E7EB;
            border-radius: 14px;
            padding: 1.2rem 1.4rem;
            margin-bottom: 0.7rem;
            box-shadow: 0 4px 14px rgba(0,0,0,0.06);
        }
        .top-result {
            border: 2px solid #0F766E;
            background: rgba(240, 253, 250, 0.92);
        }
        .rank-badge {
            display: inline-block;
            background: #0F766E;
            color: white;
            font-size: 0.72rem;
            font-weight: 700;
            padding: 2px 9px;
            border-radius: 999px;
            margin-right: 8px;
        }
        .rank-badge.top {
            background: #F59E0B;
        }
        .class-name {
            font-size: 1.05rem;
            font-weight: 600;
            color: #111827;
        }
        .class-desc {
            font-size: 0.82rem;
            color: #6B7280;
            margin-top: 2px;
        }
        .confidence-value {
            font-size: 1.3rem;
            font-weight: 700;
            color: #0F766E;
            float: right;
        }
        .stButton>button {
            background: #0F766E;
            color: white;
            border-radius: 8px;
            border: none;
        }

        /* ---------- SIDEBAR ---------- */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #134E4A 0%, #0F766E 55%, #0D9488 100%);
        }
        section[data-testid="stSidebar"] * {
            color: #F0FDFA !important;
        }
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: #FFFFFF !important;
            font-weight: 700;
        }
        .sidebar-card {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 12px;
            padding: 1rem 1.1rem;
            margin-bottom: 1rem;
            backdrop-filter: blur(4px);
        }
        .sidebar-metric {
            background: linear-gradient(135deg, #F59E0B, #D97706);
            border-radius: 12px;
            padding: 1rem;
            text-align: center;
            margin-bottom: 1rem;
            box-shadow: 0 4px 12px rgba(245,158,11,0.3);
        }
        .sidebar-metric .value {
            font-size: 2rem;
            font-weight: 800;
            color: white;
            line-height: 1;
        }
        .sidebar-metric .label {
            font-size: 0.78rem;
            color: rgba(255,255,255,0.9);
            margin-top: 4px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .class-pill {
            display: block;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            padding: 6px 10px;
            margin-bottom: 6px;
            font-size: 0.82rem;
            color: #F0FDFA !important;
            border-left: 3px solid #F59E0B;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_model(model_path: str):
    return YOLO(model_path)


def confidence_color(pct: float) -> str:
    if pct >= 70:
        return "#0F766E"   # teal — high confidence
    elif pct >= 40:
        return "#F59E0B"   # amber — moderate
    else:
        return "#9CA3AF"   # gray — low


def main():
    st.set_page_config(page_title="Skin Lesion Classifier", page_icon="🩺", layout="wide")
    inject_css()

    st.markdown(
        """
        <div class="main-header">
            <h1>🩺 Skin Lesion Classifier</h1>
            <p>YOLOv8-based 9-class dermatology image classifier — 87.5% validation accuracy</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="disclaimer-box">
            ⚠️ <b>Educational project only.</b> Not validated for clinical use — do not use
            this tool to make real medical decisions. Please consult a qualified
            dermatologist or physician for any skin concern.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### 🩺 About this model")

        st.markdown(
            """
            <div class="sidebar-metric">
                <div class="value">87.5%</div>
                <div class="label">Validation Accuracy</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="sidebar-card">
                <b>🧠 Trained on</b><br>
                Skin lesion dataset (Split_smol) + real images from the
                <a href="https://www.isic-archive.com/" style="color:#FDE68A;">ISIC Archive</a>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### 🔬 9 Classes")
        for cls in CLASS_INFO:
            st.markdown(f'<div class="class-pill">{cls.title()}</div>', unsafe_allow_html=True)

        st.markdown(
            """
            <div class="sidebar-card" style="margin-top: 1rem;">
                ⚠️ <b>Educational use only</b><br>
                Not validated for clinical diagnosis.
            </div>
            """,
            unsafe_allow_html=True,
        )

    col_left, col_right = st.columns([1, 1.2], gap="large")

    with col_left:
        st.subheader("Upload an image")
        uploaded_file = st.file_uploader(
            "Choose a skin lesion image", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded image", use_container_width=True)

    with col_right:
        st.subheader("Predictions")

        if uploaded_file is None:
            st.info("Upload an image on the left to see predictions here.")
            return

        try:
            model = load_model(MODEL_PATH)
        except Exception as e:
            st.error(
                f"Could not load model from '{MODEL_PATH}'. "
                f"Make sure best.pt is in the models/ folder. Error: {e}"
            )
            return

        with st.spinner("Analyzing image..."):
            result = model.predict(image, verbose=False)[0]

        probs = result.probs
        class_names = result.names
        top_indices = probs.top5[:5]

        for rank, idx in enumerate(top_indices, start=1):
            name = class_names[idx]
            confidence = float(probs.data[idx]) * 100
            color = confidence_color(confidence)
            desc = CLASS_INFO.get(name.lower().strip(), "")
            card_class = "result-card top-result" if rank == 1 else "result-card"
            badge_class = "rank-badge top" if rank == 1 else "rank-badge"

            st.markdown(
                f"""
                <div class="{card_class}">
                    <span class="{badge_class}">#{rank}</span>
                    <span class="confidence-value">{confidence:.1f}%</span>
                    <div class="class-name">{name.title()}</div>
                    <div class="class-desc">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(min(confidence / 100, 1.0))

        st.markdown(
            """
            <div class="disclaimer-box" style="margin-top: 1rem;">
                Remember: this is a demo classifier, not a diagnostic tool.
                Always seek professional medical advice for skin concerns.
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()