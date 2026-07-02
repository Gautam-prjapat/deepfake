import streamlit as st
import os
import tempfile
from model_utils import load_model, predict_video

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="DeepSight | ISTVT Deepfake Detector",
    page_icon="👁️",
    layout="wide"
)

# ==========================================
# CACHE THE MODEL (Prevents reloading on every click)
# ==========================================
@st.cache_resource
def get_model_and_device():
    weights_path = "istvt_master_weights.pth"
    if not os.path.exists(weights_path):
        st.error(f"Critical Error: Missing weights file at {weights_path}. Please place it in the same folder as this app.")
        st.stop()
    return load_model(weights_path)

model, device = get_model_and_device()

# ==========================================
# UI DASHBOARD
# ==========================================
st.title("👁️ DeepSight: Video Authenticity Engine")
st.markdown("Powered by the Interpretable Spatial-Temporal Video Transformer (ISTVT)")
st.divider()

# Upload section
uploaded_file = st.file_uploader("Upload a video for deepfake analysis", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # We must save the uploaded file temporarily so OpenCV can read it from a file path
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Source Video")
        st.video(video_path)

    with col2:
        st.subheader("Analysis & Verdict")
        analyze_button = st.button("Run ISTVT Analysis", type="primary", use_container_width=True)
        
        if analyze_button:
            with st.spinner("Extracting frames and running attention mechanisms..."):
                try:
                    # Run the inference math from model_utils.py
                    probability = predict_video(model, device, video_path)
                    
                    st.divider()
                    
                    # The Verdict Logic
                    if probability > 0.5:
                        confidence = probability * 100
                        st.error("🚨 **VERDICT: DEEPFAKE DETECTED**")
                        st.progress(probability)
                        st.markdown(f"**Confidence Score:** {confidence:.2f}% probability of manipulation.")
                        st.caption("The Spatial-Temporal attention heads detected synthetic blending artifacts in this sequence.")
                    else:
                        confidence = (1.0 - probability) * 100
                        st.success("✅ **VERDICT: AUTHENTIC VIDEO**")
                        st.progress(1.0 - probability)
                        st.markdown(f"**Confidence Score:** {confidence:.2f}% probability of authenticity.")
                        st.caption("No significant spatial or temporal manipulation artifacts were detected.")
                        
                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")
            
    # Clean up the temporary file so we don't clog your hard drive
    try:
        os.remove(video_path)
    except:
        pass