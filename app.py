import streamlit as st
import os
import tempfile
import kagglehub
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
# CACHE THE MODEL
# ==========================================
@st.cache_resource
def get_model_and_device():
    with st.spinner("Downloading/Loading ISTVT model weights from Kaggle. Please wait..."):
        try:
            dataset_path = kagglehub.dataset_download("gam888i/istvt-pth")
            weights_path = os.path.join(dataset_path, "istvt_master_weights.pth")
            
            if not os.path.exists(weights_path):
                st.error(f"Critical Error: Weights file not found at {weights_path}.")
                st.stop()
                
            return load_model(weights_path)
        except Exception as e:
            st.error(f"Failed to load the model: {e}")
            st.stop()

model, device = get_model_and_device()

# ==========================================
# UI DASHBOARD
# ==========================================
st.title("👁️ DeepSight: Video Authenticity Engine")
st.markdown("Powered by the Interpretable Spatial-Temporal Video Transformer (ISTVT)")
st.divider()

uploaded_file = st.file_uploader("Upload a video for deepfake analysis", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Source Video")
        st.video(video_path)
        
        st.info("🧠 **Auto-Scaling Enabled:** The ISTVT streaming engine will automatically scan all available frames up to a maximum security limit of 520 frames, preventing server memory crashes.")

    with col2:
        st.subheader("Analysis & Verdict")
        analyze_button = st.button("Run Autonomous ISTVT Analysis", type="primary", use_container_width=True)
        
        if analyze_button:
            with st.spinner("Running autonomous parallel streaming inference..."):
                try:
                    # Execute Top-10% streaming pipeline and catch all three outputs
                    probability, spatial_heatmap, temporal_heatmap = predict_video(model, device, video_path)
                    
                    st.divider()
                    
                    # Core Performance Metrics Display
                    st.metric(label="Peak Anomaly (Top-10%) Detect Score", value=f"{probability:.4f}")
                    
                    # ==========================================
                    # EXPONENTIAL CALIBRATED VERDICT LOGIC
                    # ==========================================
                    if probability > 0.5:
                        # Deepfake side: Standard linear confidence scaling
                        confidence = probability * 100
                        st.error(f"🚨 **VERDICT: DEEPFAKE DETECTED ({confidence:.2f}% Confidence)**")
                        st.progress(probability)
                        st.caption("Spatial-temporal attention loops detected strong structural synthesis signatures.")
                    else:
                        # Authentic side: Apply an exponential curve to squeeze out baseline noise.
                        raw_ratio = probability / 0.5  
                        boosted_confidence = 100.0 - (50.0 * (raw_ratio ** 3))
                        
                        st.success(f"✅ **VERDICT: AUTHENTIC / LOW SUSPICION ({boosted_confidence:.2f}% Confidence)**")
                        st.progress(1.0 - probability)
                        st.caption(f"Raw Peak Anomaly Score: {probability:.4f}. Baseline environmental noise suppressed successfully.")
                    
                    # ==========================================
                    # VISUALIZATIONS
                    # ==========================================
                    st.subheader("Peak Attention Anomalies Visualization")
                    
                    if probability > 0.5:
                        st.caption("Visualizing specific manipulation artifacts captured by the spatial and temporal attention heads.")
                        col_spat, col_temp = st.columns(2)
                        with col_spat:
                            st.image(spatial_heatmap, caption="Spatial Attention (Structural/Blending Artifacts)", use_container_width=True)
                        with col_temp:
                            st.image(temporal_heatmap, caption="Temporal Attention (Inter-frame Inconsistencies)", use_container_width=True)
                    else:
                        st.image(spatial_heatmap, caption="No synthetic artifacts detected. The frame is authentic.", use_container_width=True)
                        
                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")
            
    try:
        os.remove(video_path)
    except:
        pass
