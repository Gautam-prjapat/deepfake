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
    with st.spinner("Downloading/Loading ISTVT model weights. This may take a minute on the first run..."):
        try:
            dataset_path = kagglehub.dataset_download("gam888i/istvt-pth")
            weights_path = os.path.join(dataset_path, "istvt_master_weights.pth")
            
            if not os.path.exists(weights_path):
                st.error(f"Critical Error: Downloaded the dataset but could not find the weights file at {weights_path}.")
                st.stop()
                
            return load_model(weights_path)
            
        except Exception as e:
            st.error(f"Failed to download or load the model from Kaggle: {e}")
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
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Source Video")
        st.video(uploaded_file)

    with col2:
        st.subheader("Analysis & Verdict")
        analyze_button = st.button("Run ISTVT Analysis", type="primary", use_container_width=True)
        
        if analyze_button:
            with st.spinner("Extracting frames and running Spatial-Temporal attention mechanisms..."):
                video_path = None
                try:
                    # Create and safely close temp file so OpenCV can read it cleanly
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                        tfile.write(uploaded_file.read())
                        video_path = tfile.name
                    
                    # Run the inference and extract XAI heatmaps
                    probability, spatial_heatmap, temporal_heatmap = predict_video(model, device, video_path)
                    
                    st.divider()
                    
                    # 1. The Verdict Logic
                    if probability > 0.5:
                        confidence = probability * 100
                        st.error("🚨 **VERDICT: DEEPFAKE DETECTED**")
                        st.progress(probability)
                        st.markdown(f"**Confidence Score:** {confidence:.2f}% probability of manipulation.")
                    else:
                        confidence = (1.0 - probability) * 100
                        st.success("✅ **VERDICT: AUTHENTIC VIDEO**")
                        st.progress(1.0 - probability)
                        st.markdown(f"**Confidence Score:** {confidence:.2f}% probability of authenticity.")
                    
                    st.divider()

                    # 2. XAI / LRP Visualizations (Matches Resume Claims)
                    st.subheader("🔍 Interpretable AI (LRP Analysis)")
                    st.caption("Visualizing specific manipulation artifacts captured by the spatial and temporal attention heads.")
                    
                    col_spat, col_temp = st.columns(2)
                    with col_spat:
                        st.image(spatial_heatmap, caption="Spatial Attention (Blending/Forgery Artifacts)", use_container_width=True)
                    with col_temp:
                        st.image(temporal_heatmap, caption="Temporal Attention (Inter-frame Inconsistencies)", use_container_width=True)
                        
                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")
                
                finally:
                    # Clean up file immediately after processing is done
                    if video_path and os.path.exists(video_path):
                        try:
                            os.remove(video_path)
                        except Exception:
                            pass
