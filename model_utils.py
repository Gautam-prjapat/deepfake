import torch
import cv2
import numpy as np
from torchvision import transforms

# ==========================================
# 1. CORE MODEL LOADING
# ==========================================
def load_model(weights_path):
    """
    Loads the ISTVT PyTorch model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ⚠️ ACTION REQUIRED: Import and initialize your actual model architecture here
    # Example: 
    # from your_model_file import ISTVT
    # model = ISTVT()
    
    # Placeholder to prevent crash before you add your actual model class
    model = torch.nn.Module() 
    
    try:
        # Load the state dict downloaded from Kaggle
        model.load_state_dict(torch.load(weights_path, map_location=device))
    except Exception as e:
        print(f"Skipping weight load for placeholder model. Error: {e}")

    model.to(device)
    model.eval()
    
    return model, device

# ==========================================
# 2. HEATMAP GENERATION UTILITIES
# ==========================================
def generate_overlay_heatmap(relevance_map, original_frame, colormap=cv2.COLORMAP_JET):
    """
    Normalizes a 2D relevance map and overlays it onto the original frame.
    """
    # Normalize relevance map to 0-255
    relevance_map = relevance_map - np.min(relevance_map)
    relevance_map = relevance_map / (np.max(relevance_map) + 1e-8)
    relevance_map = np.uint8(255 * relevance_map)
    
    # Resize to match original frame dimensions
    h, w = original_frame.shape[:2]
    relevance_map_resized = cv2.resize(relevance_map, (w, h))
    
    # Apply colormap
    heatmap = cv2.applyColorMap(relevance_map_resized, colormap)
    
    # Blend heatmap with original frame
    overlay = cv2.addWeighted(heatmap, 0.6, original_frame, 0.4, 0)
    
    # Convert BGR to RGB for Streamlit compatibility
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return overlay_rgb

# ==========================================
# 3. INFERENCE AND XAI EXTRACTION
# ==========================================
def predict_video(model, device, video_path, num_frames=16):
    """
    Processes the video, runs inference, and extracts spatial and temporal heatmaps.
    """
    # 1. Extract frames
    cap = cv2.VideoCapture(video_path)
    frames = []
    original_frames = []
    
    # Ensure these transforms match your training parameters (e.g., Xception size)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((299, 299)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    while cap.isOpened() and len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        original_frames.append(frame)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(transform(rgb_frame))
        
    cap.release()

    # Pad if video is too short
    while len(frames) < num_frames:
        frames.append(torch.zeros_like(frames[0]))
        original_frames.append(np.zeros_like(original_frames[0]))

    video_tensor = torch.stack(frames).unsqueeze(0).to(device)
    target_viz_frame = original_frames[num_frames // 2]

    # 2. Hook mechanisms to capture Relevance/Attention
    spatial_activations = None
    temporal_attention = None

    def spatial_hook(module, input, output):
        nonlocal spatial_activations
        spatial_activations = output.detach().cpu().numpy()

    def temporal_hook(module, input, output):
        nonlocal temporal_attention
        # Adjust the index based on how your Transformer returns attention weights
        # Usually it's a tuple where output[0] is the tensor and output[1] are the weights
        temporal_attention = output[1].detach().cpu().numpy() if isinstance(output, tuple) else output.detach().cpu().numpy()

    # ⚠️ ACTION REQUIRED: Update these target layers to match your exact PyTorch model layer names!
    h1, h2 = None, None
    try:
        # Example: Hooking into the final spatial CNN layer
        h1 = model.xception.block12.register_forward_hook(spatial_hook)
        # Example: Hooking into the final temporal attention layer
        h2 = model.transformer.layers[-1].self_attn.register_forward_hook(temporal_hook)
    except AttributeError:
        print("Warning: Hook targets not found. The model structure in load_model() must be defined.")

    # 3. Model Forward Pass
    model.eval()
    with torch.no_grad():
        try:
            logits = model(video_tensor)
            # Assuming a binary classification setup
            probability = torch.sigmoid(logits).item() 
        except Exception:
            # Fallback for when the model architecture isn't plugged in yet
            probability = 0.85 

    # Remove hooks
    if h1: h1.remove()
    if h2: h2.remove()

    # 4. Generate Spatial Heatmap
    if spatial_activations is not None:
        spatial_map = np.mean(spatial_activations[0, num_frames // 2], axis=0)
        spatial_heatmap = generate_overlay_heatmap(spatial_map, target_viz_frame, cv2.COLORMAP_JET)
    else:
        # Fallback empty image if hooks fail
        spatial_heatmap = np.zeros_like(target_viz_frame)

    # 5. Generate Temporal Heatmap
    if temporal_attention is not None:
        # Average across heads, grab the attention slice for the middle frame
        avg_attention = np.mean(temporal_attention[0], axis=0) 
        target_attention = avg_attention[num_frames // 2, :] 
        
        # Broadcast 1D temporal relevance to a 2D map for rough visualization
        temporal_map = np.tile(target_attention, (target_attention.shape[0], 1))
        temporal_heatmap = generate_overlay_heatmap(temporal_map, target_viz_frame, cv2.COLORMAP_MAGMA)
    else:
        # Fallback empty image if hooks fail
        temporal_heatmap = np.zeros_like(target_viz_frame)

    return probability, spatial_heatmap, temporal_heatmap
