import torch
import torch.nn as nn
import timm
import cv2
import numpy as np
from torchvision import transforms

# ==========================================
# 1. ARCHITECTURE DEFINITIONS
# ==========================================
class SelfSubtract(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        baseline = x[:, 0:1, :, :] 
        residuals = x[:, 1:, :, :] - x[:, :-1, :, :]
        return torch.cat([baseline, residuals], dim=1)

class DecomposedAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.spatial_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.temporal_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.self_subtract = SelfSubtract()
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x):
        B, T, S, C = x.shape
        x_temp = self.self_subtract(x)
        x_temp = self.norm1(x_temp)
        x_temp = x_temp.permute(0, 2, 1, 3).reshape(B * S, T, C)
        temp_out, _ = self.temporal_attn(x_temp, x_temp, x_temp)
        temp_out = temp_out.reshape(B, S, T, C).permute(0, 2, 1, 3)
        x = x + temp_out 
        
        x_spat = self.norm1(x)
        x_spat = x_spat.reshape(B * T, S, C)
        spat_out, _ = self.spatial_attn(x_spat, x_spat, x_spat)
        spat_out = spat_out.reshape(B, T, S, C)
        x = x + spat_out 
        
        x = x + self.mlp(self.norm2(x))
        return x

class ISTVT(nn.Module):
    def __init__(self, num_frames=4, embed_dim=728, num_heads=8, depth=4):
        super().__init__()
        self.backbone = timm.create_model('xception', pretrained=False) 
        self.proj = nn.Conv2d(2048, embed_dim, kernel_size=1)
        
        self.temp_embed = nn.Parameter(torch.zeros(1, num_frames, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, 100, embed_dim)) 
        
        self.blocks = nn.ModuleList([DecomposedAttentionBlock(dim=embed_dim, num_heads=num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        
        features = self.backbone.forward_features(x)
        features = self.proj(features) 
        
        _, C_new, H_new, W_new = features.shape
        S = H_new * W_new 
        
        features = features.view(B, T, C_new, S).permute(0, 1, 3, 2)
        
        # Broadcast across the batch dimension (B) seamlessly
        x = features + self.temp_embed + self.pos_embed
        
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        x = x.mean(dim=[1, 2])
        return self.head(x)

# ==========================================
# 2. VISUALIZATION UTILS
# ==========================================
def generate_overlay_heatmap(relevance_map, original_frame, probability, colormap=cv2.COLORMAP_JET):
    # Silence the heatmap for authentic videos to prevent false red highlights
    if probability < 0.5:
        return original_frame

    relevance_map = relevance_map - np.min(relevance_map)
    relevance_map = relevance_map / (np.max(relevance_map) + 1e-8)
    relevance_map = np.uint8(255 * relevance_map)
    
    h, w = original_frame.shape[:2]
    relevance_map_resized = cv2.resize(relevance_map, (w, h), interpolation=cv2.INTER_CUBIC)
    
    heatmap = cv2.applyColorMap(relevance_map_resized, colormap)
    overlay = cv2.addWeighted(heatmap, 0.5, original_frame, 0.5, 0)
    return overlay

# ==========================================
# 3. AUTONOMOUS MINI-BATCH PIPELINE
# ==========================================
def load_model(weight_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ISTVT(num_frames=4)
    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval() 
    return model, device

def predict_video(model, device, video_path, max_frames=520):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((299, 299)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
    ])
    
    cap = cv2.VideoCapture(video_path)
    all_frames = []
    all_viz_frames = []
    
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Auto-scaling logic
    if total_video_frames < max_frames:
        total_frames_to_sample = total_video_frames
    else:
        total_frames_to_sample = max_frames
        
    total_frames_to_sample = max(4, (total_frames_to_sample // 4) * 4)
    step = max(1, total_video_frames // total_frames_to_sample)
    
    for i in range(total_frames_to_sample):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            all_frames.append(transform(frame_rgb))
            all_viz_frames.append(cv2.resize(frame_rgb, (299, 299)))
        else:
            all_frames.append(torch.zeros(3, 299, 299))
            all_viz_frames.append(np.zeros((299, 299, 3), dtype=np.uint8))
            
    cap.release()

    # Pack frames into 4-frame windows
    num_windows = total_frames_to_sample // 4
    window_tensors = []
    
    for w in range(num_windows):
        idx = w * 4
        window_tensors.append(torch.stack(all_frames[idx:idx+4]))

    spatial_activations_list = []
    def spatial_hook(module, input, output):
        spatial_activations_list.append(output.detach().cpu().numpy())

    h1 = model.proj.register_forward_hook(spatial_hook)
    all_probabilities = []

    # Memory-safe mini-batching (processes 4 windows / 16 frames at a time)
    CHUNK_SIZE = 4 
    
    with torch.no_grad():
        for i in range(0, num_windows, CHUNK_SIZE):
            batch_chunk = torch.stack(window_tensors[i:i+CHUNK_SIZE]).to(device)
            raw_outputs = model(batch_chunk)
            chunk_probs = torch.sigmoid(raw_outputs).cpu().numpy().flatten()
            all_probabilities.extend(chunk_probs)

    h1.remove()

    probabilities = np.array(all_probabilities)
    if len(spatial_activations_list) > 0:
        spatial_activations = np.concatenate(spatial_activations_list, axis=0)
    else:
        spatial_activations = None

    # Extract peak anomaly
    max_idx = int(np.argmax(probabilities))
    max_probability = float(probabilities[max_idx])
    
    peak_viz_window = all_viz_frames[max_idx * 4 : (max_idx * 4) + 4]
    target_viz_frame = peak_viz_window[2] 

    if spatial_activations is not None:
        global_frame_index = (max_idx * 4) + 2
        spatial_map = np.mean(spatial_activations[global_frame_index], axis=0)
        best_spatial_heatmap = generate_overlay_heatmap(spatial_map, target_viz_frame, max_probability)
    else:
        best_spatial_heatmap = target_viz_frame

    return max_probability, best_spatial_heatmap
