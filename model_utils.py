import torch
import torch.nn as nn
import timm
import cv2
from torchvision import transforms

# ==========================================
# 1. THE EXACT TRAINED ARCHITECTURE
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
    # CRITICAL: num_frames must be 4 to match your weights
    def __init__(self, num_frames=4, embed_dim=728, num_heads=8, depth=4):
        super().__init__()
        self.backbone = timm.create_model('xception', pretrained=False) # Pretrained=False because we are loading our own weights
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
        x = features + self.temp_embed + self.pos_embed
        
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        x = x.mean(dim=[1, 2])
        return self.head(x)

# ==========================================
# 2. INFERENCE PIPELINE
# ==========================================
def load_model(weight_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ISTVT(num_frames=4)
    # Load the weights into the CPU/Local GPU safely
    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval() # Set to evaluation mode (turns off dropout/batchnorm updates)
    return model, device

def extract_and_transform_video(video_path, sequence_length=4):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((299, 299)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
    ])
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // sequence_length)
    
    for i in range(sequence_length):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(transform(frame))
        else:
            frames.append(torch.zeros(3, 299, 299))
    cap.release()
    
    # Add batch dimension: Shape becomes (1, 4, 3, 299, 299)
    return torch.stack(frames).unsqueeze(0) 

def predict_video(model, device, video_path):
    video_tensor = extract_and_transform_video(video_path).to(device)
    
    with torch.no_grad(): # No gradients needed for inference
        raw_output = model(video_tensor)
        # Apply sigmoid to squash the raw output into a 0.0 to 1.0 probability
        probability = torch.sigmoid(raw_output).item() 
        
    return probability