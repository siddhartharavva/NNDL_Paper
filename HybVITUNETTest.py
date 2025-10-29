# ======================================================
# Standalone Hybrid ViT + U-Net Testing on MPS
# ======================================================

import os, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tifffile
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import matplotlib.pyplot as plt

# ---------------- Device ----------------
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"🖥 Testing on {DEVICE}")

# ---------------- Positional Encoding ----------------
def sinusoidal_positional_encoding(n_pos, dim):
    position = np.arange(n_pos)[:, None]
    div_term = np.exp(np.arange(0, dim, 2) * -(np.log(10000.0) / dim))
    pe = np.zeros((n_pos, dim), dtype=np.float32)
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe

# ---------------- ViT Block ----------------
class ViTBlock(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0])
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x

# ---------------- Center Crop ----------------
def center_crop(layer, target_spatial):
    _, _, h, w = layer.shape
    target_h, target_w = target_spatial
    diff_y = (h - target_h) // 2
    diff_x = (w - target_w) // 2
    return layer[:, :, diff_y:diff_y + target_h, diff_x:diff_x + target_w]

# ---------------- ViT + U-Net ----------------
class ViT_UNet(nn.Module):
    def __init__(self, in_channels=5, out_channels=1, base_filters=64, embed_dim=256,
                 num_heads=8, num_layers=4):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, 3, padding=1),
            nn.BatchNorm2d(base_filters), nn.ReLU(True),
            nn.Conv2d(base_filters, base_filters, 3, padding=1),
            nn.BatchNorm2d(base_filters), nn.ReLU(True)
        )
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters*2, 3, padding=1),
            nn.BatchNorm2d(base_filters*2), nn.ReLU(True),
            nn.Conv2d(base_filters*2, base_filters*2, 3, padding=1),
            nn.BatchNorm2d(base_filters*2), nn.ReLU(True)
        )
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(base_filters*2, embed_dim, 3, padding=1),
            nn.BatchNorm2d(embed_dim), nn.ReLU(True)
        )
        self.transformers = nn.ModuleList([ViTBlock(embed_dim=embed_dim, num_heads=num_heads) for _ in range(num_layers)])
        self.up2 = nn.ConvTranspose2d(embed_dim, base_filters*2, 4, 2, 1)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_filters*4, base_filters*2, 3, padding=1),
            nn.BatchNorm2d(base_filters*2), nn.ReLU(True),
            nn.Conv2d(base_filters*2, base_filters*2, 3, padding=1),
            nn.BatchNorm2d(base_filters*2), nn.ReLU(True)
        )
        self.up1 = nn.ConvTranspose2d(base_filters*2, base_filters, 4, 2, 1)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_filters*2, base_filters, 3, padding=1),
            nn.BatchNorm2d(base_filters), nn.ReLU(True),
            nn.Conv2d(base_filters, base_filters, 3, padding=1),
            nn.BatchNorm2d(base_filters), nn.ReLU(True)
        )
        self.final = nn.Conv2d(base_filters, out_channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        b = self.bottleneck_conv(p2)
        _, E, h, w = b.shape
        tokens = b.flatten(2).transpose(1, 2)
        pe = torch.from_numpy(sinusoidal_positional_encoding(tokens.shape[1], E)).to(tokens.device).unsqueeze(0)
        tokens = tokens + pe
        for blk in self.transformers:
            tokens = blk(tokens)
        b = tokens.transpose(1, 2).contiguous().view(B, E, h, w)
        d2 = self.up2(b)
        e2c = center_crop(e2, d2.shape[2:])
        d2 = torch.cat([d2, e2c], 1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        e1c = center_crop(e1, d1.shape[2:])
        d1 = torch.cat([d1, e1c], 1)
        d1 = self.dec1(d1)
        out = self.final(d1)
        if out.shape[-2:] != (H, W):
            out = F.interpolate(out, (H, W), mode='bilinear', align_corners=False)
        return out

# ---------------- Denoise Function with Downsampling ----------------
def denoise_stack(model, stack, window=5, device="mps", downscale=0.5):
    model.eval()
    stack = stack.astype(np.float32)
    stack /= (stack.max() + 1e-12)
    if downscale != 1.0:
        H, W = stack.shape[1], stack.shape[2]
        stack = np.array([F.interpolate(torch.from_numpy(f[None, None].astype(np.float32)), 
                                        scale_factor=downscale, mode='bilinear', align_corners=False)[0,0].numpy() 
                          for f in stack])
    F_, H, W = stack.shape
    half = window // 2
    out_stack = np.zeros_like(stack)
    for t in range(half, F_ - half):
        inp = stack[t - half:t + half + 1]
        inp_tensor = torch.from_numpy(inp.astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(inp_tensor)
            out = out.cpu().numpy()[0, 0]
            if downscale != 1.0:
                out = F.interpolate(torch.from_numpy(out[None, None]), size=(H, W), mode='bilinear', align_corners=False)[0,0].numpy()
        out_stack[t] = out
    out_stack[:half] = stack[:half]; out_stack[-half:] = stack[-half:]
    return out_stack

# ---------------- Paths ----------------
BASE_DIR = os.path.expanduser("/Users/siddhartharavva/Desktop/SEM 5 Local/NNDL_Paper")
VAL_DIR = os.path.join(BASE_DIR, "Validation")
MODEL_PATH = "vit_unet_denoiser.pth"

# ---------------- Load Model ----------------
model = ViT_UNet(in_channels=5, out_channels=1).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
print("✅ Model loaded")

# ---------------- Testing ----------------
f0 = tifffile.imread(os.path.join(VAL_DIR, "F0.tif")).astype(np.float32)
f0 /= f0.max() + 1e-12
results = {"F0": f0}

for fn in ["F1.tif", "F2.tif", "F3.tif"]:
    path = os.path.join(VAL_DIR, fn)
    if not os.path.exists(path): continue
    print(f"🧪 Denoising {fn} ...")
    stack = tifffile.imread(path)
    den = denoise_stack(model, stack, device=DEVICE, downscale=0.5)  # downscale 50% for MPS
    tifffile.imwrite(f"denoised_{fn}", den.astype(np.float32))
    results[fn] = den
    print(f"✅ Saved denoised_{fn}")

# ---------------- Quantitative Results ----------------
print("\n📈 Quantitative Results vs F0:")
for fn in ["F1.tif", "F2.tif", "F3.tif"]:
    if fn not in results: continue
    d = results[fn]; c = results["F0"]
    m = min(d.shape[0], c.shape[0])
    p = psnr(c[:m], d[:m], data_range=1.0)
    s = ssim(c[:m], d[:m], data_range=1.0, channel_axis=None)
    print(f"{fn}: PSNR={p:.2f}, SSIM={s:.4f}")

# ---------------- Plot Example ----------------
idx = f0.shape[0] // 2
plt.figure(figsize=(12,6))
plt.subplot(1,4,1); plt.imshow(f0[idx], cmap='gray'); plt.title("F0 (Clean)")
for i, fn in enumerate(["F1.tif", "F2.tif", "F3.tif"], 2):
    if fn in results:
        plt.subplot(1,4,i); plt.imshow(results[fn][idx], cmap='gray'); plt.title(fn)
plt.tight_layout(); plt.show()

print("✅ Testing complete. Results saved to current directory.")
