# ======================================================
# Hybrid ViT + U-Net Denoiser for Calcium Imaging (Full Pipeline)
# ======================================================

import os, glob, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tifffile
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

import platform
import matplotlib
if platform.system() == "Darwin":
    matplotlib.use("Agg")  # avoid MacOS backend crash

# ======================================================
# 1️⃣ Dataset Class
# ======================================================
class CalciumNoiseDataset(Dataset):
    def __init__(self, files, patch_size=64, window=5, patches_per_stack=400):
        self.files = files
        self.patch_size = patch_size
        self.window = window
        self.half = window // 2
        self.patches_per_stack = patches_per_stack

    def __len__(self):
        return len(self.files) * self.patches_per_stack

    def __getitem__(self, idx):
        si = idx // self.patches_per_stack
        A = tifffile.imread(self.files[si]).astype(np.float32)
        A = A / (A.max() + 1e-12)
        F, H, W = A.shape

        if F <= self.window or H < self.patch_size or W < self.patch_size:
            inp = np.zeros((self.window, self.patch_size, self.patch_size), dtype=np.float32)
            tgt = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
            return torch.from_numpy(inp), torch.from_numpy(tgt).unsqueeze(0)

        t = random.randint(self.half, F - self.half - 1)
        y = random.randint(0, H - self.patch_size)
        x = random.randint(0, W - self.patch_size)
        inp = A[t - self.half:t + self.half + 1, y:y + self.patch_size, x:x + self.patch_size]
        tgt = A[t, y:y + self.patch_size, x:x + self.patch_size]

        if random.random() < 0.5:
            inp = inp[:, :, ::-1].copy(); tgt = tgt[:, ::-1].copy()
        if random.random() < 0.5:
            inp = inp[:, ::-1, :].copy(); tgt = tgt[::-1, :].copy()

        return torch.from_numpy(inp).float(), torch.from_numpy(tgt).unsqueeze(0).float()

# ======================================================
# 2️⃣ Model Definition: Hybrid ViT + U-Net
# ======================================================
def center_crop(layer, target_spatial):
    _, _, h, w = layer.shape
    target_h, target_w = target_spatial
    diff_y = (h - target_h) // 2
    diff_x = (w - target_w) // 2
    return layer[:, :, diff_y:diff_y + target_h, diff_x:diff_x + target_w]

def sinusoidal_positional_encoding(n_pos, dim):
    position = np.arange(n_pos)[:, None]
    div_term = np.exp(np.arange(0, dim, 2) * -(np.log(10000.0) / dim))
    pe = np.zeros((n_pos, dim), dtype=np.float32)
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe

class ViTBlock(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super(ViTBlock, self).__init__()
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

class ViT_UNet(nn.Module):
    def __init__(self, in_channels=5, out_channels=1, base_filters=64, embed_dim=256,
                 num_heads=8, num_layers=4):
        super(ViT_UNet, self).__init__()
        # Encoder
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
        # Transformer blocks
        self.transformers = nn.ModuleList([
            ViTBlock(embed_dim=embed_dim, num_heads=num_heads, dropout=0.1)
            for _ in range(num_layers)
        ])
        # Decoder
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
        n = tokens.shape[1]
        pe = torch.from_numpy(sinusoidal_positional_encoding(n, E)).to(tokens.device).unsqueeze(0)
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

# ======================================================
# 3️⃣ Training Setup (Train dir only)
# ======================================================
base_dir = os.path.expanduser("/Users/siddhartharavva/Desktop/SEM 5 Local/NNDL_Paper")
train_dir = os.path.join(base_dir, "Train")
train_files = sorted(glob.glob(os.path.join(train_dir, "*.tif")))

train_dataset = CalciumNoiseDataset(train_files, patch_size=64, window=5, patches_per_stack=400)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"✅ Training on {DEVICE}")

model = ViT_UNet(in_channels=5, out_channels=1).to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

EPOCHS = 30
train_losses = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0
    for inp, tgt in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False):
        inp, tgt = inp.to(DEVICE), tgt.to(DEVICE)
        optimizer.zero_grad()
        out = model(inp)
        if out.shape[-2:] != tgt.shape[-2:]:
            tgt = F.interpolate(tgt, out.shape[-2:], mode='bilinear', align_corners=False)
        loss = criterion(out, tgt)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    scheduler.step()
    print(f"Epoch {epoch}/{EPOCHS} | Train Loss: {epoch_loss:.6f}")

torch.save(model.state_dict(), "vit_unet_denoiser.pth")
print("✅ Model saved as vit_unet_denoiser.pth")

plt.plot(train_losses); plt.title("Training Loss"); plt.show()

# ======================================================
# 4️⃣ Validation: F0–F3 Evaluation
# ======================================================
def denoise_stack(model, stack, window=5, device="cuda"):
    model.eval()
    stack = stack.astype(np.float32)
    stack /= (stack.max() + 1e-12)
    F_, H, W = stack.shape
    half = window // 2
    out_stack = np.zeros_like(stack)
    for t in range(half, F_ - half):
        inp = stack[t - half:t + half + 1]
        inp_tensor = torch.from_numpy(inp).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(inp_tensor)
            if out.shape[-2:] != (H, W):
                out = F.interpolate(out, (H, W), mode='bilinear', align_corners=False)
            out = out.cpu().numpy()[0, 0]
        out_stack[t] = out
    out_stack[:half] = stack[:half]; out_stack[-half:] = stack[-half:]
    return out_stack

input_folder = os.path.join(base_dir, "Validation")
f0 = tifffile.imread(os.path.join(input_folder, "F0.tif")).astype(np.float32)
f0 /= f0.max() + 1e-12
results = {"F0": f0}

model.eval()
for fn in ["F1.tif", "F2.tif", "F3.tif"]:
    path = os.path.join(input_folder, fn)
    if not os.path.exists(path): continue
    print(f"🧪 Denoising {fn} ...")
    stack = tifffile.imread(path)
    den = denoise_stack(model, stack, device=DEVICE)
    tifffile.imwrite(f"denoised_{fn}", den.astype(np.float32))
    results[fn] = den
    print(f"✅ Saved denoised_{fn}")

print("\n📈 Quantitative Results vs F0:")
for fn in ["F1.tif", "F2.tif", "F3.tif"]:
    if fn not in results: continue
    d = results[fn]; c = results["F0"]
    m = min(d.shape[0], c.shape[0])
    p = psnr(c[:m], d[:m], data_range=1.0)
    s = ssim(c[:m], d[:m], data_range=1.0, channel_axis=None)
    print(f"{fn}: PSNR={p:.2f}, SSIM={s:.4f}")

idx = f0.shape[0] // 2
plt.figure(figsize=(12,6))
plt.subplot(1,4,1); plt.imshow(f0[idx], cmap='gray'); plt.title("F0 (Clean)")
for i, fn in enumerate(["F1.tif", "F2.tif", "F3.tif"], 2):
    if fn in results:
        plt.subplot(1,4,i); plt.imshow(results[fn][idx], cmap='gray'); plt.title(fn)
plt.tight_layout(); plt.show()

print("✅ Training and validation complete. Results saved to current directory.")