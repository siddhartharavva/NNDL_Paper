import os
import tifffile
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from tqdm import tqdm
import torch
import torch.nn.functional as F

base_dir = "/Users/siddhartharavva/Desktop/SEM 5 Local/NNDL_Paper"
f0 = tifffile.imread(os.path.join(base_dir, "Validation", "F0.tif")).astype(np.float32)
f0 /= f0.max() + 1e-12

denoised_files = ["denoised_F1.tif", "denoised_F2.tif", "denoised_F3.tif"]

for fn in denoised_files:
    path = os.path.join(base_dir, fn)
    if not os.path.exists(path):
        print(f"{fn} not found, skipping...")
        continue

    d = tifffile.imread(path).astype(np.float32)
    d /= d.max() + 1e-12

    # Resize denoised stack to F0 shape
    d_resized = []
    for frame in tqdm(d, desc=f"Resizing frames for {fn}"):
        frame_resized = F.interpolate(torch.from_numpy(frame[None,None]).float(),
                                      size=f0.shape[1:], mode='bilinear', align_corners=False)[0,0].numpy()
        d_resized.append(frame_resized)
    d_resized = np.array(d_resized)

    m = min(d_resized.shape[0], f0.shape[0])
    p = psnr(f0[:m], d_resized[:m], data_range=1.0)
    s = ssim(f0[:m], d_resized[:m], data_range=1.0, channel_axis=None)

    print(f"{fn}: PSNR={p:.2f}, SSIM={s:.4f}")
