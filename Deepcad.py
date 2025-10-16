import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tifffile
from tqdm import tqdm
import numpy as np
import os

# -------------------------
# Dataset (Memory-Efficient Version)
# -------------------------
class DeepCADDataset(Dataset):
    """
    This Dataset reads patches directly from the TIFF file on-the-fly,
    avoiding loading the entire large image into RAM during training.
    """
    def __init__(self, noisy_path, patch_size=32, length=1000):
        if not os.path.exists(noisy_path):
            raise FileNotFoundError(f"The file was not found at: {noisy_path}")
            
        self.tif_path = noisy_path
        with tifffile.TiffFile(self.tif_path) as tif:
            shape = tif.series[0].shape
        
        # Determine the shape, assuming (Z, Y, X) or (C, Z, Y, X)
        if len(shape) == 3:
            self.data_shape = (1,) + shape  # (C, Z, Y, X)
        elif len(shape) == 4:
            self.data_shape = shape
        else:
            raise ValueError(f"Expected 3D or 4D data, but got {len(shape)}D data")

        self.patch_size = patch_size
        self.len = length

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        c, z, y, x = self.data_shape
        ps = self.patch_size

        # Defensive handling when image is smaller than requested patch
        if z < ps or y < ps or x < ps:
            z0, y0, x0 = 0, 0, 0
            z1 = min(ps, z)
            y1 = min(ps, y)
            x1 = min(ps, x)
        else:
            z0 = np.random.randint(0, z - ps + 1)
            y0 = np.random.randint(0, y - ps + 1)
            x0 = np.random.randint(0, x - ps + 1)
            z1, y1, x1 = z0 + ps, y0 + ps, x0 + ps

        # Read patch efficiently
        with tifffile.TiffFile(self.tif_path) as tif:
            arr = tif.asarray(key=slice(z0, z1))
            if arr.ndim == 3:
                patch_numpy = arr[:, y0:y1, x0:x1]
            elif arr.ndim == 4:
                patch_numpy = arr[:, :, y0:y1, x0:x1]
            else:
                raise ValueError(f"Unexpected TIFF array shape: {arr.shape}")

        if patch_numpy.ndim == 3:
            patch = torch.tensor(patch_numpy[None, ...], dtype=torch.float32)
        else:
            patch = torch.tensor(patch_numpy, dtype=torch.float32)

        mask = (torch.rand_like(patch) > 0.9).float()
        noisy_input = patch * (1 - mask)

        return noisy_input, patch, mask


# -------------------------
# Model (3D UNet)
# -------------------------
class UNet3D(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv3d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.final = nn.Conv3d(16, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        b = self.bottleneck(p2)

        u2 = self.up2(b)
        if e2.shape[2:] != u2.shape[2:]:
            e2 = self.center_crop(e2, u2.shape[2:])
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        if e1.shape[2:] != u1.shape[2:]:
            e1 = self.center_crop(e1, u1.shape[2:])
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        return self.final(d1)

    def center_crop(self, layer, target_size):
        _, _, z, y, x = layer.size()
        _, _, tz, ty, tx = target_size
        z1 = (z - tz) // 2
        y1 = (y - ty) // 2
        x1 = (x - tx) // 2
        return layer[:, :, z1:(z1 + tz), y1:(y1 + ty), x1:(x1 + tx)]


# -------------------------
# Self-supervised Training and Inference
# -------------------------
def run_unsupervised_deepcad(noisy_path, output_path):
    device = torch.device("cpu")
    dataset = DeepCADDataset(noisy_path, patch_size=32)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

    model = UNet3D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    try:
        model = torch.compile(model)
    except Exception:
        pass  # skip on MPS if unsupported

    epochs = 1
    print("--- Starting Training ---")
    for epoch in range(epochs):
        model.train()
        loop = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{epochs}]")
        for noisy_input, target, mask in loop:
            noisy_input, target, mask = noisy_input.to(device), target.to(device), mask.to(device)
            optimizer.zero_grad()
            output = model(noisy_input)
            loss = criterion(output * mask, target * mask)
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())
    print("--- Training Finished ---")

    # --- Inference ---
    print("\n--- Starting Inference ---")
    model.eval()
    with torch.no_grad():
        # Faster file loading (use npy if available)
        if noisy_path.endswith(".npy"):
            full_data_numpy = np.load(noisy_path)
        else:
            full_data_numpy = tifffile.imread(noisy_path)

        full_data = torch.tensor(full_data_numpy[None, None, ...], dtype=torch.float16 if device != torch.device("cpu") else torch.float32)
        _, _, z, y, x = full_data.shape
        original_dims = (z, y, x)

        pad_z = (4 - (z % 4)) % 4
        pad_y = (4 - (y % 4)) % 4
        pad_x = (4 - (x % 4)) % 4

        padded_data = nn.functional.pad(full_data, (0, pad_x, 0, pad_y, 0, pad_z))
        denoised_padded = torch.zeros_like(padded_data)

        chunk_z, chunk_y, chunk_x = 16, 64, 64
        for z0 in tqdm(range(0, padded_data.shape[2], chunk_z), desc="Denoising Chunks"):
            for y0 in range(0, padded_data.shape[3], chunk_y):
                for x0 in range(0, padded_data.shape[4], chunk_x):
                    z1 = min(z0 + chunk_z, padded_data.shape[2])
                    y1 = min(y0 + chunk_y, padded_data.shape[3])
                    x1 = min(x0 + chunk_x, padded_data.shape[4])

                    chunk = padded_data[:, :, z0:z1, y0:y1, x0:x1].to(device)
                    denoised_chunk = model(chunk).cpu().to(torch.float32)
                    denoised_padded[:, :, z0:z1, y0:y1, x0:x1] = denoised_chunk

        denoised = denoised_padded[:, :, :original_dims[0], :original_dims[1], :original_dims[2]]
        denoised_numpy = denoised.squeeze().numpy()

    tifffile.imwrite(output_path, denoised_numpy)
    print(f"\n--- Inference Finished ---")
    print(f"Denoised image saved to {output_path}")


if __name__ == "__main__":
    input_file = "Train/A1.tif"
    output_file = "denoised_output.tif"
    
    try:
        run_unsupervised_deepcad(input_file, output_file)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Please ensure the file path is correct and the file exists.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")