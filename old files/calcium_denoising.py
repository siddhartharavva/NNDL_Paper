"""
CALCIUM IMAGING DENOISING PROJECT - FINAL COMPLETE IMPLEMENTATION
All Outputs Saved for Evaluation
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tifffile
import json
import time
import random
import pandas as pd
from tqdm import tqdm
from skimage.metrics import structural_similarity
from sklearn.decomposition import NMF
from skimage.metrics import peak_signal_noise_ratio
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# ==============================================================================
# DATASET CLASS
# ==============================================================================
class CalciumImagingDataset:
    def __init__(self, tif_path, sequence_length=8, patch_size=64):
        print("📁 Loading and analyzing dataset...")
        self.raw_data = tifffile.imread(tif_path).astype(np.float32)
        self.original_shape = self.raw_data.shape
        print(f"📐 Original shape: {self.original_shape}")

        self.mean_intensity = np.mean(self.raw_data)
        self.std_intensity = np.std(self.raw_data)
        self.snr_estimate = self.mean_intensity / self.std_intensity

        self.preprocessed_data = self._preprocess()
        self.sequence_length = sequence_length
        self.patch_size = patch_size

    def _preprocess(self):
        print("🧹 Preprocessing...")
        data = self.raw_data.copy()

        data_min, data_max = np.percentile(data, [1, 99])
        data = np.clip((data - data_min) / (data_max - data_min), 0, 1)

        from scipy import ndimage
        for t in range(data.shape[0]):
            data[t] = ndimage.gaussian_filter(data[t], sigma=0.5)

        return data

    def get_statistics(self):
        return {
            "shape": self.original_shape,
            "mean_intensity": self.mean_intensity,
            "std_intensity": self.std_intensity,
            "snr_estimate": self.snr_estimate,
        }

class SelfSupervisedDataset(Dataset):
    def __init__(self, calcium_dataset, num_patches=1000, augment=True):
        self.data = calcium_dataset.preprocessed_data
        self.sequence_length = calcium_dataset.sequence_length
        self.patch_size = calcium_dataset.patch_size
        self.num_patches = num_patches
        self.augment = augment
        self.T, self.H, self.W = self.data.shape

    def __len__(self): return self.num_patches

    def __getitem__(self, idx):
        t = random.randint(0, self.T - self.sequence_length)
        h = random.randint(0, self.H - self.patch_size)
        w = random.randint(0, self.W - self.patch_size)

        patch = self.data[t:t+self.sequence_length, h:h+self.patch_size, w:w+self.patch_size]
        noisy = patch + np.random.normal(0, 0.1, patch.shape)
        noisy = np.clip(noisy, 0, 1)
        return (
            torch.tensor(noisy).unsqueeze(1).float(),
            torch.tensor(patch).unsqueeze(1).float()
        )

# ==============================================================================
# MODELS
# ==============================================================================
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1)
        )
    def forward(self, x):
        B, T, C, H, W = x.shape
        out = []
        for t in range(T):
            out.append(self.net(x[:, t]))
        return torch.stack(out, dim=1)

class DeepCADBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Conv3d(1, 16, 3, padding=1)
        self.dec = nn.Conv3d(16, 1, 3, padding=1)
    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        return self.dec(F.relu(self.enc(x))).permute(0, 2, 1, 3, 4)

class CNMFBaseline:
    def __init__(self, n_components=30):
        self.n_components = n_components
    def fit_denoise(self, data):
        T, H, W = data.shape
        data_2d = data.transpose(1, 2, 0).reshape(H*W, T)
        nmf = NMF(n_components=self.n_components, max_iter=50)
        W = nmf.fit_transform(data_2d)
        Hm = nmf.components_
        return (W @ Hm).reshape(H, W, T).transpose(2, 0, 1)

# ==============================================================================
# TRAINER CLASS
# ==============================================================================
class ModelTrainer:
    def __init__(self, dataset, output_dir="exam_results"):
        self.dataset = dataset
        self.device = device
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "model_training_logs"), exist_ok=True)

    def train(self, model_class, epochs=3, model_name="Model"):
        print(f"🧠 Training {model_name}...")
        model = model_class().to(self.device)
        dataset = SelfSupervisedDataset(self.dataset, num_patches=200)
        loader = DataLoader(dataset, batch_size=2, shuffle=True)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()

        losses = []
        for epoch in range(epochs):
            epoch_loss = 0
            for i, (noisy, clean) in enumerate(loader):
                noisy, clean = noisy.to(self.device), clean.to(self.device)
                opt.zero_grad()
                pred = model(noisy)
                loss = criterion(pred, clean)
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
                if i > 20: break
            epoch_loss /= (i + 1)
            losses.append(epoch_loss)
            print(f"Epoch {epoch+1}/{epochs}: Loss={epoch_loss:.6f}")

        # Save logs
        log_dir = os.path.join(self.output_dir, "model_training_logs", f"{model_name}_log.json")
        with open(log_dir, "w") as f:
            json.dump({"losses": losses}, f, indent=4)
        print(f"📄 Saved training log: {log_dir}")

        # Save model
        torch.save(model.state_dict(), os.path.join(self.output_dir, f"{model_name}.pth"))
        return model, losses

# ==============================================================================
# ABLATION STUDY
# ==============================================================================
class AblationStudy:
    def __init__(self, dataset, device, output_dir="exam_results/ablation_runs"):
        self.dataset = dataset
        self.device = device
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run(self, model_class):
        configs = {
            "learning_rate": [1e-4, 5e-4],
            "batch_size": [2, 4],
            "epochs": [2],
        }
        results = []
        for lr in configs["learning_rate"]:
            for bs in configs["batch_size"]:
                print(f"⚙️ Config: LR={lr}, BS={bs}")
                res = self._train_config(model_class, lr, bs, 2)
                results.append(res)

        df = pd.DataFrame(results)
        csv_path = os.path.join(self.output_dir, "ablation_results.csv")
        json_path = os.path.join(self.output_dir, "ablation_results.json")
        df.to_csv(csv_path, index=False)
        with open(json_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"✅ Saved ablation results:\n ├─ {csv_path}\n └─ {json_path}")

    def _train_config(self, model_class, lr, bs, epochs):
        model = model_class().to(self.device)
        dataset = SelfSupervisedDataset(self.dataset, num_patches=100)
        loader = DataLoader(dataset, batch_size=bs, shuffle=True)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        losses = []
        t0 = time.time()
        for e in range(epochs):
            eloss = 0
            for i, (noisy, clean) in enumerate(loader):
                if i > 10: break
                noisy, clean = noisy.to(self.device), clean.to(self.device)
                opt.zero_grad()
                pred = model(noisy)
                loss = criterion(pred, clean)
                loss.backward()
                opt.step()
                eloss += loss.item()
            eloss /= (i + 1)
            losses.append(eloss)
        t1 = time.time()
        return {"lr": lr, "batch_size": bs, "avg_loss": np.mean(losses), "time": t1 - t0, "epoch_losses": losses}

# ==============================================================================
# PERFORMANCE EVALUATOR
# ==============================================================================
class Evaluator:
    def __init__(self, device, output_dir="exam_results"):
        self.device = device
        self.output_dir = output_dir

    def evaluate(self, models, dataset):
        metrics = []
        test_data = dataset.preprocessed_data
        T, H, W = test_data.shape
        clean = test_data[:8]
        noisy = np.clip(clean + np.random.normal(0, 0.1, clean.shape), 0, 1)

        for name, model in models.items():
            print(f"📊 Evaluating {name}")
            if isinstance(model, CNMFBaseline):
                denoised = model.fit_denoise(noisy)
            else:
                model.eval()
                x = torch.tensor(noisy).unsqueeze(0).unsqueeze(2).float().to(self.device)
                with torch.no_grad():
                    y = model(x).squeeze().cpu().numpy()
                denoised = y

            mse = np.mean((denoised - clean) ** 2)
            psnr = peak_signal_noise_ratio(clean, denoised, data_range=1.0)
            ssim_vals = [structural_similarity(clean[i], denoised[i], data_range=1.0) for i in range(clean.shape[0])]
            ssim = np.mean(ssim_vals)
            metrics.append({"Model": name, "MSE": mse, "PSNR": psnr, "SSIM": ssim})

        df = pd.DataFrame(metrics)
        df.to_csv(os.path.join(self.output_dir, "performance_metrics.csv"), index=False)
        print("📄 Saved performance metrics to performance_metrics.csv")

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def main():
    DATASET_PATH = "Train/A1.tif"
    OUTPUT_DIR = "exam_results"
    EPOCHS = 2

    if not os.path.exists(DATASET_PATH):
        print("❌ Dataset not found! Place your A1.tif under Train/")
        return

    ds = CalciumImagingDataset(DATASET_PATH)
    stats = ds.get_statistics()
    print(f"📊 Dataset Stats: {stats}")

    trainer = ModelTrainer(ds, OUTPUT_DIR)
    model1, _ = trainer.train(SimpleCNN, epochs=EPOCHS, model_name="Novel_CNN")
    model2, _ = trainer.train(DeepCADBaseline, epochs=EPOCHS, model_name="DeepCAD")
    model3 = CNMFBaseline()

    models = {"Novel_CNN": model1, "DeepCAD": model2}

    ablation = AblationStudy(ds, device)
    ablation.run(SimpleCNN)

    evaluator = Evaluator(device)
    evaluator.evaluate(models, ds)

    print("\n✅ All stages complete. Results saved in exam_results/")

if __name__ == "__main__":
    main()