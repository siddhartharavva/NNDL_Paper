import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tifffile
from tqdm import tqdm
import pandas as pd
import time
import random
import warnings
warnings.filterwarnings('ignore')

# ===========================
# 0. Setup & Reproducibility
# ===========================
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# ============================
# 1. Data Preprocessing
# ============================

class CalciumDataPreprocessor:
    @staticmethod
    def load_and_preprocess(file_path, normalize=True):
        print(f"Loading: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        data = tifffile.imread(file_path).astype(np.float32)
        if normalize:
            p1, p99 = np.percentile(data, [1, 99])
            data = np.clip(data, p1, p99)
            data = (data - p1) / (p99 - p1)
        return data

class SelfSupervisedDataset(Dataset):
    def __init__(self, data, sequence_length=8, patch_size=64, num_patches=500):
        self.data = data
        self.sequence_length = sequence_length
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.T, self.H, self.W = data.shape

    def __len__(self):
        return self.num_patches

    def __getitem__(self, idx):
        t_start = random.randint(0, max(0, self.T - self.sequence_length))
        h_start = random.randint(0, max(0, self.H - self.patch_size))
        w_start = random.randint(0, max(0, self.W - self.patch_size))
        patch = self.data[t_start:t_start+self.sequence_length,
                         h_start:h_start+self.patch_size,
                         w_start:w_start+self.patch_size]
        clean_target = patch.copy()
        noisy_input = self._add_realistic_noise(patch)
        return (torch.from_numpy(noisy_input).unsqueeze(1).float(),
                torch.from_numpy(clean_target).unsqueeze(1).float())

    def _add_realistic_noise(self, patch):
        noisy = patch.copy()
        noise_level = np.random.uniform(0.05, 0.2)
        noisy += np.random.normal(0, noise_level, patch.shape)
        if random.random() < 0.5:
            scaled = patch * 50
            poisson_noisy = np.random.poisson(scaled) / 50.0
            noisy = 0.7 * noisy + 0.3 * poisson_noisy
        if random.random() < 0.3:
            corrupt_frame = random.randint(0, len(patch)-1)
            if random.random() < 0.5:
                noisy[corrupt_frame] *= 0.1
            else:
                noisy[corrupt_frame] = np.random.normal(0.5, 0.2, noisy[corrupt_frame].shape)
        return np.clip(noisy, 0, 1)

class ValidationDataset(Dataset):
    def __init__(self, noisy_data, clean_data, patch_size=64, sequence_length=8, num_patches=100):
        min_t = min(noisy_data.shape[0], clean_data.shape[0])
        min_h = min(noisy_data.shape[1], clean_data.shape[1])
        min_w = min(noisy_data.shape[2], clean_data.shape[2])
        self.noisy_data = noisy_data[:min_t, :min_h, :min_w]
        self.clean_data = clean_data[:min_t, :min_h, :min_w]
        self.patch_size = patch_size
        self.sequence_length = sequence_length
        self.num_patches = num_patches
        self.T, self.H, self.W = self.noisy_data.shape

    def __len__(self):
        return self.num_patches

    def __getitem__(self, idx):
        np.random.seed(idx + 42)
        t_start = random.randint(0, max(0, self.T - self.sequence_length))
        h_start = random.randint(0, max(0, self.H - self.patch_size))
        w_start = random.randint(0, max(0, self.W - self.patch_size))
        noisy_patch = self.noisy_data[t_start:t_start+self.sequence_length,
                                      h_start:h_start+self.patch_size,
                                      w_start:w_start+self.patch_size]
        clean_patch = self.clean_data[t_start:t_start+self.sequence_length,
                                      h_start:h_start+self.patch_size,
                                      w_start:w_start+self.patch_size]
        return (torch.from_numpy(noisy_patch).unsqueeze(1).float(),
                torch.from_numpy(clean_patch).unsqueeze(1).float())

# ============================
# 2. BasicScratchCNN Model
# ============================
class BasicScratchCNN(nn.Module):
    def __init__(self, in_channels=1, num_layers=4, num_features=32):
        super().__init__()
        layers = []
        features = num_features
        layers.append(nn.Conv2d(in_channels, features, 3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(features, features, 3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(features, in_channels, 3, padding=1))
        self.cnn = nn.Sequential(*layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x_reshaped = x.view(B*T, C, H, W)
        output = self.cnn(x_reshaped)
        output = output.view(B, T, C, H, W)
        return output

# ============================
# 3. Metrics & Evaluation
# ============================
def calculate_psnr(pred, target, data_range=1.0):
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(data_range / np.sqrt(mse))

def calculate_ssim(pred, target, data_range=1.0):
    ssim_scores = []
    for t in range(len(pred)):
        pred_frame = pred[t]
        target_frame = target[t]
        mu1 = np.mean(pred_frame)
        mu2 = np.mean(target_frame)
        var1 = np.var(pred_frame)
        var2 = np.var(target_frame)
        cov = np.mean((pred_frame - mu1) * (target_frame - mu2))
        c1, c2 = (0.01 * data_range)**2, (0.03 * data_range)**2
        ssim = ((2*mu1*mu2 + c1) * (2*cov + c2)) / ((mu1**2 + mu2**2 + c1) * (var1 + var2 + c2))
        ssim_scores.append(ssim)
    return np.mean(ssim_scores)

def calculate_temporal_consistency(sequence):
    if len(sequence) <= 1:
        return 1.0
    diffs = []
    for t in range(1, len(sequence)):
        diff = np.mean(np.abs(sequence[t] - sequence[t-1]))
        diffs.append(diff)
    consistency = 1.0 / (1.0 + np.mean(diffs))
    return consistency

def run_validation_evaluation(model, val_dataset, device, model_name="Model"):
    print(f"Evaluating {model_name}...")
    model.eval()
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    metrics = { 'psnr': [], 'ssim': [], 'mse': [], 'temporal_consistency': [], 'inference_time': [] }
    with torch.no_grad():
        for i, (noisy_batch, clean_batch) in enumerate(tqdm(val_loader, desc=f"Validating {model_name}")):
            if i >= 50:
                break
            noisy_batch, clean_batch = noisy_batch.to(device), clean_batch.to(device)
            start_time = time.time()
            denoised_batch = model(noisy_batch)
            inference_time = time.time() - start_time
            denoised_np = denoised_batch.cpu().numpy().squeeze()
            clean_np = clean_batch.cpu().numpy().squeeze()
            psnr_val = calculate_psnr(denoised_np, clean_np)
            ssim_val = calculate_ssim(denoised_np, clean_np)
            mse_val = np.mean((denoised_np - clean_np) ** 2)
            temp_consistency = calculate_temporal_consistency(denoised_np)
            metrics['psnr'].append(psnr_val)
            metrics['ssim'].append(ssim_val)
            metrics['mse'].append(mse_val)
            metrics['temporal_consistency'].append(temp_consistency)
            metrics['inference_time'].append(inference_time)
    summary = {}
    for metric_name, values in metrics.items():
        if values:
            summary[f'{metric_name}_mean'] = np.mean(values)
            summary[f'{metric_name}_std'] = np.std(values)
            summary[f'{metric_name}_min'] = np.min(values)
            summary[f'{metric_name}_max'] = np.max(values)
    return summary

# ============================
# 4. Ablation Study Execution
# ============================
def run_basic_cnn_ablation(train_file, validation_dir, output_dir="basiccnn_ablation_results", epochs=3):
    os.makedirs(output_dir, exist_ok=True)
    print("Loading datasets...")
    train_data = CalciumDataPreprocessor.load_and_preprocess(train_file)
    val_datasets = {}
    val_files = ['F0.tif', 'F1.tif', 'F2.tif', 'F3.tif']
    for val_file in val_files:
        val_path = os.path.join(validation_dir, val_file)
        if os.path.exists(val_path):
            data = CalciumDataPreprocessor.load_and_preprocess(val_path)
            val_datasets[val_file[:-4]] = data
            print(f"Loaded {val_file}: {data.shape}")
    if 'F0' not in val_datasets:
        raise FileNotFoundError("Ground truth F0.tif not found!")
    ground_truth = val_datasets['F0']

    ablation_configs = [
        {'num_layers': 3, 'num_features': 16, 'lr': 1e-4, 'batch_size': 2},
        {'num_layers': 4, 'num_features': 32, 'lr': 1e-4, 'batch_size': 2},
        {'num_layers': 4, 'num_features': 64, 'lr': 1e-4, 'batch_size': 1},
        {'num_layers': 5, 'num_features': 32, 'lr': 1e-5, 'batch_size': 2},
    ]
    all_results = []
    for config_idx, config in enumerate(ablation_configs):
        print(f"\nConfig {config_idx+1}/{len(ablation_configs)} | {config}")
        model = BasicScratchCNN(num_layers=config['num_layers'], num_features=config['num_features']).to(device)
        # Train
        train_dataset = SelfSupervisedDataset(train_data, sequence_length=8, patch_size=64, num_patches=200)
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
        def combined_loss(pred, target):
            mse_loss = F.mse_loss(pred, target)
            l1_loss = F.l1_loss(pred, target)
            return mse_loss + 0.1 * l1_loss
        model.train()
        history = {'losses': [], 'total_time':0}
        start_time = time.time()
        for epoch in range(epochs):
            epoch_losses = []
            bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            for batch_idx, (noisy, clean) in enumerate(bar):
                noisy, clean = noisy.to(device), clean.to(device)
                optimizer.zero_grad()
                output = model(noisy)
                loss = combined_loss(output, clean)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_losses.append(loss.item())
                bar.set_postfix({"Loss": f"{loss.item():.5f}"})
                if batch_idx >= 100: break
            avg_loss = np.mean(epoch_losses)
            history['losses'].append(avg_loss)
            scheduler.step(avg_loss)
            print(f"Epoch {epoch+1}: Loss = {avg_loss:.5f}, LR = {optimizer.param_groups[0]['lr']:.2e}")
        history['total_time'] = time.time() - start_time
        history['final_loss'] = history['losses'][-1]
        history['best_loss'] = min(history['losses'])
        # Validation & Metrics
        for val_name, val_data in val_datasets.items():
            if val_name == 'F0': continue
            val_dataset = ValidationDataset(val_data, ground_truth, patch_size=64, sequence_length=8, num_patches=50)
            metrics = run_validation_evaluation(model, val_dataset, device, f"BasicScratchCNN_{val_name}")
            result = {
                'config_id': config_idx+1,
                'num_layers': config['num_layers'],
                'num_features': config['num_features'],
                'learning_rate': config['lr'],
                'batch_size': config['batch_size'],
                'validation_set': val_name,
                'noise_level': {'F1': 'Low', 'F2': 'Medium', 'F3': 'High'}[val_name],
                **metrics,
                **{f'train_{k}': v for k, v in history.items()}
            }
            all_results.append(result)
        # Save checkpoint
        model_name = f"BasicScratchCNN_L{config['num_layers']}_F{config['num_features']}_lr{config['lr']:.0e}"
        model_path = os.path.join(output_dir, f"{model_name}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved: {model_path}")

    # Save results CSV
    results_df = pd.DataFrame(all_results)
    csv_path = os.path.join(output_dir, 'basiccnn_ablation_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # Save report
    report_lines = []
    report_lines.append("==="*20)
    report_lines.append("BasicScratchCNN Ablation Study Report")
    report_lines.append("==="*20)
    report_lines.append(f"Training data shape: {train_data.shape}")
    for val_name, val_data in val_datasets.items():
        report_lines.append(f"{val_name}: Shape={val_data.shape}")
    report_lines.append("\nBest Configurations (max PSNR per noise):")
    for val_set in ['F1', 'F2', 'F3']:
        subset = results_df[results_df['validation_set']==val_set]
        if not subset.empty:
            best_idx = subset['psnr_mean'].idxmax()
            best = subset.loc[best_idx]
            report_lines.append(f"\nNoise: {val_set} {best['noise_level']}")
            report_lines.append(f"  Layers: {best['num_layers']}, Features: {best['num_features']}")
            report_lines.append(f"  LR: {best['learning_rate']:.0e}, Batch: {best['batch_size']}")
            report_lines.append(f"  PSNR: {best['psnr_mean']:.2f} (+/- {best['psnr_std']:.2f})")
            report_lines.append(f"  SSIM: {best['ssim_mean']:.3f} (+/- {best['ssim_std']:.3f})")
            report_lines.append(f"  Inference Time: {best['inference_time_mean']:.3f} s")
    top_psnr = results_df['psnr_mean'].max()
    top_config = results_df.loc[results_df['psnr_mean'].idxmax()]
    report_lines.append("\nBest Overall:")
    report_lines.append(f"  Layers: {top_config['num_layers']}, Features: {top_config['num_features']}, LR: {top_config['learning_rate']:.0e}")
    report_lines.append(f"  PSNR: {top_psnr:.2f} dB on {top_config['validation_set']}")
    report_path = os.path.join(output_dir, 'basiccnn_ablation_report.txt')
    with open(report_path, 'w') as f:
        f.write("\n".join(report_lines))
    print(f"\nReport saved to: {report_path}\n")
    print("\n".join(report_lines[:50]))
    print("\n... (full report saved to file) ...")

# ============================
# 5. Main
# ============================
if __name__=="__main__":
    # Update these paths as required
    TRAIN_FILE = "Train/A1.tif"
    VALIDATION_DIR = "Validation"
    OUTPUT_DIR = "basiccnn_ablation_results"
    EPOCHS = 3  # Increase for better performance!
    run_basic_cnn_ablation(TRAIN_FILE, VALIDATION_DIR, OUTPUT_DIR, epochs=EPOCHS)
