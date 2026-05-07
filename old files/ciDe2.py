
"""
CALCIUM IMAGING DENOISING - COMPLETE EXAM IMPLEMENTATION
========================================================

Features:
- Novel CNN from scratch (STAU-Net) 
- DeepCAD Baseline
- Validation with ground truth (F0.tif) vs noisy data (F1-F3.tif)
- Complete ablation study with CSV logging
- Comprehensive performance evaluation

Author: Your Name
Date: October 16, 2025
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tifffile
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import time
import random
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else 
                     "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 
                     "cpu")
print(f"ðŸš€ Using device: {device}")

# ==============================================================================
# 1. DATASET LOADING AND PREPROCESSING
# ==============================================================================

class CalciumDataPreprocessor:
    """Comprehensive preprocessing for calcium imaging data"""

    @staticmethod
    def load_and_preprocess(file_path, normalize=True):
        """Load TIFF file and apply preprocessing"""
        print(f"ðŸ“‚ Loading: {file_path}")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        data = tifffile.imread(file_path).astype(np.float32)
        print(f"ðŸ“Š Original shape: {data.shape}")
        print(f"ðŸ“ˆ Value range: [{np.min(data):.3f}, {np.max(data):.3f}]")

        if normalize:
            # Robust normalization using percentiles
            p1, p99 = np.percentile(data, [1, 99])
            data = np.clip(data, p1, p99)
            data = (data - p1) / (p99 - p1)
            print(f"âœ… Normalized to [0, 1]")

        return data

    @staticmethod
    def compute_snr(data):
        """Estimate Signal-to-Noise Ratio"""
        signal_power = np.mean(data ** 2)
        noise_power = np.var(np.diff(data, axis=0))
        snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))
        return snr_db

class SelfSupervisedDataset(Dataset):
    """Dataset for self-supervised training"""

    def __init__(self, data, sequence_length=8, patch_size=64, num_patches=500):
        self.data = data
        self.sequence_length = sequence_length
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.T, self.H, self.W = data.shape

    def __len__(self):
        return self.num_patches

    def __getitem__(self, idx):
        # Random temporal and spatial sampling
        t_start = random.randint(0, max(0, self.T - self.sequence_length))
        h_start = random.randint(0, max(0, self.H - self.patch_size))
        w_start = random.randint(0, max(0, self.W - self.patch_size))

        # Extract patch
        patch = self.data[t_start:t_start+self.sequence_length,
                         h_start:h_start+self.patch_size,
                         w_start:w_start+self.patch_size]

        # Self-supervised target (clean)
        clean_target = patch.copy()

        # Create noisy input
        noisy_input = self._add_realistic_noise(patch)

        return (torch.from_numpy(noisy_input).unsqueeze(1).float(),
                torch.from_numpy(clean_target).unsqueeze(1).float())

    def _add_realistic_noise(self, patch):
        """Add realistic noise for self-supervised training"""
        noisy = patch.copy()

        # Additive Gaussian noise
        noise_level = np.random.uniform(0.05, 0.2)
        gaussian_noise = np.random.normal(0, noise_level, patch.shape)
        noisy += gaussian_noise

        # Poisson noise (shot noise)
        if random.random() < 0.5:
            scaled = patch * 50  # Scale for Poisson
            poisson_noisy = np.random.poisson(scaled) / 50.0
            noisy = 0.7 * noisy + 0.3 * poisson_noisy

        # Random frame corruption
        if random.random() < 0.3:
            corrupt_frame = random.randint(0, len(patch)-1)
            if random.random() < 0.5:
                noisy[corrupt_frame] *= 0.1  # Dark frame
            else:
                noisy[corrupt_frame] = np.random.normal(0.5, 0.2, noisy[corrupt_frame].shape)

        return np.clip(noisy, 0, 1)

class ValidationDataset(Dataset):
    """Dataset for validation with ground truth"""

    def __init__(self, noisy_data, clean_data, patch_size=64, sequence_length=8, num_patches=100):
        self.noisy_data = noisy_data
        self.clean_data = clean_data
        self.patch_size = patch_size
        self.sequence_length = sequence_length
        self.num_patches = num_patches

        # Ensure both datasets have compatible shapes
        min_t = min(noisy_data.shape[0], clean_data.shape[0])
        min_h = min(noisy_data.shape[1], clean_data.shape[1])
        min_w = min(noisy_data.shape[2], clean_data.shape[2])

        self.noisy_data = noisy_data[:min_t, :min_h, :min_w]
        self.clean_data = clean_data[:min_t, :min_h, :min_w]

        self.T, self.H, self.W = self.noisy_data.shape

    def __len__(self):
        return self.num_patches

    def __getitem__(self, idx):
        # Fixed seed for reproducible validation
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

# ==============================================================================
# 2. MODEL IMPLEMENTATIONS
# ==============================================================================

# Attention Gate Module
class AttentionGate(nn.Module):
    """Attention gate for U-Net decoder"""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# Temporal Consistency Module
class TemporalConsistencyModule(nn.Module):
    """Module for temporal consistency across frames"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        # 1D temporal convolution
        self.temp_conv1d = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.temp_norm = nn.BatchNorm1d(channels)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(channels, channels//2, batch_first=True, bidirectional=True)

        # Output projection
        self.output_proj = nn.Linear(channels, channels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape

        # Global average pooling over spatial dimensions
        temp_feat = x.mean(dim=(3, 4))  # [B, T, C]

        # 1D convolution across time
        temp_feat_conv = self.temp_conv1d(temp_feat.transpose(1, 2))  # [B, C, T]
        temp_feat_conv = self.temp_norm(temp_feat_conv).transpose(1, 2)  # [B, T, C]

        # LSTM processing
        lstm_out, _ = self.lstm(temp_feat_conv)  # [B, T, C]
        lstm_out = self.dropout(lstm_out)

        # Generate attention weights
        attention_weights = torch.softmax(self.output_proj(lstm_out), dim=-1)
        attention_weights = attention_weights.unsqueeze(-1).unsqueeze(-1)  # [B, T, C, 1, 1]

        return x * attention_weights

# Novel CNN Architecture
class NovelCNN_Advanced(nn.Module):
    """
    ðŸ§  Novel STAU-Net: Spatio-Temporal Attention U-Net

    Key innovations:
    1. Attention gates in decoder for spatial focus
    2. Temporal consistency module with LSTM
    3. Progressive refinement layers
    4. Multi-component loss function
    """

    def __init__(self, in_channels=1, base_channels=32, depth=4):
        super().__init__()

        self.base_channels = base_channels
        self.depth = depth

        print(f"ðŸ§  Initializing Novel STAU-Net:")
        print(f"   - Base channels: {base_channels}")
        print(f"   - Depth: {depth}")
        print(f"   - Input channels: {in_channels}")

        # Encoder path
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        channels = [base_channels * (2**i) for i in range(depth)]
        in_ch = in_channels

        for i, out_ch in enumerate(channels):
            self.encoders.append(self._conv_block(in_ch, out_ch))
            if i < depth - 1:
                self.pools.append(nn.MaxPool2d(2))
            in_ch = out_ch

        # Temporal consistency module
        self.temporal_module = TemporalConsistencyModule(channels[-1])

        # Decoder path with attention gates
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        self.attention_gates = nn.ModuleList()

        for i in range(depth-1, 0, -1):
            self.upconvs.append(nn.ConvTranspose2d(channels[i], channels[i-1], 2, stride=2))
            self.attention_gates.append(AttentionGate(channels[i-1], channels[i-1], channels[i-1]//2))
            self.decoders.append(self._conv_block(channels[i], channels[i-1]))

        # Progressive refinement
        self.refine1 = self._conv_block(base_channels, base_channels//2)
        self.refine2 = self._conv_block(base_channels//2, base_channels//4)

        # Final output
        self.final = nn.Conv2d(base_channels//4, in_channels, 1)

        # Learnable residual weight
        self.residual_weight = nn.Parameter(torch.tensor(0.2))

        # Print total parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"   - Total parameters: {total_params:,}")

    def _conv_block(self, in_ch, out_ch):
        """Double convolution block with BatchNorm and ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape

        # Process frames through spatial encoder
        x_reshaped = x.view(B*T, C, H, W)

        # Encoder path
        encoder_features = []
        current = x_reshaped

        for i, (encoder, pool) in enumerate(zip(self.encoders, self.pools + [None])):
            current = encoder(current)
            encoder_features.append(current)
            if pool is not None:
                current = pool(current)

        # Reshape for temporal processing
        _, C_enc, H_enc, W_enc = current.shape
        current_temporal = current.view(B, T, C_enc, H_enc, W_enc)

        # Apply temporal consistency
        current_temporal = self.temporal_module(current_temporal)
        current = current_temporal.view(B*T, C_enc, H_enc, W_enc)

        # Decoder path with attention
        for i, (upconv, attention_gate, decoder) in enumerate(zip(self.upconvs, self.attention_gates, self.decoders)):
            current = upconv(current)

            # Apply attention gate
            skip_connection = encoder_features[-(i+2)]
            attended_skip = attention_gate(current, skip_connection)

            # Concatenate and decode
            current = torch.cat([current, attended_skip], dim=1)
            current = decoder(current)

        # Progressive refinement
        refined = self.refine1(current)
        refined = self.refine2(refined)

        # Final output
        output = self.final(refined)

        # Residual connection
        output = output + self.residual_weight * x_reshaped

        # Reshape back to sequence
        output = output.view(B, T, C, H, W)

        return output

# DeepCAD Baseline
class DeepCADBaseline(nn.Module):
    """
    ðŸ”¬ DeepCAD Baseline Implementation

    3D U-Net architecture for spatio-temporal calcium imaging denoising
    Based on the original DeepCAD paper methodology
    """

    def __init__(self, in_channels=1):
        super().__init__()

        print("ðŸ”¬ Initializing DeepCAD Baseline...")

        # Encoder
        self.enc1 = self._make_3d_conv_block(in_channels, 16)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc2 = self._make_3d_conv_block(16, 32)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc3 = self._make_3d_conv_block(32, 64)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = self._make_3d_conv_block(64, 128)

        # Decoder
        self.upconv3 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec3 = self._make_3d_conv_block(128, 64)

        self.upconv2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec2 = self._make_3d_conv_block(64, 32)

        self.upconv1 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.dec1 = self._make_3d_conv_block(32, 16)

        # Final layer
        self.final = nn.Conv3d(16, in_channels, kernel_size=1)

        total_params = sum(p.numel() for p in self.parameters())
        print(f"   - Total parameters: {total_params:,}")

    def _make_3d_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x: [B, T, C, H, W] -> [B, C, T, H, W] for 3D conv
        x = x.permute(0, 2, 1, 3, 4)

        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        # Bottleneck
        bottleneck = self.bottleneck(p3)

        # Decoder
        d3 = self.upconv3(bottleneck)
        if d3.shape[2:] != e3.shape[2:]:
            d3 = F.interpolate(d3, size=e3.shape[2:], mode='trilinear', align_corners=False)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        if d2.shape[2:] != e2.shape[2:]:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode='trilinear', align_corners=False)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        if d1.shape[2:] != e1.shape[2:]:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='trilinear', align_corners=False)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        output = self.final(d1)

        # Convert back: [B, C, T, H, W] -> [B, T, C, H, W]
        output = output.permute(0, 2, 1, 3, 4)

        return output

# ==============================================================================
# 3. VALIDATION METRICS
# ==============================================================================

def calculate_psnr(pred, target, data_range=1.0):
    """Calculate Peak Signal-to-Noise Ratio"""
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(data_range / np.sqrt(mse))

def calculate_ssim(pred, target, data_range=1.0):
    """Calculate Structural Similarity Index (simplified)"""
    # Per-frame SSIM calculation
    ssim_scores = []
    for t in range(len(pred)):
        pred_frame = pred[t]
        target_frame = target[t]

        # Calculate means
        mu1 = np.mean(pred_frame)
        mu2 = np.mean(target_frame)

        # Calculate variances and covariance
        var1 = np.var(pred_frame)
        var2 = np.var(target_frame)
        cov = np.mean((pred_frame - mu1) * (target_frame - mu2))

        # SSIM formula
        c1, c2 = (0.01 * data_range)**2, (0.03 * data_range)**2
        ssim = ((2*mu1*mu2 + c1) * (2*cov + c2)) / ((mu1**2 + mu2**2 + c1) * (var1 + var2 + c2))
        ssim_scores.append(ssim)

    return np.mean(ssim_scores)

def calculate_temporal_consistency(sequence):
    """Calculate temporal consistency metric"""
    if len(sequence) <= 1:
        return 1.0

    # Frame-to-frame differences
    diffs = []
    for t in range(1, len(sequence)):
        diff = np.mean(np.abs(sequence[t] - sequence[t-1]))
        diffs.append(diff)

    # Lower differences = higher consistency
    consistency = 1.0 / (1.0 + np.mean(diffs))
    return consistency

def run_validation_evaluation(model, val_dataset, device, model_name="Model"):
    """Run comprehensive validation evaluation"""
    print(f"ðŸ” Evaluating {model_name}...")

    model.eval()
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    metrics = {
        'psnr': [],
        'ssim': [],
        'mse': [],
        'temporal_consistency': [],
        'inference_time': []
    }

    with torch.no_grad():
        for i, (noisy_batch, clean_batch) in enumerate(tqdm(val_loader, desc=f"Validating {model_name}")):
            if i >= 50:  # Limit for speed
                break

            noisy_batch = noisy_batch.to(device)
            clean_batch = clean_batch.to(device)

            # Measure inference time
            start_time = time.time()
            denoised_batch = model(noisy_batch)
            inference_time = time.time() - start_time

            # Convert to numpy
            denoised_np = denoised_batch.cpu().numpy().squeeze()
            clean_np = clean_batch.cpu().numpy().squeeze()

            # Calculate metrics
            psnr_val = calculate_psnr(denoised_np, clean_np)
            ssim_val = calculate_ssim(denoised_np, clean_np)
            mse_val = np.mean((denoised_np - clean_np) ** 2)
            temp_consistency = calculate_temporal_consistency(denoised_np)

            metrics['psnr'].append(psnr_val)
            metrics['ssim'].append(ssim_val)
            metrics['mse'].append(mse_val)
            metrics['temporal_consistency'].append(temp_consistency)
            metrics['inference_time'].append(inference_time)

    # Calculate summary statistics
    summary = {}
    for metric_name, values in metrics.items():
        if values:
            summary[f'{metric_name}_mean'] = np.mean(values)
            summary[f'{metric_name}_std'] = np.std(values)
            summary[f'{metric_name}_min'] = np.min(values)
            summary[f'{metric_name}_max'] = np.max(values)

    return summary

# ==============================================================================
# 4. ABLATION STUDY AND TRAINING
# ==============================================================================

class ComprehensiveTrainer:
    """Complete training and evaluation pipeline"""

    def __init__(self, train_file, validation_dir, output_dir="exam_results"):
        self.train_file = train_file
        self.validation_dir = validation_dir
        self.output_dir = output_dir
        self.device = device

        os.makedirs(output_dir, exist_ok=True)

        # Load datasets
        print("ðŸ“‚ Loading datasets...")
        self.train_data = CalciumDataPreprocessor.load_and_preprocess(train_file)

        # Load validation datasets
        self.val_datasets = {}
        val_files = ['F0.tif', 'F1.tif', 'F2.tif', 'F3.tif']

        for val_file in val_files:
            val_path = os.path.join(validation_dir, val_file)
            if os.path.exists(val_path):
                data = CalciumDataPreprocessor.load_and_preprocess(val_path)
                self.val_datasets[val_file[:-4]] = data  # Remove .tif extension
                print(f"âœ… Loaded {val_file}: {data.shape}")
            else:
                print(f"âš ï¸ Warning: {val_file} not found in {validation_dir}")

        if 'F0' not in self.val_datasets:
            raise FileNotFoundError("Ground truth F0.tif not found!")

        self.ground_truth = self.val_datasets['F0']

    def run_ablation_study(self, epochs=3):
        """Run comprehensive ablation study"""
        print("\nðŸ”¬ STARTING ABLATION STUDY")
        print("=" * 50)

        # Define ablation configurations
        ablation_configs = [
            # Novel CNN configurations
            {'model': 'NovelCNN', 'base_channels': 16, 'depth': 3, 'lr': 1e-4, 'batch_size': 2},
            {'model': 'NovelCNN', 'base_channels': 32, 'depth': 3, 'lr': 1e-4, 'batch_size': 2},
            {'model': 'NovelCNN', 'base_channels': 32, 'depth': 4, 'lr': 1e-4, 'batch_size': 2},
            {'model': 'NovelCNN', 'base_channels': 32, 'depth': 4, 'lr': 1e-5, 'batch_size': 2},
            {'model': 'NovelCNN', 'base_channels': 64, 'depth': 4, 'lr': 1e-4, 'batch_size': 1},

            # DeepCAD configurations
            {'model': 'DeepCAD', 'lr': 1e-4, 'batch_size': 2},
            {'model': 'DeepCAD', 'lr': 1e-5, 'batch_size': 2},
            {'model': 'DeepCAD', 'lr': 1e-4, 'batch_size': 1},
        ]

        all_results = []

        for config_idx, config in enumerate(ablation_configs):
            print(f"\nðŸ“Š Configuration {config_idx + 1}/{len(ablation_configs)}: {config}")

            try:
                # Train model with this configuration
                model, training_history = self._train_single_config(config, epochs)

                # Evaluate on all validation sets
                for val_name, val_data in self.val_datasets.items():
                    if val_name == 'F0':  # Skip ground truth for noisy input
                        continue

                    print(f"ðŸ” Evaluating on {val_name}...")

                    # Create validation dataset
                    val_dataset = ValidationDataset(
                        val_data, self.ground_truth, 
                        patch_size=64, sequence_length=8, num_patches=50
                    )

                    # Run evaluation
                    val_metrics = run_validation_evaluation(model, val_dataset, self.device, 
                                                          f"{config['model']}_{val_name}")

                    # Combine results
                    result = {
                        'config_id': config_idx + 1,
                        'model': config['model'],
                        'base_channels': config.get('base_channels', 'N/A'),
                        'depth': config.get('depth', 'N/A'),
                        'learning_rate': config['lr'],
                        'batch_size': config['batch_size'],
                        'validation_set': val_name,
                        'noise_level': {'F1': 'Low', 'F2': 'Medium', 'F3': 'High'}[val_name],
                        **val_metrics,
                        **{f'train_{k}': v for k, v in training_history.items()}
                    }

                    all_results.append(result)

                # Save model checkpoint
                model_name = f"{config['model']}_ch{config.get('base_channels', 'X')}_d{config.get('depth', 'X')}_lr{config['lr']:.0e}"
                model_path = os.path.join(self.output_dir, f"{model_name}.pth")
                torch.save(model.state_dict(), model_path)
                print(f"ðŸ’¾ Model saved: {model_path}")

            except Exception as e:
                print(f"âŒ Configuration failed: {e}")
                continue

        # Save detailed results to CSV
        results_df = pd.DataFrame(all_results)
        csv_path = os.path.join(self.output_dir, 'detailed_ablation_results.csv')
        results_df.to_csv(csv_path, index=False)
        print(f"\nðŸ“Š Detailed results saved to: {csv_path}")

        # Generate summary report
        self._generate_summary_report(results_df)

        return results_df

    def _train_single_config(self, config, epochs):
        """Train a single model configuration"""

        # Create model
        if config['model'] == 'NovelCNN':
            model = NovelCNN_Advanced(
                base_channels=config['base_channels'],
                depth=config['depth']
            )
        else:  # DeepCAD
            model = DeepCADBaseline()

        model.to(self.device)

        # Create training dataset
        train_dataset = SelfSupervisedDataset(
            self.train_data, 
            sequence_length=8, 
            patch_size=64, 
            num_patches=200  # Reduced for faster training
        )

        train_loader = DataLoader(
            train_dataset, 
            batch_size=config['batch_size'], 
            shuffle=True, 
            num_workers=0
        )

        # Training setup
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

        # Multi-component loss function
        def combined_loss(pred, target):
            mse_loss = F.mse_loss(pred, target)
            l1_loss = F.l1_loss(pred, target)

            # Temporal consistency loss
            if pred.size(1) > 1:
                temp_diff_pred = pred[:, 1:] - pred[:, :-1]
                temp_diff_target = target[:, 1:] - target[:, :-1]
                temporal_loss = F.mse_loss(temp_diff_pred, temp_diff_target)
            else:
                temporal_loss = torch.tensor(0.0, device=pred.device)

            return mse_loss + 0.1 * l1_loss + 0.05 * temporal_loss

        # Training loop
        model.train()
        training_history = {
            'losses': [],
            'learning_rates': [],
            'total_time': 0
        }

        start_time = time.time()

        for epoch in range(epochs):
            epoch_losses = []
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

            for batch_idx, (noisy, clean) in enumerate(progress_bar):
                noisy, clean = noisy.to(self.device), clean.to(self.device)

                optimizer.zero_grad()
                output = model(noisy)
                loss = combined_loss(output, clean)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()

                epoch_losses.append(loss.item())
                progress_bar.set_postfix({"Loss": f"{loss.item():.6f}"})

                # Limit batches for speed
                if batch_idx >= 20:
                    break

            avg_loss = np.mean(epoch_losses)
            training_history['losses'].append(avg_loss)
            training_history['learning_rates'].append(optimizer.param_groups[0]['lr'])

            scheduler.step(avg_loss)

            print(f"Epoch {epoch+1}: Loss = {avg_loss:.6f}, LR = {optimizer.param_groups[0]['lr']:.2e}")

        training_history['total_time'] = time.time() - start_time
        training_history['final_loss'] = training_history['losses'][-1]
        training_history['best_loss'] = min(training_history['losses'])

        return model, training_history

    def _generate_summary_report(self, results_df):
        """Generate comprehensive summary report"""

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("ðŸŽ¯ COMPREHENSIVE ABLATION STUDY REPORT")
        report_lines.append("=" * 80)

        # Dataset summary
        report_lines.append("\nðŸ“Š DATASET SUMMARY:")
        report_lines.append("-" * 30)
        report_lines.append(f"Training data shape: {self.train_data.shape}")
        report_lines.append(f"Training data SNR: {CalciumDataPreprocessor.compute_snr(self.train_data):.2f} dB")

        for val_name, val_data in self.val_datasets.items():
            snr = CalciumDataPreprocessor.compute_snr(val_data)
            report_lines.append(f"{val_name} data shape: {val_data.shape}, SNR: {snr:.2f} dB")

        # Model performance summary
        report_lines.append("\nðŸ† BEST PERFORMING CONFIGURATIONS:")
        report_lines.append("-" * 40)

        # Group by validation set and find best PSNR
        for val_set in ['F1', 'F2', 'F3']:
            subset = results_df[results_df['validation_set'] == val_set]
            if not subset.empty:
                best_psnr_idx = subset['psnr_mean'].idxmax()
                best_config = subset.loc[best_psnr_idx]

                report_lines.append(f"\n{val_set} (Noise Level: {best_config['noise_level']}):")
                report_lines.append(f"  ðŸ¥‡ Best Model: {best_config['model']}")
                if best_config['model'] == 'NovelCNN':
                    report_lines.append(f"     - Channels: {best_config['base_channels']}, Depth: {best_config['depth']}")
                report_lines.append(f"     - Learning Rate: {best_config['learning_rate']:.0e}")
                report_lines.append(f"     - PSNR: {best_config['psnr_mean']:.2f} Â± {best_config['psnr_std']:.2f} dB")
                report_lines.append(f"     - SSIM: {best_config['ssim_mean']:.3f} Â± {best_config['ssim_std']:.3f}")
                report_lines.append(f"     - Inference Time: {best_config['inference_time_mean']:.3f} s")

        # Hyperparameter analysis
        report_lines.append("\nðŸ“ˆ HYPERPARAMETER IMPACT ANALYSIS:")
        report_lines.append("-" * 40)

        # Learning rate impact
        novel_cnn_results = results_df[results_df['model'] == 'NovelCNN']
        if not novel_cnn_results.empty:
            lr_analysis = novel_cnn_results.groupby('learning_rate')['psnr_mean'].agg(['mean', 'std', 'count'])
            report_lines.append("\nLearning Rate Impact (Novel CNN):")
            for lr, stats in lr_analysis.iterrows():
                report_lines.append(f"  {lr:.0e}: PSNR = {stats['mean']:.2f} Â± {stats['std']:.2f} ({stats['count']} configs)")

        # Channel analysis
        channel_analysis = novel_cnn_results.groupby('base_channels')['psnr_mean'].agg(['mean', 'std', 'count'])
        if not channel_analysis.empty:
            report_lines.append("\nBase Channels Impact (Novel CNN):")
            for channels, stats in channel_analysis.iterrows():
                if pd.notna(channels):
                    report_lines.append(f"  {int(channels)}: PSNR = {stats['mean']:.2f} Â± {stats['std']:.2f} ({stats['count']} configs)")

        # Model comparison
        report_lines.append("\nðŸ†š MODEL COMPARISON:")
        report_lines.append("-" * 20)

        model_comparison = results_df.groupby('model').agg({
            'psnr_mean': ['mean', 'std'],
            'ssim_mean': ['mean', 'std'],
            'inference_time_mean': ['mean', 'std']
        }).round(3)

        report_lines.append("Average Performance Across All Validation Sets:")
        for model in model_comparison.index:
            report_lines.append(f"\n{model}:")
            report_lines.append(f"  PSNR: {model_comparison.loc[model, ('psnr_mean', 'mean')]:.2f} Â± {model_comparison.loc[model, ('psnr_mean', 'std')]:.2f} dB")
            report_lines.append(f"  SSIM: {model_comparison.loc[model, ('ssim_mean', 'mean')]:.3f} Â± {model_comparison.loc[model, ('ssim_mean', 'std')]:.3f}")
            report_lines.append(f"  Time: {model_comparison.loc[model, ('inference_time_mean', 'mean')]:.3f} Â± {model_comparison.loc[model, ('inference_time_mean', 'std')]:.3f} s")

        # Recommendations
        report_lines.append("\nðŸ’¡ RECOMMENDATIONS:")
        report_lines.append("-" * 20)

        best_overall = results_df.loc[results_df['psnr_mean'].idxmax()]
        report_lines.append(f"ðŸ† Best Overall Configuration:")
        report_lines.append(f"   Model: {best_overall['model']}")
        if best_overall['model'] == 'NovelCNN':
            report_lines.append(f"   Base Channels: {best_overall['base_channels']}")
            report_lines.append(f"   Depth: {best_overall['depth']}")
        report_lines.append(f"   Learning Rate: {best_overall['learning_rate']:.0e}")
        report_lines.append(f"   Best PSNR: {best_overall['psnr_mean']:.2f} dB on {best_overall['validation_set']}")

        # Save report
        report_text = "\n".join(report_lines)
        report_path = os.path.join(self.output_dir, 'comprehensive_report.txt')
        with open(report_path, 'w') as f:
            f.write(report_text)

        print(f"\nðŸ“‹ Comprehensive report saved to: {report_path}")
        print("\n" + "\n".join(report_lines))

# ==============================================================================
# 5. MAIN EXECUTION FUNCTION
# ==============================================================================

def main():
    """Main execution function for exam"""

    print("ðŸŽ¯ CALCIUM IMAGING DENOISING - COMPREHENSIVE EVALUATION")
    print("=" * 80)

    # Configuration - UPDATE THESE PATHS
    TRAIN_FILE = "Train/A1.tif"
    VALIDATION_DIR = "Validation"
    OUTPUT_DIR = "comprehensive_exam_results"
    EPOCHS = 2  # Reduced for faster demo

    print(f"ðŸ“ Training file: {TRAIN_FILE}")
    print(f"ðŸ“‚ Validation directory: {VALIDATION_DIR}")
    print(f"ðŸ“Š Output directory: {OUTPUT_DIR}")
    print(f"ðŸ”„ Epochs per configuration: {EPOCHS}")
    print(f"ðŸ’¾ Device: {device}")

    try:
        # Initialize trainer
        trainer = ComprehensiveTrainer(TRAIN_FILE, VALIDATION_DIR, OUTPUT_DIR)

        # Run ablation study
        results_df = trainer.run_ablation_study(epochs=EPOCHS)

        print("\n" + "ðŸŽ‰" * 20)
        print("âœ… COMPREHENSIVE EVALUATION COMPLETED!")
        print("ðŸŽ‰" * 20)

        print(f"\nðŸ“Š Results Summary:")
        print(f"â€¢ Total configurations tested: {len(results_df['config_id'].unique())}")
        print(f"â€¢ Total validation runs: {len(results_df)}")
        print(f"â€¢ Best PSNR achieved: {results_df['psnr_mean'].max():.2f} dB")
        print(f"â€¢ Best SSIM achieved: {results_df['ssim_mean'].max():.3f}")

        print(f"\nðŸ“ Output files generated:")
        print(f"â€¢ detailed_ablation_results.csv - Complete CSV log of all results")
        print(f"â€¢ comprehensive_report.txt - Summary report")
        print(f"â€¢ *.pth - Model checkpoints for each configuration")

        print(f"\nðŸŽ¯ Ready for exam presentation!")

        return results_df

    except FileNotFoundError as e:
        print(f"\nâŒ ERROR: {e}")
        print("\nðŸ“ Please ensure the following files exist:")
        print(f"â€¢ {TRAIN_FILE}")
        print(f"â€¢ {VALIDATION_DIR}/F0.tif (ground truth)")
        print(f"â€¢ {VALIDATION_DIR}/F1.tif (low noise)")
        print(f"â€¢ {VALIDATION_DIR}/F2.tif (medium noise)")
        print(f"â€¢ {VALIDATION_DIR}/F3.tif (high noise)")

    except Exception as e:
        print(f"\nâŒ Unexpected error: {e}")
        print("Please check your installation and file paths.")

if __name__ == "__main__":
    main()