import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from einops.layers.torch import Rearrange
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# ================= 配置=================
CFG = dict(
    data_path="velocity_field_dataset.npz", 
    spatial_size=(128, 128),                
    T_train=None,                           
    batch_size=4,                           
    epochs=2000,
    lr=1e-3,
    weight_decay=1e-5,
    patch_size=(8, 8),
    dim=192,                                
    depth=6,
    heads=4,
    mlp_dim=384,
    z_dim=128,                              
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FIG_DIR = "karst_results_autonomous"
os.makedirs(FIG_DIR, exist_ok=True)

# ================= 数据集处理 =================
class KarstNPZDataset(Dataset):
    def __init__(self, mode='train', interp_res=(128, 128)):
        if not os.path.exists(CFG['data_path']):
            raise FileNotFoundError(f"Dataset not found: {CFG['data_path']}")
        
        print(f"[{mode.upper()}] Loading sparse data and interpolating to grid...")
        raw = np.load(CFG['data_path'])
        
        t_vals = raw['times']
        x_pts = raw['x']
        y_pts = raw['y']
        u_raw = raw['U_combined'] 
        v_raw = raw['V_combined']
        
        x_min, x_max = x_pts.min(), x_pts.max()
        y_min, y_max = y_pts.min(), y_pts.max()
        
        grid_x, grid_y = np.mgrid[
            x_min:x_max:complex(0, interp_res[0]), 
            y_min:y_max:complex(0, interp_res[1])
        ]
        self.grid_x = grid_x
        self.grid_y = grid_y
        
        total_frames = len(t_vals)
        n_train = int(total_frames * 0.8)
        
        if mode == 'train':
            indices = np.arange(n_train)
        else:
            indices = np.arange(n_train, total_frames)
            
        self.t_vals = t_vals[indices]
        
        self.t_vals_norm = self.t_vals - self.t_vals[0] 
        
        self.data_grid = []
        self.mask_grid = []
        
        pts = np.stack([x_pts, y_pts], axis=1) # (N, 2)
        
        print(f"Interpolating {len(indices)} frames. This may take a moment...")
        for i in indices:
            
            u_grid = griddata(pts, u_raw[i], (grid_x, grid_y), method='linear', fill_value=0.0)
            v_grid = griddata(pts, v_raw[i], (grid_x, grid_y), method='linear', fill_value=0.0)
            
            mask = np.ones_like(u_grid)
            
            frame = np.stack([u_grid, v_grid], axis=0) # (2, H, W)
            self.data_grid.append(frame)
            self.mask_grid.append(mask)
            
        self.data_grid = np.stack(self.data_grid, axis=0) # (T, 2, H, W)
        self.mask_grid = np.stack(self.mask_grid, axis=0) # (T, H, W)
        
        # 归一化统计量
        self.mean = np.mean(self.data_grid, axis=(0, 2, 3), keepdims=True)
        self.std = np.std(self.data_grid, axis=(0, 2, 3), keepdims=True) + 1e-6
        
        # 归一化数据
        self.data_norm = (self.data_grid - self.mean) / self.std
        
        print(f"Dataset {mode} ready. Shape: {self.data_norm.shape}")

    def __len__(self):
        return 1  本

    def __getitem__(self, idx):
        T, C, H, W = self.data_norm.shape
        targets = torch.tensor(self.data_norm).float()
        
        x_coord = torch.tensor(self.grid_x).float()
        x_coord = (x_coord - x_coord.min()) / (x_coord.max() - x_coord.min())
        y_coord = torch.tensor(self.grid_y).float()
        y_coord = (y_coord - y_coord.min()) / (y_coord.max() - y_coord.min())
        
        mask = torch.tensor(self.mask_grid[0]).float()
        u0 = targets[0, 0]
        v0 = targets[0, 1]
        
        inputs = torch.stack([x_coord, y_coord, mask, u0, v0], dim=0) # (5, H, W)
        times = torch.tensor(self.t_vals_norm).float()
        
        return inputs, times, targets

# =================  Koopman =================
class KoopmanAutonomousDynamics(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        assert z_dim % 2 == 0
        self.z_dim = z_dim
        num_blocks = z_dim // 2
        
        
        self.sigma = nn.Parameter(torch.randn(num_blocks) * 0.1 - 0.5) 
        freqs = torch.exp(torch.linspace(np.log(0.1), np.log(10.0), num_blocks))
        self.omega = nn.Parameter(freqs)
        
    def forward(self, z0, t_query):
        B, N, D = z0.shape
        B_t, T = t_query.shape
        
        sig = self.sigma.view(1, 1, -1)   
        omg = self.omega.view(1, 1, -1)   
        t = t_query.view(B, T, 1)         
        
        real_part = torch.exp(sig * t) * torch.cos(omg * t) 
        imag_part = torch.exp(sig * t) * torch.sin(omg * t) 
        
        z0_pairs = z0.view(B, N, -1, 2)
        x0 = z0_pairs[..., 0]
        y0 = z0_pairs[..., 1]
        
        x0 = x0.unsqueeze(1)
        y0 = y0.unsqueeze(1)
        r = real_part.unsqueeze(2)
        i = imag_part.unsqueeze(2)
        
        zt_real = x0 * r - y0 * i
        zt_imag = x0 * i + y0 * r
        
        zt = torch.stack([zt_real, zt_imag], dim=-1).reshape(B, T, N, D)
        return zt

# ================= ViT-K=================
class ViT_Karst_Model(nn.Module):
    def __init__(self):
        super().__init__()
        dim = CFG['dim']
        p1, p2 = CFG['patch_size']
        h, w = CFG['spatial_size']
        in_ch = 5 
        
        self.patch_embed = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p1, p2=p2),
            nn.Linear(p1 * p2 * in_ch, dim),
            nn.LayerNorm(dim)
        )
        self.pos_embed = nn.Parameter(torch.randn(1, (h // p1) * (w // p2), dim) * 0.02)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, CFG['heads'], CFG['mlp_dim'], batch_first=True, dropout=0.0),
            CFG['depth']
        )
        
        self.to_z = nn.Linear(dim, CFG['z_dim'])
        self.koopman = KoopmanAutonomousDynamics(CFG['z_dim'])
        self.from_z = nn.Linear(CFG['z_dim'], dim)
        
        self.decoder = nn.Sequential(
            Rearrange('b (h w) c -> b c h w', h=h // p1, w=w // p2),
            nn.ConvTranspose2d(dim, dim // 2, 4, 2, 1),
            nn.LayerNorm([dim // 2, h // p1 * 2, w // p2 * 2]),
            nn.GELU(),
            nn.ConvTranspose2d(dim // 2, dim // 4, 4, 2, 1),
            nn.LayerNorm([dim // 4, h // p1 * 4, w // p2 * 4]),
            nn.GELU(),
            nn.ConvTranspose2d(dim // 4, 32, 4, 2, 1),
            nn.GELU(),
            nn.Conv2d(32, 2, 1, bias=True)
        )

    def forward(self, inputs, t_query):
        tokens = self.patch_embed(inputs) + self.pos_embed
        encoded = self.encoder(tokens)
        z0 = self.to_z(encoded)
        zt = self.koopman(z0, t_query)
        B, T, N, Z = zt.shape
        feat = self.from_z(zt.reshape(B * T, N, Z))
        out = self.decoder(feat)
        return out.view(B, T, 2, CFG['spatial_size'][0], CFG['spatial_size'][1])

# ================= 损失函数 =================
def reconstruction_loss(pred, target, mask=None):
    loss = F.mse_loss(pred, target, reduction='none')
    if mask is not None:
        loss = loss * mask.view(1, 1, 1, *mask.shape[-2:])
        return loss.sum() / (mask.sum() * 2 * pred.shape[1] + 1e-6)
    else:
        return loss.mean()

# ================= 主程序 =================
if __name__ == "__main__":
    train_ds = KarstNPZDataset(mode='train', interp_res=CFG['spatial_size'])
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    
    eval_ds = KarstNPZDataset(mode='eval', interp_res=CFG['spatial_size'])
    eval_ds.mean = train_ds.mean
    eval_ds.std = train_ds.std
    eval_loader = DataLoader(eval_ds, batch_size=1, shuffle=False)
    
    model = ViT_Karst_Model().to(device)
    optim = AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, CFG['epochs'])
    
    print(f"Start Training on Karst Flow...")
    
    for ep in range(CFG['epochs']):
        model.train()
        ep_loss = 0
        for inputs, times, targets in train_loader:
            inputs, times, targets = inputs.to(device), times.to(device), targets.to(device)
            optim.zero_grad()
            pred = model(inputs, times)
            loss = reconstruction_loss(pred, targets)
            loss.backward()
            optim.step()
            ep_loss += loss.item()
        scheduler.step()
        
        if (ep+1) % 50 == 0:
            print(f"Epoch {ep+1}/{CFG['epochs']} | Loss: {ep_loss:.6f}")
            
    print("Evaluating extrapolation and generating plots...")
    model.eval()
    with torch.no_grad():
        inputs, times, targets = next(iter(eval_loader))
        inputs, times = inputs.to(device), times.to(device)
        pred = model(inputs, times) # (B, T, 2, H, W)
        
        pred_np = pred.cpu().numpy() * train_ds.std + train_ds.mean
        targets_np = targets.numpy() * train_ds.std + train_ds.mean
        
        t_last_idx = -1
        
        u_pred_raw = pred_np[0, t_last_idx, 0]
        v_pred_raw = pred_np[0, t_last_idx, 1]
        u_true_raw = targets_np[0, t_last_idx, 0]
        v_true_raw = targets_np[0, t_last_idx, 1]
        
        u_pred_plot = u_pred_raw.T
        v_pred_plot = v_pred_raw.T
        u_true_plot = u_true_raw.T
        v_true_plot = v_true_raw.T
        
        speed_pred = np.sqrt(u_pred_plot**2 + v_pred_plot**2)
        speed_true = np.sqrt(u_true_plot**2 + v_true_plot**2)
        error_field = np.abs(speed_true - speed_pred)
        
        ny, nx = u_pred_plot.shape
        x = np.linspace(0, nx-1, nx)
        y = np.linspace(0, ny-1, ny)
        X, Y = np.meshgrid(x, y) 
        
        n_levels = 12
        v_max = max(np.max(speed_true), np.max(speed_pred))
        levels = np.linspace(0, v_max, n_levels + 1)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
        
        cf1 = axes[0].contourf(X, Y, speed_true, levels=levels, cmap='jet', extend='both')

        axes[0].streamplot(X, Y, u_true_plot, v_true_plot, 
                           density=1.2, color='white', linewidth=0.6, arrowsize=0.8)
        axes[0].set_title("Ground Truth", fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        cf2 = axes[1].contourf(X, Y, speed_pred, levels=levels, cmap='jet', extend='both')
        axes[1].streamplot(X, Y, u_pred_plot, v_pred_plot, 
                           density=1.2, color='white', linewidth=0.6, arrowsize=0.8)
        axes[1].set_title("Prediction (ViT-K)", fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        cbar1 = fig.colorbar(cf2, ax=axes[0:2], location='bottom', fraction=0.05, pad=0.05)
        cbar1.set_label('Velocity Magnitude |U|', fontsize=12)
        
        err_max = np.max(error_field)
        err_levels = np.linspace(0, err_max, n_levels + 1)
        
        cf3 = axes[2].contourf(X, Y, error_field, levels=err_levels, cmap='jet', extend='max')
        axes[2].set_title("Absolute Error", fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        cbar2 = fig.colorbar(cf3, ax=axes[2], location='bottom', fraction=0.05, pad=0.05)
        cbar2.set_label('Error |U_true - U_pred|', fontsize=12)
        
        png_path = os.path.join(FIG_DIR, "karst_professional_vis.png")
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"Professional visualization saved to {png_path}")
        
        for ax in axes:
            ax.set_title("")
        
        pdf_path = os.path.join(FIG_DIR, "karst_professional_vis.pdf")

        plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
        print(f"Professional visualization (no titles) saved to {pdf_path}")
        
        plt.close()