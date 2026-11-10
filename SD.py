import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, random_split
from einops.layers.torch import Rearrange
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata


# ==================== Global ====================
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['mathtext.fontset'] = 'dejavusans'

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

FIG_DIR = 'figs_stokes_darcy_multiframe96_opt2'
os.makedirs(FIG_DIR, exist_ok=True)

# ==================== Config ====================
CFG = dict(
    spatial_size=(96, 96),
    T_train=5,                   # train time discretization on [0,1]
    T_eval=10,                    # eval time discretization on [0,2]
    temporal_domain_train=(0.0, 1.0),
    temporal_domain_eval=(0.0, 2.0),
    batch_size=16,
    epochs=1200,                  
    lr=5e-4,                      
    weight_decay=2e-5,            
    amp_tempering=False,          

    # Multi-frame encoder settings
    ref_use_all=False,            
    ref_K=1,                      

    # Loss weights
    w_u1=3.0,
    w_u2=5.0,
    w_p=0.05,
    w_phi=1.0,

    # Model
    patch_size=(16, 16),
    dim=192,
    depth=6,
    heads=6,
    mlp_dim=384,
    z_dim=192,
    emb_dropout=0.1,

    # Fourier features
    fourier_freqs=(1, 2, 4),     

    # Refine head
    refine=True,                 
    refine_hidden=32,            
    refine_blocks=1,             

    # Scheduler
    warmup_epochs=10,
)

# ==================== Exact Solution ====================
def exact_solution(x: torch.Tensor, y: torch.Tensor, t: torch.Tensor):
    k = 1.0
    phi_D = (torch.exp(y) - torch.exp(-y)) * torch.sin(x) * torch.exp(-t)
    u_S1 = (k / math.pi) * torch.sin(2 * math.pi * y) * torch.cos(x) * torch.exp(-t)
    u_S2 = (-2 * k + (k / math.pi**2) * torch.sin(math.pi * y)**2) * torch.sin(x) * torch.exp(-t)
    p_S = torch.zeros_like(x)
    return u_S1, u_S2, p_S, phi_D

# ==================== Dataset ====================
class PhysicsDataset(Dataset):
    def __init__(self,
                 spatial_size=(96,96), temporal_size=20,
                 spatial_domain=((0, math.pi), (-1, 1)),
                 temporal_domain=(0,1),
                 num_samples=100,
                 mode='train',
                 channel_stats_external=None):
        super().__init__()
        self.spatial_size = spatial_size
        self.temporal_size = temporal_size
        self.spatial_domain = spatial_domain
        self.temporal_domain = temporal_domain
        self.num_samples = num_samples
        self.mode = mode

        self.x = np.linspace(spatial_domain[0][0], spatial_domain[0][1], spatial_size[0])
        self.y = np.linspace(spatial_domain[1][0], spatial_domain[1][1], spatial_size[1])
        self.t = np.linspace(temporal_domain[0], temporal_domain[1], temporal_size)

        if channel_stats_external is None:
            self.compute_normalization_params()
        else:
            self.channel_stats = channel_stats_external

    def compute_normalization_params(self):
        print('[Dataset] Computing normalization parameters...')
        X, Y, T = np.meshgrid(self.x, self.y, self.t, indexing='ij')
        X_t, Y_t, T_t = torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32), torch.tensor(T, dtype=torch.float32)
        u1, u2, p, phi = exact_solution(X_t, Y_t, T_t)
        
        mask_S = (Y < 0).astype(np.float32)
        mask_D = (Y >= 0).astype(np.float32)
        u1 = u1 * torch.tensor(mask_S)
        u2 = u2 * torch.tensor(mask_S)
        p = p * torch.tensor(mask_S)
        phi = phi * torch.tensor(mask_D)

        self.channel_stats = {
            'u1': {'max': torch.abs(u1).max().item(), 'std': u1.std().item()},
            'u2': {'max': torch.abs(u2).max().item(), 'std': u2.std().item()},
            'p' : {'max': max(torch.abs(p).max().item(), 1e-6), 'std': max(p.std().item(), 1e-6)},
            'phi':{'max': torch.abs(phi).max().item(), 'std': phi.std().item()},
        }
        for name, st in self.channel_stats.items():
            print(f"    {name:>3}: max={st['max']:.6f}, std={st['std']:.6f}")

    def normalize(self, u1, u2, p, phi):
        u1_norm = u1 / self.channel_stats['u1']['max']
        u2_norm = u2 / self.channel_stats['u2']['max']
        p_norm  = p
        phi_norm = phi / self.channel_stats['phi']['max']
        return u1_norm, u2_norm, p_norm, phi_norm

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        X, Y, T = np.meshgrid(self.x, self.y, self.t, indexing='ij')
        
        # 生成子域掩码
        mask_S = (Y < 0).astype(np.float32)  
        mask_D = (Y >= 0).astype(np.float32) 
        
        X_t, Y_t, T_t = torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32), torch.tensor(T, dtype=torch.float32)
        u1, u2, p, phi = exact_solution(X_t, Y_t, T_t)
        u1, u2, p, phi = u1.numpy(), u2.numpy(), p.numpy(), phi.numpy()
        u1 *= mask_S; u2 *= mask_S; p *= mask_S; phi *= mask_D  

        
        u1n, u2n, pn, phin = self.normalize(u1, u2, p, phi)

        # 输入特征
        X_norm = (X - self.spatial_domain[0][0]) / (self.spatial_domain[0][1] - self.spatial_domain[0][0]) * 2 - 1
        Y_norm = Y.astype(np.float32)
        T_norm = T.astype(np.float32)
        input_features = np.stack([X_norm, Y_norm, T_norm, mask_D], axis=0)  # (4,X,Y,T)

        # 前4通道为物理量，后2通道为掩码
        targets = np.stack([u1n, u2n, pn, phin, mask_S, mask_D], axis=0)  # (6,X,Y,T)
        return torch.tensor(input_features, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)

# ==================== Koopman Layer ====================
class KoopmanExp(nn.Module):
    def __init__(self, z_dim: int):
        super().__init__()
        self.R = nn.Parameter(torch.zeros(z_dim, z_dim))  # Skew-symmetric part
        self.C = nn.Parameter(torch.zeros(z_dim, z_dim))  # Positive semi-definite part
        
        self.alpha = nn.Parameter(torch.tensor(-1.0)) 

        nn.init.uniform_(self.R, -0.001, 0.001)
        nn.init.kaiming_uniform_(self.C, a=math.sqrt(5))
        with torch.no_grad():
            self.C *= 2.0  

    def _A(self):
        device = self.R.device
        
        S = 0.5 * (self.R - self.R.t())  # 反对称矩阵 (Oscillation)
        D = self.C.t() @ self.C          # 对称正定矩阵 (Dissipation)

        # A = S - D + alpha * I
        A = S - D + self.alpha * torch.eye(self.R.shape[0], device=device)
        return A

    def forward(self, z0, dt):
        A = self._A()  

        B, T = dt.shape
        d = A.shape[0]

        # 计算K矩阵：即指数矩阵的时间演化
        K = torch.matrix_exp(dt.view(-1, 1, 1) * A)  

        K = K.view(B, T, d, d)  # (B, T, d, d)

        z0e = z0.unsqueeze(1).expand(B, T, *z0.shape[1:])  # (B, T, N, d)
        zt = torch.matmul(z0e, K.transpose(-1, -2))  # (B, T, N, d)

        return zt

# ==================== Lightweight Refine Head ====================
class ResidualConvBlock(nn.Module):
    def __init__(self, channels=4, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=3, padding=1),
        )
    def forward(self, x):
        return x + self.net(x)

class RefineHead(nn.Module):
    def __init__(self, channels=4, hidden=32, blocks=1):
        super().__init__()
        self.blocks = nn.Sequential(*[ResidualConvBlock(channels, hidden) for _ in range(blocks)])

    def forward(self, x):  # x: (B,C,X,Y,T)
        B, C, X, Y, T = x.shape
        y = x.permute(0, 4, 1, 2, 3).reshape(B*T, C, X, Y)  # (B*T,C,X,Y)
        y = self.blocks(y)
        y = y.view(B, T, C, X, Y).permute(0, 2, 3, 4, 1)    # (B,C,X,Y,T)
        return y

# ==================== ViT + Multi-frame + Fourier features ====================
class ViT_KNO_MultiFrame(nn.Module):
    def __init__(self,
                 spatial_size=(96, 96),
                 patch_size=(16, 16),
                 dim=192,
                 depth=6,
                 heads=6,
                 mlp_dim=384,
                 in_channels_base=4,      
                 fourier_freqs=(1,2,4),   
                 z_dim=192,
                 out_channels=4,
                 emb_dropout=0.1,
                 refine=True,
                 refine_hidden=32,
                 refine_blocks=1):
        super().__init__()
        self.spatial_size = spatial_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.fourier_freqs = tuple(fourier_freqs)

        assert spatial_size[0] % patch_size[0] == 0
        assert spatial_size[1] % patch_size[1] == 0

        self.nx = spatial_size[0] // patch_size[0]
        self.ny = spatial_size[1] // patch_size[1]
        self.num_patches = self.nx * self.ny

        fourier_ch = 2 * len(self.fourier_freqs) * 2
        self.in_channels_space = in_channels_base + fourier_ch

        patch_dim = self.in_channels_space * patch_size[0] * patch_size[1]

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (nx px) (ny py) -> b (nx ny) (px py c)',
                      px=patch_size[0], py=patch_size[1], nx=self.nx, ny=self.ny),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
            nn.Dropout(emb_dropout),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=mlp_dim,
            dropout=emb_dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=depth)

        self.to_z = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, z_dim),
        )

        self.koopman = KoopmanExp(z_dim=z_dim)

        self.decoder = nn.Sequential(
            nn.LayerNorm(z_dim),
            nn.Linear(z_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, patch_size[0] * patch_size[1] * out_channels),
        )

        self.temporal_pool = nn.Linear(dim, 1)
        self.refine = RefineHead(out_channels, refine_hidden, refine_blocks) if refine else None

    @staticmethod
    def _fourier_features(coord_norm: torch.Tensor, freqs):
        outs = []
        for f in freqs:
            ang = 2 * math.pi * f * coord_norm
            outs += [torch.sin(ang), torch.cos(ang)]
        return torch.cat(outs, dim=1) if outs else torch.zeros_like(coord_norm)

    def forward(self, inputs, ref_use_all=False, ref_K=1):
        B, C, X, Y, T = inputs.shape
        assert C >= 4

        if ref_use_all:
            ref_idx = list(range(T))
        else:
            K = max(1, min(ref_K, T))
            ref_idx = np.linspace(0, T-1, num=K, dtype=int).tolist()
            if 0 not in ref_idx:
                ref_idx[0] = 0

        tokens_list = []
        frame_summaries = []
        for ti in ref_idx:
            base = inputs[:, [0,1,2,3], :, :, ti]  # (B,4,X,Y)
            Xn = base[:, 0:1]
            Yn = base[:, 1:2]
            fx = self._fourier_features(Xn, self.fourier_freqs)
            fy = self._fourier_features(Yn, self.fourier_freqs)
            aug = torch.cat([base, fx, fy], dim=1)
            assert aug.shape[1] == self.in_channels_space

            tokens_t = self.to_patch_embedding(aug)
            tokens_list.append(tokens_t)
            frame_summaries.append(tokens_t.mean(dim=1))

        tokens_stack = torch.stack(tokens_list, dim=1)
        summary_stack = torch.stack(frame_summaries, dim=1)

        logits = self.temporal_pool(summary_stack)
        attn = torch.softmax(logits, dim=1)
        attn = attn.unsqueeze(2)
        tokens = (tokens_stack * attn).sum(dim=1)

        tokens = tokens + self.pos_embedding
        tokens = self.dropout(tokens)
        tokens = self.transformer(tokens)
        z0 = self.to_z(tokens)

        t_grid = inputs[:, 2, 0, 0, :]
        t0 = t_grid[:, :1]
        dt = t_grid - t0

        zt = self.koopman(z0, dt)

        B_, T_, N_, d_ = zt.shape
        zt_flat = zt.reshape(B_*T_, N_, d_)
        dec = self.decoder(zt_flat)
        px, py = self.patch_size
        dec = dec.view(B_*T_, N_, px, py, self.out_channels)
        dec = dec.permute(0, 4, 1, 2, 3)

        dec = dec.view(B, T, self.out_channels, self.num_patches, px, py)
        dec = dec.permute(0, 2, 3, 4, 5, 1)
        out = dec.reshape(B, self.out_channels, self.nx*px, self.ny*py, T)

        if self.refine is not None:
            out = self.refine(out)

        return out

# ==================== Loss ==================== 
class ChannelWeightedLoss(nn.Module):
    def __init__(self, w_u1=2.0, w_u2=5.0, w_p=0.1, w_phi=1.0):
        super().__init__()
        self.weights = torch.tensor([w_u1, w_u2, w_p, w_phi])

    def forward(self, outputs, targets):
        """
        outputs: (B,4,X,Y,T) → 预测的u1/u2/p/phi
        targets: (B,6,X,Y,T) → 前4为物理量，后2为mask_S/mask_D
        """
        total = 0.0
        ch_losses = []
        B = outputs.shape[0]
        
        mask_S = targets[:, 4:5, ...]  # (B,1,X,Y,T)：u1/u2/p有效域
        mask_D = targets[:, 5:6, ...]  # (B,1,X,Y,T)：phi有效域
        domain_masks = torch.cat([mask_S, mask_S, mask_S, mask_D], dim=1)  # (B,4,X,Y,T)

        for i, w in enumerate(self.weights):
            pred = outputs[:, i:i+1, ...]
            true = targets[:, i:i+1, ...]
            
            masked_loss = F.mse_loss(pred, true, reduction='none') * domain_masks[:, i:i+1, ...]
            valid_pixels = domain_masks[:, i:i+1, ...].sum() + 1e-12  
            avg_loss = (masked_loss.sum()) / valid_pixels
            total += w * avg_loss
            ch_losses.append(float(avg_loss.item()))
        
        return total, ch_losses


# ==================== Utils ====================
def interpolate_error(error, x_range, y_range, interpolation_factor=4):
    
    x_orig = np.linspace(x_range[0], x_range[1], error.shape[0])  
    y_orig = np.linspace(y_range[0], y_range[1], error.shape[1])  
    X_orig, Y_orig = np.meshgrid(x_orig, y_orig, indexing='ij')  

    x_interp = np.linspace(x_range[0], x_range[1], error.shape[0] * interpolation_factor)
    y_interp = np.linspace(y_range[0], y_range[1], error.shape[1] * interpolation_factor)
    X_interp, Y_interp = np.meshgrid(x_interp, y_interp, indexing='ij')

    try:
        points = np.column_stack((X_orig.ravel(), Y_orig.ravel()))  
        values = error.ravel()  
        error_interp = griddata(
            points, values, (X_interp, Y_interp), 
            method='cubic',  
            fill_value=0  
        )
        smoothed_error = gaussian_filter(error_interp, sigma=1.5)  
        return smoothed_error, X_interp, Y_interp

    except Exception as e:
        return error, X_orig, Y_orig

# ==================== Trainer ====================
class Trainer:
    def __init__(self, model, train_loader, val_loader, device, cfg):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.criterion = ChannelWeightedLoss(
            w_u1=cfg['w_u1'], w_u2=cfg['w_u2'], w_p=cfg['w_p'], w_phi=cfg['w_phi']
        )
        self.optimizer = AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])

        self.base_lr = cfg['lr']
        self.warmup_epochs = int(cfg.get('warmup_epochs', 10))
        cosine_T = max(1, cfg['epochs'] - self.warmup_epochs)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cosine_T, eta_min=1e-6
        )

        self.train_loss_history, self.val_loss_history = [], []
        self.train_channel_losses, self.val_channel_losses = [], []

        self.ref_use_all = cfg['ref_use_all']
        self.ref_K = cfg['ref_K']

    def _forward(self, inputs):
        return self.model(inputs,
                          ref_use_all=self.ref_use_all,
                          ref_K=self.ref_K)

    def _set_warmup_lr(self, epoch):
        lr = self.base_lr * float(epoch + 1) / float(max(1, self.warmup_epochs))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

    def train(self, num_epochs=200):
        print('[Train] start')
        best_val = float('inf')
        self.model.train()
        for ep in range(num_epochs):
            if ep < self.warmup_epochs:
                self._set_warmup_lr(ep)
            else:
                self.scheduler.step()

            if ep % 10 == 0:
                cur_lr = self.optimizer.param_groups[0]['lr']
                print(f"[lr] {cur_lr:.3e}")

            run_loss = 0.0
            run_ch = np.zeros(4)

            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self._forward(inputs)
                loss, ch_losses = self.criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                run_loss += float(loss.item())
                run_ch += np.array(ch_losses)

            avg_loss = run_loss / len(self.train_loader)
            avg_ch = run_ch / len(self.train_loader)

            val_loss, val_ch = self.validate()

            self.train_loss_history.append(avg_loss)
            self.val_loss_history.append(val_loss)
            self.train_channel_losses.append(avg_ch)
            self.val_channel_losses.append(val_ch)

            print(f"Epoch {ep:03d} | Train {avg_loss:.6e} | Val {val_loss:.6e} | TrainCh {avg_ch}")

            if val_loss < best_val:
                best_val = val_loss
                torch.save(self.model.state_dict(), os.path.join(FIG_DIR, 'best_model_multiframe_96.pth'))
        print(f'[Train] done. Best val = {best_val:.6e}')
        return self.model

    def validate(self):
        self.model.eval()
        tot, tot_ch = 0.0, np.zeros(4)
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self._forward(inputs)
                loss, ch_losses = self.criterion(outputs, targets)
                tot += float(loss.item())
                tot_ch += np.array(ch_losses)
        self.model.train()
        return tot/len(self.val_loader), tot_ch/len(self.val_loader)

# ==================== Visualization & Evaluation ====================
def visualize_results(model, device, trainer=None, channel_stats_train=None,
                      ref_use_all=False, ref_K=1):
    dataset = PhysicsDataset(
        spatial_size=CFG['spatial_size'],
        temporal_size=CFG['T_eval'],
        spatial_domain=((0, math.pi), (-1, 1)),
        temporal_domain=CFG['temporal_domain_eval'],
        num_samples=1, mode='eval',
        channel_stats_external=channel_stats_train
    )
    inputs, targets = dataset[0]
    spatial_size_y = CFG['spatial_size'][1]
    y_coords = dataset.y  

    x_global_range = dataset.spatial_domain[0]  # x范围 (0, np.pi)
    y_global_range = dataset.spatial_domain[1]  # y范围 (-1, 1)


    split_idx = np.where(y_coords >= 0)[0][0]  
    valid_ranges = {
        'u1': (0, split_idx),    
        'u2': (0, split_idx),
        'p': (0, split_idx),
        'phi': (split_idx, spatial_size_y)  
    }
    valid_y_ranges = {
        'u1': (-1, 0),
        'u2': (-1, 0),
        'p': (-1, 0),
        'phi': (0, 1)
    }

    # 提取数据
    mask_S = targets[4:5, ...].numpy()
    mask_D = targets[5:6, ...].numpy()
    targets_np = targets[0:4, ...].numpy()  # 前4通道为物理量
    inputs = inputs.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(inputs, ref_use_all=ref_use_all, ref_K=ref_K)
    outputs_np = outputs.squeeze(0).cpu().numpy()
    T_vec = inputs[0, 2, 0, 0, :].detach().cpu().numpy()
    
    
    titles = ['u1', 'u2', 'p', 'phi']
    targets_cropped = []
    outputs_cropped = []
    for i, title in enumerate(titles):
        y_start, y_end = valid_ranges[title]
        target_crop = targets_np[i, :, y_start:y_end, :]  # 裁剪y轴
        output_crop = outputs_np[i, :, y_start:y_end, :]
        targets_cropped.append(target_crop)
        outputs_cropped.append(output_crop)

    time_points = [0.0, 0.5, 1.0, 1.5, 2.0]
    time_indices = [np.argmin(np.abs(T_vec - t)) for t in time_points]
    cmap_main = 'GnBu'
    cmap_error = 'Blues'

    for t_idx, t_val in zip(time_indices, time_points):
        fig, axes = plt.subplots(4, 3, figsize=(12, 8), dpi=600)
        fig.suptitle(f'Stokes-Darcy (Valid Subdomains) at t = {t_val}', fontsize=16, y=0.95)

        for row, (title, target_crop, output_crop) in enumerate(zip(titles, targets_cropped, outputs_cropped)):
            y_min, y_max = valid_y_ranges[title]
            true_data = target_crop[:, :, t_idx]
            pred_data = output_crop[:, :, t_idx]
            
            err_data = np.abs(true_data - pred_data)
            
            current_x_range = x_global_range
            current_y_range = (y_min, y_max)
            
            smoothed_err, _, _ = interpolate_error(
                error=err_data,
                x_range=current_x_range,
                y_range=current_y_range,
                interpolation_factor=4
            )

            ax_true = axes[row, 0]
            vmin = np.nanmin(true_data)
            vmax = np.nanmax(true_data)
            im_true = ax_true.imshow(
                true_data.T, origin='lower', cmap=cmap_main,
                extent=[current_x_range[0], current_x_range[1], current_y_range[0], current_y_range[1]]
            )
            ax_true.set_title(f'True: {title}', fontsize=12)
            ax_true.set_xlabel('x', fontsize=10)
            ax_true.set_ylabel('y', fontsize=10)
            ax_true.set_xticks(np.linspace(current_x_range[0], current_x_range[1], 4))
            ax_true.set_yticks(np.linspace(current_y_range[0], current_y_range[1], 3))
            cbar = plt.colorbar(im_true, ax=ax_true, shrink=1)
            cbar.ax.tick_params(labelsize=8)
            ax_true.set_aspect(1.5)

            ax_pred = axes[row, 1]
            im_pred = ax_pred.imshow(
                pred_data.T, origin='lower', cmap=cmap_main,
                extent=[current_x_range[0], current_x_range[1], current_y_range[0], current_y_range[1]], 
                vmin=vmin, vmax=vmax
            )
            ax_pred.set_title(f'Predicted: {title}', fontsize=12)
            ax_pred.set_xlabel('x', fontsize=10)
            ax_pred.set_ylabel('y', fontsize=10)
            ax_pred.set_xticks(np.linspace(current_x_range[0], current_x_range[1], 4))
            ax_pred.set_yticks(np.linspace(current_y_range[0], current_y_range[1], 3))
            cbar = plt.colorbar(im_pred, ax=ax_pred, shrink=1)
            cbar.ax.tick_params(labelsize=8)
            ax_pred.set_aspect(1.5)

            ax_err = axes[row, 2]
            err_vmax = np.nanmax(smoothed_err) * 0.8 if np.nanmax(smoothed_err) > 1e-8 else 1e-8
            im_err = ax_err.imshow(
                smoothed_err.T, origin='lower', cmap=cmap_error,
                extent=[current_x_range[0], current_x_range[1], current_y_range[0], current_y_range[1]],
                vmin=0, vmax=err_vmax
            )
            ax_err.set_title(f'Error: {title}', fontsize=12)
            ax_err.set_xlabel('x', fontsize=10)
            ax_err.set_ylabel('y', fontsize=10)
            ax_err.set_xticks(np.linspace(current_x_range[0], current_x_range[1], 4))
            ax_err.set_yticks(np.linspace(current_y_range[0], current_y_range[1], 3))
            cbar = plt.colorbar(im_err, ax=ax_err, shrink=1)
            cbar.ax.tick_params(labelsize=8)
            ax_err.set_aspect(1.5)

        plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05, wspace=0.25, hspace=0.5)
        save_path = os.path.join(FIG_DIR, f'multiframe96_halfplot_t_{t_val}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

    if trainer:
        plt.figure(figsize=(16, 10), dpi=600)
        ep = range(len(trainer.train_loss_history))
        colors = ["#307eb3", "#de9b61", "#299d29", "#ca3232"]
        markers = ['o', 's', '^', 'D']
          
        for i, (n, color, marker) in enumerate(zip(titles, colors, markers)):
            tr = [x[i] for x in trainer.train_channel_losses]
            plt.plot(ep, tr, '-', color=color, marker=marker, markersize=6, markevery=10, linewidth=2.0, alpha=0.8, label=f'Train {n}')

        plt.yscale('log')
        plt.grid(True, which='both', ls='-', alpha=0.3)
        plt.title('Training Progress: Per-Channel MSE Loss', fontsize=20)
        plt.xlabel('Epoch', fontsize=18)
        plt.ylabel('Weighted MSE Loss (log scale)', fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(ncol=2, fontsize=16, framealpha=0.9, loc='upper right')
        plt.tight_layout()
        save_path = os.path.join(FIG_DIR, 'multiframe96_training_progress.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=600)
        plt.close()


def evaluate_prediction_quality(model, device, channel_stats_train=None, 
                                ref_use_all=False, ref_K=1):
    print('\n[Eval] Quality on t in [0,2] including extrapolation to 2.0:')
    dataset = PhysicsDataset(
        spatial_size=CFG['spatial_size'],
        temporal_size=CFG['T_eval'],
        spatial_domain=((0, math.pi), (-1, 1)),
        temporal_domain=CFG['temporal_domain_eval'],
        num_samples=1, mode='eval',
        channel_stats_external=channel_stats_train
    )
    inputs, targets = dataset[0]
    y_coords = dataset.y
    split_idx = np.where(y_coords >= 0)[0][0]
    valid_ranges = {
        'u1': (0, split_idx), 'u2': (0, split_idx), 'p': (0, split_idx),
        'phi': (split_idx, CFG['spatial_size'][1])
    }

    inputs = inputs.unsqueeze(0).to(device)
    targets_np = targets[0:4, ...].numpy()
    with torch.no_grad():
        outputs = model(inputs, ref_use_all=ref_use_all, ref_K=ref_K)
    outputs_np = outputs.squeeze(0).cpu().numpy()
    T_vec = inputs[0, 2, 0, 0, :].detach().cpu().numpy()
    
    
    results = []
    titles = ['u1','u2','p','phi']
    for i, title in enumerate(titles):
        y_start, y_end = valid_ranges[title]
        for t_val in [0.0, 0.5, 1.0, 1.5, 2.0]:
            t_idx = np.argmin(np.abs(T_vec - t_val))
            
            true_data = targets_np[i, :, y_start:y_end, t_idx].flatten()
            pred_data = outputs_np[i, :, y_start:y_end, t_idx].flatten()
            
            valid_mask = ~np.isnan(true_data)
            true_data = true_data[valid_mask]
            pred_data = pred_data[valid_mask]
            if len(true_data) == 0:
                continue
            error = true_data - pred_data
            mse = float(np.mean(error**2))
            mae = float(np.mean(np.abs(error)))
            max_error = float(np.max(np.abs(error)))
            den = float(np.linalg.norm(true_data))
            rel_l2 = float(np.linalg.norm(error) / (den + 1e-12)) if den > 0 else float('nan')
            results.append({'channel': title, 'time': t_val, 'mse': mse, 'mae': mae,
                            'max_error': max_error, 'rel_l2': rel_l2})

    for r in results:
        print(f"{r['channel']:>3} t={r['time']:.1f} | MSE={r['mse']:.8f} | MAE={r['mae']:.8f} | Max={r['max_error']:.8f} | Rel-L2={r['rel_l2']:.6f}")
    return results

# ==================== Main ====================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    dataset_train = PhysicsDataset(
        spatial_size=CFG['spatial_size'],
        temporal_size=CFG['T_train'],
        spatial_domain=((0, math.pi), (-1, 1)),
        temporal_domain=CFG['temporal_domain_train'],
        num_samples=200, mode='train'
    )
    channel_stats_train = dataset_train.channel_stats

    n_train = int(0.8 * len(dataset_train))
    n_val = len(dataset_train) - n_train
    train_set, val_set = random_split(dataset_train, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=CFG['batch_size'], shuffle=True, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=CFG['batch_size'], shuffle=False, pin_memory=True)

    # 模型初始化
    model = ViT_KNO_MultiFrame(
        spatial_size=CFG['spatial_size'],
        patch_size=CFG['patch_size'],
        dim=CFG['dim'], depth=CFG['depth'], heads=CFG['heads'], mlp_dim=CFG['mlp_dim'],
        in_channels_base=4,
        fourier_freqs=CFG['fourier_freqs'],
        z_dim=CFG['z_dim'], out_channels=4, emb_dropout=CFG['emb_dropout'],
        refine=CFG['refine'], refine_hidden=CFG['refine_hidden'], refine_blocks=CFG['refine_blocks']
    ).to(device)

    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")

    # 训练
    trainer = Trainer(model, train_loader, val_loader, device, CFG)
    model = trainer.train(num_epochs=CFG['epochs'])

    # 保存模型
    torch.save(model.state_dict(), os.path.join(FIG_DIR, 'model_multiframe_96_last.pth'))
    print('Saved:', os.path.join(FIG_DIR, 'model_multiframe_96_last.pth'))

    # 可视化
    visualize_results(model, device, trainer, channel_stats_train=channel_stats_train,
                      ref_use_all=CFG['ref_use_all'], ref_K=CFG['ref_K'])

    # 评估
    results = evaluate_prediction_quality(model, device, channel_stats_train=channel_stats_train,
                                          ref_use_all=CFG['ref_use_all'], ref_K=CFG['ref_K'])

    # 保存评估结果
    with open(os.path.join(FIG_DIR, 'multiframe96_prediction_evaluation.txt'), 'w') as f:
        f.write("Stokes-Darcy (Multi-frame, 96x96) Prediction Quality (train on [0,1], eval includes t=2)\n")
        f.write("Channel | Time | MSE | MAE | Max Error | Relative L2\n")
        f.write("-"*90 + "\n")
        for r in results:
            f.write(f"{r['channel']} t={r['time']:.1f} | MSE: {r['mse']:.8f} | MAE: {r['mae']:.8f} | "
                    f"Max: {r['max_error']:.8f} | Rel-L2: {r['rel_l2']:.6f}\n")

    print("\nAll outputs are in:", os.path.abspath(FIG_DIR))


if __name__ == '__main__':
    main()
