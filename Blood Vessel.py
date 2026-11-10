import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from einops.layers.torch import Rearrange
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.ndimage import zoom

# ================= 配置 =================
CFG = dict(
    data_path="vessel-data50s.npy",
    mask_path="vessel-mask50.npy",
    
    spatial_size=(96, 96), 
    dt=0.04,

    # 实验设置：
    T_train=50,          # 训练 1.0s (25帧)
    T_eval=1250,         # 评估 50.0s (1250帧)

    batch_size=8,
    epochs=10000,         
    lr=5e-4,             # 主干网络学习率
    lr_freq=2e-5,        # 频率参数低学习率
    weight_decay=1e-6,   

    patch_size=(8, 8),   
    dim=192,
    depth=6,
    heads=6,
    mlp_dim=384,
    z_dim=192,           

    fourier_freqs=(1, 2, 4, 8, 16),
)


plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['mathtext.fontset'] = 'dejavusans'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FIG_DIR = "karst_results_paper_v1-noisy"
os.makedirs(FIG_DIR, exist_ok=True)


def save_figure_as_pdf(fig, png_path, dpi=300):
    pdf_path = os.path.splitext(png_path)[0] + '.pdf'
    try:
        fig.savefig(pdf_path, bbox_inches='tight', dpi=dpi, format='pdf')
        print(f"已保存PDF: {pdf_path}")
    except Exception as e:
        print(f"保存PDF失败 {pdf_path}: {e}")

# ================= 1. Dataset =================
class KarstDataset(Dataset):
    def __init__(self, mode='train'):
        if not os.path.exists(CFG['data_path']):
            print(f"Warning: {CFG['data_path']} not found.")
            self.data = np.random.rand(100, 4, 96, 96).astype(np.float32)
            self.mask = np.zeros((96, 96)).astype(np.float32)
        else:
            raw_data = np.load(CFG['data_path']).astype(np.float32)
            self.mask = np.load(CFG['mask_path']).astype(np.float32)
            
            fluid_indices = (self.mask == 0)
            u_vals = raw_data[:, 0:2, fluid_indices]
            
            self.stats = {
                'u_max': np.max(np.abs(u_vals)) + 1e-6,
                'p_max': np.percentile(np.abs(raw_data[:, 2]), 99.9) + 1e-6 
            }
            print(f"[{mode}] Norm: U_max={self.stats['u_max']:.4e}, P_max={self.stats['p_max']:.4e}")
            
            self.data = raw_data.copy()
            # 归一化
            self.data[:, 0:2] /= self.stats['u_max']
            self.data[:, 2:4] /= self.stats['p_max']

        # 计算平均流 (Mean Flow)
        mean_calc_len = 120 
        if self.data.shape[0] >= mean_calc_len:
            mean_slice = self.data[:mean_calc_len]
        else:
            mean_slice = self.data[:CFG['T_train']]
        self.mean_flow = np.mean(mean_slice, axis=0)
        
        self.dt = CFG['dt']
        
        target_len = CFG['T_train'] if mode == 'train' else CFG['T_eval']
        self.real_len = min(target_len, self.data.shape[0])
        
        x = np.linspace(0, 1, CFG['spatial_size'][0])
        y = np.linspace(0, 1, CFG['spatial_size'][1])
        self.x, self.y = np.meshgrid(x, y, indexing='ij')

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        T = self.real_len
        phys_data = self.data[:T]
        # 减去平均流，得到波动项 (Fluctuation)
        phys_fluct = phys_data - self.mean_flow[np.newaxis, ...]
        targets_phys = torch.tensor(phys_fluct).permute(1, 2, 3, 0).float()
        
        H, W = CFG['spatial_size']
        mask_rock = torch.tensor(self.mask).unsqueeze(-1).expand(-1, -1, T)
        mask_fluid = 1.0 - mask_rock
        
        targets = torch.cat([targets_phys, mask_fluid.unsqueeze(0), mask_rock.unsqueeze(0)], dim=0)
        
        inputs = torch.zeros((4, H, W, T))
        inputs[0] = torch.tensor(self.x).unsqueeze(-1).expand(-1, -1, T)
        inputs[1] = torch.tensor(self.y).unsqueeze(-1).expand(-1, -1, T)
        inputs[2] = mask_rock
        t_raw = torch.arange(T) * self.dt
        inputs[3] = t_raw.view(1, 1, T).expand(H, W, -1)
        
        return inputs, targets

# ================= Koopman =================
class KoopmanNaturalDynamics(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        assert z_dim % 2 == 0
        self.z_dim = z_dim
        num_blocks = z_dim // 2
        
        num_random = num_blocks // 2
        num_hint = num_blocks - num_random
        
        rand_part = torch.rand(num_random) * 30.0
        hint_part = torch.normal(mean=15.7, std=0.1, size=(num_hint,))
        init_freqs = torch.cat([rand_part, hint_part])
        
        self.omega = nn.Parameter(init_freqs)
        
        self.raw_gamma = nn.Parameter(torch.zeros(num_blocks) - 2.0)

    def forward(self, z0, dt):
        B, T = dt.shape
        B_z, N, D = z0.shape
        w = self.omega
        
        sigma = -F.softplus(self.raw_gamma) 
        
        dt_flat = dt.unsqueeze(-1)
        
        # 旋转部分 (e^iwt)
        theta = dt_flat * w.view(1, 1, -1)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        
        # 衰减部分 (e^sigma*t)
        scale = torch.exp(dt_flat * sigma.view(1, 1, -1))
        
        z0_pairs = z0.view(B_z, N, -1, 2).unsqueeze(1).expand(B, T, N, -1, -1)
        x = z0_pairs[..., 0]
        y = z0_pairs[..., 1]
        
        scale = scale.unsqueeze(2)
        cos_t = cos_t.unsqueeze(2)
        sin_t = sin_t.unsqueeze(2)
        
        x_new = scale * (x * cos_t - y * sin_t)
        y_new = scale * (x * sin_t + y * cos_t)
        zt = torch.stack([x_new, y_new], dim=-1).reshape(B, T, N, D)
        return zt

# ================= ViT Model =================

class SmoothConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3):
        super().__init__()
        
        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride=1, padding=0, bias=False)
        self.norm = nn.GroupNorm(8, out_c)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(self.pad(x))))

class ViT_KNO_Learnable(nn.Module):
    def __init__(self):
        super().__init__()
        dim = CFG['dim']
        p1, p2 = CFG['patch_size']
        h, w = CFG['spatial_size']
        
        base_ch = 3 
        in_ch = base_ch + base_ch * len(CFG['fourier_freqs']) * 2
        
        self.patch_embed = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p1, p2=p2),
            nn.Linear(p1 * p2 * in_ch, dim),
            nn.LayerNorm(dim)
        )
        self.pos_embed = nn.Parameter(torch.randn(1, (h // p1) * (w // p2), dim) * 0.02)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, CFG['heads'], CFG['mlp_dim'],
                                       batch_first=True, dropout=0.0),
            CFG['depth']
        )
        self.to_z = nn.Linear(dim, CFG['z_dim'])
        
        self.koopman = KoopmanNaturalDynamics(CFG['z_dim'])
        self.from_z = nn.Linear(CFG['z_dim'], dim)
        
        def make_decoder_smooth(out_ch):
            return nn.Sequential(
                Rearrange('b (h w) c -> b c h w', h=h // p1, w=w // p2),
                
                # 12 -> 24
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                SmoothConvBlock(dim, dim // 2),
                
                # 24 -> 48
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                SmoothConvBlock(dim // 2, dim // 4),
                
                # 48 -> 96
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                SmoothConvBlock(dim // 4, 32),
                
                nn.ReflectionPad2d(1),
                nn.Conv2d(32, out_ch, 3, 1, 0, bias=True)
            )

        self.decoder_u = make_decoder_smooth(2) # u, v
        self.decoder_p = make_decoder_smooth(2) # p, p_dup (保持4通道结构)
        
        nn.init.zeros_(self.decoder_p[-1].weight)
        nn.init.zeros_(self.decoder_p[-1].bias)

    def get_feats(self, inputs):
        base = inputs[:, [0, 1, 2], :, :, 0]
        feats = [base]
        for f in CFG['fourier_freqs']:
            feats.append(torch.sin(2 * math.pi * f * base))
            feats.append(torch.cos(2 * math.pi * f * base))
        return torch.cat(feats, dim=1)

    def forward(self, inputs):
        x = self.get_feats(inputs)
        tokens = self.patch_embed(x) + self.pos_embed
        z0 = self.to_z(self.encoder(tokens))
        
        t_raw = inputs[:, 3, 0, 0, :] 
        dt = t_raw - t_raw[:, 0:1]
        
        zt = self.koopman(z0, dt)
        
        B, T, N, Z = zt.shape
        feat = self.from_z(zt.reshape(B * T, N, Z))
        
        out_u = self.decoder_u(feat)
        out_p = self.decoder_p(feat)
        out = torch.cat([out_u, out_p], dim=1)
        
        return out.view(B, T, 4, CFG['spatial_size'][0], CFG['spatial_size'][1]).permute(0, 2, 3, 4, 1)

# =================  Loss  =================

def compute_gradient_central(img):
    # img: [B, C, H, W, T]
    B, C, H, W, T = img.shape
    
    # [B, C, H, W, T] -> [B, T, C, H, W] -> [B*T, C, H, W]
    img_reshaped = img.permute(0, 4, 1, 2, 3).reshape(B * T, C, H, W)
    
    img_padded = F.pad(img_reshaped, (1, 1, 1, 1), mode='replicate')
    
    dy = (img_padded[:, :, 2:, 1:-1] - img_padded[:, :, :-2, 1:-1]) / 2.0
    dx = (img_padded[:, :, 1:-1, 2:] - img_padded[:, :, 1:-1, :-2]) / 2.0
    
    # [B*T, C, H, W] -> [B, T, C, H, W] -> [B, C, H, W, T]
    dx = dx.view(B, T, C, H, W).permute(0, 2, 3, 4, 1)
    dy = dy.view(B, T, C, H, W).permute(0, 2, 3, 4, 1)
    
    return dx, dy

def domain_aware_loss(pred, target):
    
    true_phys = target[:, :4]
    mask_fluid = target[:, 4:5]
    mask_rock  = 1.0 - mask_fluid

    pred_u, pred_v = pred[:, 0:1], pred[:, 1:2]
    pred_p = pred[:, 2:3]
    
    true_u, true_v = true_phys[:, 0:1], true_phys[:, 1:2]
    true_p = true_phys[:, 2:3]

    loss_u = ((pred_u - true_u)**2 * mask_fluid).sum() / (mask_fluid.sum() + 1e-6)
    loss_v = ((pred_v - true_v)**2 * mask_fluid).sum() / (mask_fluid.sum() + 1e-6)
    
    loss_p_fluid = ((pred_p - true_p)**2 * mask_fluid).sum() / (mask_fluid.sum() + 1e-6)
    loss_p_solid = ((pred_p - true_p)**2 * mask_rock).sum() / (mask_rock.sum() + 1e-6)

    p_dx_pred, p_dy_pred = compute_gradient_central(pred_p)
    p_dx_true, p_dy_true = compute_gradient_central(true_p)
    
    loss_p_grad = (
        ((p_dx_pred - p_dx_true)**2 * mask_fluid).mean() +
        ((p_dy_pred - p_dy_true)**2 * mask_fluid).mean()
    )

    u_dx, _ = compute_gradient_central(pred_u)
    _, v_dy = compute_gradient_central(pred_v)
    div = u_dx + v_dy
    loss_div = (div**2 * mask_fluid).mean()

    
    loss = (
        20.0 * (loss_u + loss_v) +
        80.0 * loss_p_fluid +    
        80.0 * loss_p_solid +    
        40.0 * loss_p_grad +     
        5.0  * loss_div
    )
    return loss

# ================= 可视化 =================
def visualize_at_step(target, pred, mask_rock, t_idx, dt, save_name):
    t_val = t_idx * dt
    pred_phys = pred.copy()
    mask_expanded = mask_rock[:, :, np.newaxis]
    
    pred_phys[0] *= (1 - mask_expanded)
    pred_phys[1] *= (1 - mask_expanded)

    vars = ['Ux', 'Uy', 'Pressure']
    fig, axs = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)

    fig.suptitle(f"Component-wise Comparison @ t={t_val:.2f}s", fontsize=16)
    cols = ['Reference (FEM)', 'Prediction (Ours)', 'Abs Error']

    for ax, col in zip(axs[0], cols):
        ax.set_title(col, fontsize=14)
    for ax, row in zip(axs[:, 0], vars):
        ax.set_ylabel(row, fontsize=14)

    for i in range(3):
        gt = target[i, :, :, t_idx].T
        pr = pred_phys[i, :, :, t_idx].T
        er = np.abs(gt - pr)
        
        vmin, vmax = gt.min(), gt.max()
        if vmax - vmin < 1e-6: vmax += 1e-6
        
        im1 = axs[i, 0].imshow(gt, origin='lower', cmap='jet', vmin=vmin, vmax=vmax)
        axs[i, 1].imshow(pr, origin='lower', cmap='jet', vmin=vmin, vmax=vmax)
        im3 = axs[i, 2].imshow(er, origin='lower', cmap='magma')
        
        plt.colorbar(im1, ax=axs[i, 1], fraction=0.046, pad=0.04)
        plt.colorbar(im3, ax=axs[i, 2], fraction=0.046, pad=0.04)

   
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"已保存PNG: {save_name}")
    
    save_figure_as_pdf(fig, save_name, dpi=300)
    
    plt.close()

def plot_paper_quality(target_np, pred_np, mask_rock, t_idx, dt, save_name, global_t_label):
    if t_idx >= target_np.shape[-1]:
        t_idx = target_np.shape[-1] - 1

    H, W = target_np.shape[1], target_np.shape[2]

    ux_gt = target_np[0, :, :, t_idx].T
    uy_gt = target_np[1, :, :, t_idx].T
    p_gt  = target_np[2, :, :, t_idx].T
    ux_pr = pred_np[0, :, :, t_idx].T
    uy_pr = pred_np[1, :, :, t_idx].T
    p_pr  = pred_np[2, :, :, t_idx].T

    mask = mask_rock.T.astype(float)
    fluid_mask = 1.0 - mask

    ux_gt *= fluid_mask; uy_gt *= fluid_mask
    ux_pr *= fluid_mask; uy_pr *= fluid_mask

    scale = 4
    Hh, Wh = H * scale, W * scale

    ux_gt_hi = zoom(ux_gt, scale, order=3)
    uy_gt_hi = zoom(uy_gt, scale, order=3)
    p_gt_hi  = zoom(p_gt,  scale, order=3)
    ux_pr_hi = zoom(ux_pr, scale, order=3)
    uy_pr_hi = zoom(uy_pr, scale, order=3)
    p_pr_hi  = zoom(p_pr,  scale, order=3)
    fluid_mask_hi = zoom(fluid_mask, scale, order=0)

    for arr in [ux_gt_hi, uy_gt_hi, ux_pr_hi, uy_pr_hi]:
        arr[fluid_mask_hi < 0.5] = np.nan

    sp_gt_hi = np.sqrt(ux_gt_hi**2 + uy_gt_hi**2)
    sp_pr_hi = np.sqrt(ux_pr_hi**2 + uy_pr_hi**2)

    Yh2, Xh2 = np.mgrid[0:Hh, 0:Wh]
    Xn2, Yn2 = Xh2 / (Wh - 1), Yh2 / (Hh - 1)

    hires = 800
    xh = np.linspace(0, 1, hires)
    yh = np.linspace(0, 1, hires)
    Xh, Yh = np.meshgrid(xh, yh)
    
    path1_y = 0.8 - 0.4 * ((Xh+0.15)**3)
    path2_y = 0.8 - 1.4 * ((Xh+0.15)**3)
    d1 = np.abs(Yh - path1_y)
    d2 = np.abs(Yh - path2_y)
    mask_hi_geom = (d1 < 0.145) | (d2 < 0.15)

    p_min = min(p_gt_hi.min(), p_pr_hi.min())
    p_max = max(p_gt_hi.max(), p_pr_hi.max())
    v_min = 0.0
    v_max = np.nanmax(np.concatenate([sp_gt_hi.ravel(), sp_pr_hi.ravel()])) + 1e-8

    fig = plt.figure(figsize=(12, 6.5), dpi=300)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 0.08], wspace=0.15, hspace=0.15)

    ax_gt = fig.add_subplot(gs[0, 0])
    ax_pr = fig.add_subplot(gs[0, 1])
    cax_p = fig.add_subplot(gs[1, 0])
    cax_vel = fig.add_subplot(gs[1, 1])

    axes = [ax_gt, ax_pr]
    titles = ["Reference (FEM)", "Prediction (Koopman)"]
    ux_all, uy_all = [ux_gt_hi, ux_pr_hi], [uy_gt_hi, uy_pr_hi]
    p_all, sp_all = [p_gt_hi, p_pr_hi], [sp_gt_hi, sp_pr_hi]

    im_p = None
    for ax, ux_h, uy_h, p_h, sp_h, title in zip(axes, ux_all, uy_all, p_all, sp_all, titles):
        im_p = ax.imshow(p_h, extent=[0, 1, 0, 1], origin="lower", cmap="coolwarm", vmin=p_min, vmax=p_max, interpolation="bicubic")
        ax.streamplot(Xn2, Yn2, ux_h, uy_h, color=sp_h, cmap="turbo", density=2.0, linewidth=0.7, arrowsize=1.0, norm=plt.Normalize(v_min, v_max))
        ax.contour(xh, yh, mask_hi_geom.astype(float), levels=[0.5], colors="black", linewidths=1.5, zorder=10)
        ax.set_title(title, fontsize=16)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")

    phase_str = "Training Phase" if global_t_label <= (CFG['T_train']*CFG['dt']) else "Extrapolation Phase"
    fig.suptitle(f"Time t={global_t_label:.2f}s [{phase_str}]", fontsize=18)

    cb_p = fig.colorbar(im_p, cax=cax_p, orientation='horizontal')
    cb_p.set_label("Pressure field", fontsize=13)
    sm = plt.cm.ScalarMappable(cmap="turbo", norm=plt.Normalize(v_min, v_max))
    sm.set_array([])
    cb_v = fig.colorbar(sm, cax=cax_vel, orientation='horizontal')
    cb_v.set_label("Velocity magnitude", fontsize=13)

   
    plt.savefig(save_name, bbox_inches="tight", dpi=300)
    print(f"已保存PNG: {save_name}")
    
    save_figure_as_pdf(fig, save_name, dpi=300)
    
    plt.close()

# ================= Main =================
if __name__ == "__main__":
    
    try:
        import cairocffi
        print("cairocffi 已安装，支持PDF导出")
    except ImportError:
        print("cairocffi 未安装，尝试自动安装...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "cairocffi"])
    
    dataset = KarstDataset('train')
    loader = DataLoader(dataset, batch_size=CFG['batch_size'], shuffle=True)
    eval_ds = KarstDataset('eval')
    eval_loader = DataLoader(eval_ds, batch_size=1, shuffle=False)
    
    model = ViT_KNO_Learnable().to(device)

    print("Configuring Optimizer with Layer-wise LR...")
    koopman_ids = list(map(id, model.koopman.parameters()))
    base_params = filter(lambda p: id(p) not in koopman_ids, model.parameters())

    optim = AdamW([
        {'params': base_params, 'lr': CFG['lr']},
        {'params': model.koopman.parameters(), 'lr': CFG['lr_freq']}
    ], weight_decay=CFG['weight_decay'])

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, CFG['epochs'])

    print(f"Start Training ({CFG['epochs']} epochs) - With Boundary & Physics Fixes...")

    for ep in range(CFG['epochs']):
        model.train()
        loss_ep = 0.0
        for inp, tgt in loader:
            inp, tgt = inp.to(device), tgt.to(device)
            optim.zero_grad()
            pred = model(inp)
            loss = domain_aware_loss(pred, tgt)
            loss.backward()
            optim.step()
            loss_ep += loss.item()
        sched.step()
        if (ep + 1) % 50 == 0:
            print(f"Epoch {ep+1}: Loss = {loss_ep:.6f}")

    # 保存模型
    model_path = os.path.join(FIG_DIR, "karst_model_paper.pth")
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存至: {model_path}")

    print("Evaluating and Generating Paper-Quality Plots...")
    model.eval()
    with torch.no_grad():
        inp, tgt = next(iter(eval_loader))
        inp = inp.to(device)
        pred = model(inp).cpu()

    target_fluct = tgt[0].numpy()
    pred_fluct = pred[0].numpy()
    # 加上平均流
    mean_flow = dataset.mean_flow[:, :, :, np.newaxis]

    target_full = target_fluct[:4] + mean_flow
    pred_full = pred_fluct + mean_flow
    mask_rock = dataset.mask

    max_phys_time = pred_full.shape[-1] * CFG['dt']
    
    
    times_to_plot = [0.5, 1.0, 1.5, 2.0, 4.0, 10, 20, 30, 40, 48,49]

    for t_val in times_to_plot:
        if t_val < max_phys_time:
            t_idx = int(round(t_val / CFG['dt']))
            print(f"Generating plots for t={t_val}s...")
            
            visualize_at_step(
                target_full, pred_full, mask_rock, t_idx, CFG['dt'],
                os.path.join(FIG_DIR, f"components_t{t_val:.1f}s.png")
            )
            
            plot_paper_quality(
                target_full, pred_full, mask_rock, t_idx, CFG['dt'],
                os.path.join(FIG_DIR, f"paper_flow_t{t_val:.1f}s.png"),
                t_val
            )
            
    print(f"\n所有图片已保存至: {os.path.abspath(FIG_DIR)}")
    print("每个PNG文件都已生成对应的PDF文件")