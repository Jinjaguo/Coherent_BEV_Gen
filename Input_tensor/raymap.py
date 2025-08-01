import torch
import yaml
import numpy as np
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def fourier_encode(x: torch.Tensor, L: int = 6,
                   include_orig: bool = True) -> torch.Tensor:
    """
    Fourier PE: (B,6,H,W) → (B, 6 + 12L, H, W)
    """
    if x.dim() != 4 or x.size(1) != 6:
        raise ValueError("expected x = (B,6,H,W)")

    freqs = (2.0 ** torch.arange(L, device=x.device, dtype=x.dtype)) * torch.pi
    xp = x.unsqueeze(-1) * freqs.view(1, 1, 1, 1, L)               # (B,6,H,W,L)
    pe = torch.cat([torch.sin(xp), torch.cos(xp)], dim=-1)         # (B,6,H,W,2L)
    pe = pe.view(x.size(0), x.size(1) * 2 * L, x.size(2), x.size(3))  # (B,12L,H,W)

    if include_orig:
        pe = torch.cat([x, pe], dim=1)                             # (B,6+12L,H,W)
    return pe


def build_raymap(precomputed_dirs_cam: torch.Tensor,
                           T_cam2ego: torch.Tensor) -> torch.Tensor:
    """
    利用预计算的方向向量，高效生成 (1, 6, H, W) 的 raymap。
    precomputed_dirs_cam: (3, H, W) 的张量，由 precompute_dirs_cam 函数生成。
    T_cam2ego: (4, 4) 的相机到自车位姿变换矩阵。
    """
    device = precomputed_dirs_cam.device
    T_cam2ego = T_cam2ego.to(device)

    R = T_cam2ego[:3, :3]  # (3,3)
    t = T_cam2ego[:3, 3].view(3, 1, 1) # (3,1,1)

    h, w = precomputed_dirs_cam.shape[-2:]
    dirs_ego = (R @ precomputed_dirs_cam.view(3, -1)).view(3, h, w)
    orig_ego = t.view(3,1,1).expand_as(dirs_ego) # (3,H,W)

    ray = torch.cat([orig_ego, dirs_ego], dim=0).unsqueeze(0)  # (1, 6, H, W)
    return ray.contiguous()


# run once per camera at start-up
def precompute_dirs_cam(H, W, K, device=torch.device("cuda")):
    """返回 (3, H/16, W/16) 的单位方向射线，channel-first"""
    u, v = torch.meshgrid(
        torch.arange(W, dtype=torch.float32, device=device),
        torch.arange(H, dtype=torch.float32, device=device),
        indexing="xy")
    pix = torch.stack([u, v, torch.ones_like(u)], 0)                # (3,H,W)
    dirs = torch.linalg.inv(K) @ pix.view(3, -1)                    # (3,H*W)
    dirs = dirs.view(3, H, W)
    dirs = dirs / torch.linalg.norm(dirs, dim=0, keepdim=True)
    dirs = F.avg_pool2d(dirs.unsqueeze(0), 16)[0]                   # ↓16
    return dirs.contiguous()             # (3, H/16, W/16)
