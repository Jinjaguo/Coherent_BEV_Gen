import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from raymap import *

# ---------------- Ray Branch ----------------
class RayBranch(nn.Module):
    def __init__(self, L: int = 6, out_ch: int = 128, use_bn: bool = False):
        super().__init__()
        in_pe = 6 + 12 * L

        norm = (lambda c: nn.BatchNorm2d(c)) if use_bn \
               else (lambda c: nn.GroupNorm(8, c))

        self.pe_down = nn.Sequential(
            nn.Conv2d(in_pe, 64, 1, bias=False),
            norm(64),
            nn.ReLU(inplace=True)
        )
        self.body = nn.Sequential(
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_ch, 1, bias=False),
            norm(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, ray_pe: torch.Tensor) -> torch.Tensor:
        x = self.pe_down(ray_pe)   # (B,64,H,W)
        return self.body(x)        # (B,out_ch,H,W)

# ---------------- Image Branch ----------------
class ImgBranch(nn.Module):
    def __init__(self, out_ch: int = 128, pretrained: bool = True):
        super().__init__()
        rn = resnet18(pretrained=pretrained)

        self.stem   = nn.Sequential(rn.conv1, rn.bn1, rn.relu, rn.maxpool)  # /4
        self.layer1 = rn.layer1                                             # /4
        self.layer2 = rn.layer2                                             # /8
        self.layer3 = rn.layer3                                             # /16
        self.head   = nn.Conv2d(256, out_ch, 1)                             # 256→out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.head(x)             # (B,out_ch,H/16,W/16)

# ---------------- Fusion Head ----------------
class Fusion(nn.Module):
    def __init__(self, ch_img: int = 128, ch_ray: int = 128, ch_fuse: int = 256):
        super().__init__()
        self.norm_img = nn.GroupNorm(8, ch_img)
        self.norm_ray = nn.GroupNorm(8, ch_ray)
        self.fuse = nn.Sequential(
            nn.Conv2d(ch_img + ch_ray, ch_fuse, 1, bias=False),
            nn.BatchNorm2d(ch_fuse),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_fuse, ch_fuse, 3, padding=1, bias=False)
        )

    def forward(self, img_lat, ray_lat):
        x = torch.cat([self.norm_img(img_lat), self.norm_ray(ray_lat)], dim=1)
        return self.fuse(x)            # (B,ch_fuse,H/16,W/16)

# ---------------- Whole Net ----------------
class FrontViewFusionNet(nn.Module):
    def __init__(self, L: int = 6, mid_ch: int = 128, fuse_ch: int = 256):
        super().__init__()
        self.ray_branch  = RayBranch(L=L, out_ch=mid_ch)
        self.img_branch  = ImgBranch(out_ch=mid_ch, pretrained=True)
        self.fusion_head = Fusion(mid_ch, mid_ch, fuse_ch)

    def forward(self, rgb, ray_pe):
        ray_lat = self.ray_branch(ray_pe)         # (B,128,H/16,W/16)
        img_lat = self.img_branch(rgb)            # (B,128,H/16,W/16)
        return self.fusion_head(img_lat, ray_lat) # (B,256,H/16,W/16)

# ---------------- Quick Test ----------------
if __name__ == "__main__":
    cfg = yaml.safe_load(open("camera_config  s/CAM_FRONT.yaml", "r"))
    H, W = cfg["H"], cfg["W"]
    K = torch.tensor(cfg["intrinsic"], dtype=torch.float32, device=device)
    T = torch.tensor(cfg["extrinsic"], dtype=torch.float32, device=device)
    T_cam2ego = torch.inverse(T)

    ray_map      = build_raymap(H // 16, W // 16, K, T_cam2ego, device)
    ray_pe_const = fourier_encode(ray_map, L=6, include_orig=True).half()  # (1,78,H/16,W/16)

    B   = 16
    rgb = torch.randn(B, 3, H, W, device=device)

    net = FrontViewFusionNet().to(device).eval()

    with torch.no_grad(), torch.cuda.amp.autocast():
        fused = net(rgb, ray_pe_const.repeat(B, 1, 1, 1))
    print("fused shape:", fused.shape)   # 应为 (16,256,H/16,W/16)