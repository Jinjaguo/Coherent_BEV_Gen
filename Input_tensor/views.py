from front_view import *

# -------------------------------- 仅 RayMap → 几何特征 ------------------------------
class PoseViewEncoder(nn.Module):
    """
    对没有 RGB 的侧/后摄像头，仅以 RayMap(+Fourier PE) 编码相机姿态。
    输出 (B, mid_ch, H/16, W/16)
    """
    def __init__(self, L: int = 6, mid_ch: int = 128):
        super().__init__()
        self.ray_branch = RayBranch(L=L, out_ch=mid_ch)   # 复用你已有的 RayBranch

    def forward(self, ray_pe: torch.Tensor) -> torch.Tensor:
        """
        ray_pe: (B, 6+12L, H/16, W/16) —— 已做 Fourier PE
        """
        return self.ray_branch(ray_pe)


class MultiViewFusionNet(nn.Module):
    """
    front_view: RGB + RayMap
    side / rear views: 仅 RayMap
    """
    def __init__(self,
                 num_pose_views: int,
                 L: int = 6,
                 mid_ch: int = 128,
                 fuse_ch: int = 256):
        super().__init__()
        # 主视角
        self.front_net   = FrontViewFusionNet(L=L, mid_ch=mid_ch, fuse_ch=fuse_ch)

        # N 个仅姿态分支，共享各摄像头独立
        self.pose_encoder = nn.ModuleList([PoseViewEncoder(L=L, mid_ch=mid_ch)] * num_pose_views)
        self.num_pose_views = num_pose_views

        # 再次融合主视角 (fuse_ch) 与 N 个 pose_feat (mid_ch*num_pose_views)
        in_ch = fuse_ch + mid_ch * num_pose_views
        self.final_fuse = nn.Sequential(
            nn.Conv2d(in_ch, fuse_ch, 1, bias=False),
            nn.BatchNorm2d(fuse_ch), nn.ReLU(inplace=True),
            nn.Conv2d(fuse_ch, fuse_ch, 3, padding=1, bias=False)
        )

    def forward(self, rgb_front, ray_pe_front, ray_pe_list):
        """
        rgb_front   : (B,3,H,W)
        ray_pe_front: (B,6+12L,H/16,W/16)
        ray_pe_list : 长度 = num_pose_views，元素 shape 同 ray_pe_front
        """
        B = rgb_front.size(0)

        # —— 主视角
        front_feat = self.front_net(rgb_front, ray_pe_front)      # (B,256,H/16,W/16)

        # —— 其余视角（共享编码器）
        pose_feats = []
        for ray_pe in ray_pe_list:                                # len=N
            pose_feats.append(self.pose_encoder(ray_pe))          # (B,128,H/16,W/16)

        pose_feats = torch.cat(pose_feats, dim=1)                 # (B,128*N,H/16,W/16)

        # —— 最终融合
        all_feat = torch.cat([front_feat, pose_feats], dim=1)     # (B,256+128N,H/16,W/16)
        return self.final_fuse(all_feat)                          # (B,256,H/16,W/16)

