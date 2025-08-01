from front_view import *
from raymap import*
from views import *
from torch.utils.data import Dataset
import PIL.Image as Image
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision.transforms.functional import pil_to_tensor

# ---------- Pose-only encoder (不共享权重) ----------
class PoseViewEncoder(nn.Module):
    def __init__(self, L=6, mid_ch=128):
        super().__init__()
        self.ray_branch = RayBranch(L=L, out_ch=mid_ch)

    # using ray_branch see front_view.py
    def forward(self, ray_pe):                     # (B,78,h,w)
        return self.ray_branch(ray_pe)             # (B,128,h,w)


# ---------- 文本 broadcast ----------
class TextBroadcaster(nn.Module):
    #TODO：确定clip后的维度
    def __init__(self, txt_dim=1024, out_ch=64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(txt_dim, out_ch),
            nn.SiLU()
        )

    def forward(self, txt_vec, h, w):              # (B,txt_dim)
        x = self.proj(txt_vec)                     # (B,out_ch)
        return x.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)


# ---------- Multimodal conditioner ----------
class MultiCondEncoder(nn.Module):
    def __init__(self,
                 num_pose_views=5,
                 L=6,
                 mid_ch=128,
                 fuse_ch=256,
                 txt_ch=64):
        super().__init__()
        self.front = FrontViewFusionNet(L, mid_ch, fuse_ch)
        # 每个pose独立权重
        self.pose_encoders = nn.ModuleList(
            [PoseViewEncoder(L, mid_ch) for _ in range(num_pose_views)]
        )
        self.txt_broad = TextBroadcaster(1024, txt_ch)

        # 1×1 Conv 把 concat 后通道压回 fuse_ch
        cond_in = fuse_ch + num_pose_views * mid_ch + txt_ch
        self.cond_proj = nn.Conv2d(cond_in, fuse_ch, 1)

    @torch.no_grad()  # 若只做 feature 提供，可冻结梯度
    def forward(self, rgb_front, ray_front, ray_pose_list, txt_vec):
        """
        rgb_front : (B,3,H,W)
        ray_front : (B,78,h,w)
        ray_pose_list : list[Tensor] len=5, 每个 (B,78,h,w)
        txt_vec   : (B,1024)
        """
        h, w = ray_front.shape[-2:]
        # 主视角外观+几何
        front_feat = self.front(rgb_front, ray_front)           # (B,256,h,w)
        # 其余视角几何，使用 zip 将每个编码器和对应的射线输入配对处理
        pose_feats_list = [
            enc(r) for enc, r in zip(self.pose_encoders, ray_pose_list)]
        pose_feats = torch.cat(pose_feats_list, dim=1)          # (B,5*128,h,w)
        # 文本
        txt_feat = self.txt_broad(txt_vec, h, w)                # (B,64,h,w)
        # 拼接→投影
        cond = torch.cat([front_feat, pose_feats, txt_feat], 1)  # (B,960,h,w)
        return self.cond_proj(cond)                             # (B,256,h,w)

class SixCamDataset(Dataset):
    def __init__(self, frame_meta_list, cam_static_dict, cap_dir):
        """
        frame_meta_list: 长度 404，每项 dict{
            "img_front": path,
            "poses": {cam_id: {"K":..,"T_cam2ego":..}},   # six cams
            "caption_path": path_to_saved_npy            # (1024,)
        }
        cam_static_dict: {cam_id: dirs_cam_const (3,h,w)}
        """
        self.meta = frame_meta_list
        self.static = cam_static_dict

    def read_img(self, path):
        # 这个方法保持不变
        img = Image.open(path).convert("RGB")
        img = np.array(img) / 255.0
        img = img.transpose(2, 0, 1)
        return torch.from_numpy(img).float()

    def __getitem__(self, idx):
        m = self.meta[idx]
        rgb = self.read_img(m["img_front"])
        txt_vec = torch.from_numpy(np.load(m["caption_path"]))

        ray_dict = {}
        for cam_id, pose in m["poses"].items():
            # 从 self.static 中获取预计算的方向向量
            precomputed_dirs = self.static[cam_id]
            T_cam2ego = pose["T_cam2ego"]  # 获取位姿
            raw_raymap = build_raymap(precomputed_dirs, T_cam2ego)
            ray_pe = fourier_encode(raw_raymap, L=6)
            ray_dict[cam_id] = ray_pe.half()

        return rgb, txt_vec, ray_dict