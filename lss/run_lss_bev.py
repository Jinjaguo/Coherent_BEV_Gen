# run_lss_bev.py
import os
import torch
from explore import compile_data, compile_data_all
from explore import compile_model
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def save_bev_tensors(version,
                     checkpoint_path,
                     dataroot='../data/mini',
                     gpuid=0,
                     output_folder='./bev_tensors',
                     n_save=100):
    """
    批量跑valloader，保存前 n_save 个BEV预测 tensor（未sigmoid，shape [1,H,W]）
    """
    os.makedirs(output_folder, exist_ok=True)
    # 1. 准备数据
    grid_conf = { 'xbound':[ -50,50,0.5], 'ybound':[-50,50,0.5],
                  'zbound':[-10,10,20],    'dbound':[4,45,1] }
    data_aug_conf = {
        'resize_lim':(0.193,0.225), 'final_dim':(128,352),
        'rot_lim':(-5.4,5.4), 'H':900, 'W':1600,
        'rand_flip':False, 'bot_pct_lim':(0.0,0.22),
        'cams':['CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT',
                'CAM_BACK_LEFT','CAM_BACK','CAM_BACK_RIGHT'],
        'Ncams':6  # 注意：LSS 原本接6视图
    }
    loader = compile_data_all(version, dataroot,
                                data_aug_conf=data_aug_conf,
                                grid_conf=grid_conf,
                                bsz=1, nworkers=4,
                                parser_name='segmentationdata')

    # 2. 加载模型
    device = torch.device(f'cuda:{gpuid}' if gpuid>=0 else 'cpu')
    model = compile_model(grid_conf, data_aug_conf, outC=1)
    sd = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(sd)
    model.to(device).eval()
    # 3. 遍历前 n_save 帧，保存 BEV tensor
    cnt = 0
    with torch.no_grad():
        for batch in loader:
            print(f'Processing frame {cnt}')
            imgs, rots, trans, intrins, post_rots, post_trans, _ = batch
            imgs = imgs.to(device)
            rots = rots.to(device)
            trans = trans.to(device)
            intrins = intrins.to(device)
            post_rots = post_rots.to(device)
            post_trans = post_trans.to(device)

            out = model(imgs, rots, trans, intrins, post_rots, post_trans)
            bev_feat = out.squeeze(0).cpu()
            bev_np = bev_feat.mean(dim=0).numpy()
            bev_norm = (bev_np - bev_np.min()) / (bev_np.max() - bev_np.min() + 1e-5)

            import matplotlib.pyplot as plt
            plt.figure(figsize=(6, 6))
            plt.imshow(bev_norm, cmap='inferno')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f'../imgs/bev_vis_{cnt:04d}.png')
            plt.close()
            torch.save(bev_feat, os.path.join(output_folder, f'bev_{cnt:04d}.pt'))
            print(f'Saved: bev_{cnt:04d}.pt + bev_vis_{cnt:04d}.png')

            cnt += 1

if __name__=='__main__':
    # 调整下面两行：version, checkpoint_path
    save_bev_tensors(version='mini',
                     checkpoint_path='../model525000.pt',
                     dataroot='../../data',
                     gpuid=0,
                     output_folder='../bev_tensors',
                     n_save=200)
