import argparse
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
from pathlib import Path
from tqdm import tqdm

from loaders.data_loader import get_whole_loader
from loaders.model_loader import load_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

class PDERefinerEngine:
    """
    只包含推理所需的逻辑 (参考论文 Eq. 4 和 Appendix C)
    """
    def __init__(self, model, refinement_steps, min_noise_std):
        self.model = model
        self.K_max = refinement_steps
        self.sigma_min = min_noise_std
        
    def get_sigma(self, k):
        if k == 0: 
            return 0
        else:
            return self.sigma_min ** (k / self.K_max)

    @torch.no_grad()
    def predict_next(self, u_prev, params, steps_to_run):
        """
        预测下一步 u(t+1)。
        steps_to_run: 指定跑几步 refinement (0=MSE Baseline, K=Full Refiner)
        """
        batch_size = u_prev.shape[0]
        k0_norm = torch.zeros((batch_size, 1), device=device)
        cond_params_0 = torch.cat([params, k0_norm], dim=1)
        zeros = torch.zeros_like(u_prev)
        model_input = torch.cat([u_prev, zeros], dim=1)
        u_hat = self.model(model_input, cond_params_0)
        
        if steps_to_run == 0:
            return u_hat

        for k in range(1, steps_to_run + 1):
            sigma_k = self.get_sigma(k)
            k_norm = torch.full((batch_size, 1), k / self.K_max, device=device)
            cond_params = torch.cat([params, k_norm], dim=1)
            noise = torch.randn_like(u_hat)
            u_input_noisy = u_hat + noise * sigma_k
            model_input = torch.cat([u_prev, u_input_noisy], dim=1)
            pred_noise = self.model(model_input, cond_params)
            u_hat = u_input_noisy - pred_noise * sigma_k
            
        return u_hat

# --- 2. 核心评估函数 ---
def evaluate(args, engine, test_loader, save_dir):
    engine.model.eval()
    
    # 取一个 Batch 出来测试 (通常测试只需要看几条轨迹)
    # 假设 loader 返回的是 (B, T, C, H, W)
    batch_data, batch_params = next(iter(test_loader))
    batch_data, batch_params = batch_data.to(device), batch_params.to(device)
    
    B, T, C, H, W = batch_data.shape
    loss_curves = np.zeros((engine.K_max + 1, T - 1))
    
    # 存储第一个样本的轨迹用于画图: [K+1, T, C, H, W]
    viz_trajectories = [] 
    
    # 遍历不同的 Refinement 步数 (比如 k=0, k=1, ... k=K) 进行对比
    [cite_start]# [cite: 322] Trade-off between steps and performance
    for k in range(engine.K_max + 1):
        print(f"Running Rollout with k={k}...")
        
        curr_u = batch_data[:, 0] # t=0
        k_trajectory = [curr_u.cpu()] # 记录轨迹
        
        mse_list = []
        
        # [cite_start]自回归循环 [cite: 109] "unrolling the model"
        for t in range(1, T):
            gt = batch_data[:, t]
            
            # 预测
            next_u = engine.predict_next(curr_u, batch_params, steps_to_run=k)
            
            # 计算 Loss
            mse = torch.mean((next_u - gt)**2).item()
            mse_list.append(mse)
            
            # 记录第一个样本的预测
            k_trajectory.append(next_u[0].cpu())
            
            # 更新状态 (Autoregressive)
            curr_u = next_u
        
        loss_curves[k] = np.array(mse_list)
        viz_trajectories.append(torch.stack(k_trajectory, dim=0))

    # [cite_start]--- 3. 画图: Loss Curve [cite: 966] ---
    plt.figure(figsize=(10, 6))
    steps = np.arange(1, T)
    for k in range(engine.K_max + 1):
        label = f"MSE Baseline (k=0)" if k == 0 else f"Refiner (k={k})"
        plt.plot(steps, loss_curves[k], label=label, linewidth=2)
    
    plt.yscale('log')
    plt.xlabel('Time Step')
    plt.ylabel('MSE Loss (Log Scale)')
    plt.title(f'Rollout Stability Comparison ({args.taskname})')
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig(os.path.join(save_dir, "rollout_loss_comparison.png"))
    plt.close()
    
    # [cite_start]--- 4. 画图: 轨迹可视化 [cite: 94] ---
    # 选取 5 个时间点: 0, 25%, 50%, 75%, 100%
    time_indices = [0, int(T*0.25), int(T*0.5), int(T*0.75), T-1]
    
    rows = len(time_indices)
    cols = engine.K_max + 2 # GT + (K+1) predictions
    
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows), constrained_layout=True)
    
    # 获取 Ground Truth (第一个样本)
    gt_traj = batch_data[0].cpu() # (T, C, H, W)
    
    for r, t_idx in enumerate(time_indices):
        # 第一列: GT
        ax = axes[r, 0]
        ax.imshow(gt_traj[t_idx, 0], cmap='jet') # 只画 Channel 0
        if r == 0: ax.set_title("Ground Truth")
        ax.set_ylabel(f"t={t_idx}")
        ax.set_xticks([])
        ax.set_yticks([])
        
        # 后续列: 预测
        for k in range(engine.K_max + 1):
            pred_img = viz_trajectories[k][t_idx, 0] # (C, H, W) -> Channel 0
            ax = axes[r, k+1]
            ax.imshow(pred_img, cmap='jet')
            if r == 0: ax.set_title(f"k={k}")
            ax.axis('off')

    plt.savefig(os.path.join(save_dir, "rollout_viz.png"))
    print(f"Results saved to {save_dir}")


def main(args):
    # 1. 确定路径
    save_dir = Path(f"./checkpoints/{args.taskname}")
    json_path = save_dir / "args.json"
    model_path = save_dir / "model_best.pt" # 或者 checkpoint_xx.pt，看你怎么存的
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Config not found: {json_path}")
    with open(json_path, 'r') as f:
        args = SimpleNamespace(**json.load(f))
    
    print(f"Loaded config for task: {args.taskname}")
    print(f"Dataset: {args.dataset_name}, Model: {args.model_name}")
    
    # load test data
    data_dir = f"./data/{args.dataset_name}"
    test_idx = [args.total_files - 1]
    print(f"Loading Test Data from {data_dir} (Index: {test_idx})...")
    test_loader = get_whole_loader(
        data_dir=data_dir,
        dataset_type=args.dataset_name,
        file_indices=test_idx,
        batch_size=args.batch_size
    )
    
    # extract dimensions from a sample batch
    sample_batch = next(iter(test_loader))
    whole_sample, p_sample = sample_batch
    T = whole_sample.shape[1]
    in_channels = whole_sample.shape[2]
    out_channels = whole_sample.shape[2]
    param_dim = p_sample.shape[1]
    print(f"Dimensions -> In: {in_channels}, Out: {out_channels}, Param: {param_dim}")

    # load model
    print(f"Building Model ({args.model_name})...")
    model = load_model(
        model_name=args.model_name,
        num_modes=tuple(args.num_modes),
        in_channel=in_channels+out_channels,
        out_channel=out_channels,
        param_channel=param_dim,
        num_layers=args.num_layers,
        hidden_channel=args.hidden_channel
    )
    model = model.to(device)
    
    # find epoch checkpoint
    ckpts = [f for f in os.listdir(save_dir) if f.endswith('.pt')]
    if not ckpts: raise FileNotFoundError("No checkpoints found.")
    ckpts.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    last_ckpt = os.path.join(save_dir, ckpts[-1])
    print(f"Loading Weights: {last_ckpt}")
    checkpoint = torch.load(last_ckpt, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint.get('epoch', 'unknown')
    
    # eval
    engine = PDERefinerEngine(
        model, 
        refinement_steps=args.refinement_steps, 
        min_noise_std=args.min_noise_std
    )
    evaluate(args, engine, test_loader, save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--taskname', type=str, required=True, help='Name of the task folder in ./checkpoints/')
    args = parser.parse_args()
    
    main(args)