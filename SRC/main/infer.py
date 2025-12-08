import argparse
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
from tqdm import tqdm
from neuralop.losses import LpLoss

from loaders.model_loader import load_model
from loaders.data_loader import get_loader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

def visualize_10_samples(inputs, preds, gts, save_path):
    """
    画前10个样本：Input | Pred | GT | Error
    """
    inputs = inputs.detach().cpu().numpy()
    preds = preds.detach().cpu().numpy()
    gts = gts.detach().cpu().numpy()
    errors = np.abs(preds - gts)
    
    # 强制取前10个，如果不够就取全部
    num_samples = min(10, preds.shape[0])
    
    # 修改：列数从 3 变为 4，宽度适当增加
    fig, axes = plt.subplots(num_samples, 4, figsize=(13, 2.5 * num_samples))
    if num_samples == 1: axes = axes[None, :] # 处理单样本边界情况

    for i in range(num_samples):
        # 取第0个Channel进行可视化
        inp, p, g, e = inputs[i, 0], preds[i, 0], gts[i, 0], errors[i, 0]

        # 1. Input (新增)
        ax = axes[i, 0]
        im = ax.imshow(inp, cmap='jet')
        ax.set_title(f'Sample {i} - Input')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # 2. Pred (后移)
        ax = axes[i, 1]
        im = ax.imshow(p, cmap='jet')
        ax.set_title(f'Sample {i} - Pred')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # 3. GT (后移)
        ax = axes[i, 2]
        im = ax.imshow(g, cmap='jet')
        ax.set_title(f'Sample {i} - GT')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # 4. Error (后移)
        ax = axes[i, 3]
        im = ax.imshow(e, cmap='magma')
        ax.set_title(f'Sample {i} - Error')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Visualization saved to: {save_path}")

def main(args):
    # prepare
    save_dir = f"./checkpoints/{args.taskname}"
    json_path = os.path.join(save_dir, "args.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Config not found: {json_path}")
    with open(json_path, 'r') as f:
        args = SimpleNamespace(**json.load(f))

    # load test data
    data_dir = f"./data/{args.dataset_name}"
    test_idx = [args.total_files - 1]
    print(f"Loading Test Data from {data_dir} (Index: {test_idx})...")
    test_loader = get_loader(
        data_dir=data_dir,
        dataset_type=args.dataset_name,
        file_indices=test_idx,
        batch_size=args.batch_size
    )

    # extract dimensions from a sample batch
    sample_batch = next(iter(test_loader))
    (x_sample, p_sample), y_sample = sample_batch
    in_channels = x_sample.shape[1]
    out_channels = y_sample.shape[1]
    param_dim = p_sample.shape[1]
    print(f"Dimensions -> In: {in_channels}, Out: {out_channels}, Param: {param_dim}")

    # load model
    print(f"Building Model ({args.model_name})...")
    model = load_model(
        model_name=args.model_name,
        num_modes=tuple(args.num_modes),
        in_channel=in_channels,
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
    model.eval()
    total_mse = 0.0
    vis_inputs = None # 用于保存第一批数据画图
    
    loss_fn = LpLoss(d=2, p=2, reduction='mean')
    
    print("Running Inference on Full Test Set...")
    with torch.no_grad():
        for i, ((x, params), y) in tqdm(enumerate(test_loader), total=len(test_loader)):
            x, params, y = x.to(device), params.to(device), y.to(device)
            out = model(x, condition=params)
            loss = loss_fn(out, y)
            total_mse += loss.item()
            if i == 0:
                vis_inputs = (x[:10], out[:10], y[:10])
    avg_mse = total_mse / len(test_loader)
    print(f"\nResult: Average MSE on Test Set: {avg_mse:.6f}")

    # ================= 5. 展示前10个结果 =================
    if vis_inputs is not None:
        inputs, preds, gts = vis_inputs
        vis_path = os.path.join(save_dir, f"vis_result_epoch_{epoch}.png")
        visualize_10_samples(inputs, preds, gts, vis_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--taskname', type=str, required=True, help='Name of the task folder')
    args = parser.parse_args()
    
    main(args)