import argparse
import os
import torch
import time
import json
import math
import torch.nn as nn
from tqdm import tqdm
from neuralop.training import AdamW
from neuralop.losses import LpLoss
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from loaders.model_loader import load_model
from loaders.data_loader import get_loader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

def trainer(
    model,
    train_loader,
    test_loader,
    optimizer,
    scheduler,
    loss_fn_signal,
    loss_fn_noise,
    save_dir,
    epochs,
    refinement_steps,
    min_noise_std,
    save_interval=50
):  
    # Training Loop
    num_phases = refinement_steps + 1
    print(f"\n--- PDE-Refiner Engine Started. K={refinement_steps}, Sigma_min={min_noise_std} ---")
    print(f"--- Curriculum Learning Active: Split into {num_phases} Phases ---")

    # sigma table
    sigma_table = torch.zeros(refinement_steps + 1, device=device)
    for k_idx in range(1, refinement_steps + 1):
        sigma_table[k_idx] = min_noise_std ** (k_idx / refinement_steps)
    
    for epoch in range(epochs):
        # prepare
        model.train()
        train_loss = 0.0
        phase_idx = int((epoch / epochs) * num_phases)
        phase_idx = min(phase_idx, refinement_steps)
        desc_str = f"Ep {epoch+1}/{epochs} [Phase {phase_idx}/{refinement_steps}: k<= {phase_idx}]"
        pbar = tqdm(train_loader, desc=desc_str, leave=False)

        # train
        t0 = time.time()
        for (x, params), y in pbar:
            # x: u(t-1) condition
            # y: u(t) ground truth
            # params: PDE parameters (viscosity etc.)
            x, params, y = x.to(device), params.to(device), y.to(device)
            batch_size = x.shape[0]
            
            # --- PDE-Refiner Training Logic [cite: 727-736] ---
            
            # 1. 均匀采样 k ~ U(0, K) 
            # 为了便于Batch处理，这里对整个Batch使用相同的k，或者也可以对每个样本独立采样
            k = torch.randint(0, phase_idx + 1, (1,)).item()
            
            # 将 k 归一化并拼接到条件参数中，以便模型感知当前的细化步骤
            # params shape: [B, param_dim] -> [B, param_dim + 1]
            k_tensor = torch.full((batch_size, 1), k / refinement_steps, device=device)
            cond_params = torch.cat([params, k_tensor], dim=1)

            optimizer.zero_grad()

            if k == 0:
                current_estimate = torch.zeros_like(y)
                model_input = torch.cat([x, current_estimate], dim=1)
                out = model(model_input, cond_params)
                loss = loss_fn_signal(out, y)
            
            else:
                # --- Step k>0: 去噪训练  ---
                sigma_k = sigma_table[k]
                noise = torch.randn_like(y)
                y_noised = y + sigma_k * noise
                model_input = torch.cat([x, y_noised], dim=1)
                pred_noise = model(model_input, cond_params)
                loss = loss_fn_noise(pred_noise, noise)

            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
        
        train_time = time.time() - t0
        train_loss /= len(train_loader)
        
        scheduler.step()
        curr_lr = scheduler.get_last_lr()[0]

        # evaluate
        if (epoch + 1) % save_interval == 0:
            model.eval()
            k_loss_sums = {k: 0.0 for k in range(refinement_steps + 1)}
            
            with torch.no_grad():
                for (x, params), y in test_loader:
                    x, params, y = x.to(device), params.to(device), y.to(device)
                    batch_size = x.shape[0]
                    for k in range(refinement_steps + 1):
                        k_tensor = torch.full((batch_size, 1), k / refinement_steps, device=device)
                        cond_params = torch.cat([params, k_tensor], dim=1)
                        if k == 0:
                            current_estimate = torch.zeros_like(y)
                            model_input = torch.cat([x, current_estimate], dim=1)
                            out_signal = model(model_input, cond_params)
                            loss_val = loss_fn_signal(out_signal, y).item()
                            k_loss_sums[k] += loss_val  
                        else:
                            sigma_k = min_noise_std ** (k / refinement_steps)
                            noise = torch.randn_like(y)
                            y_noised = y + sigma_k * noise
                            model_input = torch.cat([x, y_noised], dim=1)
                            pred_noise = model(model_input, cond_params)
                            loss_val = loss_fn_noise(pred_noise, noise).item()
                            k_loss_sums[k] += loss_val

            num_batches = len(test_loader)
            log_strings = []
            avg_k0_loss = 0.0
            for k in range(refinement_steps + 1):
                avg_loss = k_loss_sums[k] / num_batches
                marker = "" if k <= phase_idx else "*"
                if k == 0:
                    avg_k0_loss = avg_loss
                    log_strings.append(f"k0={avg_loss:.5f}")
                else:
                    log_strings.append(f"k{k}{marker}={avg_loss:.5f}")
            loss_detail_str = " | ".join(log_strings)
            print(f"Epoch {epoch+1}: Train={train_loss:.5f} | LR={curr_lr:.6f} | Time={train_time:.1f}s")
            print(f"    >> Eval Details: {loss_detail_str}")

            ckpt_path = os.path.join(save_dir, f"checkpoint_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_k0_loss,
                'k_losses': {k: v/num_batches for k, v in k_loss_sums.items()}
            }, ckpt_path)
        else:
            print(f"Epoch {epoch+1}: Train={train_loss:.5f} | LR={curr_lr:.6f} | Time={train_time:.1f}s")


def main(args):
    # prepare directories
    save_dir = f"./checkpoints/{args.taskname}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        raise FileExistsError(f"Save directory {save_dir} already exists. Please choose a different taskname.")
    with open(os.path.join(save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    data_dir = f"./data/{args.dataset_name}"
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory {data_dir} does not exist.")

    # loading data
    print("--- Preparing Data ---")
    train_indices = list(range(args.total_files - 1))
    test_indices = [args.total_files - 1]
    train_loader = get_loader(
        data_dir=data_dir,
        dataset_type=args.dataset_name,
        file_indices=train_indices,
        batch_size=args.batch_size
    )
    test_loader = get_loader(
        data_dir=data_dir,
        dataset_type=args.dataset_name,
        file_indices=test_indices,
        batch_size=args.batch_size
    )

    # extract dimensions
    sample_batch = next(iter(train_loader))
    (x_sample, p_sample), y_sample = sample_batch
    raw_in_channels = x_sample.shape[1]
    out_channels = y_sample.shape[1]
    model_in_channels = raw_in_channels + out_channels
    param_dim = p_sample.shape[1] + 1
    print(f"Detected Dims -> Raw In: {raw_in_channels}, Model In (Concat): {model_in_channels}, Out: {out_channels}, Param Condition(+k): {param_dim}")

    # 3. 模型构建
    print(f"--- Building Model ({args.model_name}) ---")
    model = load_model(
        model_name=args.model_name,
        num_modes=tuple(args.num_modes), 
        in_channel=model_in_channels,
        out_channel=out_channels,
        param_channel=param_dim,
        num_layers=args.num_layers,
        hidden_channel=args.hidden_channel
    )
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    steps_per_phase = args.epochs // (args.refinement_steps + 1)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=steps_per_phase, 
        T_mult=1, 
        eta_min=1e-6
    )
    loss_fn_signal = LpLoss(d=2, p=2, reduction='mean')
    loss_fn_noise = nn.MSELoss() 

    trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn_signal=loss_fn_signal,
        loss_fn_noise=loss_fn_noise,
        save_dir=save_dir,
        epochs=args.epochs,
        refinement_steps=args.refinement_steps,
        min_noise_std=args.min_noise_std
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Dataset Args
    parser.add_argument('--taskname', type=str, default='burger_series')
    parser.add_argument('--dataset_name', type=str, default='burger', help='Prefix of files (ns, elastic)')
    parser.add_argument('--total_files', type=int, default=5, help='Total number of .pt files for dataset')
    
    # Model Args
    parser.add_argument('--model_name', type=str, default='CFNO', choices=['CFNO', 'FNO', 'DFNO'])
    parser.add_argument('--num_modes', type=int, nargs='+', default=[12, 12], help='Modes for FNO')
    parser.add_argument('--hidden_channel', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=3)
    
    # Training Args
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=5000)
    
    # [PDE-Refiner Specific Args]
    parser.add_argument('--refinement_steps', type=int, default=4, help='Number of refinement steps K')
    parser.add_argument('--min_noise_std', type=float, default=5e-4, help='Minimum noise standard deviation sigma_min')
    
    args = parser.parse_args()
    
    main(args)