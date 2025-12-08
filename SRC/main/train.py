import argparse
import os
import torch
import time
import json
from tqdm import tqdm
from neuralop.training import AdamW
from neuralop.losses import LpLoss
from torch.optim.lr_scheduler import CosineAnnealingLR

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
    loss_fn,
    save_dir,
    epochs,
    save_interval=50
):  
    # Training Loop
    print(f"\n--- Engine Started. Checkpointing every {save_interval} epochs ---")
    for epoch in range(epochs):
        # prepare
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        # train
        t0 = time.time()
        for (x, params), y in pbar:
            x, params, y = x.to(device), params.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x, params)
            loss = loss_fn(out, y)
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
            test_loss = 0.0
            with torch.no_grad():
                for (x, params), y in test_loader:
                    x, params, y = x.to(device), params.to(device), y.to(device)
                    out = model(x, condition=params)
                    test_loss += loss_fn(out, y).item()
            test_loss /= len(test_loader)

            print(f"Epoch {epoch+1}: Train={train_loss:.5f} | Test={test_loss:.5f} | LR={curr_lr:.6f} | Time={train_time:.1f}s")
            ckpt_path = os.path.join(save_dir, f"checkpoint_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': test_loss,
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

    # extract dimensions from a sample batch
    sample_batch = next(iter(train_loader))
    (x_sample, p_sample), y_sample = sample_batch
    in_channels = x_sample.shape[1]   # [B, C, H, W] -> C
    out_channels = y_sample.shape[1]  # [B, C, H, W] -> C
    param_dim = p_sample.shape[1]     # [B, Param_Dim] -> Param_Dim
    print(f"Detected Dimensions -> In: {in_channels}, Out: {out_channels}, Param Condition: {param_dim}")

    # 3. 模型构建 (Model Initialization)
    print(f"--- Building Model ({args.model_name}) ---")
    
    model = load_model(
        model_name=args.model_name,
        num_modes=tuple(args.num_modes), # e.g. (12, 12)
        in_channel=in_channels,
        out_channel=out_channels,
        param_channel=param_dim,
        num_layers=args.num_layers,
        hidden_channel=args.hidden_channel
    )
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    loss_fn = LpLoss(d=2, p=2, reduction='mean')

    trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        save_dir=save_dir,
        epochs=args.epochs
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Dataset Args
    parser.add_argument('--taskname', type=str, default='noFNO_elastic_4000')
    parser.add_argument('--dataset_name', type=str, default='elastic', help='Prefix of files (ns, elastic)')
    parser.add_argument('--total_files', type=int, default=11, help='Total number of .pt files for dataset')
    
    # Model Args
    parser.add_argument('--model_name', type=str, default='noFNO', choices=['CFNO', 'FNO', 'DFNO', 'noFNO'])
    parser.add_argument('--num_modes', type=int, nargs='+', default=[12, 12], help='Modes for FNO (e.g. 12 12)')
    parser.add_argument('--hidden_channel', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=6)
    
    # Training Args
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=1000)
    
    args = parser.parse_args()
    
    main(args)