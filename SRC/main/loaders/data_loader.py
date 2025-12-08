import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, List

class NextTimeDataset(Dataset):
    def __init__(self, 
                 data_dir: str, 
                 dataset_type: str,
                 file_indices: List[int]):
        """
        Args:
            data_dir (str): 数据文件夹路径
            dataset_type (str): 数据前缀，例如 "ns" 或 "elastic"
            file_indices (List[int]): 需要加载的文件索引列表。
        """
        self.data_dir = Path(data_dir)
        self.dataset_type = dataset_type
        
        # 简单检查，防止传入空列表
        if not file_indices:
            raise ValueError("file_indices list cannot be empty.")

        print(f"Loading '{self.dataset_type}' dataset ({len(file_indices)} files) from {self.data_dir}...")

        u_tensors = []
        p_tensors = []

        for idx in file_indices:
            u_filename = f"{self.dataset_type}_U_random_{idx}.pt"
            p_filename = f"{self.dataset_type}_PARAMS_{idx}.pt"
            u_path = self.data_dir / u_filename
            p_path = self.data_dir / p_filename

            if not u_path.exists() or not p_path.exists():
                raise FileNotFoundError(f"Missing file pair: {u_path} or {p_path}")

            u_data = torch.load(u_path) # (B_local, C, T, H, W)
            print(f"Loaded {u_filename} with shape {u_data.shape}")
            p_data = torch.load(p_path) # (B_local, num_params)
            print(f"Loaded {p_filename} with shape {p_data.shape}")

            assert u_data.shape[0] == p_data.shape[0], f"Batch size mismatch in {u_filename}"

            u_tensors.append(u_data)
            p_tensors.append(p_data)

        self.data = torch.cat(u_tensors, dim=0).permute(0, 2, 1, 3, 4)
        self.params = torch.cat(p_tensors, dim=0)


        # check Nan and outliers
        flat_data = self.data.reshape(self.data.shape[0], -1)
        nan_mask_data = torch.isnan(flat_data).any(dim=1)
        outlier_mask_data = (torch.abs(flat_data) > 200).any(dim=1)
        nan_mask_params = torch.isnan(self.params.reshape(self.params.shape[0], -1)).any(dim=1)
        invalid_mask = nan_mask_data | outlier_mask_data | nan_mask_params
        if invalid_mask.any():
            num_bad = invalid_mask.sum().item()
            total = self.data.shape[0]
            print(f"Warning: Found {num_bad}/{total} trajectories containing NaN or values > 200, removing them...")
            valid_mask = ~invalid_mask
            self.data = self.data[valid_mask]
            self.params = self.params[valid_mask]
            if self.data.shape[0] == 0:
                raise ValueError("Error: All data invalid (NaN or > 200)!")


        self.B, self.T, self.C, self.H, self.W = self.data.shape
        
        if self.T < 2:
             raise ValueError(f"Time dimension T={self.T} is too short for prediction.")
             
        self.samples_per_trajectory = self.T - 1

        print(f"[{self.dataset_type.upper()}] Loaded. Shape: {self.data.shape}. Total Transitions: {self.B}")

        # 计算并应用归一化
        print(f"Original Data Stats -> Max: {self.data.max():.2f}, Min: {self.data.min():.2f}, Mean: {self.data.mean():.2f}")
        self.data_mean = torch.mean(self.data)
        self.data_std = torch.std(self.data)
        self.data = (self.data - self.data_mean) / (self.data_std + 1e-6)
        print(f"Normalization Applied -> Mean: {self.data_mean:.4f}, Std: {self.data_std:.4f}")
        print(f"Normalized Data Stats -> Max: {self.data.max():.2f}, Min: {self.data.min():.2f}")

    def __len__(self):
        return self.B

    def __getitem__(self, idx) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        b = idx 
        t = torch.randint(0, self.samples_per_trajectory, (1,)).item()
        x_state = self.data[b, t]
        p_val = self.params[b]
        y_state = self.data[b, t+1]
        return (x_state, p_val), y_state
  

class WholeTimeDataset(NextTimeDataset):
    def __getitem__(self, idx):
        sequence = self.data[idx]
        p_val = self.params[idx]
        return sequence, p_val

def get_loader(data_dir, dataset_type, file_indices, batch_size):
    # 必须传入所有参数
    dataset = NextTimeDataset(
        data_dir=data_dir, 
        dataset_type=dataset_type,
        file_indices=file_indices
    )
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def get_whole_loader(data_dir, dataset_type, file_indices, batch_size):
    dataset = WholeTimeDataset(
        data_dir=data_dir, 
        dataset_type=dataset_type,
        file_indices=file_indices
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
