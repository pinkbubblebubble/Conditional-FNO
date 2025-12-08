import torch
import numpy as np
from torchdiffeq import odeint
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# Hyperparameters
LX, LY = 2.0, 2.0
NX, NY = 64, 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ElasticWaveODE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        x = torch.linspace(0, LX, NX, device=DEVICE)
        y = torch.linspace(0, LY, NY, device=DEVICE)
        X_GRID, Y_GRID = torch.meshgrid(x, y, indexing='xy')
        kx = 2 * np.pi * torch.fft.rfftfreq(NX, d=LX/NX, device=DEVICE)
        ky = 2 * np.pi * torch.fft.fftfreq(NY, d=LY/NY, device=DEVICE)
        KX, KY = torch.meshgrid(kx, ky, indexing='xy')
        self.register_buffer('X_GRID', X_GRID)
        self.register_buffer('Y_GRID', Y_GRID)
        self.register_buffer('LAPLACIAN_K', -(KX**2 + KY**2))
        self.register_buffer('GRAD_X_K', 1j * KX)
        self.register_buffer('GRAD_Y_K', 1j * KY)

    def forward(self, t, w_hat, cp, cs):
        u_hat_x = w_hat[0]
        u_hat_y = w_hat[1]
        v_hat_x = w_hat[2]
        v_hat_y = w_hat[3]
        div_u_hat = self.GRAD_X_K * u_hat_x + self.GRAD_Y_K * u_hat_y
        grad_div_x_hat = self.GRAD_X_K * div_u_hat
        grad_div_y_hat = self.GRAD_Y_K * div_u_hat
        lap_u_x_hat = self.LAPLACIAN_K * u_hat_x
        lap_u_y_hat = self.LAPLACIAN_K * u_hat_y
        A = cp**2 - cs**2
        B = cs**2
        acc_x_hat = A * grad_div_x_hat + B * lap_u_x_hat
        acc_y_hat = A * grad_div_y_hat + B * lap_u_y_hat
        dw_dt_hat = torch.stack([v_hat_x, v_hat_y, acc_x_hat, acc_y_hat], dim=0)
        
        return dw_dt_hat


if __name__ == "__main__":
    # hyperparameters
    OUTPUT_DIR = "./"
    T = 10.0
    accuracy = 1 / 256
    t_eval = np.arange(0, T + accuracy, accuracy)
    time_steps = len(t_eval)
    t_eval = torch.tensor(t_eval, dtype=torch.float32, device=DEVICE)

    SAMPLES_PER_BATCH = 10
    TOTAL_BATCHES = 1
    
    print(f"Time steps: {time_steps}")
    print(f"Total batches to generate: {TOTAL_BATCHES}")
    print(f"Samples per batch: {SAMPLES_PER_BATCH}")

    func = ElasticWaveODE()
    
    for i in range(TOTAL_BATCHES):
        current_seed = 42 + i
        rng = np.random.default_rng(seed=current_seed)
        
        print(f"\n--- Batch {i} (Seed: {current_seed}) ---")
        
        U_list = []
        PARAMS_list = []


        # generation
        for _ in tqdm(range(SAMPLES_PER_BATCH), desc=f"Generating Batch {i}"):
            cp_val = rng.uniform(0.0, 1.0)
            cs_val = rng.uniform(0.0, 1.0)
            PARAMS_list.append([cp_val, cs_val])

            # initial conditions (single sample)
            U0 = np.zeros((4, NY, NX))
            x = np.linspace(-1, 1, NX)
            y = np.linspace(-1, 1, NY)
            X, Y = np.meshgrid(x, y)
            
            num_pulses = rng.integers(1, 4)
            total_pulse = np.zeros_like(X)
            
            for _ in range(num_pulses):
                cx = rng.uniform(-1.0, 1.0)
                cy = rng.uniform(-1.0, 1.0)
                sigma = rng.uniform(0.1, 0.25)
                amplitude = rng.uniform(0.1, 0.5)
                dist_sq = (X - cx)**2 + (Y - cy)**2
                pulse = amplitude * np.exp(-dist_sq / (2 * sigma**2))
                total_pulse += pulse

            U0[0, :, :] = total_pulse  # ux
            U0[1, :, :] = total_pulse  # uy
            U0[2:4] = 0.0
            U0 = torch.tensor(U0, dtype=torch.float32, device=DEVICE)

            U0_hat = torch.fft.rfft2(U0)
            ode_wrapper = lambda t, y: func(t, y, cp_val, cs_val)
            U_hat = odeint(ode_wrapper, U0_hat, t_eval, method='rk4', options={'step_size': accuracy})
            U = torch.fft.irfft2(U_hat, s=(NY, NX))
            U_list.append(U[:, 0:2].cpu())

        U = torch.stack(U_list, dim=0).permute(0, 2, 1, 3, 4) # (B, T, C, H, W)
        PARAMS = torch.tensor(PARAMS_list, dtype=torch.float32) # (B, num_params)

        U_filename = f"{OUTPUT_DIR}/elastic_U_random_{i}.pt"
        PARAMS_filename = f"{OUTPUT_DIR}/elastic_PARAMS_{i}.pt"
        
        print(f"Batch {i} - Final U shape: {U.shape}")
        print(f"Batch {i} - Final PARAMS shape: {PARAMS.shape}")
        
        torch.save(U, U_filename)
        torch.save(PARAMS, PARAMS_filename)
        print(f"Successfully saved batch {i} to {U_filename} and {PARAMS_filename}")