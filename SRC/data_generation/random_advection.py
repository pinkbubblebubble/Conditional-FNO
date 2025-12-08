import torch
import numpy as np
from torchdiffeq import odeint
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

LX, LY = 2.0, 2.0
NX, NY = 64, 64
NU = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AdvectionODE(torch.nn.Module):
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

    def forward(self, t, u_hat, cx, cy):
        diff_term = NU * self.LAPLACIAN_K * u_hat
        adv_term = - (cx * self.GRAD_X_K + cy * self.GRAD_Y_K) * u_hat
        s = 0.2 * torch.sin(3 * np.pi * self.X_GRID) * torch.cos(3 * np.pi * self.Y_GRID) * torch.cos(4 * np.pi * t)
        s_hat = torch.fft.rfft2(s)
        return diff_term + adv_term + s_hat

if __name__ == "__main__":
    # hyperparameters
    OUTPUT_DIR = "./"
    T = 10.0
    accuracy = 1 / 512
    t_eval = np.arange(0, T + accuracy, accuracy)
    time_steps = len(t_eval)
    t_eval = torch.tensor(t_eval, dtype=torch.float32, device=DEVICE)
    
    SAMPLES_PER_BATCH = 10
    TOTAL_BATCHES = 1
    
    print(f"Time steps: {time_steps}")
    print(f"Total batches to generate: {TOTAL_BATCHES}")
    print(f"Samples per batch: {SAMPLES_PER_BATCH}")

    func = AdvectionODE()

    for i in range(TOTAL_BATCHES):
        current_seed = 42 + i
        rng = np.random.default_rng(seed=current_seed)
        
        print(f"\n--- Batch {i} (Seed: {current_seed}) ---")
        
        U_list = []
        PARAMS_list = []

        for _ in tqdm(range(SAMPLES_PER_BATCH), desc=f"Generating Batch {i}"):
            # sample parameters
            cx_val = rng.uniform(-1.0, 1.0)
            cy_val = rng.uniform(-1.0, 1.0)
            PARAMS_list.append([cx_val, cy_val])

            # initial conditions (single sample)
            U0 = rng.standard_normal((1, NY, NX))
            U0 = gaussian_filter(U0, sigma=(0, 3.0, 3.0))
            U0 = torch.tensor(U0, dtype=torch.float32, device=DEVICE)
            U0_hat = torch.fft.rfft2(U0)
            ode_wrapper = lambda t, y: func(t, y, cx_val, cy_val)
            U_hat = odeint(ode_wrapper, U0_hat, t_eval, method='rk4', options={'step_size': accuracy})
            U = torch.fft.irfft2(U_hat, s=(NY, NX))
            U_list.append(U.cpu())

        U = torch.stack(U_list, dim=0).permute(0, 2, 1, 3, 4)
        PARAMS = torch.tensor(PARAMS_list, dtype=torch.float32)

        U_filename = f"{OUTPUT_DIR}/advection_U_random_{i}.pt"
        PARAMS_filename = f"{OUTPUT_DIR}/advection_PARAMS_{i}.pt"
        
        print(f"Batch {i} - Final U shape: {U.shape}")
        print(f"Batch {i} - Final PARAMS shape: {PARAMS.shape}")
        
        torch.save(U, U_filename)
        torch.save(PARAMS, PARAMS_filename)
        print(f"Successfully saved batch {i} to {U_filename} and {PARAMS_filename}")