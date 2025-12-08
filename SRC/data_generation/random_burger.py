import torch
import numpy as np
from torchdiffeq import odeint
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# Hyperparameters
LX, LY = 2.0, 2.0
NX, NY = 64, 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BurgersODE(torch.nn.Module):
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

    def forward(self, t, w_hat, nu):
        u_hat = w_hat[0]
        v_hat = w_hat[1]
        ux_hat = self.GRAD_X_K * u_hat
        uy_hat = self.GRAD_Y_K * u_hat
        vx_hat = self.GRAD_X_K * v_hat
        vy_hat = self.GRAD_Y_K * v_hat
        u = torch.fft.irfft2(u_hat, s=(NY, NX))
        v = torch.fft.irfft2(v_hat, s=(NY, NX))
        ux = torch.fft.irfft2(ux_hat, s=(NY, NX))
        uy = torch.fft.irfft2(uy_hat, s=(NY, NX))
        vx = torch.fft.irfft2(vx_hat, s=(NY, NX))
        vy = torch.fft.irfft2(vy_hat, s=(NY, NX))
        adv_u = u * ux + v * uy
        adv_v = u * vx + v * vy
        adv_u_hat = torch.fft.rfft2(adv_u)
        adv_v_hat = torch.fft.rfft2(adv_v)
        du_dt_hat = nu * self.LAPLACIAN_K * u_hat - adv_u_hat
        dv_dt_hat = nu * self.LAPLACIAN_K * v_hat - adv_v_hat
        dw_dt_hat = torch.stack([du_dt_hat, dv_dt_hat], dim=0)
        return dw_dt_hat


if __name__ == "__main__":
    # hyperparameters
    OUTPUT_DIR = "./"
    T = 2.0 
    accuracy = 1 / 1024
    t_eval = np.arange(0, T + accuracy, accuracy)
    time_steps = len(t_eval)
    t_eval = torch.tensor(t_eval, dtype=torch.float32, device=DEVICE)

    SAMPLES_PER_BATCH = 10
    TOTAL_BATCHES = 1
    
    print(f"Time steps: {time_steps}")
    print(f"Total batches to generate: {TOTAL_BATCHES}")
    print(f"Samples per batch: {SAMPLES_PER_BATCH}")

    func = BurgersODE()

    for i in range(TOTAL_BATCHES):
        current_seed = 42 + i
        rng = np.random.default_rng(seed=current_seed)
        
        print(f"\n--- Batch {i} (Seed: {current_seed}) ---")
        
        U_list = []
        PARAMS_list = []

        for _ in tqdm(range(SAMPLES_PER_BATCH), desc=f"Generating Batch {i}"):
            nu_val = rng.uniform(0.01, 0.1)
            PARAMS_list.append([nu_val])

            # initial conditions (single sample)
            U0 = rng.standard_normal((2, NY, NX))
            U0 = gaussian_filter(U0, sigma=(0, 3.0, 3.0))
            U0 = torch.tensor(U0, dtype=torch.float32, device=DEVICE)

            U0_hat = torch.fft.rfft2(U0)
            ode_wrapper = lambda t, y: func(t, y, nu_val)
            U_hat = odeint(ode_wrapper, U0_hat, t_eval, method='rk4', options={'step_size': accuracy})
            U = torch.fft.irfft2(U_hat, s=(NY, NX))
            U_list.append(U.cpu())

        U = torch.stack(U_list, dim=0).permute(0, 2, 1, 3, 4)
        PARAMS = torch.tensor(PARAMS_list, dtype=torch.float32)

        U_filename = f"{OUTPUT_DIR}/burger_U_random_{i}.pt"
        PARAMS_filename = f"{OUTPUT_DIR}/burger_PARAMS_{i}.pt"
        
        print(f"Batch {i} - Final U shape: {U.shape}")
        print(f"Batch {i} - Final PARAMS shape: {PARAMS.shape}")
        
        torch.save(U, U_filename)
        torch.save(PARAMS, PARAMS_filename)
        print(f"Successfully saved batch {i} to {U_filename} and {PARAMS_filename}")