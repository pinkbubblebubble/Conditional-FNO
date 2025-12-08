import torch
import numpy as np
from torchdiffeq import odeint
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# Hyperparameters
LX, LY = 1.0, 1.0
NX, NY = 64, 64
NU = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NavierStokesODE(torch.nn.Module):
    def __init__(self, forcing_k=4):
        super().__init__()
        x = torch.linspace(0, LX, NX, device=DEVICE)
        y = torch.linspace(0, LY, NY, device=DEVICE)
        X_GRID, Y_GRID = torch.meshgrid(x, y, indexing='xy')
        kx = 2 * np.pi * torch.fft.rfftfreq(NX, d=LX/NX, device=DEVICE)
        ky = 2 * np.pi * torch.fft.fftfreq(NY, d=LY/NY, device=DEVICE)
        KX, KY = torch.meshgrid(kx, ky, indexing='xy')
        K_SQUARED = KX**2 + KY**2
        LAPLACIAN_K = -K_SQUARED
        INV_LAPLACIAN_K = -1.0 / (K_SQUARED + 1e-8)
        INV_LAPLACIAN_K[0, 0] = 0.0
        
        k_phys = 2 * np.pi * forcing_k
        term_y = k_phys * torch.cos(k_phys * X_GRID)
        term_x = k_phys * torch.cos(k_phys * Y_GRID)
        
        self.register_buffer('X_GRID', X_GRID)
        self.register_buffer('Y_GRID', Y_GRID)
        self.register_buffer('LAPLACIAN_K', LAPLACIAN_K)
        self.register_buffer('INV_LAPLACIAN_K', INV_LAPLACIAN_K)
        self.register_buffer('GRAD_X_K', 1j * KX)
        self.register_buffer('GRAD_Y_K', 1j * KY)
        self.register_buffer('TERM_Y_HAT', torch.fft.rfft2(term_y))
        self.register_buffer('TERM_X_HAT', torch.fft.rfft2(term_x))

    def forward(self, t, omega_hat, coef_x, coef_y):
        psi_hat = -omega_hat * self.INV_LAPLACIAN_K
        u_hat = self.GRAD_Y_K * psi_hat
        v_hat = -self.GRAD_X_K * psi_hat
        omega_x_hat = self.GRAD_X_K * omega_hat
        omega_y_hat = self.GRAD_Y_K * omega_hat
        u = torch.fft.irfft2(u_hat, s=(NY, NX))
        v = torch.fft.irfft2(v_hat, s=(NY, NX))
        omega_x = torch.fft.irfft2(omega_x_hat, s=(NY, NX))
        omega_y = torch.fft.irfft2(omega_y_hat, s=(NY, NX))
        adv = u * omega_x + v * omega_y
        adv_hat = torch.fft.rfft2(adv)
        forcing_hat = coef_y * self.TERM_Y_HAT - coef_x * self.TERM_X_HAT
        return NU * self.LAPLACIAN_K * omega_hat - adv_hat + forcing_hat

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

    func = NavierStokesODE()

    for i in range(TOTAL_BATCHES):
        current_seed = 42 + i
        rng = np.random.default_rng(seed=current_seed)
        
        print(f"\n--- Batch {i} (Seed: {current_seed}) ---")
        
        U_list = []
        PARAMS_list = []

        for _ in tqdm(range(SAMPLES_PER_BATCH), desc=f"Generating Batch {i}"):
            coef_x = rng.uniform(0.0, 2.0)
            coef_y = rng.uniform(0.0, 2.0)
            PARAMS_list.append([coef_x, coef_y])

            # initial conditions (single sample)
            U0 = rng.standard_normal((1, NY, NX))
            U0 = gaussian_filter(U0, sigma=(0, 3.0, 3.0))
            U0 = torch.tensor(U0, dtype=torch.float32, device=DEVICE) * 100

            U0_hat = torch.fft.rfft2(U0)
            ode_wrapper = lambda t, y: func(t, y, coef_x, coef_y)
            
            U_hat = odeint(ode_wrapper, U0_hat, t_eval, method='rk4', options={'step_size': accuracy})
            U = torch.fft.irfft2(U_hat, s=(NY, NX))
            U_list.append(U.cpu())


        U = torch.stack(U_list, dim=0).permute(0, 2, 1, 3, 4)
        PARAMS = torch.tensor(PARAMS_list, dtype=torch.float32)

        U_filename = f"{OUTPUT_DIR}/ns_U_random_{i}.pt"
        PARAMS_filename = f"{OUTPUT_DIR}/ns_PARAMS_{i}.pt"
        
        print(f"Batch {i} - Final U shape: {U.shape}")
        print(f"Batch {i} - Final PARAMS shape: {PARAMS.shape}")
        
        torch.save(U, U_filename)
        torch.save(PARAMS, PARAMS_filename)
        print(f"Successfully saved batch {i} to {U_filename} and {PARAMS_filename}")