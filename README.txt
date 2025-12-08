================================================================================
Conditional Fourier Neural Operators (Conditional-FNO)
================================================================================

This package implements conditional Fourier Neural Operators (FNOs) for 
parameter-dependent PDE surrogate modeling. We systematically compare three 
conditioning strategies: local feature-wise modulation, global channel-wise 
scaling, and input-level concatenation.

================================================================================
INSTALLATION
================================================================================

1. Prerequisites:
   - Python 3.8 or higher
   - CUDA-capable GPU (recommended) or CPU
   - PyTorch 1.12 or higher

2. Install dependencies:
   pip install torch torchvision torchaudio
   pip install neuralop
   pip install torchdiffeq
   pip install scipy numpy tqdm

3. Verify installation:
   python -c "import torch; import neuralop; print('Installation successful')"

================================================================================
USAGE
================================================================================

DATA GENERATION:
---------------
Generate training data for different PDE benchmarks:

  python SRC/data_generation/random_advection.py    # Advection-Diffusion
  python SRC/data_generation/random_burger.py       # Burgers Equation
  python SRC/data_generation/random_elastic.py      # Elastic Waves
  python SRC/data_generation/random_NS.py           # Navier-Stokes

Generated data will be saved in ./data/{dataset_name}/ directory as .pt files.

TRAINING:
---------
Train a conditional FNO model:

  python SRC/main/train.py \
    --taskname my_experiment \
    --dataset_name elastic \
    --total_files 11 \
    --model_name CFNO \
    --num_modes 12 12 \
    --hidden_channel 32 \
    --num_layers 6 \
    --batch_size 64 \
    --learning_rate 1e-3 \
    --epochs 1000

Available models: CFNO (Conditional FNO), FNO (Vanilla FNO), 
                  DFNO (DeepONet-FNO), noFNO (Input-level concatenation)

For series prediction with refinement:

  python SRC/main/train_series.py \
    --taskname my_series_experiment \
    --dataset_name ns \
    --refinement_steps 4 \
    --epochs 2000

INFERENCE:
---------
Run inference on trained models:

  python SRC/main/infer.py \
    --taskname my_experiment \
    --batch_size 64

For series inference:

  python SRC/main/infer_series.py \
    --taskname my_series_experiment \
    --batch_size 64

Checkpoints and results will be saved in ./checkpoints/{taskname}/ directory.

================================================================================
DEMO
================================================================================

Quick start example (Elastic Waves with CFNO):

1. Generate data:
   python SRC/data_generation/random_elastic.py

2. Train model:
   python SRC/main/train.py \
     --taskname demo_elastic \
     --dataset_name elastic \
     --total_files 11 \
     --model_name CFNO \
     --num_modes 12 12 \
     --epochs 500

3. Evaluate:
   python SRC/main/infer.py --taskname demo_elastic

Results will include visualizations and error metrics in the checkpoint directory.

================================================================================
DIRECTORY STRUCTURE
================================================================================

SRC/
  data_generation/    - Scripts for generating PDE benchmark data
  main/
    model/            - Model architectures (CFNO, FNO, etc.)
    loaders/          - Data and model loading utilities
    train.py          - Single-step training script
    train_series.py   - Series prediction training with refinement
    infer.py          - Single-step inference script
    infer_series.py   - Series inference script

DOC/
  CSE598_Conditional_FNO.pdf  - Project write-up
  [presentation slides PDF]   - Presentation slides

================================================================================
CONTACT
================================================================================

For questions or issues, please refer to the project webpage or GitHub repository.

================================================================================

