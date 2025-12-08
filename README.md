# Conditional Fourier Neural Operators

This package implements conditional Fourier Neural Operators (FNOs) for parameter-dependent PDE surrogate modeling. We systematically compare three conditioning strategies: **local feature-wise modulation**, **global channel-wise scaling**, and **input-level concatenation**.

## 📋 Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Data Generation](#data-generation)
  - [Training](#training)
  - [Inference](#inference)
- [Quick Start Demo](#quick-start-demo)
- [Directory Structure](#directory-structure)
- [Citation](#citation)
- [Links](#links)

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- PyTorch 1.12 or higher

### Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install neuralop
pip install torchdiffeq
pip install scipy numpy tqdm
```

### Verify Installation

```bash
python -c "import torch; import neuralop; print('Installation successful')"
```

## 🚀 Usage

### Data Generation

Generate training data for different PDE benchmarks:

```bash
# Advection-Diffusion
python SRC/data_generation/random_advection.py

# Burgers Equation
python SRC/data_generation/random_burger.py

# Elastic Waves
python SRC/data_generation/random_elastic.py

# Navier-Stokes
python SRC/data_generation/random_NS.py
```

Generated data will be saved in `./data/{dataset_name}/` directory as `.pt` files.

### Training

#### Single-step Training

Train a conditional FNO model:

```bash
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
```

**Available models:**
- `CFNO`: Conditional FNO (local feature-wise modulation)
- `FNO`: Vanilla FNO
- `DFNO`: DeepONet-FNO
- `noFNO`: Input-level concatenation

#### Series Prediction with Refinement

For multi-step prediction with PDE-Refiner:

```bash
python SRC/main/train_series.py \
  --taskname my_series_experiment \
  --dataset_name ns \
  --refinement_steps 4 \
  --epochs 2000
```

### Inference

#### Single-step Inference

```bash
python SRC/main/infer.py \
  --taskname my_experiment \
  --batch_size 64
```

#### Series Inference

```bash
python SRC/main/infer_series.py \
  --taskname my_series_experiment \
  --batch_size 64
```

Checkpoints and results will be saved in `./checkpoints/{taskname}/` directory.

## 🎯 Quick Start Demo

Quick start example (Elastic Waves with CFNO):

1. **Generate data:**
   ```bash
   python SRC/data_generation/random_elastic.py
   ```

2. **Train model:**
   ```bash
   python SRC/main/train.py \
     --taskname demo_elastic \
     --dataset_name elastic \
     --total_files 11 \
     --model_name CFNO \
     --num_modes 12 12 \
     --epochs 500
   ```

3. **Evaluate:**
   ```bash
   python SRC/main/infer.py --taskname demo_elastic
   ```

Results will include visualizations and error metrics in the checkpoint directory.

## 📁 Directory Structure

```
.
├── SRC/
│   ├── data_generation/    # Scripts for generating PDE benchmark data
│   └── main/
│       ├── model/          # Model architectures (CFNO, FNO, etc.)
│       ├── loaders/        # Data and model loading utilities
│       ├── train.py        # Single-step training script
│       ├── train_series.py # Series prediction training with refinement
│       ├── infer.py        # Single-step inference script
│       └── infer_series.py # Series inference script
├── DOC/
│   ├── CSE598_Conditional_FNO.pdf  # Project write-up
│   └── presentation.pdf             # Presentation slides
└── README.md                        # This file
```

## 📄 Citation

If you find this work useful, please cite:

```bibtex
@article{cai2025exploring,
  title={Exploring Conditioning Strategies for Fourier Neural Operators},
  author={Cai, Wenxi and Wang, Yimin},
  journal={arXiv preprint},
  year={2025}
}
```

## 🔗 Links

- **Project Webpage**: [Link to your webpage]
- **Paper**: [DOC/CSE598_Conditional_FNO.pdf](DOC/CSE598_Conditional_FNO.pdf)
- **Slides**: [DOC/presentation.pdf](DOC/presentation.pdf)
- **GitHub Repository**: [https://github.com/pinkbubblebubble/Conditional-FNO](https://github.com/pinkbubblebubble/Conditional-FNO)

## 📧 Contact

For questions or issues, please refer to the [project webpage](https://pinkbubblebubble.github.io/Conditional-FNO/) or open an issue on GitHub.

---

**Authors**: Wenxi Cai, Yimin Wang  
**Institution**: University of Michigan

