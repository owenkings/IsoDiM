# IsoDiM: Isometric Discrete Diffusion Model

**Bridging Discrete and Continuous: Topology-Aware Motion Generation with Finite Scalar Quantization**

## 🎯 Overview

IsoDiM is a text-to-motion generation model that combines:
- **FSQ (Finite Scalar Quantization)**: Creates a structured, topology-preserving discrete latent space
- **JiT (Joint in Time) Transformer**: 1D Transformer with self-attention for inter-frame interactions
- **X-Prediction**: Direct data prediction instead of velocity prediction, better suited for bounded FSQ space

## 🏗️ Architecture

```
Text → CLIP → MAR Transformer → JiT Diffusion Transformer → FSQ Codes → Tokenizer Decoder → Motion
```

### Key Components

| Component | Description |
|-----------|-------------|
| `IsoDiM_Tokenizer` | FSQ-based VAE for motion encoding/decoding |
| `IsoDiM_Transformer` | JiT-style Diffusion Transformer with adaLN |
| `IsoDiM` | Main model integrating MAR + Diffusion with Grid Snapping |

## 📦 Installation

```bash
conda env create -f environment.yml
conda activate IsoDiM
```

## 🚀 Quick Start

### 1. Train Tokenizer (FSQ-VAE)

```bash
python train_Tokenizer.py \
    --name IsoDiM_Tokenizer_High \
    --model IsoDiM_Tokenizer_High \
    --dataset_name t2m \
    --batch_size 256 \
    --epoch 50
```

### 2. Train IsoDiM Model

```bash
python train_IsoDiM.py \
    --name IsoDiM \
    --tokenizer_name IsoDiM_Tokenizer_High \
    --tokenizer_model IsoDiM_Tokenizer_High \
    --model IsoDiM-DiT-XL \
    --dataset_name t2m \
    --batch_size 64 \
    --epoch 500 \
    --need_evaluation
```

### 3. Generate Motion

```bash
python sample.py \
    --name IsoDiM \
    --tokenizer_name IsoDiM_Tokenizer_High \
    --model IsoDiM-DiT-XL \
    --text_prompt "a person walks forward and waves" \
    --motion_length 120
```

### 4. Evaluate

```bash
# Evaluate Tokenizer reconstruction quality
python evaluation_Tokenizer.py \
    --name IsoDiM_Tokenizer_High \
    --model IsoDiM_Tokenizer_High

# Evaluate generation quality
python evaluation_IsoDiM.py \
    --name IsoDiM \
    --tokenizer_name IsoDiM_Tokenizer_High \
    --model IsoDiM-DiT-XL
```

## 📊 Model Variants

### Tokenizer Models

| Model | FSQ Levels | Codebook Size | Dim | Use Case |
|-------|------------|---------------|-----|----------|
| `IsoDiM_Tokenizer_Small` | [5,5,5,5,5] | 3,125 | 5 | 🧪 快速测试 |
| `IsoDiM_Tokenizer_Medium` | [8,5,5,5,5] | 5,000 | 5 | 轻量实验 |
| `IsoDiM_Tokenizer_Large` | [8,6,6,5,5,5] | 36,000 | 6 | 常规训练 |
| `IsoDiM_Tokenizer_High` | [8,8,8,5,5,5] | 64,000 | 6 | ✅ **推荐** |
| `IsoDiM_Tokenizer_Ultra` | [8,8,8,8,5,5] | 102,400 | 6 | 高精度 |
| `IsoDiM_Tokenizer_Mega` | [8,8,8,8,8,5] | 163,840 | 6 | 超高精度 |
| `IsoDiM_Tokenizer_HighDim7` | [7,5,5,5,5,5,5] | 109,375 | 7 | 高维实验 |
| `IsoDiM_Tokenizer_HighDim8` | [5,5,5,5,5,5,5,5] | 390,625 | 8 | 高维实验 |

### IsoDiM Models

| Model | Depth | Hidden | Heads | Recommended |
|-------|-------|--------|-------|-------------|
| `IsoDiM-DiT-S` | 12 | 384 | 6 | |
| `IsoDiM-DiT-B` | 12 | 768 | 12 | |
| `IsoDiM-DiT-L` | 24 | 1024 | 16 | |
| `IsoDiM-DiT-XL` | 28 | 1152 | 16 | ✅ |

## 🔑 Key Features

### FSQ Topology Preservation
Unlike VQ-VAE where token ID 100 and 101 have no relationship, FSQ preserves metric structure:
- Adjacent grid points are numerically close
- Diffusion can exploit this topology for smoother generation

### X-Prediction vs V-Prediction
```
V-Prediction: Loss = ||model(x_t, t) - (x_1 - x_0)||²  # Predict velocity
X-Prediction: Loss = ||model(x_t, t) - x_1||²         # Predict clean data
```
X-prediction is better for bounded FSQ space [-1, 1].

### Grid Snapping
During inference, predictions are snapped to valid FSQ grid points for proper decoding.

## 📁 Project Structure

```
IsoDiM/
├── models/
│   ├── Quantizer.py      # FSQ module
│   ├── Tokenizer.py      # FSQ-VAE
│   ├── Transformer.py    # JiT Diffusion Transformer
│   ├── IsoDiM.py         # Main model
│   └── LengthEstimator.py
├── utils/
│   ├── datasets.py       # Data loading
│   ├── eval_utils.py     # Evaluation metrics
│   ├── train_utils.py    # Training utilities
│   └── motion_process.py # Motion processing
├── diffusions/           # Diffusion library
├── train_Tokenizer.py    # Tokenizer training
├── train_IsoDiM.py       # Main model training
├── evaluation_*.py       # Evaluation scripts
└── sample.py             # Inference script
```

## 📈 Expected Results

| Metric | Tokenizer (Recon) | IsoDiM (Gen) |
|--------|-------------------|--------------|
| FID | ~0.07 | <0.15 |
| R-Precision Top1 | ~0.47 | ~0.50 |
| Diversity | ~10.3 | ~9.5 |

## 📝 Citation

```bibtex
@article{isodim2024,
  title={Bridging Discrete and Continuous: Topology-Aware Motion Generation with Finite Scalar Quantization},
  author={...},
  year={2024}
}
```

## 🙏 Acknowledgments

This project builds upon:
- [MARDM](https://github.com/...) - Original masked autoregressive diffusion model
- [FSQ](https://arxiv.org/abs/...) - Finite Scalar Quantization
- [JiT](https://arxiv.org/abs/...) - Joint in Time architecture

