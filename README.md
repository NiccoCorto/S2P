# S2P — Speech-to-Head-Pose

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Wav2Vec2](https://img.shields.io/badge/Wav2Vec2-facebook%2Fwav2vec2--base--960h-blue)
![License](https://img.shields.io/badge/license-MIT-green)

> **Research project** — MICC (Media Integration and Communication Center), Università degli Studi di Firenze. Remote collaboration, A.A. 2025/2026.

---

## Overview

**S2P** is a research project for **audio-driven head pose prediction**: given a raw speech waveform, the model predicts the temporal sequence of 3D head rotations (Pitch, Yaw, Roll) frame by frame.

The core pipeline follows a deterministic approach:

```
Raw Audio (WAV) → Wav2Vec2 (frozen) → Bidirectional LSTM → FC → (Pitch, Yaw, Roll) per frame
```

The project explores several architectural variants and loss functions across multiple experiments, systematically investigating the **One-to-Many problem** — a fundamental challenge in audio-driven head motion synthesis where the same phonetic content can correspond to arbitrarily different head movements.

---

## Architecture

The model (`HeadPosePredictor`) is composed of:

| Component | Details |
|---|---|
| **Audio Encoder** | `facebook/wav2vec2-base-960h` — fully frozen (no fine-tuning) |
| **Layer Norm** | Applied on the 768-dim Wav2Vec2 output features |
| **Temporal Alignment** | Linear interpolation to match pose frame rate |
| **Sequence Model** | Bidirectional LSTM (hidden: 256, layers: 2) |
| **Output** | FC layer → 3 values (Pitch, Yaw, Roll) as axis-angle vectors |
| **Speaker Conditioning** | Optional One-Hot encoding (587 speakers from HDTF training set) |

Predicted rotations are represented as **axis-angle vectors**, converted to rotation matrices via the **Rodrigues formula** for rendering and loss computation.

---

## Datasets

| Dataset | Split | Samples | Notes |
|---|---|---|---|
| **HDTF** | Train / Val | ~587 unique speakers | Primary training dataset |
| **MEAD-EMOTE** | Test | ~2,062 clips (6 subjects) | Held-out quantitative evaluation |

---

## Experiments

The project is organized as a series of progressive experiments, each investigating a specific architectural or loss design choice:

| Exp | Name | Key Change | Main Finding |
|---|---|---|---|
| EXP4 | Baseline LSTM | Standard MSE on angles | Regression to mean, static output |
| EXP5 | Architecture Search | 2–4 LSTM layers, hidden 256/512 | No significant gain from depth alone |
| EXP6 | Overfitting Study | Deliberate overfit on small set | Confirms the model can learn dynamics if forced |
| EXP7–8 | Velocity Loss | Angular velocity regularization | Reduced jitter; staticness persists |
| EXP9 | One-Hot Conditioning | Speaker identity as conditioning signal | Identity acts as static bias, not dynamic style |
| EXP10 | FaceLoss | MSE on 3D face vertices (FLAME topology) instead of raw angles | More accurate spatial mean; dynamic variance still collapsed |
| EXP11–12 | VarLoss | Temporal variance penalty on 3D vertices | Partial improvement; stochasticity requires generative models |

> **Key takeaway:** L2-based deterministic models inevitably collapse to the conditional mean on stochastic one-to-many mappings. Solving this requires generative approaches (VAE, Diffusion Models) or adversarial losses (GAN).

---

## Loss Functions

| Loss | Description |
|---|---|
| **PosLoss** | MSE on predicted vs. ground-truth angles (Pitch, Yaw, Roll) |
| **VelLoss** | MSE on angular velocity (inter-frame difference) — reduces jitter |
| **FaceLoss** | MSE on 3D vertex positions (Forward Kinematics on FLAME canonical mesh) — geometrically meaningful gradients |
| **MeshVelLoss** | MSE on inter-frame 3D vertex displacement — visual jitter reduction |
| **VarLoss** | Penalizes mismatch between predicted and GT temporal variance on 3D vertices — encourages dynamic output |

---

## Evaluation Metrics

| Metric | Unit | Description |
|---|---|---|
| **MAE (Pitch/Yaw/Roll)** | degrees | Mean Absolute Error on individual rotation axes |
| **MAE Total** | degrees | Average MAE across all three axes |
| **MVE** | mm | Mean Vertex Error — average 3D distance of predicted vs. GT face vertices |
| **VelMAE** | rad/frame | Temporal smoothness error on rotation velocity |
| **VelMesh** | m/frame | Temporal smoothness error on vertex displacement |

---

## Project Structure

```
S2P/
├── Audio2Pose/
│   ├── model.py              # HeadPosePredictor (Wav2Vec2 + BiLSTM)
│   ├── train.py              # Training loop (FaceLoss, VarLoss, One-Hot, Comet ML)
│   ├── evaluate.py           # Per-sample qualitative evaluation
│   ├── render_comparison.py  # Side-by-side GT vs predicted rendering
│   ├── geometry.py           # Rodrigues formula (axis-angle → rotation matrix)
│   └── utils.py              # Shared utilities
├── config.py                 # Centralized hyperparameters (argparse)
├── data_loader.py            # Dataset loading, preprocessing, batching
├── eval_hdtf_testset.py      # Quantitative evaluation on HDTF test set
├── eval_mead_testset.py      # Quantitative evaluation on MEAD-EMOTE test set
├── render_no_scantalk.py     # Rendering pipeline (pure rotation, no ScanTalk dependency)
├── speaker_mapping.json      # HDTF speaker ID → One-Hot index (587 speakers)
├── canonical_face.npy        # FLAME canonical face template (5023 vertices)
├── Results/
│   ├── EXP4/ … EXP12/        # Per-experiment predictions, metrics, READMEs
│   └── Presentazione/        # Summary materials
├── Saves/                    # Model checkpoints (.pth)
├── Logs/                     # Training CSV logs
└── run_experiments.sh        # Shell scripts for launching experiment batches
```

---

## Prerequisites

- Python 3.10+
- PyTorch 2.x with CUDA support (recommended)
- `transformers` (HuggingFace) — for Wav2Vec2
- `librosa`, `soundfile` — audio I/O
- `numpy`, `tqdm`
- `comet_ml` — experiment tracking (optional, required only for training)
- `open3d` or equivalent — for rendering (optional)

Install dependencies:
```bash
pip install torch torchvision torchaudio
pip install transformers librosa soundfile tqdm comet_ml numpy
```

---

## Quick Start

### Training

```bash
# Training on HDTF (SSH/remote server)
python Audio2Pose/train.py \
  --mode ssh \
  --exp_name my_experiment \
  --hidden_dim 256 \
  --num_layers 2 \
  --max_epoch 200 \
  --lr 0.0001 \
  --var_loss_weight 1.0

# Local quick test (limited data)
python Audio2Pose/train.py \
  --mode local \
  --max_samples 50 \
  --max_epoch 5
```

### Evaluation (MEAD-EMOTE test set)

```bash
python eval_mead_testset.py
# or for a specific experiment:
python eval_mead_testset.py --exp EXP4-A
# force recompute even if metrics.json already exists:
python eval_mead_testset.py --force
```

### Resume from checkpoint

```bash
python Audio2Pose/train.py \
  --resume_checkpoint Saves/EXP10/FaceLoss/epoch_150.pth \
  --comet_experiment_key <comet_key>
```

---

## Key Config Parameters

| Parameter | Default | Description |
|---|---|---|
| `--hidden_dim` | 256 | LSTM hidden size |
| `--num_layers` | 2 | Number of LSTM layers |
| `--lr` | 1e-4 | Learning rate (fixed, no scheduler) |
| `--max_epoch` | 200 | Max training epochs |
| `--vel_loss_weight` | 0.0 | Weight for angular velocity loss |
| `--var_loss_weight` | 1.0 | Weight for temporal variance loss (VarLoss) |
| `--kld_weight` | 0.001 | KL divergence weight (for VAE variants) |
| `--anneal_epochs` | 10 | KLD annealing epochs |
| `--cache_data` | False | Cache preprocessed data to `.pkl` |

---

## Acknowledgements

- Architecture inspired by [Speech2Land (s2l-s2d)](https://github.com/s2l-s2d).
- Audio features extracted via [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base-960h) (Meta AI).
- Face template based on the [FLAME](https://flame.is.tue.mpg.de/) head model topology.
- Datasets: [HDTF](https://github.com/MRzzm/HDTF), [MEAD](https://wywu.github.io/projects/MEAD/MEAD.html).

---

## Author

Developed as a research internship project at **MICC — Media Integration and Communication Center**, Università degli Studi di Firenze, A.A. 2025/2026 (remote collaboration).
