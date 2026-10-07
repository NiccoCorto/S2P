# S2P — Speech-to-Head-Pose

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Wav2Vec2](https://img.shields.io/badge/Wav2Vec2-facebook%2Fwav2vec2--base--960h-blue)
![License](https://img.shields.io/badge/license-MIT-green)

> **Research project** — MICC (Media Integration and Communication Center), Università degli Studi di Firenze. Remote collaboration, A.A. 2025/2026.

---

## Demo — Ground Truth vs. Prediction
Side-by-side qualitative comparison on the test sequence `RD_AmandaStuck_000_chunk0` (16 seconds):
- **Left (GT)**: Ground Truth head pose (real human dynamics from HDTF dataset)
- **Right (Prediction)**: S2P predicted head pose (FLAME 3D mesh rendering)   

https://github.com/user-attachments/assets/c3dd8f4b-07cb-4466-a86e-51d802003b37


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

The project is organized as a series of progressive experiments, each investigating a specific architectural or loss design choice across dedicated branches:

| Exp | Branch | Name | Key Change | Main Finding |
|---|---|---|---|---|
| EXP4 | `main` | Baseline LSTM | Standard MSE on angles | Regression to mean, static output |
| EXP5 | `main` | Architecture Search | 2–4 LSTM layers, hidden 256/512 | No significant gain from depth alone |
| EXP6 | `main` | Overfitting Study | Deliberate overfit on small set | Confirms the model can learn dynamics if forced |
| EXP7–8 | `main` | Velocity Loss | Angular velocity regularization | Reduced jitter; staticness persists |
| EXP9 | `DiffPoseData` | One-Hot Conditioning | Speaker identity as conditioning signal (587 speakers) | Identity acts as static bias, not dynamic style |
| EXP10 | `FaceLoss-Approach` | FaceLoss | MSE on 3D face vertices (FLAME topology) instead of raw angles | More accurate spatial mean; dynamic variance still collapsed |
| EXP11–12 | `VarLoss` | VarLoss | Temporal variance penalty on 3D vertices | Partial improvement; stochasticity requires generative models |
| VAE Study | `One-to-many-approach` | Probabilistic VAE | Latent space $Z$, KLD loss & annealing | Posterior collapse under MSE; motivates diffusion/GANs |

> **Key takeaway:** L2-based deterministic models inevitably collapse to the conditional mean on stochastic one-to-many mappings. Solving this requires generative approaches (VAE, Diffusion Models) or adversarial losses (GAN).

---

## Repository Branches

Due to the exploratory nature of this research across multiple loss formulations and modeling paradigms, development is organized across several dedicated branches. Each branch isolates a specific methodological development:

```
main (Baseline MSE, MEAD-EMOTE eval, thesis report)
  │
  ├── FaceLoss-Approach (3D mesh vertex loss via Rodrigues FK)
  │
  └── DiffPoseData (HDTF dataset format, One-Hot speaker conditioning)
        │
        ├── One-to-many-approach (Probabilistic VAE exploration & Posterior Collapse study)
        │
        └── VarLoss (FaceLoss + One-Hot + Temporal Variance Regularization)
```

### Branch Guide

| Branch | Primary Focus | Key Novelty / Features | Associated Experiments |
|---|---|---|---|
| [`main`](https://github.com/NiccoCorto/S2P/tree/main) | **Baseline & Benchmark** | Unified evaluation pipeline on MEAD-EMOTE, thesis report (`SpeechToPose.pdf`), baseline BiLSTM | EXP4 – EXP8 |
| [`FaceLoss-Approach`](https://github.com/NiccoCorto/S2P/tree/FaceLoss-Approach) | **Geometric 3D Loss** | Replaces angular MSE with 3D Forward Kinematics on FLAME canonical face mesh (5,023 vertices) via Rodrigues formula (`geometry.py`) | EXP10 |
| [`DiffPoseData`](https://github.com/NiccoCorto/S2P/tree/DiffPoseData) | **Dataset & Speaker Conditioning** | Adaptation to HDTF dataset (DiffPose/TFHP format); speaker identity conditioning via One-Hot vectors (587 speakers); ScanTalk rendering integration | EXP9 |
| [`VarLoss`](https://github.com/NiccoCorto/S2P/tree/VarLoss) | **Variance Regularization** | Combines FaceLoss + One-Hot with temporal variance penalty (`VarLoss` / `VarLossSTD`) to prevent static pose collapse; uniform $1/N$ prior fallback for unseen test speakers | EXP10 – EXP12 |
| [`One-to-many-approach`](https://github.com/NiccoCorto/S2P/tree/One-to-many-approach) | **Generative VAE Paradigm** | Explores probabilistic synthesis with a Variational Autoencoder (Pose Encoder, latent $Z$, KL Divergence loss with annealing); documents the *Posterior Collapse* challenge | VAE Study |

### Branch Details

- **`main`**: The primary reference branch. Contains the clean baseline architecture, dataset preprocessing, full quantitative evaluation suite on the MEAD-EMOTE dataset (`eval_mead_testset.py`), qualitative demo renderer, and the project thesis report (`SpeechToPose.pdf`).
- **`FaceLoss-Approach`**: Addresses the non-linearity and metric distortion of Euler / axis-angle MSE. By rotating a static 3D canonical face mesh (`canonical_face.npy`) with predicted and target rotation matrices, loss gradients reflect true spatial Euclidean displacements rather than raw angle errors. Detailed rationale is documented in `face_loss_approach.md`.
- **`DiffPoseData`**: Ports the pipeline to the full HDTF dataset with DiffPose formatting conventions. Implements speaker-dependent conditioning using 587 one-hot speaker vectors (`speaker_mapping.json`), testing whether speaker identity alone can resolve stylistic motion ambiguities.
- **`VarLoss`**: Extends the FaceLoss formulation by adding an explicit penalty on temporal variance discrepancies on 3D vertices between prediction and ground truth. It features calibrated loss weights ($w_{\text{vel}}$, $w_{\text{var}}$) and handles unknown speakers gracefully during test-time inference with a uniform $1/N$ prior.
- **`One-to-many-approach`**: Investigates the transition from deterministic regression to probabilistic generative modeling. Introduces a VAE framework to sample diverse motion trajectories from a standard normal latent space $Z \sim \mathcal{N}(0, I)$. Empirically demonstrates how standard reconstruction losses cause posterior collapse in speech-to-pose regression (detailed in `analisi_deterministico_vs_probabilistico.md`), underscoring the necessity of diffusion models or adversarial (GAN) objectives.

To switch to any branch locally:
```bash
git checkout <branch-name>
```

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
├── render_no_scantalk.py     # Rendering pipeline (pure rotation, no ScanTalk dependency)
├── speaker_mapping.json      # HDTF speaker ID → One-Hot index (587 speakers)
└── canonical_face.npy        # FLAME canonical face template (5023 vertices)
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
