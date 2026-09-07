#!/bin/bash
export COMET_API_KEY="7UuGcSnsZJc4BxmczoZdm8O1j"
source /mnt/diskone-second/ncortini/miniconda3/etc/profile.d/conda.sh
conda activate scantalk

echo "=================================================="
echo "  S2P - Esperimento OneToMany VAE (HDTF_TFHP)"
echo "=================================================="
echo "  Iperparametri Modello VAE:"
echo "   - Epoche:         200"
echo "   - LR:             1e-4"
echo "   - LSTM Layers:    3"
echo "   - Hidden Dim:     256"
echo "   - Dropout:        0.1"
echo "   - Batch size:     1"
echo "   - vel_loss:       2.0"
echo "   - latent_dim (Z): 64"
echo "   - max KLD weight: 0.001"
echo "   - KLD anneal ep:  10"
echo "=================================================="

EXP_NAME="exp6_OneToMany_VAE"
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$BASE_DIR/Logs/$EXP_NAME"
mkdir -p "$BASE_DIR/Results/$EXP_NAME"

echo ">>> Avvio in background..."
echo "    Log: Logs/$EXP_NAME/nohup.out"

nohup python -u Audio2Pose/train.py \
    --max_epoch      200 \
    --lr             0.0001 \
    --vel_loss_weight 2.0 \
    --dropout        0.1 \
    --num_layers     3 \
    --hidden_dim     256 \
    --latent_dim     64 \
    --kld_weight     0.001 \
    --anneal_epochs  10 \
    --batch_size     1 \
    --exp_name       "$EXP_NAME" \
    --save_path      Saves/$EXP_NAME \
    --log_path       Logs/$EXP_NAME \
    --result_path    Results/$EXP_NAME \
    --cache_data \
    > "$BASE_DIR/Logs/$EXP_NAME/nohup.out" 2>&1 &

EXP_PID=$!
echo "    PID: $EXP_PID"
echo "=================================================="
