#!/bin/bash
# run_VarLoss.sh - Lancia l'esperimento VarLoss (exp10)
#
# Combina:
#   - One-Hot encoding (solo speaker di TRAIN, val/test ricevono vettore uniforme 1/N)
#   - FaceLoss (MSE sui 5023 vertici 3D ruotati + velocity)
#   - VarLoss (differenza di varianza temporale GT vs pred sui vertici 3D)
#
# Prerequisiti:
#   - Branch: VarLoss
#   - Dataset: HDTF_TFHP_Elaborated_Pose
#   - Variabile d'ambiente: COMET_API_KEY
#
# IMPORTANTE: Se esiste s2p_lazy_cache.pkl da un run precedente (DiffPoseData/OneHot),
#             eliminala prima di avviare: rm s2p_lazy_cache.pkl speaker_mapping.json
#             Il branch VarLoss rigenera il mapping usando SOLO le identità di train.

set -e

# Setup ambiente e Comet ML
export COMET_API_KEY="${COMET_API_KEY:-7UuGcSnsZJc4BxmczoZdm8O1j}"
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "scantalk" ]; then
    if [ -f "/mnt/diskone-second/ncortini/miniconda3/bin/activate" ]; then
        source /mnt/diskone-second/ncortini/miniconda3/bin/activate scantalk
    fi
fi

# --- Configurazione ---
EXP_NAME="exp10_VarLoss"
DATA_DIR="/mnt/diskone-second/DiffPoseTalk/datasets/HDTF_TFHP_Elaborated_Pose"
SAVE_PATH="Saves/${EXP_NAME}"
RESULT_PATH="Results/${EXP_NAME}"
LOG_PATH="Logs/${EXP_NAME}"

# Iperparametri modello
HIDDEN_DIM=256
NUM_LAYERS=2
DROPOUT=0.1

# Iperparametri training
LR=0.0001
MAX_EPOCH=100
BATCH_SIZE=4
PATIENCE=100

# Pesi della loss
# - vel_loss_weight=1.0: velocity loss nello spazio dei vertici (peso bilanciato)
# - var_loss_weight=5.0: VarLoss ha peso più alto perché la varianza è numericamente
#   più piccola della pos_loss; 5x dà abbastanza influenza sul gradiente.
#   Se la faccia è ancora troppo statica, aumentare a 10.0 o 50.0.
VEL_LOSS_WEIGHT=1.0
VAR_LOSS_WEIGHT=5.0

# Crea cartelle output
mkdir -p "${SAVE_PATH}" "${RESULT_PATH}" "${LOG_PATH}"

echo "======================================================"
echo " S2P — VarLoss Experiment (${EXP_NAME})"
echo " Data:      ${DATA_DIR}"
echo " Save:      ${SAVE_PATH}"
echo " Results:   ${RESULT_PATH}"
echo " Logs:      ${LOG_PATH}"
echo " vel_w:     ${VEL_LOSS_WEIGHT}"
echo " var_w:     ${VAR_LOSS_WEIGHT}"
echo "======================================================"

# AVVISO invalidazione cache
if [ -f "s2p_lazy_cache.pkl" ]; then
    echo ""
    echo "[WARN] Trovata s2p_lazy_cache.pkl — potrebbe contenere il vecchio speaker_mapping."
    echo "       Per garantire la correttezza, eliminala con:"
    echo "         rm s2p_lazy_cache.pkl speaker_mapping.json"
    echo "       e riesegui questo script."
    echo ""
fi

python Audio2Pose/train.py \
    --mode ssh \
    --data_dir "${DATA_DIR}" \
    --save_path "${SAVE_PATH}" \
    --result_path "${RESULT_PATH}" \
    --log_path "${LOG_PATH}" \
    --exp_name "${EXP_NAME}" \
    --hidden_dim ${HIDDEN_DIM} \
    --num_layers ${NUM_LAYERS} \
    --dropout ${DROPOUT} \
    --lr ${LR} \
    --max_epoch ${MAX_EPOCH} \
    --batch_size ${BATCH_SIZE} \
    --patience ${PATIENCE} \
    --vel_loss_weight ${VEL_LOSS_WEIGHT} \
    --var_loss_weight ${VAR_LOSS_WEIGHT} \
    --cache_data

echo ""
echo "======================================================"
echo " Training completato!"
echo " Modello:  ${SAVE_PATH}/best_audio2pose.pth"
echo " Risultati: ${RESULT_PATH}/"
echo "======================================================"
