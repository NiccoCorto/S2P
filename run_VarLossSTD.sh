#!/bin/bash
# run_VarLossSTD.sh - Lancia l'esperimento VarLossSTD (exp11)
#
# Combina:
#   - One-Hot encoding (solo speaker di TRAIN, val/test ricevono vettore uniforme 1/N)
#   - FaceLoss (MSE sui 5023 vertici 3D ruotati + velocity)
#   - VarLoss (differenza di varianza temporale GT vs pred sui vertici 3D)
#
# Differenze rispetto a run_VarLoss.sh (exp10):
#   I pesi sono stati calibrati in base alle medie empiriche delle loss
#   misurate su exp10 (100 epoche, Comet ML):
#
#     m_pos ≈ 2.58e-5   →  w_pos = 1.0    (fisso, riferimento)
#     m_vel ≈ 3.57e-7   →  w_vel ≈ 36     (per contribuire il 20% del totale)
#     m_var ≈ 3.57e-9   →  w_var ≈ 7200   (per contribuire il 40% del totale)
#
#   Distribuzione target delle loss pesate: 40% Pos / 20% Vel / 40% Var
#   (rapporto 4:2:4 rispetto al contributo al gradiente totale)
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
EXP_NAME="exp11_VarLossSTD"
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

# Pesi della loss — calibrati per distribuzione 40% / 20% / 40% (Pos / Vel / Var)
#
# Dato che le medie empiriche (exp10) sono:
#   PosLoss ≈ 2.58e-5,  VelLoss ≈ 3.57e-7,  VarLoss ≈ 3.57e-9
#
# Per ottenere contributi proporzionali a 4 : 2 : 4:
#   w_pos * m_pos : w_vel * m_vel : w_var * m_var = 4 : 2 : 4
#
# Fissando w_pos = 1.0:
#   w_vel = (2/4) * (m_pos / m_vel) = 0.5 * (2.58e-5 / 3.57e-7) ≈ 36
#   w_var = (4/4) * (m_pos / m_var) = 1.0 * (2.58e-5 / 3.57e-9) ≈ 7227 → arrotondato a 7200
VEL_LOSS_WEIGHT=36
VAR_LOSS_WEIGHT=7200

# Crea cartelle output
mkdir -p "${SAVE_PATH}" "${RESULT_PATH}" "${LOG_PATH}"

echo "======================================================"
echo " S2P — VarLossSTD Experiment (${EXP_NAME})"
echo " Data:      ${DATA_DIR}"
echo " Save:      ${SAVE_PATH}"
echo " Results:   ${RESULT_PATH}"
echo " Logs:      ${LOG_PATH}"
echo " vel_w:     ${VEL_LOSS_WEIGHT}   (target: 20% del gradiente totale)"
echo " var_w:     ${VAR_LOSS_WEIGHT}  (target: 40% del gradiente totale)"
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

python -u Audio2Pose/train.py \
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
