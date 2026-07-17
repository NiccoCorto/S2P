#!/bin/bash

export COMET_API_KEY="LA_TUA_API_KEY_QUI"

# avvio esperimenti in parallelo

echo "Avvio esperimento 1: velocity_loss = 0"
nohup python Audio2Pose/train.py --vel_loss_weight 0.0 --save_path Saves/exp_vel0 --log_path Logs/exp_vel0 > log_vel0.txt 2>&1 &

echo "Avvio esperimento 2: velocity_loss = 0.5"
nohup python Audio2Pose/train.py --vel_loss_weight 0.5 --save_path Saves/exp_vel0.5 --log_path Logs/exp_vel0.5 > log_vel0.5.txt 2>&1 &

echo "Avvio esperimento 3: velocity_loss = 1.0"
nohup python Audio2Pose/train.py --vel_loss_weight 1.0 --save_path Saves/exp_vel1 --log_path Logs/exp_vel1 > log_vel1.txt 2>&1 &

echo "Avvio esperimento 4: velocity_loss = 2.0"
nohup python Audio2Pose/train.py --vel_loss_weight 2.0 --save_path Saves/exp_vel2 --log_path Logs/exp_vel2 > log_vel2.txt 2>&1 &

echo "--------------------------------------------------------"
echo "Tutti e 4 gli esperimenti sono stati avviati in parallelo in background!"
echo "Puoi controllare l'andamento di un esperimento leggendo il suo file di log testuale:"
echo "  tail -f log_vel0.txt"
echo "Inoltre, puoi visualizzare i grafici live sulla dashboard di Comet ML sul sito."
