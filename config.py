"""
config.py - Configurazione centralizzata per S2P (Speech-to-Pose)
Tutti i path e gli iperparametri in un unico posto.
Uso: python Audio2Pose/train.py  (usa i default SSH)
     python Audio2Pose/train.py --mode local --data_dir ./test_data  (test locale)
"""
import argparse
import os
import torch


def get_args():
    parser = argparse.ArgumentParser(
        description='S2P: Speech-to-Pose — generazione movimenti testa da Audio'
    )

    # modalità
    parser.add_argument("--mode", type=str, default="ssh",
                        choices=["local", "ssh"],
                        help="'ssh' per training reale, 'local' per test su Windows")

    # path dati
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Cartella base del dataset. Se None, usa il default per --mode")
    parser.add_argument("--audio_subdir", type=str, default="audio",
                        help="Sottocartella degli audio dentro data_dir")
    parser.add_argument("--pose_subdir", type=str, default="pose",
                        help="Sottocartella delle pose dentro data_dir")

    # path output
    parser.add_argument("--save_path", type=str, default="Saves",
                        help="Cartella salvataggio pesi del modello")
    parser.add_argument("--result_path", type=str, default="Results",
                        help="Cartella salvataggio predizioni .npy del test")
    parser.add_argument("--log_path", type=str, default="Logs",
                        help="Cartella salvataggio log di training (CSV)")

    # identificatore esperimento (usato come nome su CometML)
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Nome univoco dell'esperimento (visibile su CometML). "
                             "Se None, viene generato automaticamente dagli iperparametri.")
    
    # parametri per resume
    parser.add_argument("--resume_checkpoint", type=str, default=None,
                        help="Path al checkpoint (.pth) da cui riprendere l'allenamento")
    parser.add_argument("--comet_experiment_key", type=str, default=None,
                        help="ID univoco dell'esperimento Comet ML (per riprendere i grafici)")

    # iperparametri modello (VAE)
    parser.add_argument("--latent_dim", type=int, default=64,
                        help="Dimensione dello spazio latente (Z) per il VAE")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Dimensione hidden LSTM (Decoder)")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Numero di layer LSTM")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout tra LSTM e FC (default PyTorch: 0.1)")

    # iperparametri training
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="Learning rate (fisso, nessuno scheduler)")
    parser.add_argument("--max_epoch", type=int, default=200,
                        help="Numero massimo di epoche")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (supportato > 1 grazie alla collate_fn con padding audio/pose)")
    parser.add_argument("--patience", type=int, default=200,
                        help="Epoche senza miglioramento prima di early stopping")
    parser.add_argument("--vel_loss_weight", type=float, default=0.0,
                        help="Peso della velocity loss (0 = disabilitata per test overfitting; "
                             "2044 = normalizzata alla stessa magnitudo di pos_loss)")
    parser.add_argument("--var_loss_weight", type=float, default=1.0,
                        help="Peso della VarLoss (differenza varianza temporale GT vs pred sui "
                             "vertici 3D ruotati). 0 = disabilitata; valori tipici: 1.0 - 10.0. "
                             "Incoraggia il modello a uscire dalla staticità centrale.")
    parser.add_argument("--kld_weight", type=float, default=0.001,
                        help="Peso massimo della KL Divergence Loss")
    parser.add_argument("--anneal_epochs", type=int, default=10,
                        help="Numero di epoche per l'annealing della KLD (parte da 0 e arriva a kld_weight)")

    # dati
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limita il numero di campioni caricati (None = tutti). "
                             "Utile per test veloci.")
    parser.add_argument("--cache_data", action="store_true",
                        help="Salva/carica dati preprocessati da cache .pkl")
    parser.add_argument("--train_split", type=float, default=0.8,
                        help="Percentuale soggetti per training")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Percentuale soggetti per validation")

    # device
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu). Se None, auto-detect")

    # scantalk integration
    parser.add_argument("--scantalk_meshes_dir", type=str, default=None,
                        help="Cartella con le mesh .ply generate da ScanTalk (per demo)")
    parser.add_argument("--scantalk_vertices_npy", type=str, default=None,
                        help="File .npy con vertici ScanTalk shape (N_frames, N_vertices, 3)")

    args = parser.parse_args()

    # auto-completamento

    # device auto-detect
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # data dir default in base alla modalità
    if args.data_dir is None:
        if args.mode == "ssh":
            args.data_dir = "/mnt/diskone-second/DiffPoseTalk/datasets/HDTF_TFHP_Elaborated_Pose"
        else:
            # locale: cartella test_data dentro al progetto
            args.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")

    # costruisci i path completi audio/pose
    args.audio_path = os.path.join(args.data_dir, args.audio_subdir)
    args.pose_path = os.path.join(args.data_dir, args.pose_subdir)

    return args


if __name__ == "__main__":
    # test: stampa la configurazione
    args = get_args()
    print("=== Configurazione S2P ===")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
