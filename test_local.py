"""
test_local.py - Test locale su Windows con dati sintetici
Verifica che tutta la pipeline (data_loader → model → train → test) funzioni
senza bisogno di GPU o del dataset EMOTE reale.
"""
import os
import sys
import numpy as np
import argparse

# Path setup
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "Audio2Pose"))


def create_synthetic_data(data_dir, n_samples=10, n_frames_range=(30, 90)):
    """Crea dati sintetici che imitano il formato EMOTE.
    
    Genera:
      - audio/*.wav  (rumore bianco a 16kHz, durata variabile)
      - pose/*.npy   (3 angoli random per frame)
    """
    audio_dir = os.path.join(data_dir, "audio")
    pose_dir = os.path.join(data_dir, "pose")
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(pose_dir, exist_ok=True)

    try:
        import soundfile as sf
    except ImportError:
        # Fallback: genera .wav con scipy
        try:
            from scipy.io import wavfile
            use_scipy = True
        except ImportError:
            print("[ERR] Serve 'soundfile' o 'scipy' per generare i .wav di test")
            print("      pip install soundfile")
            return False
        else:
            use_scipy = True
    else:
        use_scipy = False

    subjects = ["S01", "S02", "S03", "S04", "S05"]
    emotions = ["happy", "angry", "sad"]

    print(f"Generazione {n_samples} campioni sintetici in {data_dir}...")

    for i in range(n_samples):
        subj = subjects[i % len(subjects)]
        emo = emotions[i % len(emotions)]
        name = f"{subj}_{emo}_1_{i:03d}"

        # Numero di frame video (random)
        n_frames = np.random.randint(*n_frames_range)

        # Audio: rumore bianco, durata proporzionale ai frame (~30fps)
        duration_s = n_frames / 30.0
        sr = 16000
        audio = np.random.randn(int(duration_s * sr)).astype(np.float32) * 0.01

        wav_path = os.path.join(audio_dir, f"{name}.wav")
        if use_scipy:
            from scipy.io import wavfile
            wavfile.write(wav_path, sr, (audio * 32767).astype(np.int16))
        else:
            sf.write(wav_path, audio, sr)

        # Pose: 3 angoli random piccoli (simula rotazioni naturali della testa)
        # Valori tipici in radianti: ±0.3 circa
        pose = np.random.randn(n_frames, 3).astype(np.float32) * 0.1
        # Aggiungi correlazione temporale (smooth)
        for j in range(1, n_frames):
            pose[j] = 0.8 * pose[j - 1] + 0.2 * pose[j]

        npy_path = os.path.join(pose_dir, f"{name}.npy")
        np.save(npy_path, pose)

    print(f"  Generati {n_samples} file audio + pose in {data_dir}")
    print(f"  Soggetti: {subjects}")
    return True


def run_test():
    """Esegue il test locale completo."""
    print("=" * 60)
    print("  S2P — TEST LOCALE CON DATI SINTETICI")
    print("=" * 60)

    # 1. Crea dati sintetici
    test_data_dir = os.path.join(project_root, "test_data")
    if not create_synthetic_data(test_data_dir, n_samples=10):
        return

    # 2. Verifica data_loader
    print("\n--- Test Data Loader ---")
    from config import get_args
    sys.argv = [
        "test", "--mode", "local",
        "--data_dir", test_data_dir,
        "--max_epoch", "2",
        "--max_samples", "10",
        "--save_path", os.path.join(project_root, "test_Saves"),
        "--result_path", os.path.join(project_root, "test_Results"),
        "--log_path", os.path.join(project_root, "test_Logs"),
        "--device", "cpu",
    ]
    args = get_args()

    from data_loader import get_dataloaders
    loaders = get_dataloaders(args)

    print(f"\n  Train batches: {len(loaders['train'])}")
    print(f"  Valid batches: {len(loaders['valid'])}")
    print(f"  Test batches:  {len(loaders['test'])}")

    # Verifica un batch
    for audio, pose, names in loaders["train"]:
        print(f"\n  Primo batch Train:")
        print(f"    Audio shape: {audio.shape}")
        print(f"    Pose shape:  {pose.shape}")
        print(f"    File: {names[0]}")
        break

    # 3. Verifica modello (forward pass)
    print("\n--- Test Modello ---")
    from model import HeadPosePredictor
    model = HeadPosePredictor(args)
    model = model.to(args.device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parametri addestrabili: {trainable:,}")

    # Forward pass
    for audio, pose, names in loaders["train"]:
        audio = audio.to(args.device)
        with __import__('torch').no_grad():
            out = model(audio, target_seq_len=pose.size(1))
        print(f"  Output shape: {out.shape}  (atteso: {pose.shape})")
        break

    # 4. Training mini (2 epoche)
    print("\n--- Test Training (2 epoche) ---")
    from Audio2Pose.train import PoseLoss, trainer, test as run_test_fn
    import torch

    criterion = PoseLoss(vel_weight=args.vel_loss_weight)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    model = trainer(args, loaders["train"], loaders["valid"],
                    model, optimizer, criterion, scheduler)

    # 5. Test
    print("\n--- Test Inference ---")
    run_test_fn(args, model, loaders["test"])

    # 6. Verifica evaluate
    print("\n--- Test Evaluate ---")
    from Audio2Pose.evaluate import evaluate_results
    gt_dir = os.path.join(test_data_dir, "pose")
    evaluate_results(args.result_path, gt_dir,
                     os.path.join(args.log_path, "eval_test.csv"))

    # 7. Cleanup
    print("\n✅ TUTTI I TEST PASSATI!")
    print(f"   I file di test sono in: {test_data_dir}")
    print(f"   Per pulire: cancella test_data/, test_Saves/, test_Results/, test_Logs/")


if __name__ == "__main__":
    run_test()
