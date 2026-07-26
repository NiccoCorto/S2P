"""
demo_nicco.py - Pipeline end-to-end S2P: ScanTalk + Audio2Pose
Partendo da un singolo file audio, lo script:
  1. Esegue ScanTalk per animare le labbra della mesh template
  2. Esegue Audio2Pose per predire i movimenti della testa
  3. Fonde i due risultati: applica le rotazioni alle mesh di ScanTalk
  4. Salva le mesh finali pronte per il rendering

Modalità di utilizzo:
  # Pipeline completa automatica (ScanTalk + Audio2Pose)
  python demo_nicco.py --audio test.wav
      --checkpoint Saves/best_audio2pose.pth
      --scantalk_src /path/to/ScanTalk/src
      --scantalk_model /path/to/scantalk.pth.tar
      --actor_file /path/to/FLAME_sample.ply
      --output_dir Demo_Finale

  # Se ScanTalk è già stato calcolato (salta il passo 1)
  python demo_nicco.py --audio test.wav
      --checkpoint Saves/best_audio2pose.pth
      --scantalk_dir /path/alla/cartella/scantalk
      --output_dir Demo_Finale

  # Applica pose già calcolate a mesh già calcolate
  python demo_nicco.py --pose_file Results/M034.npy --scantalk_dir scantalk_output

NOTA ROTAZIONE: Le mesh FLAME hanno l'origine (0,0,0) al perno anatomico del collo.
La rotazione viene applicata attorno all'origine per un movimento naturale della testa.
"""
import os
import sys
import glob
import argparse
import numpy as np
import cv2
import trimesh
from tqdm import tqdm


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# ─────────────────────────────────────────────
# STEP 1: ScanTalk — anima le labbra
# ─────────────────────────────────────────────

def run_scantalk(audio_path, actor_file, scantalk_model_path, output_dir, device,
                 scantalk_src=None,
                 in_channels=3, out_channels=3, latent_channels=32, lstm_layers=3):
    """Esegue ScanTalk per generare le mesh animate (labbra) a partire da un audio.

    Args:
        audio_path:          Path al file .wav di input
        actor_file:          Path al template PLY (es. FLAME_sample.ply)
        scantalk_model_path: Path al checkpoint ScanTalk (.pth.tar)
        output_dir:          Cartella di output — le PLY verranno salvate in output_dir/Meshes/
        device:              'cuda' o 'cpu'
        scantalk_src:        Path alla cartella src/ di ScanTalk (aggiunta a sys.path per gli import)
        in_channels:         Iperparametro DiffusionNet (default: 3)
        out_channels:        Iperparametro DiffusionNet (default: 3)
        latent_channels:     Iperparametro DiffusionNet (default: 32)
        lstm_layers:         Numero di layer LSTM del modello ScanTalk (default: 3)

    Returns:
        str — Path alla cartella che contiene i file .ply generati (output_dir/Meshes)
    """
    print(f"\n{'='*60}")
    print("  STEP 1: ScanTalk — Generazione mesh animate (labbra)")
    print(f"{'='*60}")
    print(f"  Audio:   {audio_path}")
    print(f"  Actor:   {actor_file}")
    print(f"  Modello: {scantalk_model_path}")

    # Aggiunge scantalk_src a sys.path solo per importare 'hubert' e 'diffusion_net'.
    # Questi nomi non confliggono con nessun modulo di S2P — nessuna manipolazione
    # di sys.modules necessaria.
    if scantalk_src is not None:
        if scantalk_src not in sys.path:
            sys.path.insert(0, scantalk_src)
        diffnet_path = os.path.join(scantalk_src, 'model', 'diffusion-net', 'src')
        if os.path.isdir(diffnet_path) and diffnet_path not in sys.path:
            sys.path.insert(0, diffnet_path)

    import torch
    import torch.nn as nn
    import librosa
    from transformers import Wav2Vec2Processor

    try:
        import diffusion_net
        from hubert.modeling_hubert import HubertModel
    except ImportError as e:
        raise ImportError(
            f"Impossibile importare hubert o diffusion_net: {e}\n"
            f"Verifica che --scantalk_src punti alla cartella src/ di ScanTalk."
        )

    # ── Classe ScanTalk embeddada direttamente ──────────────────────────────
    # Copiata da ScanTalk/src/model/scantalk.py per evitare il conflitto di
    # nome con model.py di Audio2Pose (entrambi si chiamano 'model').
    class ScanTalk(nn.Module):
        def __init__(self, in_channels, out_channels, latent_channels, lstm_layers):
            super(ScanTalk, self).__init__()
            self.audio_encoder = HubertModel.from_pretrained(
                "/mnt/diskone-second/ncortini/hubert-base"
            )
            self.audio_encoder.feature_extractor._freeze_parameters()
            self.encoder = diffusion_net.layers.DiffusionNet(
                C_in=in_channels, C_out=latent_channels,
                C_width=latent_channels, N_block=4,
                outputs_at='vertices', dropout=False
            )
            self.decoder = diffusion_net.layers.DiffusionNet(
                C_in=latent_channels * 2, C_out=out_channels,
                C_width=latent_channels, N_block=4,
                outputs_at='vertices', dropout=False
            )
            nn.init.constant_(self.decoder.last_lin.weight, 0)
            nn.init.constant_(self.decoder.last_lin.bias, 0)
            self.audio_embedding = nn.Linear(768, latent_channels)
            self.lstm = nn.LSTM(
                input_size=latent_channels,
                hidden_size=int(latent_channels / 2),
                num_layers=lstm_layers,
                batch_first=True, bidirectional=True
            )

        def predict(self, audio, actor, mass, L, evals, evecs, gradX, gradY,
                    faces, dataset, hks=None):
            hidden_states = self.audio_encoder(audio, dataset).last_hidden_state
            audio_emb = self.audio_embedding(hidden_states)
            src = hks if hks is not None else actor
            actor_vertices_emb = self.encoder(
                src, mass=mass, L=L, evals=evals, evecs=evecs,
                gradX=gradX, gradY=gradY, faces=faces
            )
            latent, _ = self.lstm(audio_emb)
            combination = torch.cat([
                actor_vertices_emb.expand(
                    1, latent.shape[1], actor_vertices_emb.shape[1], actor_vertices_emb.shape[2]
                ),
                latent.unsqueeze(2).expand(
                    1, latent.shape[1], actor_vertices_emb.shape[1], latent.shape[2]
                )
            ], dim=-1).squeeze(0)
            mass  = mass.expand(latent.shape[1], mass.shape[1])
            L     = L.to_dense().expand(latent.shape[1], L.shape[1], L.shape[2])
            evals = evals.expand(latent.shape[1], evals.shape[1])
            evecs = evecs.expand(latent.shape[1], evecs.shape[1], evecs.shape[2])
            gradX = gradX.to_dense().expand(latent.shape[1], gradX.shape[1], gradX.shape[2])
            gradY = gradY.to_dense().expand(latent.shape[1], gradY.shape[1], gradY.shape[2])
            faces = faces.expand(latent.shape[1], faces.shape[1], faces.shape[2])
            pred_disp = self.decoder(
                combination, mass=mass, L=L, evals=evals, evecs=evecs,
                gradX=gradX, gradY=gradY, faces=faces
            )
            return pred_disp + actor
    # ────────────────────────────────────────────────────────────────────────

    # Crea le sottocartelle di output
    meshes_dir = os.path.join(output_dir, "Meshes")
    os.makedirs(meshes_dir, exist_ok=True)

    # Carica il processor audio di HuBERT (usato da ScanTalk)
    print("\n  Caricamento HuBERT processor...")
    processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-xlarge-ls960-ft")

    # Carica il modello ScanTalk con i pesi dal checkpoint
    print("  Caricamento modello ScanTalk...")
    scantalk_model = ScanTalk(in_channels, out_channels, latent_channels, lstm_layers).to(device)
    checkpoint = torch.load(scantalk_model_path, map_location=device)
    scantalk_model.load_state_dict(checkpoint['autoencoder_state_dict'])
    scantalk_model.eval()

    # Carica e processa l'audio
    print(f"  Processamento audio con HuBERT...")
    speech_array, sampling_rate = librosa.load(audio_path, sr=16000)
    audio_feature = np.squeeze(processor(speech_array, sampling_rate=16000).input_values)
    audio_feature = np.reshape(audio_feature, (-1, audio_feature.shape[0]))
    audio_feature = torch.FloatTensor(audio_feature).to(device=device)

    # Carica il template PLY
    print(f"  Caricamento template mesh: {actor_file}")
    actor = trimesh.load(actor_file, process=False)
    actor_vertices = actor.vertices
    actor_faces    = actor.faces
    actor_vertices_t = torch.FloatTensor(actor_vertices).to(device=device).unsqueeze(0)

    # Genera la sequenza di mesh animate
    print("  Calcolo operatori DiffusionNet e predizione mesh...")
    with torch.no_grad():
        frames, mass, L, evals, evecs, gradX, gradY = diffusion_net.geometry.compute_operators(
            actor_vertices_t.to('cpu').squeeze(0),
            faces=torch.tensor(actor_faces),
            k_eig=128
        )
        mass          = torch.FloatTensor(np.array(mass)).float().to(device).unsqueeze(0)
        evals         = torch.FloatTensor(np.array(evals)).to(device).unsqueeze(0)
        evecs         = torch.FloatTensor(np.array(evecs)).to(device).unsqueeze(0)
        L             = L.float().to(device).unsqueeze(0)
        gradX         = gradX.float().to(device).unsqueeze(0)
        gradY         = gradY.float().to(device).unsqueeze(0)
        actor_faces_t = torch.tensor(actor_faces).to(device).float().unsqueeze(0)

        gen_seq = scantalk_model.predict(
            audio_feature, actor_vertices_t.float(),
            mass, L, evals, evecs, gradX, gradY, actor_faces_t,
            'vocaset'
        )
        gen_seq = gen_seq.cpu().detach().numpy()

    # Salva le mesh animate frame per frame
    print(f"  Salvataggio {len(gen_seq)} mesh in {meshes_dir}...")
    for k in tqdm(range(len(gen_seq)), desc="Salvataggio PLY ScanTalk"):
        tri_mesh = trimesh.Trimesh(
            np.array(gen_seq[k]), np.asarray(actor_faces), process=False
        )
        tri_mesh.export(os.path.join(meshes_dir, f"tst{str(k).zfill(3)}.ply"))

    print(f"  ✓ ScanTalk completato! {len(gen_seq)} mesh salvate in: {meshes_dir}")
    return meshes_dir


# ─────────────────────────────────────────────
# STEP 2: Audio2Pose — predici le rotazioni
# ─────────────────────────────────────────────

def predict_head_rotations(audio_path, checkpoint_path, device):
    """Predice le rotazioni della testa (Pitch, Yaw, Roll) dall'audio.

    Args:
        audio_path:      Path al file .wav di input
        checkpoint_path: Path al checkpoint Audio2Pose (.pth)
        device:          'cuda' o 'cpu'

    Returns:
        np.ndarray shape (N_frames, 3)
    """
    print(f"\n{'='*60}")
    print("  STEP 2: Audio2Pose — Predizione movimenti testa")
    print(f"{'='*60}")
    print(f"  Audio:      {audio_path}")
    print(f"  Checkpoint: {checkpoint_path}")

    from utils import predict_pose_from_audio
    from model import HeadPosePredictor
    import torch

    model = HeadPosePredictor()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()

    rotations = predict_pose_from_audio(model, audio_path, device=device)
    print(f"  ✓ Predizione completata! Shape: {rotations.shape}")
    return rotations


# ─────────────────────────────────────────────
# Caricamento mesh ScanTalk da cartella
# ─────────────────────────────────────────────

def load_scantalk_meshes(scantalk_dir):
    """Carica le mesh .ply generate da ScanTalk dalla cartella Meshes/.

    Args:
        scantalk_dir: Cartella di output di ScanTalk (contiene Meshes/)
                      oppure cartella con direttamente i .ply

    Returns:
        list di trimesh.Trimesh, ordinata per nome
    """
    meshes_subdir = os.path.join(scantalk_dir, "Meshes")
    search_dir = meshes_subdir if os.path.isdir(meshes_subdir) else scantalk_dir

    mesh_files = sorted(glob.glob(os.path.join(search_dir, '*.ply')))

    if len(mesh_files) == 0:
        raise FileNotFoundError(
            f"Nessuna mesh .ply trovata in {search_dir}\n"
            f"Assicurati che ScanTalk abbia generato le mesh in questa cartella."
        )

    print(f"  Trovate {len(mesh_files)} mesh .ply in {search_dir}")
    meshes = []
    for mf in tqdm(mesh_files, desc="Caricamento mesh ScanTalk"):
        meshes.append(trimesh.load(mf, process=False))

    return meshes, mesh_files


def load_vertices_npy(npy_path):
    """Carica vertici da un file .npy con shape (N_frames, N_vertices, 3)."""
    vertices = np.load(npy_path)
    if vertices.ndim != 3 or vertices.shape[2] != 3:
        raise ValueError(
            f"Shape inattesa per vertici: {vertices.shape}. "
            f"Atteso (N_frames, N_vertices, 3)"
        )
    print(f"  Caricati {vertices.shape[0]} frame di vertici da {npy_path}")
    return [vertices[i] for i in range(vertices.shape[0])]


# ─────────────────────────────────────────────
# STEP 3: Fusione — applica rotazioni alle mesh
# ─────────────────────────────────────────────

def apply_rotations(meshes_or_vertices, rotations, output_dir, mesh_faces=None,
                    pivot_origin=True):
    """Applica le rotazioni predette dal modello alle mesh/vertici di ScanTalk.

    Args:
        meshes_or_vertices: lista di trimesh.Trimesh o lista di np.ndarray (N_v, 3)
        rotations:          np.ndarray shape (N_frames, 3) — Pitch, Yaw, Roll per frame
        output_dir:         Cartella dove salvare le mesh ruotate
        mesh_faces:         Se vertici sono ndarray, servono le facce per ricostruire la mesh
        pivot_origin:       Se True (default), ruota attorno all'origine (0,0,0) — corretto per
                            mesh FLAME dove l'origine è al perno anatomico del collo.
                            Se False, ruota attorno al baricentro dei vertici.

    Returns:
        int — Numero di frame processati
    """
    print(f"\n{'='*60}")
    print("  STEP 3: Fusione — Applicazione rotazioni testa alle mesh")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)

    n_meshes = len(meshes_or_vertices)
    n_poses  = len(rotations)
    min_frames = min(n_meshes, n_poses)

    if n_meshes != n_poses:
        print(f"  Allineamento: {n_meshes} mesh ScanTalk vs {n_poses} frame di posa "
              f"→ uso {min_frames} frame")

    animated_vertices_list = []

    for i in tqdm(range(min_frames), desc="Fusione ScanTalk + Audio2Pose"):
        item = meshes_or_vertices[i]

        if isinstance(item, trimesh.Trimesh):
            vertices = item.vertices.copy()
            faces    = item.faces
        else:
            vertices = item.copy()
            faces    = mesh_faces

        rot = rotations[i]

        # perno di rotazione: origine (0,0,0) per mesh FLAME
        t_center = np.zeros(3) if pivot_origin else np.mean(vertices, axis=0)

        R, _ = cv2.Rodrigues(rot.astype(np.float64))
        rotated_vertices = R.dot((vertices - t_center).T).T + t_center

        if faces is not None:
            rotated_mesh = trimesh.Trimesh(
                vertices=rotated_vertices, faces=faces, process=False
            )
        else:
            rotated_mesh = trimesh.PointCloud(rotated_vertices)

        output_filepath = os.path.join(output_dir, f"frame_{i:05d}.ply")
        rotated_mesh.export(output_filepath)
        animated_vertices_list.append(rotated_vertices)

    # Salva il file .npy completo per il rendering
    npy_output_path = os.path.join(output_dir, "mesh_animata.npy")
    arr_to_save = np.array(animated_vertices_list)
    np.save(npy_output_path, arr_to_save)
    print(f"  ✓ Vertici animati salvati in: {npy_output_path} (shape: {arr_to_save.shape})")

    return min_frames


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='S2P Demo: Pipeline end-to-end ScanTalk + Audio2Pose',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi di utilizzo:

  # Pipeline completa automatica (ScanTalk + Audio2Pose):
  python demo_nicco.py \\
      --audio /path/to/audio.wav \\
      --checkpoint Saves/best_audio2pose.pth \\
      --scantalk_src /mnt/.../ScanTalk/src \\
      --scantalk_model /mnt/.../scantalk_masked_velocity_loss.pth.tar \\
      --actor_file /mnt/.../ScanTalk/src/examples/FLAME_sample.ply \\
      --output_dir Demo_Finale

  # ScanTalk già calcolato (salta il passo 1):
  python demo_nicco.py \\
      --audio /path/to/audio.wav \\
      --checkpoint Saves/best_audio2pose.pth \\
      --scantalk_dir /path/alla/cartella/scantalk \\
      --output_dir Demo_Finale

  # Solo applica pose pre-calcolate a mesh pre-calcolate:
  python demo_nicco.py \\
      --pose_file Results/M034.npy \\
      --scantalk_dir scantalk_output \\
      --output_dir Demo_Finale
        """
    )

    # ── Input Audio ──────────────────────────────────────
    parser.add_argument("--audio", type=str, default=None,
                        help="File audio .wav di input")
    parser.add_argument("--pose_file", type=str, default=None,
                        help="File .npy con pose già calcolate (N_frames x 3). "
                             "Se fornito, salta Audio2Pose.")

    # ── ScanTalk (auto) ───────────────────────────────────
    parser.add_argument("--scantalk_src", type=str, default=None,
                        help="Path alla cartella src/ di ScanTalk "
                             "(es. /mnt/.../ScanTalk/src). "
                             "Necessario per la pipeline automatica.")
    parser.add_argument("--scantalk_model", type=str, default=None,
                        help="Path al checkpoint ScanTalk (.pth.tar). "
                             "Necessario per la pipeline automatica.")
    parser.add_argument("--actor_file", type=str, default=None,
                        help="Path al template PLY (es. FLAME_sample.ply). "
                             "Necessario per la pipeline automatica.")

    # Iperparametri DiffusionNet (usati internamente da ScanTalk — non cambiare)
    parser.add_argument("--latent_channels", type=int, default=32)
    parser.add_argument("--in_channels",     type=int, default=3)
    parser.add_argument("--out_channels",    type=int, default=3)
    parser.add_argument("--lstm_layers",     type=int, default=3)

    # ── ScanTalk (manuale, se già calcolato) ──────────────
    parser.add_argument("--scantalk_dir", type=str, default=None,
                        help="Cartella output ScanTalk già esistente (con Meshes/). "
                             "Se fornito, ScanTalk NON viene rieseguito.")
    parser.add_argument("--vertices_npy", type=str, default=None,
                        help="File .npy con vertici ScanTalk (N_frames x V x 3). "
                             "Alternativa a --scantalk_dir.")

    # ── Audio2Pose ────────────────────────────────────────
    parser.add_argument("--checkpoint", type=str, default="Saves/best_audio2pose.pth",
                        help="Path al checkpoint Audio2Pose (.pth)")
    parser.add_argument("--device", type=str,
                        default="cuda" if __import__('torch').cuda.is_available() else "cpu")

    # ── Output ────────────────────────────────────────────
    parser.add_argument("--output_dir", type=str, default="Demo_Finale",
                        help="Cartella dove salvare le mesh finali (labbra + testa)")
    parser.add_argument("--fps", type=int, default=30,
                        help="FPS del video (usato dal renderer esterno)")

    args = parser.parse_args()

    # ── Validazione input ─────────────────────────────────
    if args.audio is None and args.pose_file is None:
        parser.error("Serve almeno uno tra --audio e --pose_file.")

    scantalk_already_done = (args.scantalk_dir is not None or args.vertices_npy is not None)

    if not scantalk_already_done and args.audio is not None:
        # Modalità pipeline automatica: ScanTalk va eseguito
        missing = []
        if args.scantalk_src   is None: missing.append("--scantalk_src")
        if args.scantalk_model is None: missing.append("--scantalk_model")
        if args.actor_file     is None: missing.append("--actor_file")
        if missing:
            parser.error(
                f"Per la pipeline automatica servono: {', '.join(missing)}\n"
                f"Oppure passa --scantalk_dir se ScanTalk è già stato eseguito."
            )

    print(f"\n{'='*60}")
    print(f"  S2P — Pipeline end-to-end")
    print(f"  Device: {args.device}")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*60}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── STEP 1: ScanTalk ──────────────────────────────────
    if scantalk_already_done:
        print(f"\n  [STEP 1] Mesh ScanTalk già presenti — caricamento da disco...")
        if args.scantalk_dir:
            meshes, _ = load_scantalk_meshes(args.scantalk_dir)
            items = meshes
        else:
            items = load_vertices_npy(args.vertices_npy)
        mesh_faces = None
    else:
        # Esegui ScanTalk in automatico, salva nella sottocartella scantalk_meshes/
        scantalk_output_dir = os.path.join(args.output_dir, "scantalk_meshes")
        meshes_dir = run_scantalk(
            audio_path=args.audio,
            actor_file=args.actor_file,
            scantalk_model_path=args.scantalk_model,
            output_dir=scantalk_output_dir,
            device=args.device,
            scantalk_src=args.scantalk_src,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            latent_channels=args.latent_channels,
            lstm_layers=args.lstm_layers,
        )
        meshes, _ = load_scantalk_meshes(scantalk_output_dir)
        items = meshes
        mesh_faces = None

    # ── STEP 2: Audio2Pose ────────────────────────────────
    if args.pose_file:
        print(f"\n  [STEP 2] Caricamento pose pre-calcolate da: {args.pose_file}")
        rotations = np.load(args.pose_file)
        print(f"  Shape: {rotations.shape}")
    else:
        rotations = predict_head_rotations(
            audio_path=args.audio,
            checkpoint_path=args.checkpoint,
            device=args.device,
        )
        # Salva le pose predette per riferimento futuro
        pose_save = os.path.join(args.output_dir, "predicted_pose.npy")
        np.save(pose_save, rotations)
        print(f"  Pose salvate in: {pose_save}")

    # ── STEP 3: Fusione ───────────────────────────────────
    n_processed = apply_rotations(
        items, rotations, args.output_dir, mesh_faces=mesh_faces
    )

    print(f"\n{'='*60}")
    print(f"  Pipeline completata!")
    print(f"  {n_processed} mesh finali (labbra animate + testa in movimento)")
    print(f"  salvate in: {args.output_dir}")
    print(f"  Per il video, lancia il tuo script di rendering su: {args.output_dir}/mesh_animata.npy")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()