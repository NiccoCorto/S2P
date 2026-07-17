"""
demo_nicco.py - Script di fusione ScanTalk + Audio2Pose
Prende le mesh 3D generate da ScanTalk (labbra animate, testa ferma)
e vi applica le rotazioni della testa predette dal modello Audio2Pose.

3 Modalità di utilizzo:
  1. --pose_file + --scantalk_dir  → Applica un file .npy di pose a mesh .ply già generate
  2. --audio + --scantalk_dir      → Pipeline end-to-end: audio → predici pose → applica
  3. --pose_file + --vertices_npy  → Applica pose a vertici salvati come .npy (N_frames, V, 3)

NOTA ROTAZIONE: Le mesh FLAME di ScanTalk hanno l'origine (0,0,0) posizionata
al perno anatomico del collo. La rotazione viene applicata attorno all'origine e non
al baricentro dei vertici, per ottenere un movimento naturale della testa.
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


def load_scantalk_meshes(scantalk_dir):
    """Carica le mesh .ply generate da ScanTalk dalla cartella Meshes/.
    
    Args:
        scantalk_dir: Cartella di output di ScanTalk (contiene Meshes/, Images/, demo.mp4)
                      oppure cartella con direttamente i .ply
    
    Returns:
        list di trimesh.Trimesh, ordinata per nome
    """
    # cerca mesh nella sottocartella Meshes/ (struttura standard ScanTalk)
    meshes_subdir = os.path.join(scantalk_dir, "Meshes")
    if os.path.isdir(meshes_subdir):
        search_dir = meshes_subdir
    else:
        search_dir = scantalk_dir

    mesh_files = sorted(glob.glob(os.path.join(search_dir, '*.ply')))

    if len(mesh_files) == 0:
        raise FileNotFoundError(
            f"Nessuna mesh .ply trovata in {search_dir}\n"
            f"Assicurati che ScanTalk abbia generato le mesh in questa cartella."
        )

    print(f"Trovate {len(mesh_files)} mesh .ply in {search_dir}")
    meshes = []
    for mf in tqdm(mesh_files, desc="Caricamento mesh"):
        meshes.append(trimesh.load(mf, process=False))

    return meshes, mesh_files


def load_vertices_npy(npy_path):
    """Carica vertici da un file .npy con shape (N_frames, N_vertices, 3).
    
    Args:
        npy_path: Path al file .npy dei vertici ScanTalk
    
    Returns:
        list di np.ndarray shape (N_vertices, 3)
    """
    vertices = np.load(npy_path)
    if vertices.ndim != 3 or vertices.shape[2] != 3:
        raise ValueError(
            f"Shape inattesa per vertici: {vertices.shape}. "
            f"Atteso (N_frames, N_vertices, 3)"
        )
    print(f"Caricati {vertices.shape[0]} frame di vertici da {npy_path}")
    return [vertices[i] for i in range(vertices.shape[0])]


def apply_rotations(meshes_or_vertices, rotations, output_dir, mesh_faces=None,
                    pivot_origin=True):
    """Applica le rotazioni predette dal modello alle mesh/vertici di ScanTalk.
    
    Args:
        meshes_or_vertices: lista di trimesh.Trimesh o lista di np.ndarray (N_v, 3)
        rotations: np.ndarray shape (N_frames, 3) — Pitch, Yaw, Roll per frame
        output_dir: Cartella dove salvare le mesh ruotate
        mesh_faces: Se vertici sono ndarray, servono le facce per ricostruire la mesh
        pivot_origin: Se True (default), ruota attorno all'origine (0,0,0) — corretto per
                      mesh FLAME dove l'origine è al perno anatomico del collo.
                      Se False, ruota attorno al baricentro dei vertici.
    
    Returns:
        int — Numero di frame processati
    """
    os.makedirs(output_dir, exist_ok=True)

    # allineiamo il numero di frame prendendo il minimo tra i due
    n_meshes = len(meshes_or_vertices)
    n_poses = len(rotations)
    min_frames = min(n_meshes, n_poses)

    if n_meshes != n_poses:
        print(f" Allineamento: {n_meshes} mesh, {n_poses} frame di posa "
              f" uso {min_frames} frame")

    print(f"\nApplicazione rotazioni della testa a {min_frames} frame...")

    animated_vertices_list = []

    for i in tqdm(range(min_frames), desc="Rotazione"):
        item = meshes_or_vertices[i]

        # ottieni i vertici
        if isinstance(item, trimesh.Trimesh):
            vertices = item.vertices.copy()
            faces = item.faces
        else:
            vertices = item.copy()
            faces = mesh_faces

        # prendi i 3 angoli per questo frame
        rot = rotations[i]

        # perno di rotazione:
        # - FLAME: l'origine (0,0,0) è posizionata al perno anatomico (base cranio/collo)
        #   ruotare attorno all'origine produce un movimento naturale della testa
        # - Se pivot_origin=False: usa il baricentro (solo per mesh non-FLAME)
        if pivot_origin:
            t_center = np.zeros(3)
        else:
            t_center = np.mean(vertices, axis=0)

        # matrice di Rotazione di Rodrigues
        R, _ = cv2.Rodrigues(rot.astype(np.float64))

        # applica la rotazione a tutti i vertici attorno al perno
        rotated_vertices = R.dot((vertices - t_center).T).T + t_center

        # crea e salva la nuova mesh
        if faces is not None:
            rotated_mesh = trimesh.Trimesh(
                vertices=rotated_vertices, faces=faces, process=False
            )
        else:
            rotated_mesh = trimesh.PointCloud(rotated_vertices)

        output_filepath = os.path.join(output_dir, f"frame_{i:05d}.ply")
        rotated_mesh.export(output_filepath)
        
        animated_vertices_list.append(rotated_vertices)

    # salva il file .npy completo per poterlo usare con render_npy.py
    npy_output_path = os.path.join(output_dir, "mesh_animata.npy")
    np.array_to_save = np.array(animated_vertices_list)
    np.save(npy_output_path, np.array_to_save)
    print(f"  Vertici animati salvati in: {npy_output_path} (shape: {np.array_to_save.shape})")

    return min_frames


def main():
    parser = argparse.ArgumentParser(
        description='S2P Demo: Unisci ScanTalk + Audio2Pose',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi di utilizzo:
  # Applica pose predette a mesh ScanTalk
  python demo_nicco.py --pose_file Results/M034_disgusted_2_001.npy --scantalk_dir scantalk_output

  # Pipeline end-to-end: da audio a mesh ruotate
  python demo_nicco.py --audio test.wav --scantalk_dir scantalk_output --checkpoint Saves/best_audio2pose.pth

  # Applica pose a vertici .npy 
  python demo_nicco.py --pose_file Results/M034_disgusted_2_001.npy --vertices_npy scantalk_vertices.npy
        """
    )

    # input
    parser.add_argument("--audio", type=str, default=None,
                        help="File audio .wav (per modalità end-to-end)")
    parser.add_argument("--pose_file", type=str, default=None,
                        help="File .npy con le pose predette (shape N_frames x 3)")
    parser.add_argument("--scantalk_dir", type=str, default=None,
                        help="Cartella output ScanTalk (con Meshes/ e Images/)")
    parser.add_argument("--vertices_npy", type=str, default=None,
                        help="File .npy con vertici ScanTalk (shape N_frames x V x 3)")

    # output
    parser.add_argument("--output_dir", type=str, default="Demo_Finale",
                        help="Dove salvare le mesh ruotate")
    parser.add_argument("--render_video", action="store_true",
                        help="Genera anche un video .mp4 delle mesh (richiede pyrender)")
    parser.add_argument("--fps", type=int, default=30,
                        help="FPS del video di output")

    # modello (per modalità end-to-end)
    parser.add_argument("--checkpoint", type=str, default="Saves/best_audio2pose.pth",
                        help="Path al checkpoint del modello (per --audio)")
    parser.add_argument("--device", type=str,
                        default="cuda" if __import__('torch').cuda.is_available() else "cpu")

    args = parser.parse_args()

    # validaztion di input
    if args.audio is None and args.pose_file is None:
        parser.error("Serve almeno uno tra --audio e --pose_file")

    if args.scantalk_dir is None and args.vertices_npy is None:
        parser.error("Serve almeno uno tra --scantalk_dir e --vertices_npy")

    # prendi le rotazioni
    if args.pose_file:
        print(f"\n Caricamento pose da: {args.pose_file}")
        rotations = np.load(args.pose_file)
        print(f"   Shape: {rotations.shape}")
    else:
        # end-to-end mode: predici le pose dall'audio
        print(f"\n end-to-end mode: predizione pose da audio")
        print(f"   Audio: {args.audio}")

        from utils import predict_pose_from_audio
        from model import HeadPosePredictor
        import torch

        # carica il modello
        model = HeadPosePredictor()
        model.load_state_dict(
            torch.load(args.checkpoint, map_location=args.device)
        )
        model = model.to(args.device)
        model.eval()

        rotations = predict_pose_from_audio(
            model, args.audio, device=args.device
        )
        print(f"   Pose predette: {rotations.shape}")

        # salva le pose predette
        pose_save = os.path.join(args.output_dir, "predicted_pose.npy")
        os.makedirs(args.output_dir, exist_ok=True)
        np.save(pose_save, rotations)
        print(f"   Pose salvate in: {pose_save}")

    # carica le mesh di scantalk
    mesh_faces = None
    if args.scantalk_dir:
        meshes, mesh_files = load_scantalk_meshes(args.scantalk_dir)
        items = meshes
    else:
        vertex_list = load_vertices_npy(args.vertices_npy)
        items = vertex_list
        # per vertici nudi servirebbero le facce — ma salvando come PointCloud funziona

    # applica le rotazioni
    n_processed = apply_rotations(
        items, rotations, args.output_dir, mesh_faces=mesh_faces
    )

    print(f"\n Cerchio completato! {n_processed} mesh finali (Labbra + Testa) "
          f"salvate in: {args.output_dir}")

if __name__ == "__main__":
    main()