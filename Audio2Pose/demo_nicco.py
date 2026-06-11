"""
demo_nicco.py - Script di fusione ScanTalk + Audio2Pose
Prende le mesh 3D generate da ScanTalk (labbra animate, testa ferma)
e vi applica le rotazioni della testa predette dal modello Audio2Pose.

3 Modalità di utilizzo:
  1. --pose_file + --scantalk_dir  → Applica un file .npy di pose a mesh .ply già generate
  2. --audio + --scantalk_dir      → Pipeline end-to-end: audio → predici pose → applica
  3. --pose_file + --vertices_npy  → Applica pose a vertici salvati come .npy (N_frames, V, 3)

NOTA ROTAZIONE: Le mesh FLAME (usate da ScanTalk) hanno l'origine (0,0,0) posizionata
al perno anatomico del collo. La rotazione viene applicata attorno all'origine, NON
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
    # Prova prima nella sottocartella Meshes/ (struttura standard ScanTalk)
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

    # Allineiamo il numero di frame (prendiamo il minimo tra i due)
    n_meshes = len(meshes_or_vertices)
    n_poses = len(rotations)
    min_frames = min(n_meshes, n_poses)

    if n_meshes != n_poses:
        print(f"  [INFO] Allineamento: {n_meshes} mesh, {n_poses} frame di posa "
              f"→ uso {min_frames} frame")

    print(f"\nApplicazione rotazioni della testa a {min_frames} frame...")

    for i in tqdm(range(min_frames), desc="Rotazione"):
        item = meshes_or_vertices[i]

        # Ottieni i vertici
        if isinstance(item, trimesh.Trimesh):
            vertices = item.vertices.copy()
            faces = item.faces
        else:
            vertices = item.copy()
            faces = mesh_faces

        # Prendi i 3 angoli per questo frame
        rot = rotations[i]

        # Perno di rotazione:
        # - FLAME: l'origine (0,0,0) è posizionata al perno anatomico (base cranio/collo)
        #   → ruotare attorno all'origine produce un movimento naturale della testa
        # - Se pivot_origin=False: usa il baricentro (solo per mesh non-FLAME)
        if pivot_origin:
            t_center = np.zeros(3)
        else:
            t_center = np.mean(vertices, axis=0)

        # Matrice di Rotazione di Rodrigues
        R, _ = cv2.Rodrigues(rot.astype(np.float64))

        # Applica la rotazione a tutti i vertici attorno al perno
        rotated_vertices = R.dot((vertices - t_center).T).T + t_center

        # Crea e salva la nuova mesh
        if faces is not None:
            rotated_mesh = trimesh.Trimesh(
                vertices=rotated_vertices, faces=faces, process=False
            )
        else:
            rotated_mesh = trimesh.PointCloud(rotated_vertices)

        output_filepath = os.path.join(output_dir, f"frame_{i:05d}.ply")
        rotated_mesh.export(output_filepath)

    return min_frames


def render_video(meshes_dir, output_video, fps=30):
    """Renderizza un video dalle mesh .ply ruotate (opzionale, richiede pyrender).
    
    Args:
        meshes_dir: Cartella con le mesh frame_XXXXX.ply
        output_video: Path del video di output (.mp4)
        fps: Framerate del video
    """
    # IMPORTANTE: impostare il backend PRIMA di importare pyrender
    # su server SSH headless (senza schermo fisico) serve EGL o osmesa
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    try:
        import pyrender
    except ImportError:
        print("[WARN] pyrender non installato, video non generato")
        print("       Installa con: pip install pyrender")
        return

    mesh_files = sorted(glob.glob(os.path.join(meshes_dir, 'frame_*.ply')))
    if not mesh_files:
        print("Nessuna mesh trovata per il rendering video")
        return

    print(f"\nRendering video ({len(mesh_files)} frame a {fps} FPS)...")

    # Prova a usare ffmpeg per assemblare il video
    frames = []
    for mf in tqdm(mesh_files, desc="Rendering"):
        mesh = trimesh.load(mf, process=False)

        # Rendering con pyrender
        scene = pyrender.Scene(
            ambient_light=[0.3, 0.3, 0.3],
            bg_color=[255, 255, 255]
        )
        material = pyrender.material.MetallicRoughnessMaterial(
            alphaMode='BLEND',
            baseColorFactor=[0.3, 0.3, 0.3, 1.0],
            metallicFactor=0.8,
            roughnessFactor=0.8
        )
        render_mesh = pyrender.Mesh.from_trimesh(
            mesh, material=material, smooth=True
        )
        scene.add(render_mesh)

        # Camera
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        t_center = np.mean(mesh.vertices, axis=0)
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = t_center + [0, 0, 0.5]
        scene.add(camera, pose=camera_pose)

        # Luce
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
        scene.add(light, pose=camera_pose)

        renderer = pyrender.OffscreenRenderer(800, 800)
        color, _ = renderer.render(scene)
        frames.append(color)
        renderer.delete()

    # Scrivi il video con OpenCV
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()
    print(f"Video salvato: {output_video}")


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

  # Applica pose a vertici .npy (formato alternativo)
  python demo_nicco.py --pose_file Results/M034_disgusted_2_001.npy --vertices_npy scantalk_vertices.npy
        """
    )

    # Input
    parser.add_argument("--audio", type=str, default=None,
                        help="File audio .wav (per modalità end-to-end)")
    parser.add_argument("--pose_file", type=str, default=None,
                        help="File .npy con le pose predette (shape N_frames x 3)")
    parser.add_argument("--scantalk_dir", type=str, default=None,
                        help="Cartella output ScanTalk (con Meshes/ e Images/)")
    parser.add_argument("--vertices_npy", type=str, default=None,
                        help="File .npy con vertici ScanTalk (shape N_frames x V x 3)")

    # Output
    parser.add_argument("--output_dir", type=str, default="Demo_Finale",
                        help="Dove salvare le mesh ruotate")
    parser.add_argument("--render_video", action="store_true",
                        help="Genera anche un video .mp4 delle mesh (richiede pyrender)")
    parser.add_argument("--fps", type=int, default=30,
                        help="FPS del video di output")

    # Modello (per modalità end-to-end)
    parser.add_argument("--checkpoint", type=str, default="Saves/best_audio2pose.pth",
                        help="Path al checkpoint del modello (per --audio)")
    parser.add_argument("--device", type=str,
                        default="cuda" if __import__('torch').cuda.is_available() else "cpu")

    args = parser.parse_args()

    # --- Validazione input ---
    if args.audio is None and args.pose_file is None:
        parser.error("Serve almeno uno tra --audio e --pose_file")

    if args.scantalk_dir is None and args.vertices_npy is None:
        parser.error("Serve almeno uno tra --scantalk_dir e --vertices_npy")

    # --- 1. Ottieni le rotazioni ---
    if args.pose_file:
        print(f"\n📐 Caricamento pose da: {args.pose_file}")
        rotations = np.load(args.pose_file)
        print(f"   Shape: {rotations.shape}")
    else:
        # Modalità end-to-end: predici le pose dall'audio
        print(f"\n🎤 Modalità end-to-end: predizione pose da audio")
        print(f"   Audio: {args.audio}")

        from utils import predict_pose_from_audio
        from model import HeadPosePredictor
        import torch

        # Carica il modello
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

        # Salva le pose predette
        pose_save = os.path.join(args.output_dir, "predicted_pose.npy")
        os.makedirs(args.output_dir, exist_ok=True)
        np.save(pose_save, rotations)
        print(f"   Pose salvate in: {pose_save}")

    # --- 2. Carica le mesh/vertici di ScanTalk ---
    mesh_faces = None
    if args.scantalk_dir:
        meshes, mesh_files = load_scantalk_meshes(args.scantalk_dir)
        items = meshes
    else:
        vertex_list = load_vertices_npy(args.vertices_npy)
        items = vertex_list
        # Per vertici nudi servirebbero le facce — ma salvando come PointCloud funziona

    # --- 3. Applica le rotazioni ---
    n_processed = apply_rotations(
        items, rotations, args.output_dir, mesh_faces=mesh_faces
    )

    print(f"\n🎉 Cerchio completato! {n_processed} mesh finali (Labbra + Testa) "
          f"salvate in: {args.output_dir}")

    # --- 4. Rendering video (opzionale) ---
    if args.render_video:
        video_path = os.path.join(args.output_dir, "demo_s2p.mp4")
        render_video(args.output_dir, video_path, fps=args.fps)


if __name__ == "__main__":
    main()