"""
render_comparison.py — Pipeline di confronto finale S2P vs Ground Truth

Per ogni esperimento (exp_vel0, exp_vel0.5, exp_vel1, exp_vel2) questo script:
  1. Trova automaticamente il checkpoint dell'ultima epoca addestrata.
  2. Predice gli angoli di rotazione testa dall'audio con Audio2Pose.
  3. Applica la rotazione ai vertici statici di ScanTalk (--vertices).
  4. Renderizza i vertici GT del dottorando (--vertices_pose) con PyRender.
  5. Renderizza la nostra predizione con PyRender.
  6. Unisce i due video affiancati in un unico confronto.mp4 con ffmpeg.

Uso:
  python Audio2Pose/render_comparison.py \\
      --audio /path/to/audio.wav \\
      --vertices /path/to/vertices/M034_disgusted_001.npy \\
      --vertices_pose /path/to/vertices_pose/M034_disgusted_001.npy \\
      --template ScanTalk/src/examples/FLAME_sample.ply \\ # thanos ha troppi vertici (9k)
      --saves_dir Saves \\
      --results_dir Results \\
      --fps 30
"""
import os
import sys
import glob
import argparse
import tempfile
import numpy as np
import cv2
import trimesh
import torch
import librosa

os.environ['PYOPENGL_PLATFORM'] = 'egl'

import pyrender
from subprocess import call
from tqdm import tqdm


class MeshLike:
    # fix per GLUT, non possibile per ssh, classe che replica interfaccia .v e .f necessarie per il renderer

    def __init__(self, v, f):
        self.v = np.array(v, dtype=np.float64)
        self.f = np.array(f)

# path per importare model e utils del progetto S2P
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import HeadPosePredictor
from utils import predict_pose_from_audio, apply_rotation_to_vertices


# RENDERING (estratto da ScanTalk/src/demo.py)


def render_mesh_helper(mesh, t_center, rot=np.zeros(3)):
    """Renderizza una singola mesh 3D in un frame 2D (800x800 px) usando PyRender."""
    camera_params = {
        'c': np.array([400, 400]),
        'k': np.array([-0.19816071, 0.92822711, 0, 0, 0]),
        'f': np.array([4754.97941935 / 2, 4754.97941935 / 2])
    }
    frustum = {'near': 0.01, 'far': 3.0, 'height': 800, 'width': 800}
    intensity = 2.0

    rotated_v = cv2.Rodrigues(rot)[0].dot((np.array(mesh.v) - t_center).T).T + t_center
    mesh_copy = MeshLike(rotated_v, mesh.f)

    primitive_material = pyrender.material.MetallicRoughnessMaterial(
        alphaMode='BLEND',
        baseColorFactor=[0.3, 0.3, 0.3, 1.0],
        metallicFactor=0.8,
        roughnessFactor=0.8
    )

    tri_mesh = trimesh.Trimesh(vertices=mesh_copy.v, faces=mesh_copy.f)
    render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=primitive_material, smooth=True)

    scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[255, 255, 255])

    camera = pyrender.IntrinsicsCamera(
        fx=camera_params['f'][0], fy=camera_params['f'][1],
        cx=camera_params['c'][0], cy=camera_params['c'][1],
        znear=frustum['near'], zfar=frustum['far']
    )

    scene.add(render_mesh, pose=np.eye(4))
    scene.add(camera, pose=[[1,0,0,0],[0,1,0,0],[0,0,1,1],[0,0,0,1]])

    light_color = np.array([1., 1., 1.])
    light = pyrender.DirectionalLight(color=light_color, intensity=intensity)
    angle = np.pi / 6.0
    pos = np.array([0, 0, 1.0])
    for rot_vec in [np.zeros(3), [angle,0,0], [-angle,0,0], [0,-angle,0], [0,angle,0]]:
        lp = np.eye(4)
        lp[:3, 3] = cv2.Rodrigues(np.array(rot_vec))[0].dot(pos)
        scene.add(light, pose=lp.copy())

    flags = pyrender.RenderFlags.SKIP_CULL_FACES
    try:
        r = pyrender.OffscreenRenderer(
            viewport_width=frustum['width'], viewport_height=frustum['height']
        )
        color, _ = r.render(scene, flags=flags)
        r.delete()
    except Exception as e:
        print(f"  pyrender: frame fallito — {e}")
        color = np.zeros((frustum['height'], frustum['width'], 3), dtype='uint8')

    return color[..., ::-1]  # RGB → BGR per OpenCV


def render_vertices_to_video(vertices_array, faces, audio_path, out_video_path, fps=30):
    """Renderizza una sequenza di vertici (N, V, 3) in un video .mp4 con audio.

    Args:
        vertices_array: np.ndarray (N_frames, N_vertices, 3)
        faces:          np.ndarray (F, 3) — facce della topologia FLAME
        audio_path:     Path al file .wav da allegare al video
        out_video_path: Path di output del .mp4 finale
        fps:            Frame per secondo del video
    """
    out_dir = os.path.dirname(out_video_path)
    os.makedirs(out_dir, exist_ok=True)

    tmp_video = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir=out_dir, delete=False)
    writer = cv2.VideoWriter(
        tmp_video.name,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps, (800, 800), True
    )

    center = np.mean(vertices_array[0], axis=0)
    n_frames = vertices_array.shape[0]

    for i in tqdm(range(n_frames), desc=f"  Rendering {os.path.basename(out_video_path)}"):
        mesh = MeshLike(vertices_array[i], faces)
        frame = render_mesh_helper(mesh, center)
        writer.write(frame)

    writer.release()

    # attacca l'audio con ffmpeg
    cmd = (
        '/usr/bin/ffmpeg -y '
        f'-i {audio_path} '
        f'-i {tmp_video.name} '
        '-vcodec h264 -ac 2 -channel_layout stereo '
        f'-pix_fmt yuv420p -ar 22050 {out_video_path}'
    ).split()
    call(cmd)
    os.remove(tmp_video.name)
    print(f"  Salvato: {out_video_path}")


def merge_side_by_side(left_video, right_video, output_path, label_left="GT", label_right="Pred"):
    """Affianca due video .mp4 orizzontalmente in un unico file confronto.mp4."""
    print(f"\n  Unione affiancata → {output_path}")
    cmd = (
        f'/usr/bin/ffmpeg -y '
        f'-i {left_video} -i {right_video} '
        f'-filter_complex "[0:v]drawtext=text={label_left}:fontsize=40:fontcolor=black:'
        f'x=10:y=10[l];[1:v]drawtext=text={label_right}:fontsize=40:fontcolor=black:'
        f'x=10:y=10[r];[l][r]hstack=inputs=2[v]" '
        f'-map "[v]" -map 0:a '
        f'-c:v h264 -c:a copy {output_path}'
    )
    call(cmd, shell=True)
    print(f"  Confronto salvato: {output_path}")




def find_last_epoch_checkpoint(saves_dir, exp_name):
    """Trova il file audio2pose_epoch_N.pth con l'N più alto dentro saves_dir/exp_name."""
    exp_dir = os.path.join(saves_dir, exp_name)
    pattern = os.path.join(exp_dir, "audio2pose_epoch_*.pth")
    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(
            f"Nessun checkpoint trovato in {exp_dir}. "
            f"Assicurati che il training sia completato."
        )

    # estrai il numero di epoca dal nome file e prendi il massimo
    def epoch_num(f):
        base = os.path.basename(f)  # audio2pose_epoch_40.pth
        return int(base.replace("audio2pose_epoch_", "").replace(".pth", ""))

    best = max(files, key=epoch_num)
    print(f"  Checkpoint selezionato: {best} (epoca {epoch_num(best)})")
    return best


def run_experiment(exp_name, vel_label, args):
    """Pipeline completa per un singolo esperimento.

    Args:
        exp_name:  Nome della cartella Saves (es. "exp_vel0")
        vel_label: Etichetta leggibile per la cartella output (es. "0")
        args:      Namespace con tutti i path e le config
    """
    print(f"\n{'='*60}")
    print(f" Esperimento: {exp_name}  (velocity_loss = {vel_label})")
    print(f"{'='*60}")

    # cartella di output per questo esperimento
    out_subdir = os.path.join(args.results_dir, f"esperimento_{vel_label}_vel")
    os.makedirs(out_subdir, exist_ok=True)

    #  1. carica checkpoint dell'ultima epoca 
    checkpoint_path = find_last_epoch_checkpoint(args.saves_dir, exp_name)

    # 2. carica il modello Audio2Pose 
    print("  Caricamento modello Audio2Pose...")
    model = HeadPosePredictor()
    model.load_state_dict(torch.load(checkpoint_path, map_location=args.device))
    model = model.to(args.device)
    model.eval()

    #  3. Carica i vertici GT e template faces 
    print("  Caricamento vertici...")
    vertices_static = np.load(args.vertices)      # (N, V, 3) — ScanTalk senza pose
    vertices_pose   = np.load(args.vertices_pose) # (N, V, 3) — GT 

    template = trimesh.load(args.template, process=False)
    faces = np.array(template.faces)

    print(f"  vertices (statici):  {vertices_static.shape}")
    print(f"  vertices_pose (GT):  {vertices_pose.shape}")

    #  4. Predici le rotazioni dall'audio 
    print("  Predizione rotazioni testa dall'audio...")
    n_target_frames = vertices_static.shape[0]
    rotations = predict_pose_from_audio(
        model, args.audio, device=args.device, target_frames=n_target_frames
    )
    print(f"  Rotazioni predette: {rotations.shape}")

    # salva le pose predette 
    np.save(os.path.join(out_subdir, "predicted_pose.npy"), rotations)

    #  5. Applica la rotazione ai vertici statici ScanTalk 
    print("  Applicazione rotazione testa ai vertici ScanTalk...")
    n_frames = min(vertices_static.shape[0], rotations.shape[0])
    vertices_rotated = np.zeros((n_frames, vertices_static.shape[1], 3))

    for i in range(n_frames):
        vertices_rotated[i] = apply_rotation_to_vertices(
            vertices_static[i], rotations[i], use_origin=True
        )

    #  6. Rendering GT 
    print("\n Rendering Ground Truth (vertices_pose)...")
    gt_video_path = os.path.join(out_subdir, "gt_video.mp4")
    render_vertices_to_video(
        vertices_pose[:n_frames], faces, args.audio, gt_video_path, fps=args.fps
    )

    #  7. Rendering Predizione
    print("\n  Rendering Predizione Audio2Pose...")
    pred_video_path = os.path.join(out_subdir, "pred_video.mp4")
    render_vertices_to_video(
        vertices_rotated, faces, args.audio, pred_video_path, fps=args.fps
    )

    #  8. Unisci i due video affiancati 
    confronto_path = os.path.join(out_subdir, "confronto.mp4")
    merge_side_by_side(
        gt_video_path, pred_video_path, confronto_path,
        label_left="GT", label_right=f"vel={vel_label}"
    )

    print(f"\n  Esperimento {exp_name} completato. Output in: {out_subdir}")


def main():
    parser = argparse.ArgumentParser(
        description="S2P — Rendering e confronto GT vs Audio2Pose"
    )

    # input dati
    parser.add_argument("--audio", type=str, required=True,
                        help="File .wav del parlante")
    parser.add_argument("--vertices", type=str, required=True,
                        help="File .npy vertici ScanTalk statici (N_frames, V, 3)")
    parser.add_argument("--vertices_pose", type=str, required=True,
                        help="File .npy vertici GT dottorando (N_frames, V, 3)")
    parser.add_argument("--template", type=str,
                        default="../ScanTalk/src/examples/FLAME_sample.ply",
                        help="Mesh template per recuperare le facce della topologia FLAME")

    # path progetto
    parser.add_argument("--saves_dir", type=str, default="Saves",
                        help="Cartella radice dei checkpoint (contiene exp_vel*/)")
    parser.add_argument("--results_dir", type=str, default="Results",
                        help="Cartella di output dove salvare i video")

    # config video
    parser.add_argument("--fps", type=int, default=30,
                        help="Frame per secondo del video di output")

    # device
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")

    # lista esperimenti: nome cartella e label
    parser.add_argument("--experiments", type=str, nargs="+",
                        default=["exp_vel0:0", "exp_vel0.5:0.5", "exp_vel1:1", "exp_vel2:2"],
                        help="Lista di esperimenti nel formato nome_cartella:label_vel")

    args = parser.parse_args()

    print(f"\nS2P — Rendering Confronto")
    print(f"  Audio:          {args.audio}")
    print(f"  Vertices:       {args.vertices}")
    print(f"  Vertices GT:    {args.vertices_pose}")
    print(f"  Template:       {args.template}")
    print(f"  Device:         {args.device}")
    print(f"  Esperimenti:    {args.experiments}\n")

    for exp_str in args.experiments:
        parts = exp_str.split(":")
        exp_name  = parts[0]
        vel_label = parts[1] if len(parts) > 1 else parts[0]
        run_experiment(exp_name, vel_label, args)

    print(f"\n{'='*60}")
    print(f" Tutti gli esperimenti completati!")
    print(f" I video confronto.mp4 si trovano in: {args.results_dir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
