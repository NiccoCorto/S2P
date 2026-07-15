"""
render_npy.py - Script per renderizzare file .npy di vertici in formato video (Point Cloud)
Utile per renderizzare 'vertices_pose' o 'vertices' del dataset EMOTE, che sono sprovvisti di facce (topologia mesh).
"""
import os
import argparse
import numpy as np
import cv2
from tqdm import tqdm
def render_with_pyrender(vertices, output_mp4, fps=30):
    """Renderizza usando pyrender (modalità PointCloud). Veloce ma richiede OpenGL/EGL."""
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    try:
        import pyrender
    except ImportError:
        print("[ERR] pyrender non installato. Installa con: pip install pyrender")
        return False
        
    num_frames = vertices.shape[0]
    frames = []
    
    # Calcola un centro medio per la camera (basato sul primo frame)
    mean_center = np.mean(vertices[0], axis=0)
    
    print(f"Rendering di {num_frames} frame con Pyrender...")
    renderer = pyrender.OffscreenRenderer(800, 800)
    
    for i in tqdm(range(num_frames)):
        scene = pyrender.Scene(bg_color=[0, 0, 0])
        
        pts = vertices[i]
        colors = np.ones((pts.shape[0], 4))  # Colore bianco opaco
        
        # Aggiungiamo i punti come Point Cloud
        m = pyrender.Mesh.from_points(pts, colors=colors)
        scene.add(m)
        
        # Impostiamo la camera di fronte al volto
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        camera_pose = np.eye(4)
        # La distanza Z (0.5) potrebbe dover essere aggiustata a seconda della scala dei vertici
        camera_pose[:3, 3] = mean_center + [0, 0, 0.4] 
        scene.add(camera, pose=camera_pose)
        
        color, _ = renderer.render(scene)
        frames.append(color)
        
    renderer.delete()
    
    # Salvataggio Video
    print(f"Salvataggio video in {output_mp4}...")
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_mp4, fourcc, fps, (w, h))
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()
    return True
def render_with_matplotlib(vertices, output_mp4, fps=30):
    """Renderizza usando matplotlib (lento, ma infallibile sui server SSH senza GPU)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    
    num_frames = vertices.shape[0]
    print(f"Rendering di {num_frames} frame con Matplotlib (potrebbe richiedere minuti)...")
    
    fig = plt.figure(figsize=(6, 6))
    # Colore di sfondo nero per miglior contrasto
    fig.patch.set_facecolor('black')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    
    # Disabilita gli assi
    ax.set_axis_off()
    
    # Calcola i limiti globali in modo che la camera non "salti"
    max_val = np.max(np.abs(vertices))
    ax.set_xlim([-max_val, max_val])
    ax.set_ylim([-max_val, max_val])
    ax.set_zlim([-max_val, max_val])
    
    # Inizializza lo scatter col primo frame
    scatter = ax.scatter(vertices[0, :, 0], vertices[0, :, 1], vertices[0, :, 2], 
                         s=0.5, c='white', alpha=0.8)
    
    # Imposta la vista (dipende dal sistema di coordinate)
    # Per EMOTE/FLAME solitamente la Z è avanti e la Y è su
    ax.view_init(elev=0, azim=-90)
    
    def update(frame):
        # matplotlib 3D scatter non accetta (N,3) in set_offsets, ma 2D. 
        # Per 3D bisogna impostare ._offsets3d
        scatter._offsets3d = (vertices[frame, :, 0], vertices[frame, :, 1], vertices[frame, :, 2])
        return scatter,
        
    ani = animation.FuncAnimation(fig, update, frames=num_frames, blit=False)
    
    print(f"Salvataggio video in {output_mp4}...")
    ani.save(output_mp4, fps=fps, writer='ffmpeg')
    plt.close(fig)
    return True
def main():
    parser = argparse.ArgumentParser(description="Renderizza un file .npy (Vertices) in formato .mp4")
    parser.add_argument("--npy", type=str, required=True, help="Path al file .npy (es. vertices_pose/M034_xxx.npy)")
    parser.add_argument("--out", type=str, required=True, help="Path del video di output (es. output.mp4)")
    parser.add_argument("--fps", type=int, default=30, help="Framerate")
    parser.add_argument("--backend", type=str, choices=["pyrender", "matplotlib"], default="pyrender",
                        help="Backend di rendering. Pyrender è veloce ma richiede OpenGL/EGL sul server. Matplotlib è lento ma sicuro.")
    
    args = parser.parse_args()
    
    print(f"Caricamento {args.npy}...")
    try:
        vertices = np.load(args.npy)
    except Exception as e:
        print(f"Errore nel caricamento del file: {e}")
        return
        
    if vertices.ndim != 3 or vertices.shape[2] != 3:
        print(f"Errore: Il file deve avere shape (Frames, N_vertici, 3). Trovata: {vertices.shape}")
        return
        
    if args.backend == "pyrender":
        success = render_with_pyrender(vertices, args.out, args.fps)
        if not success:
            print("Fallback su Matplotlib...")
            render_with_matplotlib(vertices, args.out, args.fps)
    else:
        render_with_matplotlib(vertices, args.out, args.fps)
        
    print("Fatto!")
if __name__ == "__main__":
    main()