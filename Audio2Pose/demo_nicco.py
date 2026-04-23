import os
import glob
import numpy as np
import cv2
import trimesh
from tqdm import tqdm
import argparse

def applica_rotazione(scantalk_meshes_dir, pose_npy_path, output_dir):
    # 1. Creiamo la cartella di output
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Carichiamo le rotazioni predette dal TUO modello
    # Shape attesa: (numero_frame, 3) -> Pitch, Yaw, Roll
    rotations = np.load(pose_npy_path)
    
    # 3. Troviamo tutte le mesh generate da ScanTalk
    mesh_files = sorted(glob.glob(os.path.join(scantalk_meshes_dir, '*.ply')))
    
    if len(mesh_files) == 0:
        print(f"Errore: Nessuna mesh .ply trovata in {scantalk_meshes_dir}")
        return
        
    print(f"Trovate {len(mesh_files)} mesh di ScanTalk e {len(rotations)} frame di posa.")
    
    # Allineiamo il numero di frame (prendiamo il minimo tra i due)
    min_frames = min(len(mesh_files), len(rotations))
    
    print("Applicazione rotazioni della testa in corso...")
    
    for i in tqdm(range(min_frames)):
        # A. Carica la singola mesh di ScanTalk
        mesh = trimesh.load(mesh_files[i], process=False)
        
        # B. Prendi i 3 angoli per questo specifico frame
        rot = rotations[i] 
        
        # C. Trova il centro della testa (t_center) per ruotare sul perno giusto
        t_center = np.mean(mesh.vertices, axis=0)
        
        # D. LA MAGIA MATEMATICA DI FEDERICO: Matrice di Rotazione di Rodrigues
        R, _ = cv2.Rodrigues(rot)
        
        # Applica la rotazione a tutti i vertici della faccia
        mesh.vertices = R.dot((mesh.vertices - t_center).T).T + t_center
        
        # E. Salva la nuova mesh inclinata
        output_filepath = os.path.join(output_dir, f"frame_{i:04d}.ply")
        mesh.export(output_filepath)
        
    print(f"\nCerchio completato! Le mesh finali (Labbra + Testa) sono in: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Unisci ScanTalk e Audio2Pose')
    parser.add_argument("--scantalk_dir", type=str, required=True, help='Cartella con le mesh .ply di ScanTalk')
    parser.add_argument("--pose_file", type=str, required=True, help='File .npy generato dal tuo Audio2Pose')
    parser.add_argument("--output_dir", type=str, default="Demo_Finale", help='Dove salvare il risultato')
    
    args = parser.parse_args()
    applica_rotazione(args.scantalk_dir, args.pose_file, args.output_dir)