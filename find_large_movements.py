import os
import pickle
import numpy as np

def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    cache_file = os.path.join(project_dir, "s2p_cache.pkl")
    
    if not os.path.exists(cache_file):
        print(f"Errore: File cache non trovato in {cache_file}")
        print("Assicurati di lanciare lo script sullo stesso server dove hai fatto il training.")
        return

    print(f"Caricamento cache da {cache_file}...")
    with open(cache_file, "rb") as f:
        cached = pickle.load(f)
    
    train_data = cached["train"]
    test_data = cached["test"]

    def get_top_movements(data_list, top_k=5):
        scores = []
        for item in data_list:
            pose = item["pose"]  # shape (N_frames, 3)
            # Calcola la varianza per ogni asse (Pitch, Yaw, Roll)
            variances = np.var(pose, axis=0)
            # Somma delle varianze come score complessivo di "movimento ampio"
            total_variance = np.sum(variances)
            
            # Oppure calcoliamo l'escursione massima (max - min)
            ranges = np.ptp(pose, axis=0)
            total_range = np.sum(ranges)
            
            scores.append({
                "name": item["name"],
                "score": total_variance,
                "range": total_range,
                "frames": len(pose)
            })
        
        # Ordina per score decrescente
        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores[:top_k]

    print("\n--- TOP 5 FILE CON MOVIMENTI AMPI (TRAINING SET) ---")
    top_train = get_top_movements(train_data)
    for i, res in enumerate(top_train):
        print(f"{i+1}. {res['name']} | Varianza Totale: {res['score']:.4f} | Range Massimo: {res['range']:.4f} | Frame: {res['frames']}")

    print("\n--- TOP 5 FILE CON MOVIMENTI AMPI (TEST SET) ---")
    top_test = get_top_movements(test_data)
    for i, res in enumerate(top_test):
        print(f"{i+1}. {res['name']} | Varianza Totale: {res['score']:.4f} | Range Massimo: {res['range']:.4f} | Frame: {res['frames']}")

if __name__ == "__main__":
    main()
