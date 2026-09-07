import torch

def axis_angle_to_matrix(r):
    """
    Converte vettori axis-angle (es. estratti da cv2.Rodrigues) in matrici di rotazione 3x3.
    Formula di Eulero-Rodrigues differenziabile, ottimizzata per PyTorch.
    
    Args:
        r: Tensor di shape (..., 3) contenente i vettori di rotazione [Pitch, Yaw, Roll].
        
    Returns:
        R: Tensor di shape (..., 3, 3) con le matrici di rotazione.
    """
    # Calcola l'angolo di rotazione (norma del vettore)
    theta = torch.norm(r, dim=-1, keepdim=True) # (..., 1)
    
    # Asse di rotazione normalizzato (divisione sicura)
    u = r / torch.clamp(theta, min=1e-6) # (..., 3)
    
    # Componenti dell'asse
    ux, uy, uz = u[..., 0], u[..., 1], u[..., 2]
    zero = torch.zeros_like(ux)
    
    # Matrice antisimmetrica (Cross-product matrix K)
    K = torch.stack([
        torch.stack([zero, -uz, uy], dim=-1),
        torch.stack([uz, zero, -ux], dim=-1),
        torch.stack([-uy, ux, zero], dim=-1)
    ], dim=-2) # (..., 3, 3)
    
    # Matrice Identità
    I = torch.eye(3, device=r.device, dtype=r.dtype).expand_as(K)
    
    theta = theta.unsqueeze(-1) # (..., 1, 1) per broadcasting corretto
    
    # K^2
    K_sq = torch.matmul(K, K)
    
    # Formula di Rodrigues: R = I + sin(theta) * K + (1 - cos(theta)) * K^2
    R = I + torch.sin(theta) * K + (1 - torch.cos(theta)) * K_sq
    
    # Per angoli minuscoli, R approssima l'Identità.
    # Questo previene gradienti NaN causati da divisioni instabili rimosse sopra.
    mask = (theta < 1e-6).expand_as(R)
    R = torch.where(mask, I, R)
    
    return R
