# Esperimento 10: VarLoss (One-Hot + FaceLoss + VarLoss)

## Obiettivo

Combinare tre meccanismi per produrre movimenti della testa più vivaci e identità-dipendenti:

1. **One-Hot Encoding** degli speaker (solo identità di *train*)
2. **FaceLoss** — errore sui 5023 vertici 3D ruotati (Forward Kinematics differenziabile)
3. **VarLoss** — penalizza la differenza tra varianza temporale GT e predizione (calcolata sui vertici 3D)

## Motivazione

Gli esperimenti precedenti mostravano una tendenza alla **staticità centrale**: il modello prediceva pose quasi costanti attorno alla posizione media. La VarLoss attacca direttamente questo problema penalizzando sequenze predette con varianza temporale diversa dalla GT.

## Setup

| Parametro | Valore |
|---|---|
| Branch | `VarLoss` |
| Dataset | `HDTF_TFHP_Elaborated_Pose` |
| Split | `splits/train.txt`, `splits/val.txt`, `splits/test.txt` |
| Speaker mapping | **Solo identità di train** (875 video → N identità uniche) |
| Hidden dim | 256 |
| LSTM layers | 2 |
| Dropout | 0.1 |
| LR | 1e-4 (fisso) |
| Batch size | 4 |
| Max epoch | 200 |

## Loss

```
total_loss = face_pos_loss + vel_weight * face_vel_loss + var_weight * var_loss
```

### FaceLoss
- Carica `canonical_face.npy` (5023 vertici FLAME)
- Applica rotazione axis-angle con la formula di Rodrigues differenziabile
- MSE sui vertici ruotati (posizione + velocità tra frame consecutivi)
- Gestisce padding mask

### VarLoss
- Calcola la varianza temporale dei vertici 3D ruotati per ogni sequenza
- MSE tra `Var_t(V_pred)` e `Var_t(V_target)` (shape: B × N × 3)
- Incoraggia il modello a produrre sequenze con la stessa "ampiezza di oscillazione" della GT

## Gestione Speaker Sconosciuti

| Caso | Vettore speaker |
|---|---|
| Identità in train | One-hot `e_i` (vettore standard) |
| Identità non in train (val/test) | Uniforme `1/N` (prior uniforme, nessuna identità favorita) |

Questo approccio è **matematicamente corretto**: non favorisce nessuna identità specifica e mantiene l'informazione audio come driver principale della predizione.

## Differenze rispetto agli esperimenti precedenti

| Feature | exp8 (OneHot) | exp9 (FaceLoss) | **exp10 (VarLoss)** |
|---|---|---|---|
| Loss | PosLoss (angoli) | FaceLoss (vertici) | **FaceLoss + VarLoss** |
| One-Hot | ✓ (tutti gli ID) | ✗ | **✓ (solo train ID)** |
| Speaker sconosciuti | — | — | **1/N uniforme** |
| VarLoss | ✗ | ✗ | **✓** |

## Come avviare

```bash
# 1. Passa al branch corretto
git checkout VarLoss

# 2. IMPORTANTE: elimina la cache obsoleta
rm -f s2p_lazy_cache.pkl speaker_mapping.json

# 3. Avvia l'esperimento
./run_VarLoss.sh
```

## Tuning del peso VarLoss

Il peso `--var_loss_weight` controlla quanto la VarLoss influenza il training.

| Valore | Effetto atteso |
|---|---|
| `0.0` | VarLoss disabilitata (equivalente a FaceLoss + OneHot puro) |
| `1.0` | Partenza conservativa, leggera spinta verso varianza GT |
| `5.0` | Pressione moderata, consigliato se la faccia rimane statica |
| `10.0` | Pressione forte, rischio di instabilità se pos_loss diventa secondaria |

## Note tecniche

- Il Forward Kinematics (Rodrigues) viene eseguito **una sola volta** per batch: FaceLoss e VarLoss
  condividono i tensori `V_pred` e `V_target` tramite `CombinedLoss`.
- La varianza è calcolata **per sequenza** (non sul batch intero), rispettando le lunghezze reali
  (i frame paddati sono esclusi dal calcolo).
- L'identità `extract_identity()` usa il formato HDTF: `RD_NomeCognome_000` → identità = `NomeCognome`.
