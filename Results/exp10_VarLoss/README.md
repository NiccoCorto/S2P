# Esperimento 10: VarLoss (One-Hot + FaceLoss + VarLoss)

## Obiettivo

Combinare tre meccanismi per produrre movimenti della testa più vivaci e identità-dipendenti:

1. **One-Hot Encoding** degli speaker (solo identità di *train*, isolando val e test)
2. **FaceLoss** — errore calcolato sui 5023 vertici 3D ruotati (Forward Kinematics differenziabile) anziché direttamente sugli angoli
3. **VarLoss** — nuovo termine che penalizza la differenza di varianza temporale tra GT e predizione, misurata sui vertici 3D

La motivazione principale è combattere la **staticità centrale**: i modelli precedenti tendevano a predire pose quasi costanti attorno alla posizione media, probabilmente perché la sola pos loss viene minimizzata restando vicini alla media del dataset.

---

## Differenze rispetto a DiffPoseData (panoramica)

| Feature | DiffPoseData (exp8 OneHot) | **exp10 VarLoss** |
|---|---|---|
| Loss principale | PosLoss — MSE sugli angoli axis-angle | **FaceLoss — MSE sui vertici 3D ruotati** |
| Velocity loss | Su angoli (Δangolo_t) | **Sui vertici 3D (ΔV_t)** |
| VarLoss | Assente | **Aggiunta: MSE tra Var_t(V_pred) e Var_t(V_target)** |
| One-Hot speaker | Tutti gli ID (train + val + test) | **Solo ID di train; val/test → 1/N uniforme** |
| geometry.py | Assente | **Aggiunto: axis_angle_to_matrix (Rodrigues)** |

---

## Cambiamenti implementativi dettagliati

### `Audio2Pose/model.py` — gestione speaker sconosciuti

In `DiffPoseData`, il `speaker_mapping.json` veniva costruito su **tutti** i file audio del dataset (train + val + test), per cui ogni identità aveva sempre un ID valido. Il one-hot era quindi sempre un vettore standard `e_i`.

In questo branch il mapping viene costruito **solo sulle identità di training** (vedi `data_loader.py`). Le identità di val/test non viste in train ricevono `speaker_id = -1` dal dataloader.

Il nuovo metodo `_build_speaker_vector` gestisce questo caso:

```python
def _build_speaker_vector(self, speaker_ids, target_len):
    N = self.num_speakers
    # default: vettore uniforme 1/N per tutti (gestisce speaker=-1)
    speaker_vec = torch.full((B, N), fill_value=1.0 / N, ...)

    # sovrascrive solo le righe con ID validi (>= 0) con il one-hot puntuale
    known_mask = speaker_ids >= 0
    if known_mask.any():
        speaker_vec[known_mask] = F.one_hot(speaker_ids[known_mask], N).float()

    return speaker_vec.unsqueeze(1).expand(-1, target_len, -1)
```

**Perché `1/N` invece di zeri?**  
Il vettore di zeri annullerebbe completamente il contributo del canale speaker, il che è un caso speciale non previsto nell'embedding. Il vettore uniforme `1/N` è il **prior non-informativo**: il modello riceve un segnale speaker "neutro", senza favorire alcuna identità specifica. È matematicamente la media di tutti i possibili one-hot, coerente con il fatto che non sappiamo l'identità del soggetto.

---

### `Audio2Pose/train.py` — CombinedLoss (FaceLoss + VarLoss)

#### Perché FaceLoss invece di PosLoss?

La `PosLoss` originale calcola MSE direttamente sugli angoli axis-angle `(pitch, yaw, roll)`. Il problema è che la relazione tra errore angolare e spostamento spaziale percepito **non è lineare**: un errore di 1° sul roll ha un effetto molto diverso da un errore di 1° sul pitch, a seconda della posizione della testa.

La `FaceLoss` calcola l'errore nello **spazio dei vertici 3D**, dove ogni spostamento è direttamente proporzionale alla distanza euclidea percepita sullo schermo.

#### Forward Kinematics (condiviso tra FaceLoss e VarLoss)

Per efficienza, `CombinedLoss` esegue la rotazione **una sola volta** per batch:

```python
# Converte angoli axis-angle → matrici di rotazione 3×3
R_pred   = axis_angle_to_matrix(predictions)   # (B, T, 3, 3)
R_target = axis_angle_to_matrix(target)         # (B, T, 3, 3)

# Ruota i 5023 vertici della faccia canonica con ogni rotazione
# canonical_face: (N=5023, 3)  →  V: (B, T, N, 3)
V_pred   = canonical_face @ R_pred.transpose(-1, -2)
V_target = canonical_face @ R_target.transpose(-1, -2)
```

La formula di Rodrigues differenziabile (`axis_angle_to_matrix`) permette il backpropagation attraverso la rotazione.

**Nota sulle coordinate FLAME**: poiché sia predizione che GT vengono ruotate a partire dalla stessa `canonical_face`, l'offset assoluto della mesh si annulla nel confronto. La loss misura la differenza di orientamento, non la posizione assoluta nello spazio.

#### FaceLoss (pos + vel)

```
pos_loss = MSE(V_pred[frame_reali], V_target[frame_reali])
vel_loss = MSE(ΔV_pred[frame_reali], ΔV_target[frame_reali])
         dove ΔV_t = V_t - V_{t-1}   (differenze temporali sui vertici)
```

La padding mask garantisce che i frame aggiunti dal collate (zero-padding) non contribuiscano alla loss.

#### VarLoss — il nuovo termine

**Motivazione**: una loss che misura solo posizione e velocità frame-per-frame può essere minimizzata da predizioni quasi costanti (media del dataset). Aggiungere un termine che penalizza la differenza di **varianza temporale** costringe il modello a riprodurre anche l'ampiezza delle oscillazioni presenti nel GT.

**Calcolo**:

Per ogni sequenza `b` nel batch, usando solo i `real_len` frame reali (escludendo il padding):

```
V_pred[b]    → shape (real_len, N, 3)
V_target[b]  → shape (real_len, N, 3)

var_pred[b]   = Var_t(V_pred[b])    → shape (N, 3)   (varianza lungo dim=0, cioè il tempo)
var_target[b] = Var_t(V_target[b])  → shape (N, 3)

var_loss = MSE(stack(var_pred), stack(var_target))   → scalare
```

La varianza è calcolata come varianza di popolazione (`unbiased=False`) per evitare instabilità numeriche con sequenze molto corte. Il risultato scalare è la distanza quadratica media tra la "distribuzione spaziale" predetta e quella del GT.

**Intuitivamente**: se il GT ha la testa che oscilla ±10° e il modello predice ±1°, `var_target >> var_pred` → `var_loss` alta → gradiente che spinge ad aumentare l'ampiezza del movimento.

**Loss totale**:
```
total = pos_loss + vel_weight × vel_loss + var_weight × var_loss
```

---

## Iperparametri — Esperimento 1

| Parametro | Valore | Note |
|---|---|---|
| `hidden_dim` | 256 | Invariato da DiffPoseData |
| `num_layers` | 2 | Invariato da DiffPoseData |
| `dropout` | 0.1 | Invariato da DiffPoseData |
| `lr` | 1e-4 | Fisso, nessuno scheduler |
| `batch_size` | 4 | Invariato |
| `max_epoch` | 200 | Invariato |
| `patience` | 200 | Early stopping disabilitato di fatto |
| **`vel_loss_weight`** | **1.0** | Peso della velocity loss sui vertici |
| **`var_loss_weight`** | **5.0** | Peso della VarLoss — più aggressivo per combattere la staticità |

**Motivazione dei pesi**:
- `vel_weight = 1.0`: la velocity loss è nello stesso spazio dei vertici (mm²), un peso di 1.0 è un punto di partenza bilanciato.
- `var_weight = 5.0`: la varianza tende ad essere numericamente più piccola della pos_loss (per via della quadratura), quindi un peso più alto (5×) è necessario per darle abbastanza influenza sul gradiente. Se la faccia è ancora troppo statica, aumentare a 10.

---

## Metriche monitorate su CometML

| Metrica | Significato |
|---|---|
| `train_pos_loss` | MSE sui vertici 3D (frame reali di train) |
| `train_vel_loss` | MSE sulle differenze di vertici tra frame consecutivi |
| `train_var_loss` | MSE tra varianza temporale GT e pred sui vertici |
| `val_pos_loss` | Come sopra, su validation |
| `val_vel_loss` | Come sopra, su validation |
| `val_var_loss` | Come sopra, su validation |

---

## Come avviare

```bash
git checkout VarLoss

# IMPORTANTE: elimina la cache obsoleta (contiene il vecchio speaker_mapping con tutti gli ID)
rm -f s2p_lazy_cache.pkl speaker_mapping.json

./run_VarLoss.sh
```

---

## Tuning del peso VarLoss (esperimenti futuri)

| `var_weight` | Effetto atteso |
|---|---|
| `0.0` | VarLoss disabilitata (= FaceLoss + OneHot puro) |
| `1.0` | Pressione lieve, conservativa |
| **`5.0`** | **← Partenza (exp10 default)** — pressione moderata |
| `10.0` | Pressione forte; monitorare che `pos_loss` non esploda |
| `50.0` | Solo se la staticità persiste; rischio instabilità |

---

## Struttura della CombinedLoss

```
CombinedLoss.forward(predictions, target, lengths)
│
├─ axis_angle_to_matrix(predictions) → R_pred   (B, T, 3, 3)
├─ axis_angle_to_matrix(target)      → R_target (B, T, 3, 3)
├─ canonical_face @ R_pred^T         → V_pred   (B, T, N, 3)  ← FK una sola volta
├─ canonical_face @ R_target^T       → V_target (B, T, N, 3)
│
├─ FaceLoss
│   ├─ pos_loss = MSE(V_pred[mask], V_target[mask])
│   └─ vel_loss = MSE(ΔV_pred[mask], ΔV_target[mask])
│
├─ VarLoss
│   ├─ per ogni b: var_pred[b]   = Var_t(V_pred[b, :real_len])    → (N, 3)
│   ├─ per ogni b: var_target[b] = Var_t(V_target[b, :real_len])  → (N, 3)
│   └─ var_loss = MSE(stack(var_pred), stack(var_target))
│
└─ total = pos_loss + 1.0 × vel_loss + 5.0 × var_loss
```
