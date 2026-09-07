# Esperimento 9: Face Loss (Vertex Loss / Geodesic Loss)

Questo documento illustra l'ultimo approccio implementato sul branch isolato `FaceLoss-Approach`, ideato per correggere le falle intrinseche della "Mean Squared Error" calcolata in radianti.

## 1. Il Problema degli Angoli (Perché la MSE fallisce)

Fino all'Esperimento 5, la rete neurale veniva addestrata misurando l'errore matematico in radianti (o vettori Axis-Angle / Eulero) per i parametri Pitch, Yaw e Roll. 
Tuttavia, **gli angoli non sono lineari nel mondo fisico**. Un minuscolo errore di $0.1$ radianti sul Pitch potrebbe non significare nulla se la faccia sta guardando dritto, ma potrebbe tradursi in uno spostamento fisico enorme di naso e occhi se la faccia è molto ruotata (es. vicina al Gimbal Lock).
In breve: la rete riceveva "punizioni" (Loss) che non corrispondevano al reale fastidio visivo umano.

## 2. Cosa abbiamo implementato (Forward Kinematics)

Per risolvere questo problema **senza appesantire la rete** (che continua virtuosamente a predire solo 3 compattissimi numeri: Pitch, Yaw, Roll), abbiamo spostato la complessità *all'interno* del calcolo dell'errore (la Loss).

1. **La Faccia Canonica**: Abbiamo estratto una "faccia manichino" statica (il primo frame di *AmandaStuck* dal dataset `vertices`), formata da 5023 punti nello spazio 3D. Questa faccia pesa pochissimo ($\sim 60$ KB) e risiede fissa nella memoria della GPU.
2. **PyTorch Rodrigues**: Abbiamo scritto un modulo di geometria pura (`Audio2Pose/geometry.py`) per trasformare i 3 valori predetti dalla rete in reali Matrici di Rotazione ($3 \times 3$) in modo differenziabile.
3. **Proiezione 3D (Vertex Loss)**: 
   - Ad ogni frame, la `PoseLoss` ruota matematicamente tutti i 5023 vertici della "faccia manichino" usando gli angoli *predetti*.
   - Fa la stessa cosa usando gli angoli *reali (Ground Truth)*.
   - Infine, calcola l'errore (MSE) misurando **la reale distanza spaziale (in scala)** tra i vertici predetti e i vertici reali.

## 3. Cosa ci si aspetta (Risultati attesi)

### A. Metriche Diverse
Le metriche di training loggate nel file `.csv` (`train_pos_loss` e `train_vel_loss`) assumeranno valori completamente diversi rispetto al passato. Non saranno più nell'ordine di $0.02$ (radianti quadratici), ma nell'ordine di $0.00004$ (distanza quadratica dei punti nello spazio 3D). Questo è un segnale di corretto funzionamento.

### B. Movimenti più fedeli
Essendo penalizzata sull'effettiva discrepanza spaziale dei vertici, la rete è costretta a dare più importanza agli assi di rotazione che muovono di più il volto. Ci aspettiamo che, sebbene la rete rimanga deterministica (e quindi tendente alla posa media statica), la statica media che apprenderà sarà percettivamente "migliore" e più bilanciata di quella appresa con la pura loss su angoli, portando a risultati visivi potenzialmente più naturali o a una posizione di riposo più biologicamente plausibile.

### C. La questione "Immobilità" (One-to-Many)
Questo approccio risolve le non-linearità geometriche, ma **non risolve il problema intrinseco del One-to-Many**. La MSE penalizzerà comunque la rete se proverà a fare un cenno della testa nel millisecondo sbagliato rispetto alla Ground Truth. 
L'esperimento serve a dotare la tesi di una Baseline Deterministica *assolutamente inattaccabile dal punto di vista accademico e matematico* prima di procedere con approcci probabilistici (VAE o Diffusione).
