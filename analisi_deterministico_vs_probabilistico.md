# Confronto Architetturale: Modello Deterministico vs Modello Probabilistico (VAE)

Questo documento traccia l'evoluzione e le differenze fondamentali tra l'approccio **deterministico** (utilizzato ad esempio nell'esperimento `exp5_HDTF_3L_h256_vel2`) e l'approccio **probabilistico VAE** (utilizzato in `exp6_OneToMany_VAE`), analizzando i cambiamenti nel codice e il loro impatto sui risultati finali.

---

## 1. Modello Deterministico (MSELoss + VelLoss)

### 1.1 Architettura e Logica (Il codice precedente)
Nel modello originale, l'architettura era progettata come una rigida **mappatura 1-a-1 (One-to-One)**:
- **Input**: Feature estratte dall'audio tramite `Wav2Vec2`.
- **Rete**: Una serie di layer LSTM bidirezionali che prendevano direttamente le feature audio.
- **Output**: 3 angoli (Pitch, Yaw, Roll) per ogni frame.

### 1.2 La Funzione di Loss
```python
Total_Loss = PosLoss_MSE + (lambda * VelLoss_MSE)
```
- **PosLoss**: Penalizzava la distanza assoluta tra l'angolo predetto e la Ground Truth.
- **VelLoss**: Penalizzava le derivate temporali (per evitare tremolii ad alta frequenza).

### 1.3 Il Risultato Finale
Poiché nella realtà a uno stesso audio corrispondono molti movimenti possibili (non c'è una mappatura 1:1 rigida), la rete si trovava punita duramente dall'MSE ogni volta che generava un movimento leggermente sfasato rispetto al video originale. 
**Risultato**: La rete imparava in fretta che **stare ferma sulla media matematica della posa** era la strategia più sicura per minimizzare l'MSE. Il volto risultava molto fluido (grazie alla VelLoss) ma **praticamente immobile**.

---

## 2. Modello Probabilistico (Variational Autoencoder - VAE)

Per risolvere il problema "One-to-Many", il codice è stato evoluto in un VAE introducendo uno spazio latente probabilistico.

### 2.1 Cambiamenti nel Codice (Architettura in `model.py`)
L'architettura è stata divisa in tre componenti:
1. **Pose Encoder**: Aggiunto un nuovo modulo LSTM che durante il training legge la posa reale (Ground Truth) e genera una distribuzione statistica globale: media (`mu`) e varianza (`logvar`).
2. **Spazio Latente ($Z$)**: Estraiamo un vettore casuale $Z$ (es. di dimensione 64) da questa distribuzione.
3. **Decoder modificato**: Il Decoder ora non riceve solo l'Audio. L'input del Decoder diventa la **concatenazione tra Audio e il vettore Latente $Z$**.
   *In Inferenza*, non avendo la Ground Truth, campioniamo semplicemente $Z$ da una normale standard $\mathcal{N}(0,1)$, aspettandoci che $Z$ fornisca lo "stile" del movimento e l'Audio il ritmo.

### 2.2 Cambiamenti nel Codice (La Funzione di Loss in `train.py`)
```python
kld_loss = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
Total_Loss = PosLoss_MSE + (lambda * VelLoss_MSE) + (beta * KLD_Loss)
```
- **KLD Loss**: È stata aggiunta la divergenza di Kullback-Leibler. Questa loss "tira" la distribuzione predetta dal Pose Encoder verso una distribuzione standard $\mathcal{N}(0,1)$, in modo che in inferenza campionare da $\mathcal{N}(0,1)$ abbia senso.
- **KLD Annealing**: È stato aggiunto un meccanismo per cui il peso `beta` parte da 0 e aumenta gradualmente nelle prime 10 epoche per non destabilizzare la rete all'inizio.

### 2.3 Il Risultato Finale (Posterior Collapse)
Nonostante l'architettura sia ora teoricamente in grado di generare movimenti diversi (cambiando $Z$), **il risultato all'epoca 160 e 200 è un volto ancora immobile**. 
Cosa significa questo? Significa che il modello ha sofferto del fenomeno noto come **Posterior Collapse** (Collasso del VAE):
1. La **MSE Loss** è ancora predominante. Il Decoder sa che se prova ad usare il vettore casuale $Z$ per generare movimenti ampi, rischia di andare fuori fase rispetto alla Ground Truth e subire una penalità MSE disastrosa.
2. Il Decoder impara quindi ad **ignorare completamente l'input $Z$**.
3. Poiché il Decoder ignora $Z$, il Pose Encoder smette di immagazzinarvi informazioni, portando `mu` e `logvar` esattamente a 0 (minimizzando la KLD Loss gratis).
4. **Conclusione**: Il VAE regredisce e si trasforma, nei fatti, nella stessa identica rete deterministica precedente. L'output è la posa statica media, e le metriche quantitative sono virtualmente identiche a quelle della versione non probabilistica.

---

## 3. Prossimi Passi (Come uscire dall'immobilità)

La lezione appresa da questo confronto è che **cambiare architettura (aggiungendo il VAE) non è sufficiente se non si cambia la metrica di valutazione (MSE) che terrorizza la rete**.
Per sbloccare il movimento, è necessario abbandonare il paradigma della riproduzione millimetrica:
- **Approccio Generativo Avversariale (GAN)**: Sostituire o affiancare la MSE con un Discriminatore che valuti solo il "realismo umano" del movimento, non la sua esatta sovrapposizione al video reale.
- **Approccio a Diffusione**: Modelli di denoising iterativo che sono nativamente immuni al Posterior Collapse e riescono a mappare il One-to-Many mantenendo la stabilità.
