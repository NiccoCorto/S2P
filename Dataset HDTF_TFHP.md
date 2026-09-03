# Dataset HDTF_TFHP

## 1. Information from DiffPoseTalk Paper
*Source: DiffPoseTalk paper, section 4.1 Datasets*

We introduce a new dataset— Talking Face with Head Poses (TFHP) — which contains **1,052 videos** of **588 subjects**, totaling **26.5 hours**. 

In TFHP, 348 videos are collected from the downloading script provided by High-Definition Talking Face (HDTF) dataset [Zhang et al. 2021]. Compared with HDTF, our TFHP dataset is more diversified in content, featuring video clips from lectures, online courses, interviews, and news programs, thereby capturing a wider array of speaking styles and head movements. 

Moreover, all videos are converted to **25 fps**. In total, approximately **2,385,000 frames** of FLAME parameters are reconstructed from the videos with our carefully designed data processing pipeline. We split the combined dataset by speakers, resulting in **460 for training, 64 for validation, and 64 for testing**.

---

## 2. Federico's Notes & Analysis

Federico processed the DiffPoseTalk dataset with head poses. The dataset path is:
`/mnt/diskone-second/DiffPoseTalk/datasets/HDTF_TFHP_Elaborated_Pose`

### Integration Details
- The folder is organized with the exact same format as the `MEAD_EMOTE` dataset, making integration with existing code very straightforward.
- A new `splits` folder has been added, containing the three splits with their corresponding filenames (`train.txt`, `val.txt`, `test.txt`).

### Characteristics
The most notable difference is the **greater temporal stability** of the dataset. Additionally, the sequences are less numerous but significantly longer.

### Comparison: HDTF_TFHP vs Mead_EMOTE

| Metric | Mead_EMOTE | HDTF_TFHP | HDTF vs Mead |
| :--- | :--- | :--- | :--- |
| **Number of sequences** | 25,411 | 1,113 | — |
| **Shortest sequence** | 1.0 s | 9.9 s | 10× longer |
| **25th percentile** | 3.2 s | 45.0 s | 14× longer |
| **Median** | 4.0 s | 58.0 s | 15× longer |
| **Mean** | 4.3 s | 85.0 s | 20× longer |
| **75th percentile** | 5.2 s | 70.5 s | 14× longer |
| **90th percentile** | 6.7 s | 227.3 s | 34× longer |
| **Longest sequence** | 17.3 s | 437.9 s (7.3 min) | 25× longer |
| **Total dataset duration**| 30.53 h | 26.27 h | 0.86× |

> [!WARNING]
> Because sequences are significantly longer (up to 7.3 minutes), processing them in a single pass can easily lead to GPU OOM (Out Of Memory) issues.

---

## 3. Our Pre-Processing and Code Modifications

To handle the excessively long sequences, we went through a few iterations in `S2P/data_loader.py`:

### Failed Attempts
1. **Direct Loading (OOM)**: Initially, we tried to load all sequences into memory as usual. However, due to the extreme length of some videos (up to 7.3 minutes), this immediately caused a System/GPU OOM (Out Of Memory) crash.
2. **Monolithic Caching**: Next, we attempted to cache the entire dataset features at once, but the data was simply too large to be cached efficiently in a single block without exhausting system resources.

### Final Solution: Lazy Loading & Disk Chunking
After these failed attempts, instead of setting a maximum sequence threshold and discarding longer videos, we implemented a **Disk Chunking** strategy combined with **Lazy Loading**:

1. **Chunking**: During the preprocessing phase (in `read_data`), we automatically divide long audio and pose sequences into smaller, manageable blocks of **20 seconds** (`chunk_seconds = 20`).
2. **Disk Caching**: These chunks are saved to the disk in a temporary `dataset_chunks` directory.
3. **Lazy Evaluation**: The `PoseDataset` class now only loads the numpy arrays (`audio_features` and `pose_target`) into memory exactly when `__getitem__` is called, dramatically reducing RAM usage.

### Handling Splits
To accommodate Federico's pre-computed splits, we added specific logic to look for the `splits/` directory:
- If a `splits/` folder is found inside the dataset directory, the code reads `train.txt`, `val.txt`, and `test.txt`.
- It dynamically maps each generated 20-second chunk (`base_key_chunkN`) back to its original `base_key` to ensure it falls into the correct split, avoiding data leakage between training and testing.
