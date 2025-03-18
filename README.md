# Chameleon: A Dataset for Segmenting and Attacking Obfuscated Power Traces in Side-Channel Analysis

**Chameleon** is a dataset designed for side-channel analysis of obfuscated power traces. 
It contains real-world power traces collected from a 32-bit RISC-V System-on-Chip 
implementing four hiding countermeasures: Dynamic Frequency Scaling (DFS), 
Random Delay (RD), Morphing (MRP), and Chaffing (CHF). The dataset also includes side-channel power traces without any active countermeasure (BASE).
Each side-channel trace includes multiple cryptographic operations 
interleaved with general-purpose applications.

The Chameleon dataset is available on 🤗 [Hugging Face](https://huggingface.co/datasets/hardware-fab/Chameleon).

<div align="center">
   <img src="./images/chameleon_logo.png" alt="Chameleon Logo" width="150">
</div>

## How to Download

Full dataset:  
⚠ **WARNING**: Full dataset requires more than 600 GB of space.
```python 
from huggingface_hub import snapshot_download
snapshot_download(repo_id="hardware-fab/Chameleon", repo_type="dataset", local_dir="<download_path>")
```

One sub-dataset of choice:
```python 
from huggingface_hub import snapshot_download
snapshot_download(repo_id="hardware-fab/Chameleon", repo_type="dataset", local_dir="<download_path>", allow_patterns="<sub_dataset>/*")
```
Replace `<sub_dataset>` with `BASE`, `DFS`, `RD`, `MRP`, `CHF`.


## Dataset Structure

The dataset is divided per hiding countermeasure. Each file has the following structure:
* **Data:** The data are power traces of 134,217,550 time samples.
 BASE, DFS, RD, MRP, and CHF sub-dataset
 contain 256, 256, 512, 512, 1024, and 256 data respectively.
 The traces capture the SoC execution of AES encryptions interleaved with general-purpose applications.
* **Metadata:** The metadata are divided into three groups:
  * **Ciphers:** This group contains the AES inputs:
    * `key`: The secret key used for AES encryption.
    * `plaintexts`: The plaintext used for the AES encryption.
  * **Pinpoints:** This group contains the start and end time samples of each AES execution in each trace file.
    * `start`: The starting sample of the AES encryption. It takes values ranging from 0 to 134,217,550.
    * `end`: The ending sample of the AES encryption. It takes values ranging from 0 to 134,217,550.
  * **Frequencies:** This group provides labels for each power trace, indicating the frequency changes during the measurement.
      _Notably, this metadata is only available for the sub-datasets with DFS enabled_. Each metadata has two fields:
    * `samples`: This field denotes the time sample at which a frequency change happens, with integer values ranging from 0 to 134,217,550.
    * `frequencies`: This field specifies the new operating frequency starting from the corresponding sample. It can take floating values from 5MHz to 100MHz.
   
> More details available on 🤗 [Hugging Face](https://huggingface.co/datasets/hardware-fab/Chameleon).

## Note
This work is part of [1] available [online]().

© 2024 hardware-fab

> [1] D. Galli, G. Chiari, and D. Zoni, "Chameleon: A Dataset for Segmenting and Attacking Obfuscated Power Traces in Side-Channel Analysis," in IACR Transactions on Cryptographic Hardware and Embedded Systems, 2025(3)
