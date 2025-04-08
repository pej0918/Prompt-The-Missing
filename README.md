# Prompt the Missing: Efficient and Robust Audio-Visual Classification under Uncertain Modalities
[![CVPR 2025 Workshop](https://img.shields.io/badge/CVPR_2025-Workshop-blue)](https://cvpr2025.thecvf.com/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)

> 📣 Accepted at **CVPR 2025 Workshop, TMM-OpenWorld 2025**  
> 🔧 Official PyTorch implementation of our paper, **Prompt the Missing**  
> ⚡️ Parameter-Efficient and Robust Audio-Visual Classification under Modality Uncertainty

---

## 📖 Overview

**Prompt the Missing** introduces a parameter-efficient method for robust **audio-visual classification** under **uncertain or missing modalities**.

Instead of modifying the backbone or requiring full fine-tuning, we inject **learnable prompt tokens** at both input and attention levels. These prompts allow dynamic adaptation to modality corruption while keeping the pretrained model frozen.

---

## 🔥 Key Highlights

- ✅ **Missing Modality Resilience**: Handles complete, partial, or noisy modality loss (audio or visual).
- 🧠 **Dual-Level Prompt Injection**: Injects learnable tokens at both input and attention levels.
- ⚙️ **Backbone-Freezing**: No backbone retraining required — lightweight and plug-and-play.
- ⚡ **Highly Efficient**:
  - 90% fewer parameters than full fine-tuning
  - 96% faster training time
  - 82.3% less memory usage
- 📈 **Superior Robustness**: Outperforms CAV-MAE, LoRA, and Adapter in missing-modality settings.
- 🧪 **Flexible Evaluation**: Supports both Concat-based (uncertain conditions) and Situation-specific prompting.

---

## 🧠 Architecture Overview

![image](https://github.com/user-attachments/assets/ce6cc96a-83a2-4b5f-83f4-b581fb08c24a)

- **Input-Level Prompt**: Injected before modality encoders.  
- **Attention-Level Prompt**: Appended into cross-attention keys and values.  
- **Fusion Module**: Enables bidirectional refinement between audio and visual streams.

---

## 📁 Datasets

We evaluate our framework on two audio-visual benchmarks:

- 🎧 [UrbanSound8K-AV](https://www.kaggle.com/datasets/lingyueguo/urbansound8k-av)
- 📷 [CIFAR10-AV](https://www.kaggle.com/datasets/lingyueguo/cifar10-av)

---

## 📈 Performance Summary

### UrbanSound8K-AV

| Noise Type        | CAV-MAE | LoRA | Adapter | Prompt (Ours) |
|-------------------|--------:|-----:|--------:|--------------:|
| Complete          |   0.99  | 0.98 |   0.98  |       **0.99** |
| Noise to Audio    |   0.69  | 0.84 |   0.88  |       **0.94** |
| Noise to Vision   |   0.83  | **0.84** |   0.81  |        0.83    |
| Noise to Both     |   0.71  | 0.80 |   0.81  |       **0.88** |
| **Average**       |   0.80  | 0.87 |   0.87  |       **0.91** |

### CIFAR10-AV

| Noise Type        | CAV-MAE | LoRA | Adapter | Prompt (Ours) |
|-------------------|--------:|-----:|--------:|--------------:|
| Complete          |   0.93  | 0.92 |   0.91  |       **0.93** |
| Noise to Audio    |   0.66  | 0.77 |   0.81  |       **0.89** |
| Noise to Vision   |   0.70  | 0.76 |   **0.85** |     0.84    |
| Noise to Both     |   0.60  | **0.79** | 0.72  |        0.78    |
| **Average**       |   0.72  | 0.81 |   0.82  |       **0.86** |

---

## 📊 Prompt Usage Strategy

| Strategy           | UrbanSound8K-AV | CIFAR10-AV |
|--------------------|------------------|------------|
| Concat-Based       | 0.76             | 0.74       |
| Situation-Specific | **0.87**         | **0.86**   |

> Even without degradation labels, our model generalizes via attention-based prompt routing.

---

## ⚡️ Efficiency & Parameter Comparison

| Method             | Params (M) | Memory (GiB) | Time/Epoch | Accuracy |
|--------------------|-----------:|-------------:|------------:|----------:|
| Full Fine-Tuning   |      86.4  |        95.1  |     60.0 s  |     0.88  |
| LoRA               |      12.6  |        27.2  |     10.1 s  |     0.86  |
| Adapter            |      15.4  |        32.8  |     12.5 s  |     0.87  |
| **Prompt (Ours)**  |   **3.2**  |   **17.8**   | **2.4 s**   | **0.88**  |

---

## ⚙️ Training & Evaluation Setup
> Note: Although the script name is run_cavmae.py, this implementation corresponds to our proposed method (Prompt the Missing) with customized prompt learning modules.

### ✨ Prompt Learning (Robust Training)
Train the prompt-based model with noisy or uncertain modality settings:

```bash
# Train with Prompt Learning
python -W ignore src/run_cavmae_pl.py \
  --model cav-mae-ft \
  --dataset vggsound \
  --data-train /path/to/train.json \
  --data-val /path/to/test.json \
  --exp-dir ./exp_test \
  --label-csv /path/to/class_labels_indices_urban.csv \
  --n_class 10 \
  --lr 1e-3 \
  --n-epochs 5 \
  --batch-size 128 \
  --finetuned_path /path/to/best_audio_model.pth \
  --proportion 0.3 \
  --dataset_mean 0 \
  --dataset_std 1 \
  --target_length 1024 \
  --mode train
```
### 🧪 Evaluation (Missing Modality Setting)
To evaluate the trained prompt model (Concat-Based) :

```bash
python -W ignore src/run_cavmae_pl.py \
  --model cav-mae-ft \
  --dataset vggsound \
  --data-train /path/to/train.json \
  --data-val /path/to/test.json \
  --exp-dir ./exp_test \
  --label-csv /path/to/class_labels_indices_urban.csv \
  --n_class 10 \
  --lr 1e-3 \
  --n-epochs 5 \
  --batch-size 128 \
  --finetuned_path /path/to/noise_to_both.pth \
  --proportion 0.3 \
  --dataset_mean 0 \
  --dataset_std 1 \
  --target_length 1024 \
  --mode eval
```

---

## 📚 Citation

```
TBD
```

## 🙏 Acknowledgement

This work builds upon the [CAV-MAE (Contrastive Audio-Visual Masked Autoencoder)](https://github.com/YuanGongND/cav-mae) framework.  
We sincerely thank the original authors for open-sourcing their code and pre-trained models, which made this research possible.
