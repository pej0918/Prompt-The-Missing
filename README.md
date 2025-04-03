# Prompt the Missing: Efficient and Robust Audio-Visual Classification under Uncertain Modalities
[![CVPR 2025 Workshop](https://img.shields.io/badge/CVPR_2025-Workshop-blue)](https://cvpr2025.thecvf.com/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)

> 📣 Accepted at **CVPR 2025 Workshop, TMM-OpenWorld 2025**  
> 🔧 Official PyTorch implementation of our paper, **Prompt the Missing**  
> 🧪 Lightweight, Training-Efficient, and Modality-Aware Prompt Learning Framework

---

## 📖 Introduction

**Prompt the Missing** addresses the challenge of missing or corrupted modalities in real-world **audio-visual classification**. Instead of retraining large models for every scenario, our method introduces **learnable prompt tokens** at both **input** and **attention** levels to dynamically adapt to **uncertain modality availability**—**without modifying the backbone model**.

We simulate modality degradation using a **case-wise training strategy**, enabling the model to learn generalizable representations for a wide range of real-world missing-modality settings.

---


## 🔥 Key Highlights

- ✅ **End-to-End Framework for Missing Modality**
  - Designed to handle **uncertain and unpredictable modality loss**, including complete, partial, and noisy cases.
- 🧠 **Dual-Level Prompt Injection**
  - Introduces **learnable prompts** at both **input-level** and **attention-level**, improving robustness to missing modalities.
- ⚙️ **Backbone-Freezing & Plug-and-Play**
  - Does **not require retraining** of the backbone model. Works with pretrained models through efficient prompt injection.
- 📦 **Parameter-Efficient and Scalable**
  - Adds **<1% extra parameters**, reduces **memory usage by 82.3%**, and cuts **training time by 96%**, enabling fast deployment and training.
- 🧪 **Flexible Inference Modes**
  - Supports both **Concat-Based** and **Situation-Specific** evaluation strategies for different deployment scenarios.
- 📈 **Superior Performance**
  - Outperforms full fine-tuning baselines by **up to +10%** in noisy and missing modality settings on the UrbanSound8K-AV benchmark.


---

## 🧠 Architecture Overview

![image](https://github.com/user-attachments/assets/7bf37eff-3dfa-4a12-8960-d9a87b2c76a6)

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

| Noise Type        | CAV-MAE Accuracy | Ours (Prompt Learning) |
|-------------------|------------------|------------------------|
| Complete          | 0.99             | 0.99                   |
| Noise to Audio    | 0.69             | **0.94**               |
| Noise to Vision   | 0.83             | **0.83**               |
| Noise to Both     | 0.71             | **0.88**               |
| **Average**       | 0.80             | **0.91**               |

### CIFAR10-AV

| Noise Type        | CAV-MAE Accuracy | Ours (Prompt Learning) |
|-------------------|------------------|------------------------|
| Complete          | 0.93             | 0.93                   |
| Noise to Audio    | 0.66             | **0.89**               |
| Noise to Vision   | 0.70             | **0.84**               |
| Noise to Both     | 0.60             | **0.78**               |
| **Average**       | 0.72             | **0.86**               |

---

## 📊 Prompt Usage Strategy Comparison

| Evaluation Strategy     | UrbanSound8K-AV | CIFAR10-AV |
|-------------------------|------------------|-------------|
| Concat-Based            | 0.76             | 0.74        |
| Situation-Specific      | **0.87**         | **0.86**    |

> ✅ Even without knowing which modality is missing, our model generalizes well using prompt attention routing.

---

## ⚡️ Efficiency

| Method          | Memory (GiB) | Time per Epoch |
|------------------|--------------|-----------------|
| Full Fine-Tuning | 95.12        | 60.0 sec        |
| Prompt Learning  | **17.85**    | **2.4 sec**     |

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
