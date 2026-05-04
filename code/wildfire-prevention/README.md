<div align="center">

# 🔥 Wildfire Prevention with Compact VLMs

### Fine-tuning Vision-Language Models on Sentinel-2 satellite imagery to predict wildfire risk — running on consumer hardware (RTX 3060)

<img src="./assets/cover.png" alt="Wildfire Prevention with VLMs" width="800"/>

> Replicating Liquid AI's [wildfire-prevention cookbook](https://github.com/Liquid4All/cookbook/tree/main/examples/wildfire-prevention) with **LoRA / QLoRA on a 12 GB GPU** — and benchmarking 4 different VLM families head-to-head.

[![HuggingFace Dataset](https://img.shields.io/badge/🤗_Dataset-damianGil/wildfire--prevention-yellow)](https://huggingface.co/datasets/damianGil/wildfire-prevention)
[![Unsloth](https://img.shields.io/badge/Powered_by-Unsloth-orange)](https://github.com/unslothai/unsloth)
[![Comet ML](https://img.shields.io/badge/Tracking-Comet_ML-blueviolet)](https://www.comet.com/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

</div>

---

## ⚡ TL;DR

A satellite captures two images of the same place — RGB (natural colour) and SWIR (infrared, reveals dry vegetation). A fine-tuned VLM looks at both and emits a **structured JSON** with 6 fields predicting wildfire risk:

```json
{
  "risk_level": "high",
  "dry_vegetation_present": true,
  "urban_interface": true,
  "steep_terrain": true,
  "water_body_present": false,
  "image_quality_limited": false
}
```

Why is this hard? Because **standard VLMs have never seen Sentinel-2 SWIR imagery in pretraining**. The vision encoder needs to learn to read the orange/red dryness patterns. We use **LoRA with vision layers unfrozen** — and beat the cookbook's full fine-tuning result running on a consumer GPU.

---

## 🏆 Results

Evaluated on 172 test samples (cross-region temporal split). Ground truth from `claude-opus-4-6`.

| Model | Strategy | Overall | risk_level | VRAM | Train time |
|---|---|---|---|---|---|
| 🥇 **Qwen2.5-VL-3B** | LoRA r=64, vision-on, 3 ep | **0.890** | 0.814 | ~7 GB | ~1.5 h |
| 🥈 **Qwen2.5-VL-3B** | **LoRA r=16**, vision-on, early-stop | **0.886** | 0.820 | ~5 GB | ~1.0 h |
| 🥉 **LFM2.5-VL-450M** | LoRA r=64, vision-on, 3 ep | **0.840** | — | ~4 GB | ~50 min |
| 📚 Cookbook (Liquid AI) | full fine-tune (H100) | 0.840 | 0.760 | ~80 GB | — |
| 🎯 Claude Opus 4.6 | the labeling teacher (ceiling) | 0.99 | 0.99 | API | — |

**Key takeaway**: bigger model + LoRA on consumer hardware **beats full fine-tuning of a smaller model on H100**. The trick is unfreezing the vision encoder so it can learn SWIR.

---

## 🎯 Why This Project Is Interesting

### 1. **Multimodal, structured output, in-domain to out-of-domain transfer**
Two images per sample, six structured fields per output. The vision encoder must adapt to a new modality (multispectral satellite) it has never seen.

### 2. **Consumer hardware**
Everything runs on **RTX 3060 12 GB**. No H100, no Modal credits. WSL2 + Ubuntu + Unsloth + LoRA.

### 3. **Beat the original cookbook with fewer resources**
The original cookbook does full fine-tuning on a Modal H100 (paid). We match its accuracy with LoRA on a $300 GPU.

### 4. **Comparative study across 4 VLM families**
Not just one model — we benchmark LFM2.5-VL, Qwen2.5-VL (3B & 7B), and Gemma 3 4B side by side on the same task, with consistent hyperparameters.

---

## 🛠 Stack

| Component | Tool |
|---|---|
| Fine-tuning framework | [Unsloth](https://github.com/unslothai/unsloth) (FastVisionModel) |
| Trainer | [TRL SFTTrainer](https://github.com/huggingface/trl) |
| Adapter method | [PEFT LoRA / QLoRA](https://github.com/huggingface/peft) |
| Experiment tracking | [Comet ML](https://www.comet.com/) |
| Dataset hosting | [HuggingFace Datasets](https://huggingface.co/datasets) |
| Environment | WSL2 (Ubuntu 22.04) + uv + Python 3.12 |
| Model serving (post-training) | GGUF export → llama.cpp / Ollama |

---

## 📂 Project Structure

```
code/wildfire-prevention/
├── README.md                          ← you are here
├── .env.example                       ← template for HF_TOKEN, COMET_API_KEY
├── pyproject.toml                     ← deps (uv-managed)
├── assets/
│   └── cover.png                      ← hero image (generated, see prompt below)
├── notebooks/
│   ├── 01_finetuning_lfm2.5-vl-450M.ipynb  ← baseline tutorial (full walkthrough)
│   ├── 02_train_qwen25_vl_3b.ipynb         ← Qwen2.5-VL-3B (LoRA r=16)
│   ├── 02_train_all_models.ipynb           ← train ALL models in one go
│   ├── 03_train_qwen25_vl_7b.ipynb         ← Qwen2.5-VL-7B (QLoRA r=64)
│   ├── 04_train_gemma3_4b.ipynb            ← Gemma 3 4B (LoRA r=16)
│   └── 05_comparison_report.ipynb          ← reads results/ and plots everything
├── scripts/
│   └── clone_dataset.py               ← clone Paulescu/wildfire-prevention to your HF
├── results/                           ← per-model JSON metrics (commited)
│   ├── lfm25_vl_450m.json
│   ├── qwen25_vl_3b.json
│   └── ...
├── data/                              ← downloaded dataset (gitignored)
├── outputs/                           ← LoRA checkpoints (gitignored)
└── configs/                           ← optional YAML configs (legacy)
```

---

## 🚀 Quick Start

### 1. Pre-requisites

- **Linux or WSL2** (Unsloth doesn't play nice on Windows native).
- **NVIDIA GPU with ≥ 6 GB VRAM** (RTX 3060+ recommended).
- `uv` installed.
- HuggingFace account with a write token.
- (Optional) Comet ML account for experiment tracking.

See `code/docs/wsl2-setup.md` for a step-by-step WSL2 setup if you're on Windows.

### 2. Setup

```bash
# Clone repo and enter
cd code/wildfire-prevention

# Install dependencies
uv sync

# Configure secrets
cp .env.example .env
# Edit .env and put your HF_TOKEN and COMET_API_KEY
```

### 3. Clone the dataset to your HF account

```bash
uv run scripts/clone_dataset.py --target YOUR_HF_USER/wildfire-prevention
```

### 4. Train your first model

Open `notebooks/01_finetuning_lfm2.5-vl-450M.ipynb` and run all cells. It downloads the model, trains LoRA adapters, evaluates on the test set, and saves a JSON report.

### 5. Train them all

For the ambitious path: `notebooks/02_train_all_models.ipynb` runs the loop over all 3 modern VLMs with their optimal configs. ~5-7 hours unattended on RTX 3060.

### 6. Generate the comparative report

After all JSONs are in `results/`, open `notebooks/05_comparison_report.ipynb` and run it. Tables, bar charts, heatmap by field, Pareto plots — all generated from your local results.

---

## 🔬 What Each Notebook Does

| Notebook | Purpose |
|---|---|
| `01_finetuning_lfm2.5-vl-450M.ipynb` | **Tutorial-style** walkthrough of the full pipeline (dataset → JSONL → train → eval). Best for learning. |
| `02_train_qwen25_vl_3b.ipynb` | Single-model training script for Qwen2.5-VL-3B with the optimized config (r=16, alpha=32, early stopping). |
| `02_train_all_models.ipynb` | All-in-one: trains the 3 modern VLMs sequentially with memory clean-up between them. Lanza y olvida. |
| `03_train_qwen25_vl_7b.ipynb` | Single-model script for Qwen2.5-VL-7B with QLoRA (4-bit) due to VRAM. |
| `04_train_gemma3_4b.ipynb` | Single-model script for Gemma 3 4B IT. |
| `05_comparison_report.ipynb` | Reads all JSONs in `results/` and generates the comparative analysis. |

---

## 💡 Lessons Learned

- **`finetune_vision_layers=True` is critical** when the visual modality is OOD (Sentinel-2 SWIR is alien to standard VLM pretraining). Without it, accuracy on SWIR-dependent fields (`dry_vegetation`, `steep_terrain`) collapses by 30+ points.
- **Bigger model → smaller LoRA rank**. For 3B+ models, `r=16` matches `r=64` while using less VRAM. Less is more.
- **`alpha = 2*r` when r is small**, `alpha = r` when r is large. Compensates the lower capacity by amplifying the contribution.
- **QLoRA needs higher rank** than LoRA. The 4-bit quantization eats some precision; LoRA must compensate. Use `r=64` for QLoRA, `r=16` for plain LoRA.
- **Early stopping needs a `threshold`**, not just patience. Default `threshold=0.0` lets training run forever if eval_loss decreases by 0.00001 per epoch. Use `threshold=0.001`.

---

## 📚 References

- 🍳 [Liquid AI Cookbook — wildfire-prevention example](https://github.com/Liquid4All/cookbook/tree/main/examples/wildfire-prevention)
- 🤗 [Original dataset by Paulescu](https://huggingface.co/datasets/Paulescu/wildfire-prevention)
- 🐌 [Unsloth Vision SFT notebook (LFM2.5-VL)](https://docs.unsloth.ai/get-started/unsloth-notebooks)
- 📄 [LoRA paper](https://arxiv.org/abs/2106.09685) · [QLoRA paper](https://arxiv.org/abs/2305.14314)
- 🛰 [Sentinel-2 mission overview](https://sentinel.esa.int/web/sentinel/missions/sentinel-2)

---

<div align="center">

**Built with curiosity. Trained on a desk. Inspired by satellites.**

</div>
