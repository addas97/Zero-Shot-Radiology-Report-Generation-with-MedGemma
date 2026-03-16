# Zero-Shot-Radiology-Report-Generation-with-MedGemma
A minimal end-to-end pipeline for zero-shot chest X-ray report generation using Vision-Language Models, evaluated on a 234-image subset of the CheXpert validation dataset.

## Overview

This project implements the three-stage CheXpert evaluation pipeline from the [CheXpert paper](https://arxiv.org/abs/1901.07031):

1. **Report Generation** — MedGemma-4B generates radiology reports from chest X-ray images
2. **Label Extraction** — Qwen2.5-3B extracts binary CheXpert disease labels from generated reports
3. **Evaluation** — BERTScore (text quality) and per-disease F1 (classification performance) are computed across 14 pathologies

---

## Pipeline

```
CheXpert Images (234)
        │
        ▼  src/runners/run_eval.py
        │  Model: google/medgemma-4b-it (vLLM)
        │
predictions.jsonl
        │
        ▼  src/runners/extract_chexpert_labels.py
        │  Model: Qwen/Qwen2.5-3B-Instruct
        │
labels.csv
        │
        ▼  src/runners/compute_metrics.py
        │
metrics_bertscore_summary.csv
metrics_f1_by_disease.csv
metrics_f1_summary.csv
```

---

## Results

### Baseline (minimal prompt)

| Metric | Value |
|---|---|
| BERTScore Precision | 0.6745 |
| BERTScore Recall | 0.7900 |
| **BERTScore F1** | **0.7273** |
| Macro-F1 | 0.3964 |
| Micro-F1 | 0.5134 |

### Improved (structured clinical prompt)

| Metric | Value | Δ |
|---|---|---|
| BERTScore Precision | 0.7386 | +0.064 |
| BERTScore Recall | 0.8086 | +0.019 |
| **BERTScore F1** | **0.7714** | **+0.044** |
| Macro-F1 | 0.3523 | -0.044 |
| Micro-F1 | 0.4809 | -0.033 |

### Per-Disease F1 (Improved Prompt)

| Disease | F1 | Support |
|---|---|---|
| No Finding | 0.3089 | 38 |
| Enlarged Cardiomediastinum | 0.5166 | 109 |
| Cardiomegaly | 0.4130 | 68 |
| Lung Opacity | 0.4881 | 126 |
| Lung Lesion | 0.0000 | 1 |
| Edema | 0.3103 | 45 |
| Consolidation | 0.4286 | 33 |
| Pneumonia | 0.0000 | 8 |
| Atelectasis | 0.3925 | 80 |
| Pneumothorax | 0.0000 | 8 |
| Pleural Effusion | 0.6964 | 67 |
| Pleural Other | None | 1 |
| Fracture | None | 0 |
| Support Devices | 0.6734 | 107 |

---

## Improvement: Structured Prompt Engineering

To improve upon the baseline zero-shot pipeline, we applied structured prompt engineering to MedGemma without any fine-tuning or additional computational cost. The original prompt was minimal and unstructured, providing no guidance on report format or which pathologies to assess. The revised prompt instructs MedGemma to generate a two-section clinical report — a **Findings** section describing observations across the lungs, pleura, cardiac silhouette, mediastinum, and support devices, followed by an **Impression** section explicitly addressing all 14 CheXpert disease categories. Additionally, the label extraction prompt for Qwen2.5-3B was augmented with explicit negation-handling rules to reduce false positives from negated mentions (e.g., *"no pleural effusion"*).

The structured prompt yielded a notable improvement in BERTScore F1 from 0.727 to **0.771** (+4.4%), indicating higher semantic alignment with ground-truth reports. Classification F1 showed mixed results due to a known limitation of pipeline-based evaluation: the structured prompt causes MedGemma to explicitly enumerate absent findings, which the downstream Qwen label extractor occasionally misinterprets as positive despite negation-aware prompting — highlighting the sensitivity of the extraction stage to linguistic negation.

---

## Repository Structure

```
BINF4008_HW2/
├── configs/
│   ├── chexpert.yaml          # Stage 1 config (model, prompt, data paths)
│   └── compute_metrics.yaml   # Stage 3 config (metric paths, BERTScore settings)
├── src/
│   ├── data/
│   │   ├── clean.py           # Preprocessing pipeline for CheXpert metadata
│   │   └── preprocessing.py   # CSV reader and data utilities
│   ├── models/
│   │   ├── __init__.py        # Model registry
│   │   └── medgemma.py        # MedGemma vLLM wrapper
│   └── runners/
│       ├── run_eval.py        # Stage 1: report generation
│       ├── extract_chexpert_labels.py  # Stage 2: label extraction
│       └── compute_metrics.py # Stage 3: BERTScore + classification F1
└── experiments/
    └── chexpert/medgemma/
        ├── predictions.jsonl
        ├── labels.csv
        ├── metrics_bertscore_summary.csv
        ├── metrics_f1_by_disease.csv
        └── metrics_f1_summary.csv
```

---

## Setup

### Requirements

- Python 3.10+
- NVIDIA GPU with ≥15 GB VRAM (A100 recommended; T4 not supported due to bfloat16 requirement)
- Hugging Face account with access to [google/medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it)

### Install

```bash
pip install vllm transformers accelerate sentencepiece bert-score cleantext pyyaml tqdm Pillow
```

### Hugging Face Authentication

MedGemma is a gated model. Accept the license at [huggingface.co/google/medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it), then:

```bash
huggingface-cli login
```

### Configure Data Paths

Edit `configs/chexpert.yaml` and set:

```yaml
dataset:
  base_dir: "/path/to/images"       # folder containing patient64xxx/ subdirectories
metadata: "/path/to/metadata.csv"   # 234-row CheXpert validation CSV
```

---

## Running the Pipeline

```bash
# Stage 1 — Generate reports (~15 min on A100)
python -m src.runners.run_eval --config configs/chexpert.yaml

# Stage 2 — Extract labels
python -m src.runners.extract_chexpert_labels \
  --predictions experiments/chexpert/medgemma/predictions.jsonl \
  --ground-truth /path/to/label.csv \
  --output-csv   experiments/chexpert/medgemma/labels.csv \
  --model-id     Qwen/Qwen2.5-3B-Instruct \
  --include-text

# Stage 3 — Compute metrics
python -m src.runners.compute_metrics --config configs/compute_metrics.yaml
```

---

## Running on Google Colab

Open `CheXpert_Colab.ipynb` in [Google Colab](https://colab.research.google.com) and select **Runtime → Change runtime type → A100 GPU**.

> ⚠️ T4 GPUs are not supported — MedGemma (Gemma3 architecture) requires bfloat16 which needs compute capability ≥ 8.0.

---

## Implementation Notes

### `build_report_prompt` (`run_eval.py`)
Combines the system role, user task template, and per-image clinical indication into a single structured prompt. A deduplication guard prevents the indication from being echoed when it matches the template.

### `load_transformers_model` (`extract_chexpert_labels.py`)
Loads Qwen2.5-3B with `padding_side="left"` (required for correct batch generation in decoder-only LLMs) and `pad_token = eos_token` (Qwen has no dedicated pad token).

### `batch_extract` (`extract_chexpert_labels.py`)
Slices `output_ids[:, prompt_len:]` before decoding to strip the echoed prompt tokens, ensuring only newly generated JSON is decoded and parsed into disease labels.

### `_compute_prf` (`compute_metrics.py`)
Returns `None` (not `0`) when a denominator is zero, so macro-averaging correctly skips absent disease classes rather than deflating scores with phantom zeros.

### Macro vs Micro F1 (`compute_metrics.py`)
Macro-F1 weights each of 14 diseases equally (important for clinical fairness across rare conditions). Micro-F1 weights every sample–label pair equally, dominated by high-prevalence labels like Support Devices and Pleural Effusion.

---

## Dataset

[CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) — Stanford ML Group. 234 images sampled from the validation split. Labels cover 14 pathologies with binary annotations (1 = positive, 0 = negative/uncertain).

---

## Models

| Stage | Model | Parameters | Framework |
|---|---|---|---|
| Report Generation | [google/medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it) | 4B | vLLM |
| Label Extraction | [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | 3B | Transformers |
| BERTScore | [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) | 66M | bert-score |

---

## Acknowledgements

Pipeline structure based on the starter code from [BINF4008_HW2](https://github.com/zilinjing/BINF4008_HW2). CheXpert dataset from [Irvin et al., 2019](https://arxiv.org/abs/1901.07031).
