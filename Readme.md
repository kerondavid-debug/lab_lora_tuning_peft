# LoRA Tuning with PEFT 🤗

![Python](https://img.shields.io/badge/python-3.11-blue)
![PEFT](https://img.shields.io/badge/peft-0.8.2-orange)
![Transformers](https://img.shields.io/badge/transformers-HuggingFace-yellow)
![License](https://img.shields.io/badge/license-MIT-green)

Fine-tunes [`bigscience/bloom-560m`](https://huggingface.co/bigscience/bloom-560m)
on the [`databricks/databricks-dolly-15k`](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
instruction dataset using **LoRA (Low-Rank Adaptation)** via Hugging Face's
[`peft`](https://huggingface.co/docs/peft) library, then compares base vs.
fine-tuned outputs.

## Table of Contents

- [Background](#background)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Notebook Structure](#notebook-structure)
- [Exercise](#exercise)
- [Notes](#notes)
- [License](#license)

## Background

LoRA factors a weight update into two small low-rank matrices instead of
training the full matrix. Example: a 50x50 matrix (2,500 params) can be
approximated by a 2x50 and a 50x2 matrix pair (200 params total) — a 92%
reduction. On real LLMs this can shrink trainable parameters to as little
as ~0.02% of the full model, with results close to full fine-tuning.

## Requirements

- GPU runtime (tested on Colab with an L4 GPU)
- Python 3.11
- `peft==0.8.2`
- `datasets==2.16.1`
- `ipywidgets==7.7.5`
- `transformers`, `torch` (provided by the Colab runtime)

## Installation

```bash
pip install -q peft==0.8.2
pip install -q datasets==2.16.1
pip install ipywidgets==7.7.5
```

## Usage

```bash
git clone <this-repo-url>
cd <this-repo>
jupyter notebook lab_lora_tuning_peft.ipynb
```

Or open directly in [Google Colab](https://colab.research.google.com/) with
`Runtime > Change runtime type > GPU`, then **Run all**. Cells must run
top to bottom — later cells reuse objects (`tokenizer`, `foundation_model`,
`train_sample`, `peft_model`, `inputs`, `get_outputs`) created earlier.

## Notebook Structure

| Step | Cells | What happens |
|------|-------|---------------|
| 1 | Load model & tokenizer | `bloom-560m`, float32, left padding for generation |
| 2 | Baseline generation | Prompt the un-tuned model to see pre-fine-tuning behavior |
| 3 | Prepare dataset | Format Dolly-15k as `### Instruction / ### Context / ### Response`, tokenize with label-masking, sample 2,000 rows |
| 4 | LoRA config | `r=8`, `lora_alpha=16`, `target_modules=["query_key_value"]`, `lora_dropout=0.05` |
| 5 | Train | `Trainer`, 3 epochs, `lr=3e-4` (~10–12 min on Colab GPU); saves to `./bloom-dolly-lora` |
| 6 | Reload & compare | Adapter on vs. off (`disable_adapter()`) on a new prompt |
| 7 | Exercise | Second LoRA run + report template (see below) |

## Exercise

A second adapter is trained on the same data/epochs with higher capacity:

```python
lora_config_v2 = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query_key_value", "dense"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

The notebook prints v1 vs. v2 outputs on the same prompt and includes a
report template comparing trainable-parameter count, training loss, and
output quality between the two runs.

## Notes

- `get_peft_model()` modifies its base model in place — the exercise cell
  reloads a fresh copy of BLOOM so the two LoRA runs are a fair comparison.
- Padding side is `"right"` during tokenization/training and `"left"` for
  generation; the notebook switches it back and forth at the right points.

## License

MIT — adjust as needed for your repo.
