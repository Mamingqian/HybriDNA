# HybriDNA

A Hybrid Transformer-Mamba2 Long-Range DNA Language Model

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![arXiv](https://img.shields.io/badge/arXiv-2502.10807-b31b1b.svg)](https://arxiv.org/abs/2502.10807)

## Overview

HybriDNA is a DNA foundation model that combines the efficiency of **Mamba-2** state space models with the expressiveness of **Grouped Query Attention** in a hybrid architecture. The model supports sequences up to **131,074 bp** and is designed for various genomic tasks.

## Model Variants

| Model | Parameters | Hidden Size | Layers | Max Length | HuggingFace |
|-------|------------|-------------|--------|------------|-------------|
| HybriDNA-300M | 300M | 1024 | 24 | 131,074 bp | [Mishamq/HybriDNA-300M](https://huggingface.co/Mishamq/HybriDNA-300M) |
| HybriDNA-3B | 3B | 4096 | 16 | 131,074 bp | [Mishamq/HybriDNA-3B](https://huggingface.co/Mishamq/HybriDNA-3B) |
| HybriDNA-7B | 7B | 4096 | 32 | 131,074 bp | [Mishamq/HybriDNA-7B](https://huggingface.co/Mishamq/HybriDNA-7B) |

## Requirements

- Python >= 3.11
- CUDA >= 12.4

See `requirements.txt` for full dependencies.

## Installation

```bash
pip install -r requirements.txt
```

Or install core packages manually:

```bash
pip install torch==2.6.0 transformers==4.57.0
pip install mamba-ssm==2.2.6 causal-conv1d==1.5.3 flash-attn==2.8.3
pip install einops==0.8.1 triton==3.2.0 safetensors==0.6.2
```

## Usage

### 1. Embedding Extraction

```python
from transformers import AutoTokenizer, AutoModel
import torch

model_name = "Mishamq/HybriDNA-300M"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

sequence = "ACGTACGTACGTACGT"
inputs = tokenizer(sequence, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state  # (batch, seq_len, hidden_size)

# Mean pooling for sequence-level embedding
sequence_embedding = embeddings.mean(dim=1)  # (batch, hidden_size)
```

### 2. Generation with Cache

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "Mishamq/HybriDNA-300M"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.config.use_cache = True  # Enable cache for efficient generation

prompt = "ACGTACGT"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.8,
)
generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated)
```

### 3. Finetuning with RC Echo (Reverse-Complement Echo)

RC Echo is a technique that processes both forward and reverse-complement strands of DNA sequences, improving classification performance by leveraging the inherent symmetry of double-stranded DNA.

```python
from transformers import AutoTokenizer, AutoConfig, Trainer, TrainingArguments
import torch

model_name = "Mishamq/HybriDNA-300M"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Import RC Echo model variant from HuggingFace
# RC Echo processes both forward and reverse-complement strands
from modeling_hybridna import HybriDNAForSequenceClassificationRCEcho

config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, num_labels=2)
model = HybriDNAForSequenceClassificationRCEcho.from_pretrained(
    model_name,
    config=config,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

# Tokenize dataset
def tokenize_fn(examples):
    return tokenizer(examples["sequence"], padding="max_length",
                     truncation=True, max_length=1024)

train_dataset = dataset["train"].map(tokenize_fn, batched=True)
eval_dataset = dataset["test"].map(tokenize_fn, batched=True)

# Training
training_args = TrainingArguments(
    output_dir="./hybridna_finetuned",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    learning_rate=2e-5,
    bf16=True,
    eval_strategy="steps",
    eval_steps=500,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train()
```

## Citation

If you use HybriDNA in your research, please cite:

```bibtex
@article{ma2025hybridna,
  title={HybriDNA: A Hybrid Transformer-Mamba2 Long-Range DNA Language Model},
  author={Ma, Mingqian and Liu, Guoqing and Cao, Chuan and Deng, Pan and Dao, Tri and Gu, Albert and Jin, Peiran and Yang, Zhao and Xia, Yingce and Luo, Renqian and others},
  journal={arXiv preprint arXiv:2502.10807},
  year={2025}
}
```

## License

Apache 2.0
