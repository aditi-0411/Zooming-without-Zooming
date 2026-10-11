# ZwZ — Zooming-without-Zooming
### Multimodal Surgical Visual Question Answering via Knowledge Distillation

## Overview

ZwZ distills a large multimodal teacher model's reasoning about fine-grained
surgical detail into a small student model (Qwen2-VL-2B), with the goal of
answering questions about small tools, tissue, and anatomical structures in
laparoscopic frames without physically zooming in.

## Pipeline

![ZwZ pipeline](docs/pipeline.png)

## Dataset

- **CholecSeg8k** (CC BY-NC-SA 4.0): 8,080 annotated laparoscopic
  cholecystectomy frames from 17 video clips
- 24,100 fine-grained crops extracted by an automated SAM + ResNet50 pipeline
- 24,100 teacher-generated VQA labels

## Model

- **Base:** Qwen2-VL-2B-Instruct, 4-bit quantized
- **Adaptation:** LoRA (r=16, alpha=32) on attention/MLP projections
  (0.83% trainable parameters)
- **Custom fusion module:** a secondary high-frequency (DWT) visual branch and
  a Transformer cross-attention decoder, designed to add fine-grained spatial
  detail to the base model's visual token stream before language generation
- **Training:** 36,202 steps (2 epochs) over 18,101 samples, checkpointed and
  resumed across multiple free-tier GPU sessions

## Evaluation

Run on 100 held-out test samples. The test split is separated by source video
to prevent near-duplicate-frame leakage.

| Metric | Value |
|---|---|
| Samples evaluated | 100 |
| Generation success rate | 100/100 |
| Mean semantic similarity vs. teacher | 0.38 |
| Min / max | 0.04 / 1.00 |

Similarity is sentence-embedding cosine similarity (`all-MiniLM-L6-v2`)
between short free-text labels, which is a coarse proxy for answer quality.

## Findings and next steps

- **Ablation of the high-frequency branch.** Zeroing the DWT branch's
  contribution at inference produced identical generated text in 15/15
  held-out samples, and gradient analysis showed its contribution to the loss
  was negligible in this training run. Matching its feature scale to the base
  encoder's tokens did not change its gradient in a diagnostic check. The
  root cause was not conclusively established.
- **Generic outputs.** The student often returns short generic labels (for
  example a single organ name) where the teacher's labels are more specific.
- **Next steps:** give the high-frequency branch its own attention
  projections and/or an auxiliary loss that requires it to carry information,
  then retrain and re-run the ablation.

## Engineering highlights

- Fully resumable, checkpoint-driven pipeline at every stage (SAM generation,
  teacher labeling, training), built to survive session interruptions on
  constrained compute
- Automated dataset path resolution across heterogeneous cloud environments
- Rate-limit- and quota-aware batched API calling for teacher labeling
- Systematic debugging with gradient-flow analysis, attention-weight
  profiling, and output-level ablation, including diagnosing silent gradient
  loss under gradient checkpointing

## Repository contents

```
README.md
LICENSE
docs/pipeline.png            pipeline diagram
src/model_architecture.py    architecture-level outline of the model
```

Training code, hyperparameter configurations, and model weights are
maintained privately.

## License

Copyright (c) 2026 Aditi. All rights reserved. Provided for portfolio
viewing. Built on CholecSeg8k (CC BY-NC-SA 4.0).
