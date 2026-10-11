# ZwZ — Zooming-without-Zooming
### Multimodal surgical visual question answering via knowledge distillation

> Research prototype on a public dataset. Not clinically validated.

## Contents

1. [Problem statement](#1-problem-statement)
2. [Approach at a glance](#2-approach-at-a-glance)
3. [Pipeline components](#3-pipeline-components)
4. [Dataset and splits](#4-dataset-and-splits)
5. [Training](#5-training)
6. [Evaluation](#6-evaluation)
7. [Findings, limitations and next steps](#7-findings-limitations-and-next-steps)
8. [Engineering notes](#8-engineering-notes)
9. [Repository contents](#9-repository-contents)
10. [References and license](#10-references-and-license)

---

## 1. Problem statement

In laparoscopic surgery the video feed is the surgeon's only view of the
operative field. The structures that matter most, such as instrument tips,
thin ligaments, small vessels and narrow tissue boundaries, often occupy a
small fraction of the frame. Two things make them hard to inspect:

- **Physical zoom costs time and attention.** Moving the scope or changing
  magnification interrupts the procedure.
- **General vision-language models lose small detail.** They resize the whole
  frame to a fixed input resolution, so a small region ends up with very few
  pixels and visual tokens.

**Question this project asks:** can a pipeline automatically find small
regions, "zoom" into them digitally, and let a small, deployable model answer
questions about them, using a much larger model as the source of supervision?

**Scope.** The project builds and trains this pipeline end to end on a public
dataset and evaluates the student against the teacher's labels. It does not
include clinical validation, and it does not report inference-latency
measurements.

## 2. Approach at a glance

![ZwZ pipeline](docs/pipeline.png)

1. **Find** small candidate regions in each frame (SAM, then an area filter).
2. **Select** the most informative crops (ResNet50 scoring, top-K per frame)
   and enlarge them ("digital zoom").
3. **Label** every crop with a large teacher model (Gemini), producing an
   answer, a short reasoning trace and a confidence value.
4. **Distill** those labels into a small student (Qwen2-VL-2B with LoRA),
   which also receives an extra high-frequency visual branch intended to carry
   fine edge and texture detail.
5. **Evaluate** on held-out frames from videos the student never saw.

## 3. Pipeline components

| Stage | Component | Role |
|---|---|---|
| Region proposal | Segment Anything (SAM, ViT-B) | Automatic mask generation over each frame |
| Fine-grained filter | Area ratio < 10% | Keeps only small regions |
| Crop selection | ResNet50 (ImageNet-pretrained) | Ranks crops, keeps top-3 per frame |
| Digital zoom | Resize to 448 x 448 | Gives the model a magnified view |
| Teacher | Gemini 3.5 Flash-Lite | Answer, reasoning, confidence per crop |
| Student backbone | Qwen2-VL-2B-Instruct, 4-bit | Generates the final answer |
| Adaptation | LoRA | Trains under 1% of parameters |
| Extra visual branch | Discrete Wavelet Transform (DWT) | High-frequency detail tokens |
| Fusion + decoder | Cross-attention decoder with a learned gate | Refines the backbone's visual tokens |

### 3.1 Region proposal (SAM)
SAM's automatic mask generator proposes candidate segments on every frame.
Masks are stored in compressed run-length form, because storing full boolean
masks for 8,080 frames would be impractical.

### 3.2 Fine-grained filter
Each mask's area is divided by the frame area and only masks under 10% are
kept, which pushes the pipeline toward small structures. After filtering, an
average frame had about 11 candidate boxes, and every frame had at least one.

### 3.3 Crop scoring and digital zoom
A pretrained ResNet50 scores each candidate crop by classifier confidence, used
as a rough proxy for "contains something recognisable". The top 3 crops per
frame are kept and resized to 448 x 448. Crops are **regenerated from the
frame and bounding box on demand** rather than saved as image files, which
avoided writing about 24,000 images to disk.

### 3.4 Teacher labelling
Gemini 3.5 Flash-Lite labels each crop. To stay within free-tier limits, six
crops are sent per request and the response is parsed from a fixed structured
format (answer, reasoning, confidence). The pipeline tracks a daily quota and
backs off during sustained service errors. Result: 24,100 labelled crops,
mean teacher-reported confidence 0.89, with 4,488 distinct answer strings.

Questions are deliberately **class-agnostic**. The raw mask values in the
released CholecSeg8k files do not match the class IDs in the dataset paper,
and no verified mapping was available, so the teacher identifies content
visually and the question never depends on a label lookup.

### 3.5 Student model
Qwen2-VL-2B-Instruct is loaded in 4-bit and adapted with LoRA on its attention
and MLP projections. Training uses the teacher's answer as the target, with
prompt tokens masked so that loss is computed on the answer only.

### 3.6 High-frequency (DWT) branch
A 1-level wavelet transform of the crop yields a high-frequency band that
standard patch-based encoders tend to smooth over. That band is pooled and
projected into the language model's embedding space as additional "edge
tokens", to be fused with the backbone's own visual tokens.

### 3.7 Fusion and decoder
The backbone's visual tokens and the edge tokens are concatenated, tagged with
learned modality-type embeddings, and passed to a cross-attention decoder. The
decoder uses the backbone's own tokens as queries, so its output has exactly
the same number of tokens and drops back into the model's embedding pipeline
unchanged. A learned gate controls how much refinement is blended in per token.

## 4. Dataset and splits

- **CholecSeg8k**: 8,080 laparoscopic cholecystectomy frames from 17 video
  clips (CC BY-NC-SA 4.0).
- **Crops:** 24,100 (top-3 per frame).
- **Training set:** 18,101 labelled samples after dropping unparsed responses
  and splitting.
- **Split by source video, not by random crop.** Adjacent frames are
  near-duplicates, so a random split would leak almost-identical images across
  train, validation and test. Videos are divided roughly 80/10/10.

## 5. Training

| Item | Value |
|---|---|
| Steps | 36,202 (2 epochs) |
| Samples per step | 1 |
| Quantization | 4-bit backbone |
| Trainable parameters | about 0.83% for LoRA, plus the fusion module |

Training ran across many free-tier GPU sessions. Checkpoints were written
periodically and mirrored to Hugging Face Hub, with automatic resume, so an
interrupted session cost at most a few hundred steps.

Training and validation loss tracked each other closely (no overfitting) and
plateaued early, at around 6.3.

## 6. Evaluation

**Protocol.** For each held-out crop, the student generates an answer from the
crop and question alone. The answer is compared with the teacher's label by
cosine similarity of sentence embeddings (`all-MiniLM-L6-v2`).

| Metric | Value |
|---|---|
| Samples evaluated | 100 |
| Generation success rate | 100 / 100 |
| Mean semantic similarity vs. teacher | 0.38 |
| Min / max | 0.04 / 1.00 |

**Reading the number.** Embedding similarity between short free-text labels is
a coarse proxy. A student answer such as "Liver" against a teacher label of
"Liver parenchyma" scores moderately even though a clinician would call it
broadly correct, while two unrelated labels can still share some similarity.
The figure is best read as relative evidence, not as accuracy.

## 7. Findings, limitations and next steps

**Findings**
- **The DWT branch had no measurable effect.** Zeroing its contribution at
  inference produced identical generated text in 15/15 held-out samples, and
  gradient analysis showed its contribution to the loss was negligible.
  Matching its feature scale to the backbone's tokens did not change its
  gradient. The root cause was not conclusively established.
- **Generic outputs.** The student often returns short generic labels (for
  example a single organ name) where the teacher's labels are more specific.
- **Loss plateau.** Loss stopped improving early, which is consistent with the
  two points above.

**Limitations**
- Single dataset, single procedure type, one teacher.
- Labels come from a model, not from clinicians, so the student inherits the
  teacher's errors.
- No latency benchmark and no clinical evaluation.

**Next steps**
- Give the high-frequency branch its own attention projections and/or an
  auxiliary loss that requires it to carry information, then retrain and
  re-run the ablation.
- Evaluate with a task-appropriate metric (for example category-level match
  against a surgical vocabulary) alongside embedding similarity.
- Benchmark inference latency before making any speed claims.

## 8. Engineering notes

- **Resumable at every stage.** Mask generation, teacher labelling and
  training each persist progress incrementally and skip completed work on
  restart.
- **Storage discipline.** Crops are regenerated on demand instead of stored,
  and old checkpoints are pruned locally and on the Hub.
- **Environment portability.** Stored dataset paths are remapped on load,
  because mount locations differ across cloud notebook environments.
- **Quota-aware API use.** Batched requests, daily quota tracking and backoff
  during sustained service errors.
- **Debugging that mattered:**
  - Reentrant gradient checkpointing silently zeroed gradients to the custom
    modules while LoRA kept training, which looks exactly like a loss plateau.
  - Optimizer state cannot be loaded across optimizer types; doing so failed
    only later, as a cryptic key error on every step.
  - Automatic multi-GPU placement split model layers across devices and
    collided with the custom modules.

## 9. Repository contents

```
README.md                    this document
LICENSE
docs/pipeline.png            pipeline diagram
src/model_architecture.py    architecture-level outline of the model
```

Training code, hyperparameter configurations and model weights are maintained
privately.

## 10. References and license

- Hong et al., *CholecSeg8k: A Semantic Segmentation Dataset for Laparoscopic
  Cholecystectomy Based on Cholec80*, arXiv:2012.12453.
- Kirillov et al., *Segment Anything*, 2023.
- Qwen team, *Qwen2-VL*.
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021.

Copyright (c) 2026 Aditi. All rights reserved. Provided for portfolio viewing.
CholecSeg8k is licensed separately under CC BY-NC-SA 4.0.
