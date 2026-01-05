# Model selection & layer-count rules

*A simple, practical README to decide which model to use for which problem and how many layers (or blocks) to start with. Written in simple language and quick to follow.*

---

## Quick decision chart

| Problem type                                       |                                              Recommended model family |                                   Start depth (conv/blocks / layers) |                                                                  Dataset size guide |
| -------------------------------------------------- | --------------------------------------------------------------------: | -------------------------------------------------------------------: | ----------------------------------------------------------------------------------: |
| Image classification (small images, e.g., 128×128) |            Simple CNN or transfer learn from small ResNet / MobileNet |             Small: 2–3 conv blocks → 1–2 FC; Medium: 4–6 conv blocks | Small: <5k → keep tiny; Medium: 5k–50k → medium; Large: >50k → deeper or pretrained |
| Image classification (natural images)              |      Pretrained ResNet / EfficientNet / MobileNet (transfer learning) |       Use whole pretrained network, optionally add 1–2 top FC layers |                                           Prefer transfer learning for <100k images |
| Object detection / Segmentation                    |                             YOLO / Faster R-CNN / U-Net (specialized) |               Use standard architectures (don’t invent from scratch) |                         Requires many labeled boxes/masks; use pretrained backbones |
| Audio classification (spectrograms)                |   2D-CNN on spectrograms (like image models) or pretrained audio nets |                      Small: 2–3 conv blocks; Medium: 4–6 conv blocks |                Small: <5k samples → light; Medium: 5k–50k → medium; else pretrained |
| Text classification / NLP                          | Pretrained Transformers (BERT) or simple RNN / 1D-CNN for short tasks | Transformer: use pretrained (no new layers needed) ; RNN: 1–3 layers |    Small text corpora: fine-tune small pretrained models; tiny tasks: simple models |
| Time series / 1D signals                           |               1D-CNN or LSTM/GRU or Transformer (depending on length) |                   1D-CNN: 2–5 conv blocks; RNN: 1–3 recurrent layers |                        Small series: light models; long-range: consider Transformer |
| Tabular data                                       |                   Gradient boosting (XGBoost/LightGBM) or shallow MLP |                          If MLP used: 2–4 dense layers (small width) |                      For tabular, tree models often beat deep nets unless data huge |
| Anomaly detection                                  |                                      Autoencoders or one-class models |              Autoencoder depth mirrors input complexity (2–4 layers) |                                                   Use small models if few anomalies |
| Very large-scale problems                          |               Use proven, scalable architectures or pretrained models |            Use deep networks (ResNet-50/101+, EfficientNet variants) |                                             Large datasets + compute available only |

---

## Simple rules (short and practical)

1. **Start small.** Begin with a small network that trains quickly. If the model underfits, increase depth or width.
2. **Use transfer learning** for images and text whenever data is limited. Fine-tuning a pretrained model usually beats building deep models from scratch.
3. **Diagnose first:**

   * If **train low & val low** → underfitting → add capacity (layers/filters).
   * If **train high & val low** → overfitting → reduce capacity or increase regularization / get more data.
4. **Prefer blocks, not single layers.** A common block is `Conv → BatchNorm → ReLU` (repeat 1–2 times) → Pool. Use these blocks repeatedly.
5. **Avoid huge dense layers.** Replace large `Flatten → big FC` with `GlobalAveragePooling` + small FC.
6. **BatchNorm helps.** Add `BatchNorm` after convolution and before activation to stabilize training.
7. **Use Dropout carefully.** Dropout is good for dense layers; less important after conv blocks if BatchNorm and augmentation are used.
8. **Data matters more than layers.** If you can add real data or augment data properly, do that before making the model huge.
9. **Prefer simpler models for tabular data.** XGBoost/LightGBM are often better than deep nets for structured features.

---

## Layer-count rules of thumb (by dataset size)

* **Tiny dataset** (< 5k samples):

  * Images / Audio: 2–3 conv blocks, small filters (16→32→64), avoid large FCs.
  * Text: use pretrained tiny transformers or simple classifiers.
  * Tabular: tree models or tiny MLP (2 layers).

* **Small–Medium dataset** (5k–50k samples):

  * Images / Audio: 4–6 conv blocks (32→64→128 channels), use data augmentation and batchnorm.
  * Text: fine-tune small pretrained Transformer (distilBERT) or LSTM with embeddings.

* **Large dataset** (>50k samples):

  * Use deeper networks or full pretrained models (ResNet-50/101, EfficientNet). You can add more blocks (6–12 conv blocks) but prefer pretrained backbones.

---

## Practical model-design patterns (easy recipes)

### Basic CNN block (use this everywhere for images/spectrograms):

```
[Conv2d -> BatchNorm -> ReLU]
repeat 1-2 times
MaxPool2d(2)
```

Start with 16 or 32 filters and double filters after each pooling.

### Example: Tiny CNN (good start for small datasets)

```
Input: (1,128,256)
Conv(1->16) -> BN -> ReLU
Conv(16->16) -> BN -> ReLU
Pool
Conv(16->32) -> BN -> ReLU
Conv(32->32) -> BN -> ReLU
Pool
GlobalAvgPool
FC(64) -> Dropout -> Output
```

Rough depth: 4 conv layers + 1 small FC.

### Example: Medium CNN (more capacity)

```
Conv(1->32) x2 -> Pool
Conv(32->64) x2 -> Pool
Conv(64->128) x2 -> Pool
GlobalAvgPool -> FC(256) -> Dropout -> Output
```

Depth: 6 conv layers, moderate FC.

### For audio spectrograms

* Treat spectrograms as images and use the same CNN blocks.
* Start with fewer filters if dataset is small.
* Use time/frequency augmentation (time masking, freq masking).

### For NLP

* **Small/medium**: fine-tune a pretrained Transformer (BERT, DistilBERT) — add 1 linear layer on top.
* **Very small**: use pretrained embeddings + an LSTM (1–2 layers) or simple averaging + linear layer.

### For tabular

* Try `XGBoost` first.
* If using MLP: `Input -> Dense(128) -> ReLU -> Dense(64) -> ReLU -> Output` and use dropout/batchnorm depending on overfitting.

---

## Regularization & training tips (must do checklist)

* Use **BatchNorm** after conv layers.
* Use **data augmentation** (images: flips, crops, color jitter; audio: time-stretch, masking).
* Use **early stopping** on validation metric.
* Use **weight decay** (L2) in optimizer.
* Use **dropout** (0.2–0.5) on dense layers.
* Use **learning rate schedule** (ReduceLROnPlateau or CosineLR) and warmup if using Transformers.
* Monitor **train vs val** curves and confusion matrix.

---

## Examples: starting configs (copy-paste friendly)

### 1) Tiny image/audio (few thousand samples)

* Start: Tiny CNN with 3–4 conv layers.
* Filters: [16, 32, 64]
* Pool after each 1–2 convs.
* Use GlobalAvgPool + FC(64) -> Output.

### 2) Medium image/audio (10k–50k)

* Start: Medium CNN with 6 conv layers or fine-tune small pretrained net.
* Filters: [32,64,128]
* Use BatchNorm, Dropout 0.3, augmentation.

### 3) Text classification (small-medium)

* Start: DistilBERT or fine-tune BERT-base with learning rate 2e-5, add 1 linear layer for classes.

### 4) Tabular (typical)

* Start: XGBoost. If insisting on NN: MLP with 2 hidden layers (128→64).

---

## Final short checklist (before adding layers)

1. Do you have enough data? If no → get more or use transfer learning.
2. Are you underfitting? If yes → increase network capacity.
3. Are you overfitting? If yes → reduce capacity or increase regularization.
4. Try BatchNorm and small blocks before stacking many layers.
5. Prefer pretrained models for complex problems.

---
