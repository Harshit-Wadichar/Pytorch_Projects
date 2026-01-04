# Legume Leaf Lesion Classification

## Project overview

This project classifies legume leaf images into three states:

* **healthy**
* **angular_leaf_spot**
* **bean_rust**

A pretrained GoogleNet (Inception v1) model is adapted and trained on the provided dataset to recognise these three conditions.

## Dataset

* **Training images:** 1034
* **Calibration (validation) images:** 133
* **Total size:** ~155 MB

The images are photographs of legume leaves labeled with one of the three states above.

## How the project works (conceptual)

1. **Prepare the data:** images are organized and a consistent preprocessing pipeline is applied so every image has the same size and numeric format.
2. **Use a pretrained model:** a GoogleNet model pretrained on a large dataset is reused because it already understands useful visual features.
3. **Adapt the classifier:** the final layer is replaced so the model outputs three class scores (one per lesion state).
4. **Train and validate:** the model is trained on the training set and evaluated on the calibration set using standard classification metrics.
5. **Inference:** given a new leaf image, the trained model predicts which of the three states the leaf belongs to.

## Evaluation and expected outputs

Measure model performance using common metrics:

* **Accuracy** — overall correctness
* **Precision / Recall / F1-score** — per-class performance
* **Confusion matrix** — where the model confuses classes

Report validation results clearly (numbers for each metric and, if possible, per-class breakdown).

## Practical tips

* With a relatively small dataset, consider using data augmentation (rotations, flips, small color changes) to improve generalization.
* If one class is much less frequent than the others, consider techniques to handle class imbalance (for example: weighted loss or balanced sampling).
* Start by training only the final classification layer and then consider fine-tuning more layers if needed.
* Monitor validation metrics to detect overfitting.

## Files and simple project structure

Keep the project layout minimal and easy to understand:

* `data/` — image folders or annotation file
* `models/` — saved model checkpoints
* `notebooks/` or `reports/` — results and plots
* `README.md` — this file

## Reproducibility checklist

* Note the Python and library versions used.
* Record preprocessing steps (image size, normalization) and any augmentation settings.
* Save the final trained model and the exact mapping from numeric labels to class names.
