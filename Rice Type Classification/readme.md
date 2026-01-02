# Rice Type Classification

## Project overview

This project trains a tiny neural network to classify rice type from measured physical features of rice grains. The model reads a CSV file (riceClassification.csv), scales numeric columns, trains a simple feedforward network, evaluates it, plots training curves, and lets you do a single-sample prediction.

## About the dataset

**Context (from dataset owner):**
This dataset was created for rice classification and is intended for educational use and practice. The dataset is a modified resource with labels such as `Jasmine` (1) and `Gonen` (0).

**Content & Columns (all numeric):**

* `id` (dropped before training)
* `Area`
* `MajorAxisLength`
* `MinorAxisLength`
* `Eccentricity`
* `ConvexArea`
* `EquivDiameter`
* `Extent`
* `Perimeter`
* `Roundness`
* `AspectRation` *(note: may be a typo for AspectRatio in some datasets)*
* `Class` (target label — e.g., 0 or 1 for binary classification)

> All attributes are numeric variables. The `Class` column is the target.

---

## What this model does (plain language)

* Takes the measured features of rice grains (the numeric columns above) as input.
* Learns patterns that separate the rice types (for example, Jasmine vs Gonen) during training.
* Outputs a single probability (value between 0 and 1) that indicates how likely the input belongs to class 1.
* Rounds that probability to 0 or 1 to give the final predicted class.

**Use cases**

* Educational experiments on classification and neural networks.
* Quick prototype for understanding how grain features map to rice type.
* Learning preprocessing, model training, evaluation, and single-sample inference.

**Not recommended for production** without further validation: dataset is small and the model is intentionally simple.

---

## Model architecture (diagram)

Below is a simple diagram showing the model and the data flow. This is the same architecture used in the code.

```
[INPUT FEATURES]  (10 features: Area, MajorAxisLength, ... )
        │
        ▼
  [Linear layer]  X.shape[1]  -->  HIDDEN_NEURONS (10)
        │
        ▼
  [Linear layer]  HIDDEN_NEURONS (10) --> 1
        │
        ▼
  [Sigmoid]
        │
        ▼
  OUTPUT: probability (0.0 - 1.0)
        │
        ▼
  Class prediction: round(probability) -> 0 or 1
```

**Simple data flow diagram**

```
CSV file (riceClassification.csv)
    ↓
pandas -> DataFrame
    ↓ (drop id, dropna)
Normalization (divide each column by its max)
    ↓
Split → train / val / test
    ↓
PyTorch Dataset & DataLoader
    ↓
Model (input -> hidden -> output -> sigmoid)
    ↓
Train (BCELoss, Adam)
    ↓
Evaluate on validation and test sets
    ↓
Plot metrics + single-sample prediction (user inputs scaled values)
```

---

## Training details (as in code)

* **Input features**: 10 numeric columns (all normalized to max absolute = 1).
* **Model**: 2 linear layers + sigmoid output.

  * `Linear(X.shape[1], HIDDEN_NEURONS)`
  * `Linear(HIDDEN_NEURONS, 1)`
  * `Sigmoid()` for probability output.
* **Loss function**: `BCELoss()` (Binary Cross-Entropy Loss).
* **Optimizer**: `Adam` with learning rate `1e-3`.
* **Batch size**: 32
* **Epochs**: 10
* **Device**: Uses GPU (`cuda`) if available, otherwise CPU.

---

## How to interpret model output

* The model returns a single number between 0 and 1 for every sample.
* Example: `0.87` means the model predicts class `1` with 87% probability.
* The code uses `round()` on the probability to map to `0` or `1`.
* If you need full probability output, use the printed probability instead of rounding.

---

## Important notes & limitations

1. **Binary vs. multi-class**: The current code uses a single output node and `BCELoss`, which only works correctly when the task is binary (two classes). If your `Class` column has more than two values (C > 2), you must change the model and loss (see 'Adapting to multi-class').

2. **Label format**: For `BCELoss`, labels must be floats `0.0` or `1.0`. If labels are integers or strings mapping to multiple classes, convert them.

3. **Normalization method**: The code normalizes each column by dividing by its max value. This is simple and effective for many cases, but other scaling (StandardScaler, MinMaxScaler) may work better depending on distributions.

4. **Small model**: This network is intentionally small (10 hidden neurons). If the problem is complex or data is larger, consider deeper/wider networks or other ML models (Random Forest, XGBoost).

5. **Arbitrary loss averaging**: The training code divides total loss by `1000` for plotting. For correct averages, divide by the number of batches or samples.

6. **Data leakage risk**: Always ensure normalization and any preprocessing that uses dataset statistics is fit only on training data (the current code uses the whole dataset's max values). In strict experiments, compute scalers on training set only and apply to val/test.

---

## How to adapt to multi-class (if dataset has > 2 rice types)

1. Map class labels to integers `0..C-1`.
2. Change final layer to `nn.Linear(HIDDEN_NEURONS, C)` (output dimension = number of classes).
3. Remove `Sigmoid()` from the model (leave raw logits).
4. Change loss to `nn.CrossEntropyLoss()`.
5. For predictions, use `pred = logits.argmax(dim=1)` to get the predicted class index.
6. Ensure labels are `torch.long` dtype for `CrossEntropyLoss`.

---

## Quick checklist before running

* Confirm `Class` has 2 unique labels. If not, follow the multi-class steps.
* Ensure `riceClassification.csv` path is correct (the code downloads to `/content/` when run in Colab).
* Convert `Class` to `0/1` floats if using `BCELoss`.
* Optionally compute scaler only on training set for stricter evaluation.

---

## Possible improvements & experiments

* Try `CrossEntropyLoss` + multi-output network for multi-class.
* Replace normalization with `sklearn.preprocessing.StandardScaler()` fit on training data.
* Add an extra hidden layer or increase `HIDDEN_NEURONS` to test model capacity.
* Use `sklearn` models (RandomForest, LogisticRegression) to compare baseline performance.
* Use k-fold cross-validation for more robust performance estimates.

---

## Example single-sample inference (what the code does)

1. User enters raw values for each of the 10 features.
2. Code scales each value by the same column max from the original data.
3. The scaled values are converted to a `torch.Tensor` and passed to `model`.
4. Model returns a probability and the code prints a rounded class.

---

## References

* Dataset source (as provided): rice type classification (modified). The original resource includes labels like Jasmine and Gonen.

---
