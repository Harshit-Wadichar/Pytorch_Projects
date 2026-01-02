# How to create a simple deep learning model with PyTorch 

This README explains, in simple language, how to build, train, and evaluate a basic neural network using PyTorch. It keeps things short and practical so you can run a model quickly and understand each step.

---

## Prerequisites

* Python installed (3.8+ recommended).
* Install PyTorch (CPU or GPU) and utilities:

```bash
pip install torch torchvision pandas numpy matplotlib
```

---

## Step 1 — Prepare your data

1. Read your data into a table (use `pandas.read_csv`).
2. Clean missing values (`dropna`) and drop useless columns (like `id`).
3. Separate features `X` and labels `y`.
4. Normalize or scale features so numbers are small (e.g., divide by max or use `StandardScaler`).
5. Split into training, validation, and test sets (use `sklearn.model_selection.train_test_split`).

**Why:** neural networks learn faster when inputs are scaled and when you hold out validation/test sets to check generalization.

---

## Step 2 — Make a PyTorch Dataset and DataLoader

* Convert numpy arrays into a `torch.utils.data.Dataset` so PyTorch can access samples by index.
* Wrap the dataset in a `DataLoader` to create batches and shuffle data.

```python
from torch.utils.data import Dataset, DataLoader

class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

loader = DataLoader(SimpleDataset(X_train, y_train), batch_size=32, shuffle=True)
```

---

## Step 3 — Define the model (a small neural network)

* Subclass `nn.Module`. Define layers in `__init__` and the forward pass in `forward`.

```python
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, in_features, hidden=16):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, 1)  # for binary
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.sigmoid(x)

model = SimpleNet(in_features=X.shape[1])
```

---

## Step 4 — Choose loss and optimizer

* For binary problems: `nn.BCELoss()` (or `nn.BCEWithLogitsLoss()` if last layer returns logits).
* For multi-class: `nn.CrossEntropyLoss()` and final layer should output `num_classes` logits.
* Optimizer: `torch.optim.Adam(model.parameters(), lr=1e-3)`.

---

## Step 5 — Training loop (simple)

1. Loop over epochs.
2. For each batch: forward → compute loss → `loss.backward()` → `optimizer.step()` → `optimizer.zero_grad()`.
3. Track loss and accuracy for train and validation.

```python
for epoch in range(epochs):
    model.train()
    for Xb, yb in loader:
        preds = model(Xb).squeeze(1)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    # validation: model.eval() + torch.no_grad()
```

**Why `model.eval()` and `torch.no_grad()`?** They turn off training behavior (like dropout and gradient tracking) so validation is faster and correct.

---

## Step 6 — Evaluate and save the model

* After training, evaluate on the test set and compute accuracy or other metrics.
* Save model weights with `torch.save(model.state_dict(), 'model.pth')` and load with `model.load_state_dict(torch.load('model.pth'))`.

---

## Common pitfalls & tips

* **Mismatch of label format:** `BCELoss` needs floats 0.0/1.0; `CrossEntropyLoss` needs integer class labels.
* **Normalization leak:** Fit scalers only on the training set, then apply to val/test.
* **Batch dimension:** For single sample inference, add batch dim (`unsqueeze(0)`) if model expects batches.
* **Device (GPU/CPU):** Move model and tensors to the same device (`model.to(device)` and `tensor.to(device)`).

---

## Minimal runnable example (outline)

1. Load data, split and scale.
2. Create `Dataset` and `DataLoader`.
3. Define `SimpleNet`.
4. Train for a few epochs.
5. Evaluate and save.

This flow is exactly what the provided rice-classification script follows; you can use that script as a working template.

---
