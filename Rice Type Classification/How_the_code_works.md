# Rice Classification — Line-by-line explanation

Below is the original code split into small chunks. Under each line (or small group of lines) you'll find a clear, plain-language explanation of what it does.

```python
# --- Imports: libraries the code needs ---
import opendatasets as od
od.download("https://www.kaggle.com/datasets/mssmartypants/rice-type-classification")

import torch # Torch main framework
import torch.nn as nn # Used for getting the NN Layers
from torch.optim import Adam # Adam Optimizer
from torch.utils.data import Dataset, DataLoader # Dataset class and DataLoader for creating the objects
from torchsummary import summary # Visualize the model layers and number of parameters
from sklearn.model_selection import train_test_split # Split the dataset (train, validation, test)
from sklearn.metrics import accuracy_score # Calculate the testing Accuracy
import matplotlib.pyplot as plt # Plotting the training progress at the end
import pandas as pd # Data reading and preprocessing
import numpy as np # Mathematical operations
```

**Explanation (imports):**

* `import opendatasets as od` and `od.download(...)` download the dataset from Kaggle to your local environment (or Colab). This fetches the CSV file.
* `torch`, `torch.nn`, `Adam`, `Dataset`, `DataLoader`, `summary`: these are PyTorch tools for building, training, and inspecting neural networks.
* `train_test_split` helps split data into training/validation/test sets.
* `accuracy_score` can compute accuracy but this code ends up calculating accuracy manually.
* `matplotlib.pyplot` is for plotting graphs.
* `pandas` reads and manipulates the CSV data.
* `numpy` is used for numeric array handling.

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

**Explanation:**

* This line checks if a GPU is available. If yes, it uses `'cuda'` (GPU) to speed up training; otherwise it uses `'cpu'` (your computer's processor).

```python
data_df = pd.read_csv("/content/rice-type-classification/riceClassification.csv") # Read the data
data_df.dropna(inplace = True) # Drop missing/null values
data_df.drop(["id"], axis =1, inplace = True) # Drop Id column
print("Output possibilities: ", data_df["Class"].unique()) # Possible Outputs
print("Data Shape (rows, cols): ", data_df.shape) # Print data shape
data_df.head() # Print/visualize the first 5 rows of the data
```

**Explanation:**

* `pd.read_csv(...)` loads the CSV file into a table called a DataFrame named `data_df`.
* `dropna(inplace=True)` removes any rows that have missing values.
* `drop(["id"], axis=1, inplace=True)` removes the `id` column because it usually doesn't help prediction.
* `data_df["Class"].unique()` prints the different classes (labels) found in the data — useful to know how many types of rice there are.
* `data_df.shape` prints how many rows and columns the data has.
* `data_df.head()` shows the first 5 rows so you can inspect the data.

```python
original_df = data_df.copy() # Creating a copy of the original Dataframe to use to normalize inference
```

**Explanation:**

* `original_df` stores a copy of the data before normalization so we can later use its column max values to scale new user inputs the same way.

```python
for column in data_df.columns:
    data_df[column] = data_df[column]/data_df[column].abs().max() # Divide by the maximum of the column which will make max value of each column is 1
```

**Explanation:**

* This loop normalizes every column so that the largest absolute value in each column becomes 1. Normalization helps neural networks train more reliably.

```python
X = np.array(data_df.iloc[:,:-1]) # Get the inputs, all rows and all columns except last column (output)
Y = np.array(data_df.iloc[:, -1]) # Get the outputs, all rows and last column only (output column)
```

**Explanation:**

* `X` contains all features (input data) as a NumPy array. `iloc[:,:-1]` means every row, all columns except the last one.
* `Y` contains the target labels (what we want to predict) — the last column in the table.

```python
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3) # Create the training split
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size = 0.5) # Create the validation split
```

**Explanation:**

* First `train_test_split` splits data into training (70%) and a temporary `X_test` (30%).
* Second split divides that temporary 30% into half: test and validation (so test=15%, val=15% of full data). This results in: 70% train, 15% validation, 15% test.

```python
print("Training set is: ", X_train.shape[0], " rows which is ", round(X_train.shape[0]/data_df.shape[0],4)*100, "%")
print("Validation set is: ",X_val.shape[0], " rows which is ", round(X_val.shape[0]/data_df.shape[0],4)*100, "%")
print("Testing set is: ",X_test.shape[0], " rows which is ", round(X_test.shape[0]/data_df.shape[0],4)*100, "%")
```

**Explanation:**

* These lines print the number of rows in train/validation/test sets and their percentage of total data.

```python
class dataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype = torch.float32).to(device)
        self.Y = torch.tensor(Y, dtype = torch.float32).to(device)

    def __len__(self):
        return len(self.X)
    def __getitem__(self, index):
        return self.X[index], self.Y[index]
```

**Explanation:**

* This defines a PyTorch `Dataset` class named `dataset`.
* `__init__` converts NumPy arrays to PyTorch tensors, sets their type to `float32`, and moves them to the chosen `device` (CPU or GPU).
* `__len__` returns how many samples there are.
* `__getitem__` returns one sample (input and label) when PyTorch asks for it by index. This is used by the `DataLoader`.

```python
training_data = dataset(X_train, y_train)
validation_data = dataset(X_val, y_val)
testing_data = dataset(X_test, y_test)
```

**Explanation:**

* Create three `dataset` objects for training, validation, and testing.

```python
BATCH_SIZE = 32
EPOCHS = 10
HIDDEN_NEURONS = 10
LR = 1e-3

train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle= True)
validation_dataloader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle= True)
testing_dataloader = DataLoader(testing_data, batch_size=BATCH_SIZE, shuffle= True)
```

**Explanation:**

* Hyperparameters are set:

  * `BATCH_SIZE`: number of samples per training step.
  * `EPOCHS`: how many times the model will pass through the entire training set.
  * `HIDDEN_NEURONS`: the size of the hidden layer.
  * `LR`: learning rate controls how big weight updates are.
* `DataLoader` wraps a `Dataset` and provides batches during the training loop. `shuffle=True` randomizes data order each epoch.

```python
class MyModel(nn.Module):
    def __init__(self):

        super(MyModel, self).__init__()

        self.input_layer = nn.Linear(X.shape[1], HIDDEN_NEURONS)
        self.linear = nn.Linear(HIDDEN_NEURONS, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x

model = MyModel().to(device)
summary(model, (X.shape[1],))
```

**Explanation:**

* `MyModel` defines the neural network structure:

  * `input_layer`: a linear transformation from the number of input features (`X.shape[1]`) to `HIDDEN_NEURONS`.
  * `linear`: transforms the hidden neurons down to 1 output.
  * `sigmoid`: squashes the single output into a value between 0 and 1 (a probability).
* `forward` defines the computation when data passes through the model.
* `model = MyModel().to(device)` creates the model and moves it to CPU/GPU.
* `summary(...)` prints a table of layers and number of parameters.

```python
criterion = nn.BCELoss()
optimizer = Adam(model.parameters(), lr= LR)

total_loss_train_plot = []
total_loss_validation_plot = []
total_acc_train_plot = []
total_acc_validation_plot = []
```

**Explanation:**

* `criterion` is the loss function. `BCELoss` (Binary Cross Entropy) expects outputs between 0 and 1 and targets as floats 0.0 or 1.0.
* `optimizer` is Adam, which updates the model weights using gradients.
* Lists are initialized to store loss and accuracy values for plotting after training.

```python
for epoch in range(EPOCHS):
    total_acc_train = 0
    total_loss_train = 0
    total_acc_val = 0
    total_loss_val = 0
    ## Training and Validation
    for data in train_dataloader:

        inputs, labels = data

        prediction = model(inputs).squeeze(1)

        batch_loss = criterion(prediction, labels)

        total_loss_train += batch_loss.item()

        acc = ((prediction).round() == labels).sum().item()

        total_acc_train += acc

        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

**Explanation (training loop):**

* `for epoch in range(EPOCHS)`: repeat the training `EPOCHS` times.
* `for data in train_dataloader`: loop over batches. Each `data` is a batch of `(inputs, labels)`.
* `prediction = model(inputs).squeeze(1)`: forward pass through model, `squeeze(1)` removes the extra dimension so shape matches labels.
* `batch_loss = criterion(prediction, labels)`: compute loss for this batch.
* `total_loss_train += batch_loss.item()`: add scalar loss to running total.
* `acc = ((prediction).round() == labels).sum().item()`: round probabilities to 0/1 and count correct predictions in batch.
* `total_acc_train += acc`: add batch correct count to total.
* `batch_loss.backward()`: compute gradients for model parameters.
* `optimizer.step()`: update model parameters using gradients and the optimizer algorithm.
* `optimizer.zero_grad()`: clear gradients so they don't accumulate into next step.

> Note: A more common order is `optimizer.zero_grad()` before `loss.backward()`, but this code clears gradients after updating which still works if gradients have been used.

```python
    ## Validation
    with torch.no_grad():
        for data in validation_dataloader:
            inputs, labels = data

            prediction = model(inputs).squeeze(1)

            batch_loss = criterion(prediction, labels)

            total_loss_val += batch_loss.item()

            acc = ((prediction).round() == labels).sum().item()

            total_acc_val += acc

    total_loss_train_plot.append(round(total_loss_train/1000, 4))
    total_loss_validation_plot.append(round(total_loss_val/1000, 4))
    total_acc_train_plot.append(round(total_acc_train/(training_data.__len__())*100, 4))
    total_acc_validation_plot.append(round(total_acc_val/(validation_data.__len__())*100, 4))

    print(f'''Epoch no. {epoch + 1} Train Loss: {total_loss_train/1000:.4f} Train Accuracy: {(total_acc_train/(training_data.__len__())*100):.4f} Validation Loss: {total_loss_val/1000:.4f} Validation Accuracy: {(total_acc_val/(validation_data.__len__())*100):.4f}''')
    print("="*50)
```

**Explanation (validation & logging):**

* `with torch.no_grad()` turns off gradient tracking (faster and no memory used) because we are only evaluating.
* Validation loop computes prediction, loss, and number of correct predictions for validation batches, added to totals.
* The code stores average-ish metrics in lists for plotting. Note: dividing by `1000` is arbitrary — not the exact average. It's just scaling for the plot here.
* Accuracy stored by dividing total correct by number of samples, multiplied by 100 to get percent.
* Print a summary line for the epoch giving train and validation loss and accuracy.

```python
with torch.no_grad():
  total_loss_test = 0
  total_acc_test = 0
  for data in testing_dataloader:
    inputs, labels = data

    prediction = model(inputs).squeeze(1)

    batch_loss_test = criterion((prediction), labels)
    total_loss_test += batch_loss_test.item()
    acc = ((prediction).round() == labels).sum().item()
    total_acc_test += acc

print(f"Accuracy Score is: {round((total_acc_test/X_test.shape[0])*100, 2)}%")
```

**Explanation (testing):**

* Another `torch.no_grad()` block evaluates the model on the test dataset.
* Sums test losses and correct counts.
* Prints the final accuracy on the test set as a percentage.

```python
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

axs[0].plot(total_loss_train_plot, label='Training Loss')
axs[0].plot(total_loss_validation_plot, label='Validation Loss')
axs[0].set_title('Training and Validation Loss over Epochs')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].set_ylim([0, 2])
axs[0].legend()

axs[1].plot(total_acc_train_plot, label='Training Accuracy')
axs[1].plot(total_acc_validation_plot, label='Validation Accuracy')
axs[1].set_title('Training and Validation Accuracy over Epochs')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Accuracy')
axs[1].set_ylim([0, 100])
axs[1].legend()

plt.tight_layout()

plt.show()
```

**Explanation (plots):**

* Creates two subplots side-by-side.
* Left subplot: training & validation loss curves.
* Right subplot: training & validation accuracy curves.
* `plt.show()` displays the plots.

```python
area = float(input("Area: "))/original_df['Area'].abs().max()
MajorAxisLength = float(input("Major Axis Length: "))/original_df['MajorAxisLength'].abs().max()
MinorAxisLength = float(input("Minor Axis Length: "))/original_df['MinorAxisLength'].abs().max()
Eccentricity = float(input("Eccentricity: "))/original_df['Eccentricity'].abs().max()
ConvexArea = float(input("Convex Area: "))/original_df['ConvexArea'].abs().max()
EquivDiameter = float(input("EquivDiameter: "))/original_df['EquivDiameter'].abs().max()
Extent = float(input("Extent: "))/original_df['Extent'].abs().max()
Perimeter = float(input("Perimeter: "))/original_df['Perimeter'].abs().max()
Roundness = float(input("Roundness: "))/original_df['Roundness'].abs().max()
AspectRation = float(input("AspectRation: "))/original_df['AspectRation'].abs().max()

my_inputs = [area, MajorAxisLength, MinorAxisLength, Eccentricity, ConvexArea, EquivDiameter, Extent, Perimeter, Roundness, AspectRation]

print("="*20)
model_inputs = torch.Tensor(my_inputs).to(device)
prediction = (model(model_inputs))
print(prediction)
print("Class is: ", round(prediction.item()))
```

**Explanation (single-sample prediction):**

* The code asks the user to input values for each of the 10 features.
* Each input is divided by the maximum value of that column in the original dataframe — so the input is scaled the same way as training data.
* `my_inputs` is the list of these scaled values.
* `torch.Tensor(my_inputs).to(device)` converts this list to a PyTorch tensor and moves it to GPU/CPU.
* `model(model_inputs)` runs the model on that single sample and returns a probability.
* `round(prediction.item())` converts the probability to 0 or 1 and prints it as class.

---

## Final short notes (important):

* If `Class` has more than 2 categories, you must use a multi-class setup (change final layer and loss to `CrossEntropyLoss`).
* Labels must be numeric 0 or 1 for `BCELoss`. If they are strings, convert them to numbers first.
* The code moves data to `device` inside `Dataset`. That’s okay but some prefer moving data in the training loop for clarity.
* The loss averaging uses `/1000` which is arbitrary — for correct averages divide by the number of batches or samples.

---
