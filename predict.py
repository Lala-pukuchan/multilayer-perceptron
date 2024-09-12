import numpy as np
import pandas as pd
from classes.MLP import MLP


# Load the model
mlp = MLP(layers=[])  # Initialize with empty layers, will be overwritten by load_model
mlp.load_model("resources/mlp_model.pkl")

# Load the test data from CSV
data = pd.read_csv("./resources/predict.csv", header=None)

# Find the column that contains 'M' or 'B'
diagnosis_column = None
for col in data.columns:
    if data[col].isin(["M", "B"]).any():
        diagnosis_column = col
        break

if diagnosis_column is None:
    raise ValueError("Failed to find the diagnosis column.")

# Set the column names
columns = [
    "feature" + str(i) if i != diagnosis_column else "diagnosis"
    for i in range(len(data.columns))
]
data.columns = columns

# Split the features and labels
X_test = data.drop(columns=["diagnosis"])
y_test = (
    data["diagnosis"].apply(lambda x: 1 if x == "M" else 0).values
)  # Mを1, Bを0に変換

# Standardize the test data
X_test = (X_test - X_test.mean()) / X_test.std()

# Perform prediction
mlp.forward_propagation(X_test.values)
predictions = mlp.a[-1]

# Convert y_test to 2D array
num_samples = y_test.shape[0]
num_classes = 2
y_test_2d = np.zeros((num_samples, num_classes))
y_test_2d[np.arange(num_samples), y_test] = 1

# Compute loss using y_test_2d
loss = mlp.compute_loss(y_test_2d, predictions)

accuracy = mlp.compute_accuracy(y_test_2d, predictions)

# Evaluate using binary cross-entropy error function
print(f"Predictions: {predictions}")
print(f"Binary Cross-Entropy Loss: {loss:.4f}")

# Evaluate using accuracy
print(f"Accuracy: {accuracy:.4f}")
