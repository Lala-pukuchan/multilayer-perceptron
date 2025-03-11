import numpy as np
import pandas as pd
from classes.MLP import MLP
from util import create_data

# Load the model
mlp = MLP(layers=[])  # Initialize with empty layers, will be overwritten by load_model
mlp.load_model("resources/mlp_model.pkl")

# Load the test data from CSV (separate.pyで作成されたdata_test.csv)
data = pd.read_csv("./resources/data_test.csv", header=None)

# Process the data using create_data (assumes format: Column 0: ID, Column 1: diagnosis, Column 2+: features)
X_test, y_test_2d = create_data(data)

# Perform prediction using the test features
mlp.forward_propagation(X_test.values)
predictions = mlp.a[-1]

# Compute loss and accuracy using the one-hot encoded labels (y_test_2d) and predictions
loss = mlp.compute_loss(y_test_2d, predictions)
accuracy = mlp.compute_accuracy(y_test_2d, predictions)

print(f"Predictions: {predictions}")
print(f"Binary Cross-Entropy Loss: {loss:.4f}")
print(f"Accuracy: {accuracy:.4f}")
