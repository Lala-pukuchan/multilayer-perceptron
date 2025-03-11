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

# Calculate Recall and F1 Score for positive class (assumed label = 1)
pred_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_test_2d, axis=1)

# True Positives, False Positives, False Negatives
TP = np.sum((pred_labels == 1) & (true_labels == 1))
FP = np.sum((pred_labels == 1) & (true_labels == 0))
FN = np.sum((pred_labels != 1) & (true_labels == 1))

recall = TP / (TP + FN + 1e-9)         # Recall = TP / (TP + FN)
precision = TP / (TP + FP + 1e-9)        # Precision = TP / (TP + FP)
f1_score = 2 * precision * recall / (precision + recall + 1e-9)  # F1 Score = 2 * (precision*recall)/(precision+recall)

# Print results
print(f"Predictions: {predictions}")
print(f"Binary Cross-Entropy Loss: {loss:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")
