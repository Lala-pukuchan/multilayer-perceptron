import numpy as np


def create_data(data):
    """
    Create training/validation data according to the new data format
    Data format:
      Column 0: ID
      Column 1: diagnosis (M or B)
      Column 2+: features
    """
    # Extract diagnosis from column 1 as labels, convert M to 1 and B to 0
    y = data.iloc[:, 1].apply(lambda x: 1 if x == "M" else 0)

    # Extract features from column 2 onwards (ID is not used)
    X = data.iloc[:, 2:]

    # Standardize features
    X = (X - X.mean()) / X.std()

    # One-hot encoding (not like 0 -> [1, 0], 1 -> [0, 1], but here converts using index=label)
    num_samples = y.shape[0]
    num_classes = 2
    y_2d_array = np.zeros((num_samples, num_classes))
    y_2d_array[np.arange(num_samples), y.values] = 1

    return X, y_2d_array
