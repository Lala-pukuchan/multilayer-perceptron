import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets with the given test size ratio.
    """
    n_samples = len(X)
    n_test = int(n_samples * test_size)

    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    X_train = X.iloc[train_indices]
    y_train = y.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_test = y.iloc[test_indices]

    return X_train, X_test, y_train, y_test


def main():
    """
    read the data and split into training and testing sets
    """
    # read the data
    data = pd.read_csv("./resources/data.csv", header=None)

    # find the column that contains 'M' or 'B'
    diagnosis_column = None
    for col in data.columns:
        if data[col].isin(["M", "B"]).any():
            diagnosis_column = col
            break

    if diagnosis_column is None:
        raise ValueError("Failed to find the diagnosis column.")

    # set the column names
    columns = [
        "feature" + str(i) if i != diagnosis_column else "diagnosis"
        for i in range(len(data.columns))
    ]
    data.columns = columns

    # split the features and labels
    X = data.drop(columns=["diagnosis"])
    y = data["diagnosis"].apply(lambda x: 1 if x == "M" else 0)  # Mを1, Bを0に変換

    # split the data into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Combine features and target for training and validation sets
    train_data = pd.concat([X_train, y_train], axis=1)
    valid_data = pd.concat([X_valid, y_valid], axis=1)

    # save the data
    train_data.to_csv("./resources/data_training.csv", index=False)
    valid_data.to_csv("./resources/data_validation.csv", index=False)


if __name__ == "__main__":
    main()
