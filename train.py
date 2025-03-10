import argparse
from classes.MLP import MLP
import numpy as np
import pandas as pd
import os


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


def train(
    hidden_layers_neuron,
    epochs,
    batch_size,
    learning_rate,
):
    """
    Train MLP model
    """
    # import data
    data_train = pd.read_csv("resources/data_training.csv")
    X_train, y_train_2d_array = create_data(data_train)
    data_valid = pd.read_csv("resources/data_test.csv")
    X_valid, y_valid_2d_array = create_data(data_valid)

    if X_train is None:
        raise ValueError("create_data returned None")

    # define layers
    # example layers is [30, 24, 24, 2]
    layers = [X_train.shape[1], *hidden_layers_neuron, 2]

    # print parameters
    print(f"layers: {layers}")
    print(f"epochs: {epochs}")
    print(f"batch_size: {batch_size}")
    print(f"learning_rate: {learning_rate}")
    print(f"y_train_2d_array: {y_train_2d_array.shape}")

    # initialize MLP
    mlp = MLP(layers)
    mlp.fit(X_train, y_train_2d_array, X_valid, y_valid_2d_array, epochs, learning_rate)
    mlp.save_model("resources/mlp_model.pkl")


def main():

    # check file exists
    if not os.path.exists("resources/data_training.csv") or not os.path.exists(
        "resources/data_test.csv"
    ):
        print("Please execute separate.py first")
        return

    # parse arguments
    parser = argparse.ArgumentParser(description="Train MLP model")
    parser.add_argument(
        "--layer",
        nargs="+",
        type=int,
        default=[24, 24],
        help="Layer size and neuron size of each in hidden layer. Examples: '--layer 24 24' for two hidden layers, '--layer 32 16 8' for three layers with decreasing size",
    )
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument(
        "--loss", type=str, default="binary_crossentropy", help="Loss function"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=0.1, help="Learning rate"
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        type=str,
        default=["accuracy", "precision", "recall", "f1"],
        help="Evaluation metrics. Available options: accuracy, precision, recall, f1.",
    )

    args = parser.parse_args()

    train(
        hidden_layers_neuron=args.layer,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()
