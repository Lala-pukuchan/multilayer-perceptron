import argparse
from classes.MLP import MLP
import numpy as np
import pandas as pd
import os


def create_data(data):
    """
    Create data for training/validation
    """

    print("data: ", data)

    # split data to X and y
    X = data.drop("diagnosis", axis=1)
    y = data["diagnosis"]

    # Standarize data
    X = (X - X.mean()) / X.std()

    # convert to 2d numpy array
    num_samples = y.shape[0]
    num_classes = 2
    y_2d_array = np.zeros((num_samples, num_classes))
    y_2d_array[np.arange(num_samples), y] = 1

    return X, y_2d_array


def train(
    hidden_layers_neuron,
    epochs,
    loss,
    batch_size,
    learning_rate,
):
    """
    Train MLP model
    """
    # import data
    data_train = pd.read_csv("resources/data_training.csv")
    X_train, y_train_2d_array = create_data(data_train)
    data_valid = pd.read_csv("resources/data_validation.csv")
    X_valid, y_valid_2d_array = create_data(data_valid)

    if X_train is None:
        raise ValueError("create_data returned None")

    # define layers
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

    if not os.path.exists("resources/data_training.csv") or not os.path.exists(
        "resources/data_validation.csv"
    ):
        print("resources/data_training.csv not found.")
        return

    parser = argparse.ArgumentParser(description="Train MLP model")
    parser.add_argument(
        "--hidden_layers_neuron",
        nargs="+",
        type=int,
        default=[24, 24],
        help="Layer size and neuron size of each in hidden layer",
    )
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument(
        "--loss", type=str, default="binary_crossentropy", help="Loss function"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=0.1, help="Learning rate"
    )

    args = parser.parse_args()

    train(
        hidden_layers_neuron=args.hidden_layers_neuron,
        epochs=args.epochs,
        loss=args.loss,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()
