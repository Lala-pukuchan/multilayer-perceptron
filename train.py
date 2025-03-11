import argparse
from classes.MLP import MLP
import pandas as pd
import os
from util import create_data


def train(
    hidden_layers_neuron,
    epochs,
    batch_size,
    learning_rate,
    early_stop,
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

    # Print shapes
    print(f"x_train shape: {X_train.shape}")
    print(f"x_valid shape: {X_valid.shape}")

    # define layers
    layers = [X_train.shape[1], *hidden_layers_neuron, 2]

    # initialize MLP
    mlp = MLP(layers)
    mlp.fit(X_train, y_train_2d_array, X_valid, y_valid_2d_array, epochs, learning_rate, early_stop=early_stop)

    print("> saving model 'resources/mlp_model.pkl' to disk...")
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
        "--early_stop",
        type=str,
        choices=['True', 'False'],
        default='False',
        help="Enable early stopping (default: False)"
    )

    args = parser.parse_args()

    # Convert early_stop string to boolean
    early_stop = args.early_stop == 'True'

    # Print chosen arguments and defaults
    print("\nSelected parameters:")
    print(f"Hidden layers: {args.layer} (default: [24, 24])")
    print(f"Epochs: {args.epochs} (default: 1000)")
    print(f"Batch size: {args.batch_size} (default: 32)")
    print(f"Learning rate: {args.learning_rate} (default: 0.1)")
    print(f"Loss function: {args.loss} (default: binary_crossentropy)")
    print(f"Early stopping: {early_stop} (default: False)")
    print("\n")

    train(
        hidden_layers_neuron=args.layer,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        early_stop=early_stop,
    )


if __name__ == "__main__":
    main()
