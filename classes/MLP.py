import numpy as np
import matplotlib.pyplot as plt
import pickle


class MLP:
    def __init__(self, layers):
        """
        initialize the MLP
        """
        self.layers = layers
        self.weights = []
        self.biases = []
        self.deltas = []
        self.a = []
        self.m = 0
        for i in range(len(layers) - 1):
            self.weights.append(np.random.randn(layers[i], layers[i + 1]) * 0.1)
            self.biases.append(np.zeros((1, layers[i + 1])))

    def sigmoid(self, x):
        """
        sigmoid function
        1 / (1 + e^-x)
        """
        # use soft max here
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """
        derivative of the sigmoid function
        x * (1 - x)
        """
        return x * (1 - x)

    def softmax(self, x):
        """
        softmax function
        """
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def forward_propagation(self, X):
        """
        forward propagation
        input -> hidden -> output layer
        """
        # input layer
        self.a = [X]
        self.m = X.shape[0]

        # for hidden layers
        for i in range(len(self.weights) - 1):
            # linear function: y = a * w + b
            z = np.dot(self.a[i], self.weights[i]) + self.biases[i]
            # activation function: 1 / (1 + e^-x)
            a = self.sigmoid(z)
            self.a.append(a)

        # for output layer
        # linear function: y = a * w + b
        z = np.dot(self.a[-1], self.weights[-1]) + self.biases[-1]
        # softmax function
        a = self.softmax(z)
        self.a.append(a)

    def back_propagation(self, y, learning_rate):
        """
        back propagation
        """
        # calculate loss for each layer
        deltas = []

        # output layer
        deltas.append(self.a[-1] - y)

        # hidden layers (from the last hidden layer to the first hidden layer)
        for i in range(len(self.a) - 2, 0, -1):
            delta = np.dot(deltas[-1], self.weights[i].T) * self.sigmoid_derivative(
                self.a[i]
            )
            deltas.append(delta)
        deltas.reverse()

        # gradient descent
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * np.dot(self.a[i].T, deltas[i]) / self.m
            self.biases[i] -= (
                learning_rate * np.sum(deltas[i], axis=0, keepdims=True) / self.m
            )

    def compute_loss(self, y, y_pred):
        """
        Compute the binary cross entropy loss
        """
        y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
        loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        return loss

    def compute_accuracy(self, y, y_pred):
        """
        Compute the accuracy
        """
        predictions = np.argmax(y_pred, axis=1)
        labels = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == labels)
        return accuracy

    def fit(
        self,
        X_train,
        y_train,
        X_valid,
        y_valid,
        epochs,
        learning_rate,
        batch_size=32,
        early_stop=False,
        patience=10,
    ):
        """
        Train the model using mini-batch gradient descent.
        """
        # Convert to numpy arrays if needed
        if hasattr(X_train, "values"):
            X_train = X_train.values
        if hasattr(X_valid, "values"):
            X_valid = X_valid.values

        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        best_val_loss = float("inf")
        patience_counter = 0

        n_samples = X_train.shape[0]
        num_batches = int(np.ceil(n_samples / batch_size))

        for epoch in range(epochs):
            # Shuffle training data at the beginning of each epoch
            permutation = np.random.permutation(n_samples)
            X_train_shuffled = X_train[permutation]
            y_train_shuffled = y_train[permutation]

            epoch_loss = 0
            epoch_accuracy = 0

            for batch in range(num_batches):
                start = batch * batch_size
                end = min(start + batch_size, n_samples)
                X_batch = X_train_shuffled[start:end]
                y_batch = y_train_shuffled[start:end]

                # Forward propagation for the current batch
                self.forward_propagation(X_batch)
                batch_loss = self.compute_loss(y_batch, self.a[-1])
                batch_accuracy = self.compute_accuracy(y_batch, self.a[-1])

                # Accumulate loss and accuracy weighted by batch size
                epoch_loss += batch_loss * (end - start)
                epoch_accuracy += batch_accuracy * (end - start)

                # Back propagation on the current batch
                self.back_propagation(y_batch, learning_rate)

            # Average epoch metrics
            epoch_loss /= n_samples
            epoch_accuracy /= n_samples
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_accuracy)

            # Evaluate on the full validation set
            self.forward_propagation(X_valid)
            val_loss = self.compute_loss(y_valid, self.a[-1])
            val_accuracy = self.compute_accuracy(y_valid, self.a[-1])
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            print(
                f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f} - val_loss: {val_loss:.4f} - acc: {epoch_accuracy:.4f} - val_acc: {val_accuracy:.4f}"
            )

            if early_stop:
                # Early stopping: only reset counter if improvement > 0.001
                if val_loss < best_val_loss - 0.001:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(
                            f"Early stopping at epoch {epoch+1}: Validation loss did not improve for {patience} consecutive epochs."
                        )
                        break

        # Plotting the results using the recorded epochs
        epochs_range = range(1, len(train_losses) + 1)
        plt.figure(figsize=(12, 5))

        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, train_losses, label="Training Loss")
        plt.plot(epochs_range, val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss over Epochs")
        plt.legend()

        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, train_accuracies, label="Training Accuracy")
        plt.plot(epochs_range, val_accuracies, label="Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Accuracy over Epochs")
        plt.legend()

        plt.tight_layout()
        plt.show()


    def save_model(self, file_path):
        """
        Save the model to a file
        """
        model_data = {
            "layers": self.layers,
            "weights": self.weights,
            "biases": self.biases,
        }
        with open(file_path, "wb") as file:
            pickle.dump(model_data, file)

    def load_model(self, file_path):
        """
        Load the model from a file
        """
        with open(file_path, "rb") as file:
            model_data = pickle.load(file)
            self.layers = model_data["layers"]
            self.weights = model_data["weights"]
            self.biases = model_data["biases"]
