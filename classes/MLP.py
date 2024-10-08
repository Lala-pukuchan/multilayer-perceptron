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

    def fit(self, X_train, y_train, X_valid, y_valid, epochs, learning_rate):
        """
        train the model
        """
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        for epoch in range(epochs):
            # forward propagation on training data
            self.forward_propagation(X_train)

            # compute training loss and accuracy
            train_loss = self.compute_loss(y_train, self.a[-1])
            train_accuracy = self.compute_accuracy(y_train, self.a[-1])

            # back propagation
            self.back_propagation(y_train, learning_rate)

            # forward propagation on validation data
            self.forward_propagation(X_valid)

            # compute validation loss and accuracy
            val_loss = self.compute_loss(y_valid, self.a[-1])
            val_accuracy = self.compute_accuracy(y_valid, self.a[-1])

            # store losses and accuracies
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)

            # print losses and accuracies
            print(f"Epoch {epoch+1}/{epochs} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - acc: {train_accuracy:.4f} - val_acc: {val_accuracy:.4f}")

        # Plotting the results
        epochs_range = range(1, epochs + 1)

        plt.figure(figsize=(12, 5))

        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, train_losses, label='Training Loss')
        plt.plot(epochs_range, val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss over Epochs')
        plt.legend()

        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, train_accuracies, label='Training Accuracy')
        plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy over Epochs')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def save_model(self, file_path):
        """
        Save the model to a file
        """
        model_data = {
            'layers': self.layers,
            'weights': self.weights,
            'biases': self.biases
        }
        with open(file_path, 'wb') as file:
            pickle.dump(model_data, file)

    def load_model(self, file_path):
        """
        Load the model from a file
        """
        with open(file_path, 'rb') as file:
            model_data = pickle.load(file)
            self.layers = model_data['layers']
            self.weights = model_data['weights']
            self.biases = model_data['biases']
