import numpy as np


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

    def fit(self, X, y, epochs, learning_rate):
        """
        train the model
        """
        # for each epoch
        for epoch in range(epochs):
            # forward propagation
            self.forward_propagation(X)

            # compute loss
            loss = self.compute_loss(y, self.a[-1])

            # back propagation
            self.back_propagation(X, y, learning_rate)

            # print loss
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

    def print_layers(self):
        """
        print the layers
        """
        for i in range(len(self.layers)):
            print(f"Layer {i}: {self.layers[i]}")

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            print(f"\nLayer {i} to {i+1}:")
            print(f"Weight shape: {w.shape}")
            print(f"Weight sample:\n{w}")
            print(f"Bias shape: {b.shape}")
            print(f"Bias sample:\n{b}")
