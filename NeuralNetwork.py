import numpy as np
import wandb


hyperparameter_defaults = dict(
        learning_rate=0.5,
        epochs=100,
    )
wandb.init(config=hyperparameter_defaults, project="assignment 1")
config = wandb.config


# Sigmoid activation function
def sigmoid(input):
    return 1 / (1 + np.exp(-input))


# Derivative of Sigmoid function
def sigmoid_derivative(input):
    return sigmoid(input) * (1 - sigmoid(input))


# tanh activation function
def tanh(input):
    return np.tanh(input)


# class that defines the neural network
class ANN():
    def __init__(self, Wxh, Why, Bh, By):
        self.Wxh = Wxh
        self.Why = Why
        self.Bh = Bh
        self.By = By

    def training(self, X, Y):
        costs = []
        for epoch in range(config.epochs):
            # Feedforward
            z2 = np.dot(X, self.Wxh) + self.Bh
            a2 = sigmoid(z2)

            z3 = np.dot(a2, self.Why) + self.By
            a3 = sigmoid(z3)

            # Backpropagation
            # Phase1
            dcost_dz3 = a3 - Y
            dz3_dtheta2 = a2
            cost_theta2 = np.dot(dz3_dtheta2.T, dcost_dz3)
            cost_b2 = dcost_dz3

            # Phase2
            dz3_da2 = self.Why
            dcost_da2 = np.dot(dcost_dz3, dz3_da2.T)
            da2_dz2 = sigmoid_derivative(z2)
            dz2_dtheta1 = X
            cost_theta1 = np.dot(dz2_dtheta1.T, da2_dz2 * dcost_da2)
            cost_b1 = dcost_da2 * da2_dz2

            # Weights update
            self.Wxh -= config.learning_rate * cost_theta1
            self.Bh -= config.learning_rate * cost_b1.sum(axis=0)

            self.Why -= config.learning_rate * cost_theta2
            self.By -= config.learning_rate * cost_b2.sum(axis=0)

        return self.Wxh, self.Bh, self.Why, self.By, costs

    def predict(self, Wxh, Why, Bh, By, X):
        m = np.shape(X)[0]  # number of test instances
        prediction = np.zeros(m)
        # Hidden layer (layer2)
        z2 = np.dot(X, Wxh) + Bh
        a2 = sigmoid(z2)
        # Output layer (layer3)
        z3 = np.dot(a2, Why) + By
        a3 = sigmoid(z3)

        for i in range(np.shape(a3)[0]):
            prediction[i] = np.argmax(a3[i])

        return prediction, a3


if __name__ == "__main__":
    X_test = np.array(
        [[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1]])
    Y_test = np.argwhere(X_test == 1)[:, -1]
    X_train = np.copy(X_test)
    Y_train = np.copy(X_train)

    Wxh = np.random.rand(8, 3)
    Why = np.random.rand(3, 8)
    Bh = np.random.randn(3)
    By = np.random.randn(8)
    ann = ANN(Wxh, Why, Bh, By)

    Wxh, Bh, Why, By, costs = ann.training(X_train, Y_train)
    pred, a3 = ann.predict(Wxh, Why, Bh, By, X_test)
    accuracy = (pred == Y_test).mean() * 100
    wandb.log({'Training Set Accuracy: ': accuracy})
    wandb.run.name = "Lr" + str(config.learning_rate) + "_Epochs" + str(config.epochs)
