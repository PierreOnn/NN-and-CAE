import numpy as np
import math


# Sigmoid activation function
def sigmoid(input):
    return 1/(1+np.exp(-input))


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

    def training(self, X, Y, learning_rate, epochs):
        costs = []
        for epoch in range(epochs):
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
            self.Wxh -= learning_rate * cost_theta1
            self.Bh -= learning_rate * cost_b1.sum(axis=0)

            self.Why -= learning_rate * cost_theta2
            self.By -= learning_rate * cost_b2.sum(axis=0)

            # Cost calculation
            if epoch % 200 == 0:
                loss = np.sum(-Y * np.log(a3))
                costs.append(loss)

        return self.Wxh, self.Bh, self.Why, self.By, costs

    def predict(self, Wxh, Why, Bh, By, X):
        m = np.shape(X)[0]  # number of test instances
        prediction = np.zeros(m)
        # hidden layer (layer2)
        z2 = np.dot(X, Wxh) + Bh
        a2 = sigmoid(z2)
        # output layer (layer3)
        z3 = np.dot(a2, Why) + By
        a3 = sigmoid(z3)

        for i in range(np.shape(a3)[0]):
            prediction[i] = np.argmax(a3[i])

        return prediction, a3


# This is just code to test the neural network, so not needed for the actual program
if __name__ == "__main__":
    X = np.array(
        [[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1]])
    Y = np.array(
        [[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1]])
    Wxh = np.random.rand(8, 3)
    Why = np.random.rand(3, 8)
    Bh = np.random.randn(3)
    By = np.random.randn(8)
    ann = ANN(Wxh, Why, Bh, By)

    Wxh, Bh, Why, By, costs = ann.training(X, Y, learning_rate=0.01, epochs=50000)
    pred, a3 = ann.predict(Wxh, Why, Bh, By, X)

    print("predicted: ", pred)
    print("real: ", Y)
    print('Training Set Accuracy: ', (pred == X).mean() * 100, "%")
    print("b1: \n", Bh)
    print("Theta1: \n", Wxh)
    print("b2: \n", By)
    print("Theta2: \n", Why)
    print("a3: \n", np.round(a3, 2))
