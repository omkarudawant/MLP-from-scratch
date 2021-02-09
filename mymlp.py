import numpy as np

np.random.seed(0)
np.warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats.stats import pearsonr
from sklearn.model_selection import train_test_split
from random import random


class MLP(object):
    def __init__(self, architecture: list):

        # create a generic representation of the layers
        layers = architecture

        # create random connection weights for the layers
        weights = []
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i + 1])
            weights.append(w)
        self.weights = weights

        # save derivatives per layer
        derivatives = []
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i + 1]))
            derivatives.append(d)
        self.derivatives = derivatives

        # save activations per layer
        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations

    def _relu(self, x):
        return x * (x > 0).any()

    def _relu_derivative(self, x):
        return 1 if (x >= 0).all() else 0

    def forward_propagate(self, inputs):
        """Computes forward propagation of the network based on input signals.
        Args:
            inputs (ndarray): Input signals
        Returns:
            activations (ndarray): Output values
        """

        # the input layer activation is just the input itself
        activations = inputs

        # save the activations for backpropogation
        self.activations[0] = activations

        # iterate through the network layers
        for i, w in enumerate(self.weights):
            # calculate matrix multiplication between previous activation and weight matrix
            net_inputs = np.dot(activations, w)

            # apply sigmoid activation function
            activations = self._relu(net_inputs)

            # save the activations for backpropogation
            self.activations[i + 1] = activations

        # return output layer activation
        return activations

    def back_propagate(self, error, regularization):
        """Backpropogates an error signal.
        Args:
            error (ndarray): The error to backprop.
        Returns:
            error (ndarray): The final error of the input
        """

        # iterate backwards through the network layers
        for i in reversed(range(len(self.derivatives))):

            # get activation for previous layer
            activations = self.activations[i + 1]

            # apply sigmoid derivative function
            delta = error * self._relu_derivative(activations)

            # reshape delta as to have it as a 2d array
            delta_re = delta.reshape(delta.shape[0], -1).T

            # get activations for current layer
            current_activations = self.activations[i]

            # reshape activations as to have them as a 2d column matrix
            current_activations = current_activations.reshape(
                current_activations.shape[0], -1
            )

            # save derivative after applying matrix multiplication
            self.derivatives[i] = np.dot(current_activations, delta_re)

            # backpropogate the next error with L2 regularization
            error = np.dot(delta, self.weights[i].T) + regularization * np.sum(
                self.weights[i]
            )

    def _create_mini_batches(self, X, y, batch_size):
        # create mini batches
        batches = list()
        combined = np.hstack((X, y.reshape(-1, 1)))
        np.random.shuffle(combined)

        num_batches = combined.shape[0] // batch_size
        i = 0
        for i in range(num_batches):
            mini_batch = combined[i * batch_size : (i + 1) * batch_size, :]
            X_mini = mini_batch[:, :-1]
            Y_mini = mini_batch[:, -1].reshape(-1)
            batches.append((X_mini, Y_mini))

        if combined.shape[0] % batch_size != 0:
            mini_batch = combined[i * batch_size : combined.shape[0]]
            X_mini = mini_batch[:, :-1]
            Y_mini = mini_batch[:, -1].reshape(-1)
            batches.append((X_mini, Y_mini))
        return batches

    def train(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        epochs,
        batch_size,
        learning_rate,
        regularization,
        plot_error=False,
        verbose=0,
    ):
        """Trains model running forward prop and backprop
        Args:
            X: inputs (ndarray)
            y: targets (ndarray)
            epochs (int): Num. epochs we want to train the network for
            learning_rate (float): Step to apply to gradient descent
        """
        train_error = list()
        test_error = list()

        training_error = 0
        testing_error = 0

        # now enter the training loop
        for i in range(epochs):
            sum_errors = 0
            train_batches = self._create_mini_batches(
                X=X_train, y=y_train, batch_size=batch_size
            )
            test_batches = self._create_mini_batches(
                X=X_test, y=y_test, batch_size=batch_size
            )

            total = 0
            # iterate through all the mini batches
            for batch_index, batch in enumerate(train_batches):
                X_mini, y_mini = batch
                total = X_mini.shape[0]
                for j, input in enumerate(X_mini):
                    target = y_mini[j]

                    # activate the network!
                    pred_train = self.forward_propagate(input)

                    error = target - pred_train

                    self.back_propagate(error, regularization)

                    # now perform gradient descent on the derivatives to update weights
                    self.gradient_descent(learning_rate)

                    # keep track of the MSE for reporting later
                    sum_errors += np.average((target - pred_train) ** 2)

                # Epoch complete, report the training error
                if verbose == 2:
                    print(
                        "Batch {} | Error: {}".format(
                            batch_index + 1, sum_errors / total
                        )
                    )
            train_error.append(sum_errors / total)
            # Epoch complete, report the training error
            if verbose:
                print(
                    " * Error after epoch {}: {}\n-------------".format(
                        i + 1, sum_errors / total
                    )
                )

            for batch_index, batch in enumerate(test_batches):
                X_mini, y_mini = batch
                total = X_mini.shape[0]
                for j, input in enumerate(X_mini):
                    target = y_mini[j]

                    # activate the network!
                    pred_test = self.forward_propagate(input)

                    error = target - pred_test

                    # keep track of the MSE for reporting later
                    sum_errors += np.average((target - pred_test) ** 2)

            test_error.append(sum_errors / total)

        if plot_error:
            self._plot_errors(
                train_error=train_error, test_error=test_error,
            )

        print("Training complete!")

    def gradient_descent(self, learningRate=1):
        """Learns by descending the gradient
        Args:
            learningRate (float): How fast to learn.
        """
        # update the weights by stepping down the gradient
        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            weights += derivatives * learningRate

    def _plot_errors(self, train_error, test_error):
        plt.plot(range(len(train_error)), train_error, label="Train error")
        plt.plot(range(len(test_error)), test_error, label="Test error")
        plt.xlabel("Steps")
        plt.ylabel("Error")
        plt.title("Train Vs Test Error plot")
        plt.legend(loc="best")
        plt.savefig("error_plot.png")

    def score(self, true, pred):
        assert len(true) == len(pred), "Lengths of true and pred does not match"
        if not type(true) is np.ndarray and not type(pred) is np.ndarray:
            true = np.array(true)
            pred = np.array(pred)
        numerator = np.square(true - pred)
        mse = np.mean(numerator)
        return mse


if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser(
        "Script for training custom mlp with mini batch GD and L2 regularization"
    )
    arg_parser.add_argument(
        "--epochs", help="Integer, Number of training epochs", required=True
    )
    arg_parser.add_argument("--lr", help="Float, Learning Rate", required=True)
    arg_parser.add_argument("--l2", help="Float, L2 Regularization", required=True)
    arg_parser.add_argument(
        "--batch_size", help="Integer, Batch size for mini batch GD", required=True
    )
    arg_parser.add_argument("--verbose", help="Integer, Verbose level, Can be 1 or 2")
    args = arg_parser.parse_args()

    # Create f=x*y where x, y -> [0, 10]
    x1 = np.random.randint(low=0, high=11, size=1500)
    x2 = np.random.randint(low=0, high=11, size=1500)

    X = np.array([x1, x2]).T
    y = x1 * x2

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # create a Multilayer Perceptron Object
    mlp = MLP(architecture=[X_train.shape[1], 500, 600, 1])

    verbose = eval(args.verbose) if args.verbose != 1 else 1
    # train network
    mlp.train(
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=eval(args.epochs),
        learning_rate=eval(args.lr),
        regularization=eval(args.l2),
        batch_size=eval(args.batch_size),
        verbose=verbose,
        plot_error=True,
    )

    # get a prediction
    output = mlp.forward_propagate(X_test)

    # print MSE score
    print("MSE: ", mlp.score(true=y_test, pred=output))
