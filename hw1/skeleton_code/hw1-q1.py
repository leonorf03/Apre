#!/usr/bin/env python

# Deep Learning Homework 1

import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import time
import utils

def softmax(x):
    x = x - np.max(x)  # Subtract max value for numerical stability
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)


def relu(x):
	return np.maximum(0, x)

class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        pass

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        # Q1.1 (a)
        eta = kwargs.get("learning_rate", 1)
        y_i_hat = np.argmax(self.W.dot(x_i))
        if y_i_hat != y_i:
            self.W[y_i, :] += eta * x_i
            self.W[y_i_hat] -= eta * x_i


class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001, l2_penalty=0.0, **kwargs):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        # Q1.2 (a,b)
        if(l2_penalty == 0):
            y_hat_i = softmax(self.W.dot(x_i))
            y_i_one_hot = np.zeros(y_hat_i.shape)
            y_i_one_hot[y_i] = 1
            self.W = self.W + learning_rate * (y_i_one_hot - y_hat_i)[:, None].dot(x_i[:, None].T)
        else:
            y_hat_i = softmax(self.W.dot(x_i)) 
            y_i_one_hot = np.zeros(y_hat_i.shape)  
            y_i_one_hot[y_i] = 1

            gradient = (y_i_one_hot - y_hat_i)[:, None].dot(x_i[:, None].T) - l2_penalty * self.W
            self.W = self.W + learning_rate * gradient

"""
class MLP(object):
    # Q3.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size):
        # Initialize an MLP with a single hidden layer.
        units = [n_features, hidden_size, n_classes]
        self.W = [np.random.normal(0.1, 0.01, (units[1], units[0])), 
                  np.random.normal(0.1, 0.01, (units[2], units[1]))]
        self.b = [np.zeros(units[1]), np.zeros(units[2])]

    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.
        num_layers = len(self.W)
        hiddens = []
        for i in range(num_layers):
            h = X if i == 0 else hiddens[i-1]
            z = self.W[i].dot(h) + self.b[i]
            if i < num_layers-1:  # Assume the output layer has no activation.
                hiddens.append(relu(z))
        output = z
        # For classification this is a vector of logits (label scores).
        # For regression this is a vector of predictions.
        return output, hiddens

    def evaluate(self, x, y):
        
        ##X (n_examples x n_features)
        ##y (n_examples): gold labels
        
        # Identical to LinearModel.evaluate()
        accuracy = 0
        for x_i, y_i in zip(x, y):
            predicted_labels, _ = self.predict(x_i)
            accuracy += np.argmax(predicted_labels) == y_i
        return accuracy / x.shape[0]

    def backward(self, x, y, output, hiddens, loss_function='cross_entropy'):
        
        grad_weights = []
        grad_biases = []
        
        for i in range(len(self.W) - 1, -1, -1):
            h = x if i == 0 else hiddens[i-1]
            if i == len(self.W) - 1:
                if loss_function == 'cross_entropy':
                    grad_z = softmax(output) - y
                elif loss_function == 'squared':
                    grad_z = output - y
            else:
                relu_derivs = np.array([k > 0 for k in hiddens[i-1]]) # RELU derivative
                grad_z = self.W[i+1].T.dot(grad_z) * relu_derivs

            # Gradient of hidden parameters.
            grad_weights.append(grad_z[:, None].dot(h[:, None].T))
            grad_biases.append(grad_z)

        grad_weights.reverse()
        grad_biases.reverse()
        return grad_weights, grad_biases

    def update_parameters(self, grad_weights, grad_biases, learning_rate): 
        for i in range(len(self.W)):
            self.W[i] -= learning_rate * grad_weights[i]
            self.b[i] -= learning_rate * grad_biases[i]

    def train_epoch(self, X, y, learning_rate=0.001, loss_function='cross_entropy'):
        total_loss = 0
        for x_i, y_i in zip(X, y):
            output, hiddens = self.predict(x_i)

            # Create one-hot encoding for the target
            y_i_one_hot = np.zeros(output.shape)
            y_i_one_hot[y_i] = 1

            # Compute gradients
            grad_weights, grad_biases = self.backward(x_i, y_i_one_hot, output, hiddens, loss_function=loss_function)
            self.update_parameters(grad_weights, grad_biases, learning_rate=learning_rate)

            # Compute the cross-entropy loss with epsilon to avoid log(0)
            total_loss += -np.sum(y_i_one_hot * np.log(softmax(output) + 1e-10))

        return total_loss / len(X)  # Return average loss
"""

class MLP(object):
    def __init__(self, n_classes, n_features, hidden_size):
        # Initialize an MLP with a single hidden layer.
        # Q1.3 (a)
        units = [n_features, hidden_size, n_classes]
        self.W = [np.random.normal(0.1, 0.01, (units[1], units[0])), 
                  np.random.normal(0.1, 0.01, (units[2], units[1]))]
        self.b = [np.zeros(units[1]), np.zeros(units[2])]

    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes.
        # Q1.3 (a)
        num_layers = len(self.W)
        hiddens = []
        for i in range(num_layers):
            h = X if i == 0 else hiddens[i-1]
            z = self.W[i].dot(h) + self.b[i]
            if i < num_layers-1:  # Output layer has no activation
                hiddens.append(relu(z))
        output = z # Last layer
        return output, hiddens

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible
        """
        accuracy = 0
        for x_i, y_i in zip(X, y):
            predicted_labels, _ = self.predict(x_i)
            accuracy += np.argmax(predicted_labels) == y_i
        return accuracy / X.shape[0]
    
    def backward(self, x, y, output, hiddens, loss_function='cross_entropy'):
        grad_weights = []
        grad_biases = []
        
        for i in range(len(self.W) - 1, -1, -1):
            h = x if i == 0 else hiddens[i-1]
            if i == len(self.W) - 1:
                if loss_function == 'cross_entropy':
                    grad_z = softmax(output) - y
                # elif loss_function == 'squared':
                #    grad_z = output - y
            else:
                relu_derivs = np.array([k > 0 for k in hiddens[i-1]]) # RELU derivative
                grad_z = self.W[i+1].T.dot(grad_z) * relu_derivs

            # Gradient of hidden parameters.
            grad_weights.append(grad_z[:, None].dot(h[:, None].T))
            grad_biases.append(grad_z)

        grad_weights.reverse()
        grad_biases.reverse()
        return grad_weights, grad_biases
    
    def update_params(self, grad_weights, grad_biases, learning_rate): 
        for i in range(len(self.W)):
            self.W[i] -= learning_rate * grad_weights[i]
            self.b[i] -= learning_rate * grad_biases[i]

    def train_epoch(self, X, y, learning_rate=0.001, **kwargs):
        """
        Dont forget to return the loss of the epoch.
        """
        # Q1.3 (a)
        total_loss = 0
        for x_i, y_i in zip(X, y):
            output, hiddens = self.predict(x_i)

            y_i_one_hot = np.zeros(output.shape)
            y_i_one_hot[y_i] = 1

            grad_weights, grad_biases = self.backward(x_i, y_i_one_hot, output, hiddens, 'cross_entropy')
            self.update_params(grad_weights, grad_biases, learning_rate=learning_rate)

            # Compute the cross-entropy loss with epsilon to avoid log(0)
            total_loss += -np.sum(y_i_one_hot * np.log(softmax(output) + 1e-10))

        return total_loss / len(X)  # Return average loss


def plot(epochs, train_accs, val_accs, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs, train_accs, label='train')
    plt.plot(epochs, val_accs, label='validation')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')

def plot_loss(epochs, loss, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, loss, label='train')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')


def plot_w_norm(epochs, w_norms, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('W Norm')
    plt.plot(epochs, w_norms, label='train')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=100,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    parser.add_argument('-l2_penalty', type=float, default=0.0,)
    parser.add_argument('-data_path', type=str, default='intel_landscapes.v2.npz',)
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_dataset(data_path=opt.data_path, bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]
    n_classes = np.unique(train_y).size
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)
    epochs = np.arange(1, opt.epochs + 1)
    train_loss = []
    weight_norms = []
    valid_accs = []
    train_accs = []

    start = time.time()

    print('initial train acc: {:.4f} | initial val acc: {:.4f}'.format(
        model.evaluate(train_X, train_y), model.evaluate(dev_X, dev_y)
    ))
    
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        if opt.model == 'mlp':
            loss = model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        else:
            model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate,
                l2_penalty=opt.l2_penalty,
            )
        
        train_accs.append(model.evaluate(train_X, train_y))
        valid_accs.append(model.evaluate(dev_X, dev_y))
        if opt.model == 'mlp':
            print('loss: {:.4f} | train acc: {:.4f} | val acc: {:.4f}'.format(
                loss, train_accs[-1], valid_accs[-1],
            ))
            train_loss.append(loss)
        elif opt.model == "logistic_regression":
            weight_norm = np.linalg.norm(model.W)
            print('train acc: {:.4f} | val acc: {:.4f} | W norm: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1], weight_norm,
            ))
            weight_norms.append(weight_norm)
        else:
            print('train acc: {:.4f} | val acc: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1],
            ))
    elapsed_time = time.time() - start
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print('Training took {} minutes and {} seconds'.format(minutes, seconds))
    print('Final test acc: {:.4f}'.format(
        model.evaluate(test_X, test_y)
        ))

    # plot
    plot(epochs, train_accs, valid_accs, filename=f"Q1-{opt.model}-accs.pdf")
    if opt.model == 'mlp':
        plot_loss(epochs, train_loss, filename=f"Q1-{opt.model}-loss.pdf")
    elif opt.model == 'logistic_regression':
        plot_w_norm(epochs, weight_norms, filename=f"Q1-{opt.model}-w_norms.pdf")
    with open(f"Q1-{opt.model}-results.txt", "w") as f:
        f.write(f"Final test acc: {model.evaluate(test_X, test_y)}\n")
        f.write(f"Training time: {minutes} minutes and {seconds} seconds\n")


if __name__ == '__main__':
    main()
