#!/usr/bin/env python

# Deep Learning Homework 1

import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')

import time
import utils

# run Q2.1 -  python hw1-q2.py logistic_regression -epochs 100 -batch_size 32
# run Q2.2.a - python hw1-q2.py mlp

class LogisticRegression(nn.Module):

    def __init__(self, n_classes, n_features, **kwargs):
        """
        n_classes (int)
        n_features (int)

        The __init__ should be used to declare what kind of layers and other
        parameters the module has. For example, a logistic regression module
        has a weight matrix and bias vector. For an idea of how to use
        pytorch to make weights and biases, have a look at
        https://pytorch.org/docs/stable/nn.html
        """
        super().__init__()
        # In a pytorch module, the declarations of layers needs to come after
        # the super __init__ line, otherwise the magic doesn't work.

        # Define the weights and biases for logistic regression
        self.linear = nn.Linear(n_features, n_classes)

    def forward(self, x, **kwargs):
        """
        x (batch_size x n_features): a batch of training examples

        Every subclass of nn.Module needs to have a forward() method. forward()
        describes how the module computes the forward pass. In a log-lineear
        model like this, for example, forward() needs to compute the logits
        y = Wx + b, and return y (you don't need to worry about taking the
        softmax of y because nn.CrossEntropyLoss does that for you).

        One nice thing about pytorch is that you only need to define the
        forward pass -- this is enough for it to figure out how to do the
        backward pass.
        """
        return self.linear(x)


class FeedforwardNetwork(nn.Module):
    def __init__(
            self, n_classes, n_features, hidden_size, layers,
            activation_type, dropout, **kwargs):
        """
        n_classes (int)
        n_features (int)
        hidden_size (int)
        layers (int)
        activation_type (str)
        dropout (float): dropout probability

        As in logistic regression, the __init__ here defines a bunch of
        attributes that each FeedforwardNetwork instance has. Note that nn
        includes modules for several activation functions and dropout as well.
        """
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(n_features, hidden_size))
        
        # Hidden layers
        for _ in range(layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_size, n_classes))
        
        # Activation function
        if activation_type == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError("Unsupported activation type")
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, **kwargs):
        """
        x (batch_size x n_features): a batch of training examples

        This method needs to perform all the computation needed to compute
        the output logits from x. This will include using various hidden
        layers, pointwise nonlinear functions, and dropout.
        """
        for i, layer in enumerate(self.layers[:-1]):  # Apply hidden layers 
            x = layer(x)
            x = self.activation(x)
            x = self.dropout(x)
        x = self.layers[-1](x)  # Apply output layer
        return x


def train_batch(X, y, model, optimizer, criterion, **kwargs):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    model: a PyTorch defined model
    optimizer: optimizer used in gradient step
    criterion: loss function

    To train a batch, the model needs to predict outputs for X, compute the
    loss between these predictions and the "gold" labels y using the criterion,
    and compute the gradient of the loss with respect to the model parameters.

    Check out https://pytorch.org/docs/stable/optim.html for examples of how
    to use an optimizer object to update the parameters.

    This function should return the loss (tip: call loss.item()) to get the
    loss as a numerical value that is not part of the computation graph.
    """
    optimizer.zero_grad()
    
    # Forward pass: predict y_hat
    y_hat = model(X)
    loss = criterion(y_hat, y)
    loss.backward()
    
    optimizer.step()

    return loss.item()


def predict(model, X):
    """X (n_examples x n_features)"""
    scores = model(X)  # (n_examples x n_classes)
    predicted_labels = scores.argmax(dim=-1)  # (n_examples)
    return predicted_labels

@torch.no_grad()
def evaluate(model, X, y, criterion):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    """
    model.eval()
    logits = model(X)
    loss = criterion(logits, y)
    loss = loss.item()
    y_hat = logits.argmax(dim=-1)
    n_correct = (y == y_hat).sum().item()
    n_possible = float(y.shape[0])
    model.train()
    return loss, n_correct / n_possible

def train_and_evaluate(model, train_dataloader, dev_X, dev_y, test_X, test_y, criterion, optimizer, epochs):
    train_losses = []
    valid_losses = []
    valid_accs = []

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_train_losses = []
        for X_batch, y_batch in train_dataloader:
            loss = train_batch(X_batch, y_batch, model, optimizer, criterion)
            epoch_train_losses.append(loss)

        train_loss = torch.tensor(epoch_train_losses).mean().item()
        train_losses.append(train_loss)

        val_loss, val_acc = evaluate(model, dev_X, dev_y, criterion)
        valid_losses.append(val_loss)
        valid_accs.append(val_acc)

        print(f"Epoch {epoch}/{epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")

    elapsed_time = time.time() - start_time
    _, test_acc = evaluate(model, test_X, test_y, criterion)

    return train_losses, valid_losses, valid_accs, test_acc, elapsed_time

def plot(epochs, plottables, filename=None, ylim=None):
    """Plot the plottables over the epochs.
    
    Plottables is a dictionary mapping labels to lists of values.
    """
    plt.clf()
    plt.xlabel('Epoch')
    for label, plottable in plottables.items():
        plt.plot(epochs, plottable, label=label)
    plt.legend()
    if ylim:
        plt.ylim(ylim)
    if filename:
        plt.savefig(filename, bbox_inches='tight')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=100, type=int,
                        help="Number of epochs to train for.")
    parser.add_argument('-batch_size', default=32, type=int,
                        help="Size of training batch.")
    parser.add_argument('-l2_decay', type=float, default=0.01)
    parser.add_argument('-momentum', type=float, default=0.0,
                        help="Momentum factor for SGD optimizer (default: 0.0).")
    parser.add_argument('-optimizer', type=str, default='sgd',
                        choices=['sgd', 'adam'],
                        help="Choose optimizer for training (default: sgd).")
    parser.add_argument('-data_path', type=str, default='intel_landscapes.v2.npz',)
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    # Load dataset
    data = utils.load_dataset(opt.data_path)
    dataset = utils.ClassificationDataset(data)
    dev_X, dev_y = dataset.dev_X, dataset.dev_y
    test_X, test_y = dataset.test_X, dataset.test_y

    n_classes = torch.unique(dataset.y).shape[0]
    n_feats = dataset.X.shape[1]

    # Train Logistic Regression if selected
    if opt.model == 'logistic_regression':
        print("Training Logistic Regression Model...")
        best_final_val_acc = 0
        best_lr = None
        best_test_acc = 0

        # Evaluate different learning rates
        learning_rates = [0.00001, 0.001, 0.1]
        results = {}

        for lr in learning_rates:
            print(f"\nTraining with learning rate: {lr}")
            model = LogisticRegression(n_classes, n_feats)
            optimizer = torch.optim.SGD(
                model.parameters(), lr=lr, weight_decay=opt.l2_decay
            )
            criterion = nn.CrossEntropyLoss()

            # Train and validate the model
            train_losses, valid_losses, valid_accs = [], [], []
            for epoch in range(1, opt.epochs + 1):
                epoch_train_losses = []
                for X_batch, y_batch in DataLoader(dataset, batch_size=opt.batch_size, shuffle=True):
                    loss = train_batch(X_batch, y_batch, model, optimizer, criterion)
                    epoch_train_losses.append(loss)

                train_loss = torch.tensor(epoch_train_losses).mean().item()
                val_loss, val_acc = evaluate(model, dev_X, dev_y, criterion)

                train_losses.append(train_loss)
                valid_losses.append(val_loss)
                valid_accs.append(val_acc)

                print(f"Epoch {epoch:03d}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")

            # Check best validation accuracy
            final_val_acc = valid_accs[-1]
            if final_val_acc > best_final_val_acc:
                best_final_val_acc = final_val_acc
                best_lr = lr
                _, best_test_acc = evaluate(model, test_X, test_y, criterion)

            # Store results for this learning rate
            results[lr] = {
                "train_losses": train_losses,
                "valid_losses": valid_losses,
                "valid_accs": valid_accs,
            }

            print(f"Learning Rate {lr} Results: Final Val Acc = {final_val_acc:.4f}, Test Acc = {best_test_acc:.4f}")

        print("\nFinal Results:")
        print(f"Best Learning Rate: {best_lr}")
        print(f"Best Validation Accuracy: {best_final_val_acc:.4f}")
        print(f"Test Accuracy with Best LR: {best_test_acc:.4f}")

        # Plot results
        epochs = torch.arange(1, opt.epochs + 1)
        for lr, result in results.items():
            plot(
                epochs,
                {"Train Loss": result["train_losses"], "Valid Loss": result["valid_losses"]},
                filename=f"logistic_regression_lr-{lr}-losses.pdf"
            )
            plot(
                epochs,
                {"Valid Accuracy": result["valid_accs"]},
                filename=f"logistic_regression_lr-{lr}-accuracy.pdf"
            )

    # Train Feedforward Network if selected
    elif opt.model == 'mlp':
        print("Training Feedforward Network with default batch...")
        # Default hyperparameters for MLP
        default_hyperparams = {
            'epochs': 200,
            'learning_rate': 0.002,
            'hidden_size': 200,
            'layers': 2,
            'dropout': 0.3,
            'batch_size': 64,
            'activation': 'relu',
            'l2_decay': 0.0,
        }

        # Train with Default Hyperparameters
        train_dataloader = DataLoader(dataset, batch_size=default_hyperparams['batch_size'], shuffle=True, generator = torch.Generator().manual_seed(42))
        model = FeedforwardNetwork(
            n_classes=n_classes,
            n_features=n_feats,
            hidden_size=default_hyperparams['hidden_size'],
            layers=default_hyperparams['layers'],
            activation_type=default_hyperparams['activation'],
            dropout=default_hyperparams['dropout']
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=default_hyperparams['learning_rate'], weight_decay=default_hyperparams['l2_decay'])
        criterion = nn.CrossEntropyLoss()

        train_losses_default, valid_losses_default, valid_accs_default, test_acc_default, elapsed_time = train_and_evaluate(
            model, train_dataloader, dev_X, dev_y, test_X, test_y, criterion, optimizer, default_hyperparams['epochs']
        )
        
        print("Training Feedforward Network with batch 512...")
        # Train with batch_size=512
        train_dataloader_512 = DataLoader(dataset, batch_size=512, shuffle=True, generator = torch.Generator().manual_seed(42))
        model = FeedforwardNetwork(
            n_classes=n_classes,
            n_features=n_feats,
            hidden_size=default_hyperparams['hidden_size'],
            layers=default_hyperparams['layers'],
            activation_type=default_hyperparams['activation'],
            dropout=default_hyperparams['dropout']
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=default_hyperparams['learning_rate'], weight_decay=default_hyperparams['l2_decay'])

        train_losses_512, valid_losses_512, valid_accs_512, test_acc_512, elapsed_time_512 = train_and_evaluate(
            model, train_dataloader_512, dev_X, dev_y, test_X, test_y, criterion, optimizer, default_hyperparams['epochs']
        )

        # Plot results for Feedforward Network
        epochs = torch.arange(1, default_hyperparams['epochs'] + 1)
        plot(
            epochs,
            {"Train Loss (Default)": train_losses_default, "Train Loss (Batch=512)": train_losses_512},
            filename="feedforward_loss_comparison.pdf"
        )
        plot(
            epochs,
            {"Valid Loss (Default)": valid_losses_default, "Valid Loss (Batch=512)": valid_losses_512},
            filename="feedforward_valid_loss_comparison.pdf"
        )
        plot(
            epochs,
            {"Valid Accuracy (Default)": valid_accs_default, "Valid Accuracy (Batch=512)": valid_accs_512},
            filename="feedforward_accuracy_comparison.pdf"
        )

        print("\nFeedforward Network Results:")
        print(f"Default Batch Size: Test Accuracy = {test_acc_default:.4f}")
        print(f"Default Batch Size: Time of Execution = {elapsed_time:.4f}")
        print(f"Batch Size 512: Test Accuracy = {test_acc_512:.4f}")
        print(f"Batch Size 512: Time of Execution = {elapsed_time_512:.4f}")

if __name__ == "__main__":
    main()
