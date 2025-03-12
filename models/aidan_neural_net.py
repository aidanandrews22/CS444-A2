"""Neural network model."""

from typing import Sequence

import numpy as np


class NeuralNetwork:
    """A multi-layer fully-connected neural network. The net has an input
    dimension of N, a hidden layer dimension of H, and output dimension C.
    We train the network with a MLE loss function. The network uses a ReLU
    nonlinearity after each fully connected layer except for the last.
    The outputs of the last fully-connected layer are passed through
    a sigmoid.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        num_layers: int,
        opt: str,
    ):
        """Initialize the model. Weights are initialized to small random values
        and biases are initialized to zero. Weights and biases are stored in
        the variable self.params, which is a dictionary with the following
        keys:
        W1: 1st layer weights; has shape (D, H_1)
        b1: 1st layer biases; has shape (H_1,)
        ...
        Wk: kth layer weights; has shape (H_{k-1}, C)
        bk: kth layer biases; has shape (C,)
        Parameters:
            input_size: The dimension D of the input data
            hidden_size: List [H1,..., Hk] with the number of neurons Hi in the
                hidden layer i
            output_size: output dimension C
            num_layers: Number of fully connected layers in the neural network
            opt: option for using "SGD" or "Adam" optimizer (Adam is Extra Credit)
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers
        self.opt = opt
        self.lr = 0.05  # Lower initial learning rate
        self.batch_size = 128  # Mini-batch size
        self.decay_rate = 0.9  # More gradual decay
        self.momentum = 0.9  # Momentum coefficient
        self.decay_steps = 500  # Apply decay every N steps
        self.step_count = 0  # Track update steps
        self.t = 0

        assert len(hidden_sizes) == (num_layers - 1)
        sizes = [input_size] + hidden_sizes + [output_size]

        self.params = {}
        for i in range(1, num_layers + 1):
            # He initialization for ReLU
            self.params["W" + str(i)] = np.random.randn(
                sizes[i - 1], sizes[i]
            ) * np.sqrt(2 / sizes[i - 1])
            self.params["b" + str(i)] = np.zeros(sizes[i])

            # Init momentum buffers
            self.params["mW" + str(i)] = np.zeros((sizes[i - 1], sizes[i]))
            self.params["mb" + str(i)] = np.zeros(sizes[i])

    def linear(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fully connected (linear) layer.
        Parameters:
            W: the weight matrix
            X: the input data
            b: the bias
        Returns:
            the output
        """
        return (X @ W) + b

    def linear_grad(
        self, W: np.ndarray, X: np.ndarray, de_dz: np.ndarray
    ) -> np.ndarray:
        """Gradient of linear layer
        Parameters:
            W: the weight matrix
            X: the input data
            de_dz: the gradient of loss
        Returns:
            de_dw, de_db, de_dx
            where
                de_dw: gradient of loss with respect to W
                de_db: gradient of loss with respect to b
                de_dx: gradient of loss with respect to X
        """
        return [np.dot(X.T, de_dz), np.sum(de_dz, axis=0), np.dot(de_dz, W.T)]

    def relu(self, X: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output
        """
        return np.maximum(0, X)  # More efficient

    def relu_grad(self, X: np.ndarray) -> np.ndarray:
        """Gradient of Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output data
        """
        return np.where(X > 0, 1.0, 0.0)

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        # Numerical stability improvement
        x = np.clip(x, -88.0, 88.0)  # Prevent overflow

        sigmoid = np.empty_like(x)
        mask1 = x < 0
        sigmoid[mask1] = np.exp(x[mask1]) / (1 + np.exp(x[mask1]))

        mask2 = ~mask1
        sigmoid[mask2] = 1 / (1 + np.exp(-x[mask2]))

        return sigmoid

    def sigmoid_grad(self, X: np.ndarray) -> np.ndarray:
        s = self.sigmoid(X)
        return s * (1 - s)

    def mse(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        return np.mean((y - p) ** 2)

    def mse_grad(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        return (2 / y.shape[0]) * (p - y)  # Correct factor with batch size

    def mse_sigmoid_grad(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        return (2 / y.shape[0]) * (p - y) * p * (1 - p)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the outputs for all of the data samples.
        Hint: this function is also used for prediction.
        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training or
                testing sample
        Returns:
            Matrix of shape (N, C)
        """
        self.outputs = {}
        self.outputs["L0"] = X

        for i in range(1, self.num_layers + 1):
            output = self.linear(
                self.params["W" + str(i)],
                self.outputs["L" + str(i - 1)],
                self.params["b" + str(i)],
            )
            self.outputs["UL" + str(i)] = output
            if i < self.num_layers:
                output = self.relu(output)
            else:
                output = self.sigmoid(output)
            self.outputs["L" + str(i)] = output

        return self.outputs["L" + str(self.num_layers)]

    def backward(self, y: np.ndarray) -> float:
        """Perform back-propagation and compute the gradients and losses.
        Parameters:
            y: training value targets
        Returns:
            Total loss for this batch of training samples
        """
        self.gradients = {}

        upstream_gradient = self.mse_sigmoid_grad(
            y, self.outputs["L" + str(self.num_layers)]
        )
        final_gradient = self.linear_grad(
            self.params["W" + str(self.num_layers)],
            self.outputs["L" + str(self.num_layers - 1)],
            upstream_gradient,
        )
        self.gradients["W" + str(self.num_layers)] = final_gradient[0]
        self.gradients["b" + str(self.num_layers)] = final_gradient[1]
        self.gradients["X" + str(self.num_layers)] = final_gradient[2]

        for i in range(self.num_layers - 1, 0, -1):
            upstream_gradient = self.gradients["X" + str(i + 1)] * self.relu_grad(
                self.outputs["UL" + str(i)]
            )
            gradient = self.linear_grad(
                self.params["W" + str(i)],
                self.outputs["L" + str(i - 1)],
                upstream_gradient,
            )

            self.gradients["W" + str(i)] = gradient[0]
            self.gradients["b" + str(i)] = gradient[1]
            self.gradients["X" + str(i)] = gradient[2]

        return np.mean(self.mse(y, self.outputs["L" + str(self.num_layers)]))

    def train_on_batch(self, X_batch, y_batch, lr):
        """Train model on a single mini-batch"""
        self.forward(X_batch)
        loss = self.backward(y_batch)
        self.update(lr)
        return loss

    def train_epoch(self, X, y, lr):
        """Train for one epoch using mini-batches"""
        indices = np.random.permutation(len(X))
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        losses = []
        for i in range(0, len(X), self.batch_size):
            X_batch = X_shuffled[i : i + self.batch_size]
            y_batch = y_shuffled[i : i + self.batch_size]
            loss = self.train_on_batch(X_batch, y_batch, lr)
            losses.append(loss)

        return np.mean(losses)

    def update(
        self, lr: float = 0.001, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8
    ):
        """Update the parameters of the model using the previously calculated
        gradients.
        Parameters:
            lr: Learning rate
            b1: beta 1 parameter (for Adam)
            b2: beta 2 parameter (for Adam)
            eps: epsilon to prevent division by zero (for Adam)
        """
        self.step_count += 1

        # Learning rate schedule - decay every decay_steps
        if self.step_count % self.decay_steps == 0:
            lr = lr * self.decay_rate

        if self.opt == "SGD":
            for i in range(1, self.num_layers + 1):
                # Update with momentum
                self.params["mW" + str(i)] = (
                    self.momentum * self.params["mW" + str(i)]
                    - lr * self.gradients["W" + str(i)]
                )
                self.params["mb" + str(i)] = (
                    self.momentum * self.params["mb" + str(i)]
                    - lr * self.gradients["b" + str(i)]
                )

                self.params["W" + str(i)] += self.params["mW" + str(i)]
                self.params["b" + str(i)] += self.params["mb" + str(i)]
        elif self.opt == "Adam":
            # TODO: (Extra credit) implement Adam optimizer here
            self.t += 1
            for i in range(1, self.num_layers + 1):
                # Clip gradients for stability
                dW = np.clip(self.gradients["W" + str(i)], -1, 1)
                db = np.clip(self.gradients["b" + str(i)], -1, 1)

                # Update biased first moment estimate
                self.params["m_W" + str(i)] = (
                    b1 * self.params["m_W" + str(i)] + (1 - b1) * dW
                )
                self.params["m_b" + str(i)] = (
                    b1 * self.params["m_b" + str(i)] + (1 - b1) * db
                )

                # Update biased second raw moment estimate
                self.params["v_W" + str(i)] = b2 * self.params["v_W" + str(i)] + (
                    1 - b2
                ) * (dW**2)
                self.params["v_b" + str(i)] = b2 * self.params["v_b" + str(i)] + (
                    1 - b2
                ) * (db**2)

                # Compute bias-corrected first moment estimate
                m_W_corrected = self.params["m_W" + str(i)] / (1 - b1**self.t)
                m_b_corrected = self.params["m_b" + str(i)] / (1 - b1**self.t)

                # Compute bias-corrected second raw moment estimate
                v_W_corrected = self.params["v_W" + str(i)] / (1 - b2**self.t)
                v_b_corrected = self.params["v_b" + str(i)] / (1 - b2**self.t)

                # Update parameters
                self.params["W" + str(i)] -= (
                    lr * m_W_corrected / (np.sqrt(v_W_corrected) + eps)
                )
                self.params["b" + str(i)] -= (
                    lr * m_b_corrected / (np.sqrt(v_b_corrected) + eps)
                )
        else:
            raise NotImplementedError

    def update_lr(self):
        """Legacy learning rate update - keep for compatibility"""
        self.lr *= self.decay_rate
