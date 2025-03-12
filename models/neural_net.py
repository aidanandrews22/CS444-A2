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
        self.lr = 0.05 # 0.2  
        
        assert len(hidden_sizes) == (num_layers - 1)
        sizes = [input_size] + hidden_sizes + [output_size]

        self.params = {}
        for i in range(1, num_layers + 1):
            self.params["W" + str(i)] = np.random.randn(sizes[i - 1], sizes[i]) / np.sqrt(sizes[i - 1])
            self.params["b" + str(i)] = np.zeros(sizes[i])
            
            # TODO: (Extra Credit) You may set parameters for Adam optimizer here

    def linear(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fully connected (linear) layer.
        Parameters:
            W: the weight matrix
            X: the input data
            b: the bias
        Returns:
            the output
        """ 
        # TODO: implement me
        return (X @ W) + b # X.T @ W.T? 
    
    def linear_grad(self, W: np.ndarray, X: np.ndarray, de_dz: np.ndarray) -> np.ndarray:
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
        # TODO: implement me
        # return [np.dot(de_dz, X.T), np.sum(de_dz), np.dot(W.T, de_dz)]
        return [np.dot(X.T, de_dz), np.sum(de_dz, axis = 0), np.dot(de_dz, W.T)]

    def relu(self, X: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output
        """
        # TODO: implement me
        return np.where(X > 0, X, 0) 

    def relu_grad(self, X: np.ndarray) -> np.ndarray:
        """Gradient of Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output data
        """
         # TODO: implement me
        return np.where(X > 0, 1, 0) 

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        # TODO ensure that this is numerically stable
        # sigmoid = np.zeros_like(x) 

        # mask1 = x < 0
        # sigmoid[mask1] = np.exp(x[mask1] / (1 + np.exp(x[mask1])))

        # mask2 = x >= 0          
        # sigmoid[mask2] = 1 / (1 + np.exp(-x[mask2]))          

        # return sigmoid 
        sigmoid = np.empty_like(x)
        mask1 = x < 0
        sigmoid[mask1] = np.exp(x[mask1]) / (1 + np.exp(x[mask1]))
        
        mask2 = x >= 0
        sigmoid[mask2] = 1 / (1 + np.exp(-x[mask2]))
        
        return sigmoid
    
    def sigmoid_grad(self, X: np.ndarray) -> np.ndarray:
        # TODO implement this
        return self.sigmoid(X) * (1 - self.sigmoid(X)) 

    def mse(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        # TODO implement this
        # sum = 0 
        # for i in range(len(y)):
        #   sum += np.square(y[i] - p[i])
        # return sum / (2 * len(y)) 
        return np.mean((y - p) ** 2)
    
    def mse_grad(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        # TODO implement this
        return (1 / y.size) * (p - y) 
     
    def mse_sigmoid_grad(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        # TODO implement this                    
        # return (1 / len(y)) * (p - y) * p * (1 - p)  
        return (2 / y.size) * (p - y) * p * (1 - p)        

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the outputs for all of the data samples.
        Hint: this function is also used for prediction.
        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training or
                testing sample
        Returns:
            Matrix of shape (N, C) 
        """
        # TODO: implement me. You'll want to store the output of each layer in
        # self.outputs as it will be used during back-propagation. You can use
        # the same keys as self.params. You can use functions like
        # self.linear, self.relu, and self.mse in here.

        self.outputs = {}
        self.outputs["L0"] = X 

        for i in range(1, self.num_layers + 1):
          output = self.linear(self.params["W" + str(i)], self.outputs["L" + str(i - 1)], self.params["b" + str(i)])
          self.outputs["UL" + str(i)] = output 
          if (i < self.num_layers):
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
        # TODO: implement me. You'll want to store the gradient of each
        # parameter in self.gradients as it will be used when updating each
        # parameter and during numerical gradient checks. You can use the same
        # keys as self.params. You can add functions like self.linear_grad,
        # self.relu_grad, and self.softmax_grad if it helps organize your code.

        self.gradients = {}

        # final_gradient = self.linear_grad(self.params["W" + str(self.num_layers)], self.outputs["L" + str(self.num_layers)], self.mse_sigmoid_grad(y, self.outputs["UL" + str(self.num_layers)]))
        # self.num_layers - 1? 
        upstream_gradient = self.mse_sigmoid_grad(y, self.outputs["L" + str(self.num_layers)])  
        final_gradient = self.linear_grad(self.params["W" + str(self.num_layers)], self.outputs["L" + str(self.num_layers - 1)], upstream_gradient)
        self.gradients["W" + str(self.num_layers)] = final_gradient[0]
        self.gradients["b" + str(self.num_layers)] = final_gradient[1]
        self.gradients["X" + str(self.num_layers)] = final_gradient[2]

        for i in range(self.num_layers - 1, 0, -1):
          upstream_gradient = self.gradients["X" + str(i + 1)] * self.relu_grad(self.outputs["UL" + str(i)])
          gradient = self.linear_grad(self.params["W" + str(i)], self.outputs["L" + str(i - 1)], upstream_gradient)
          
          self.gradients["W" + str(i)] = gradient[0]
          self.gradients["b" + str(i)] = gradient[1]
          self.gradients["X" + str(i)] = gradient[2]

        return np.mean(self.mse(y, self.outputs["L" + str(self.num_layers)])) 

        # upstream_gradient = self.gradients["X" + str(i + 1)]
        # if (i > 1):
        #  upstream_gradient *= self.relu_grad(self.outputs["UL" + str(i)])
        
        # gradient = self.linear_grad(self.params["W" + str(i)], self.outputs["L" + str(i - 1)], self.gradients["X" + str(i + 1)])
        # if (i > 1):
        #   gradient[2] *= self.relu_grad(self.outputs["UL" + str(i)])

    def update(
        self,
        lr: float = 0.001,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8
    ):
        """Update the parameters of the model using the previously calculated
        gradients.
        Parameters:
            lr: Learning rate
            b1: beta 1 parameter (for Adam)
            b2: beta 2 parameter (for Adam)
            eps: epsilon to prevent division by zero (for Adam)
        """
        if self.opt == 'SGD':
            # TODO: implement SGD optimizer here
            for i in range(1, self.num_layers + 1):
              self.params["W" + str(i)] -= lr * self.gradients["W" + str(i)]  
              self.params["b" + str(i)] -= lr * self.gradients["b" + str(i)] 
        elif self.opt == 'Adam':
            # TODO: (Extra credit) implement Adam optimizer here
            pass
        else:
            raise NotImplementedError
    
    def update_lr(self):
      self.lr *= 0.65 # .87  
        