import numpy as np

from deeplearning.layers import *
from deeplearning.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
        self.params["W1"] = np.random.normal(loc = 0, scale = weight_scale, size = (input_dim, hidden_dim))
        self.params["b1"] = np.zeros(hidden_dim)
        self.params["W2"] = np.random.normal(loc = 0, scale = weight_scale, size = (hidden_dim, num_classes))
        self.params["b2"] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################

        out, fr_cache = affine_relu_forward(X, self.params["W1"], self.params["b1"])
        scores, f_cache = affine_forward(out, self.params["W2"], self.params["b2"])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization on the weights,    #
        # but not the biases.                                                      #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        
        loss, dx_final = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(np.square(self.params["W1"])) 
                                  + np.sum(np.square(self.params["W2"])))
        ab_dx, ab_dw, ab_db = affine_backward(dx_final, f_cache)
        arb_dx, arb_dw, arb_db = affine_relu_backward(ab_dx, fr_cache)
        
        grads["W1"] = arb_dw + self.reg * self.params["W1"]
        grads["b1"] = arb_db
        grads["W2"] = ab_dw + self.reg * self.params["W2"]
        grads["b2"] = ab_db
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. For a network with L layers,
    the architecture will be

    {affine - relu} x (L - 1) - affine - softmax

    where the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3 * 32 * 32, num_classes=10,reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        """

        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        ############################################################################
        self.params["W1"] = np.random.normal(loc = 0, scale = weight_scale, size = (input_dim, hidden_dims[0]))
        self.params["b1"] = np.zeros(hidden_dims[0])
        
        for num in range(1, self.num_layers-1):
            text = num+1
            weight_index = "W" + str(text)
            bias_index = "b" + str(text)
            
            self.params[weight_index] = np.random.normal(loc = 0, scale = weight_scale, size = (hidden_dims[num-1], hidden_dims[num]))
            self.params[bias_index] = np.zeros(hidden_dims[num])
        
        
        weight_index = "W" + str(self.num_layers)
        bias_index = "b" + str(self.num_layers)
    
        self.params[weight_index] = np.random.normal(loc = 0, scale = weight_scale, size = (hidden_dims[-1], num_classes))
        self.params[bias_index] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #                                                       #
        ############################################################################
        outputs = {}
        
        outputs["out1"], outputs["cache1"] = affine_relu_forward(X, self.params["W1"], self.params["b1"])
        
        for num in range(2, self.num_layers):
            out_index_before = "out" + str(num-1)
            cache_index_before = "cache" + str(num-1)
            out_index = "out" + str(num)
            cache_index = "cache" + str(num)
            weight_index = "W" + str(num)
            bias_index = "b" + str(num)
            
            outputs[out_index], outputs[cache_index] = affine_relu_forward(outputs[out_index_before], self.params[weight_index], self.params[bias_index])
            
        out_index_before = "out" + str(self.num_layers-1)
        out_index = "out" + str(self.num_layers)
        cache_index = "cache" + str(self.num_layers)
        weight_index = "W" + str(self.num_layers)
        bias_index = "b" + str(self.num_layers)
        
        outputs[out_index], outputs[cache_index] = affine_forward(outputs[out_index_before], self.params[weight_index], self.params[bias_index])
        
        scores = outputs[out_index]
            
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization on the         #
        # weights, but not the biases.                                             #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        
        loss, dx_final = softmax_loss(scores, y)
        for num in range(1, self.num_layers+1):
            loss += 0.5 * self.reg *(np.sum(np.square(self.params["W" + str(num)]))) 
                  
        pregrads = {}
                  
        cache_index = "cache" + str(self.num_layers)     
        dx_index = "dx" + str(self.num_layers)    
        dw_index = "dw" + str(self.num_layers)    
        db_index = "db" + str(self.num_layers)
                                     
        pregrads[dx_index], pregrads[dw_index], pregrads[db_index] = affine_backward(dx_final, outputs[cache_index])
        
        for i in range(self.num_layers-1, 0, -1):
            cache_index = "cache" + str(i)
            dx_index_plus = "dx" + str(i+1)
            dx_index = "dx" + str(i)
            dw_index = "dw" + str(i)
            db_index = "db" + str(i)
                                     
            pregrads[dx_index], pregrads[dw_index], pregrads[db_index] = affine_relu_backward(pregrads[dx_index_plus], outputs[cache_index])
        
        for i in range(1, self.num_layers+1):
            w_index = "W" + str(i)
            b_index = "b" + str(i)
            dw_index = "dw" + str(i)
            db_index = "db" + str(i)
                                     
            grads[w_index] = pregrads[dw_index] + self.reg * self.params[w_index]
            grads[b_index] = pregrads[db_index]
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
