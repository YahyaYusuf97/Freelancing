import numpy as np
class LinearLayer(object):
    def __init__(self, n_inputs, n_units, rng, name):
        """
        Linear (dense, fully-connected) layer.
        :param n_inputs:
        :param n_units:
        :param rng: random number generator used for initialization
        :param name:
        """
        super(LinearLayer, self).__init__()
        self.n_inputs = n_inputs
        self.n_units = n_units
        self.rng = rng
        self.name = name
        self.initialize()
        self.input = None
        
    def has_params(self):
        return True

    def forward(self, X):
        """
        Forward message.
        :param X: layer inputs, shape (n_samples, n_inputs)
        :return: layer output, shape (n_samples, n_units)
        """
        #TODO to be done
        self.input = X
        Z = np.dot(X, self.W)+self.b
        return Z
        """
        self.input = X
        Z = np.dot(X, self.W)+self.b
        self.input = X
        return Z
    """

    def delta(self, Y, delta_next):
        """
        Computes delta (dl/d(layer inputs)), based on delta from the following layer. The computations involve backward
        message.
        :param Y: output of this layer (i.e., input of the next), shape (n_samples, n_units)
        :param delta_next: delta vector backpropagated from the following layer, shape (n_samples, n_units)
        :return: delta vector from this layer, shape (n_samples, n_inputs)
        """
        # TODO to be done
        delta_current = np.dot(delta_next,self.W.T)
        return delta_current
        
           

    def grad(self, X, delta_next):
        """
        Gradient averaged over all samples. The computations involve parameter message.
        :param X: layer input, shape (n_samples, n_inputs)
        :param delta_next: delta vector backpropagated from the following layer, shape (n_samples, n_units)
        :return: a list of two arrays [dW, db] corresponding to gradients of loss w.r.t. weights and biases, the shapes
        of dW and db are the same as the shapes of the actual parameters (self.W, self.b)
        
        there is no method dedicated to compute the
        parameter message, instead, you will find grad method computing the
        gradient of loss with respect to all layer parameters.
        """
        # TODO to be done
        #self.input = X
        #mmkn n transpose el x not sure
        self.dW = np.dot(X.T,delta_next)
        self.db = np.mean(delta_next.T, axis=1, keepdims=True).squeeze()
        return [self.dW , self.db]


    def initialize(self):
        """
        Perform He's initialization (https://arxiv.org/pdf/1502.01852.pdf). This method is tuned for ReLU activation
        function. Biases are initialized to 1 increasing probability that ReLU is not initially turned off.
        """
        scale = np.sqrt(2.0 / self.n_inputs)
        self.W = self.rng.normal(loc=0.0, scale=scale, size=(self.n_inputs, self.n_units))
        self.b = np.ones(self.n_units)

    def update_params(self, dtheta):
        """
        Updates weighs and biases.
        :param dtheta: contains a two element list of weight and bias updates the shapes of which corresponds to self.W
        and self.b
        """
        assert len(dtheta) == 2, len(dtheta)
        dW, db = dtheta
        assert dW.shape == self.W.shape, dW.shape
        assert db.shape == self.b.shape, db.shape
        self.W += dW
        self.b += db
