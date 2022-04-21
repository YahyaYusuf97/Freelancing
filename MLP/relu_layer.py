import numpy as np
class ReLULayer(object):
    def __init__(self, name):
        super(ReLULayer, self).__init__()
        self.name = name
        
    def has_params(self):
        return False

    def forward(self, X):
        # TODO to be done
        A = np.maximum(0,X)
        self.input=X
        return A


    def delta(self, Y, delta_next):
        # TODO to be done
        # calculate relu der.
        # multiply it (*) delta_next
        # return the multipliction result
        delta_current = np.heaviside(self.input,0) *(delta_next)
        return delta_current
  