import numpy as np
class SoftmaxLayer(object):
    def __init__(self, name):
        super(SoftmaxLayer, self).__init__()
        self.name = name

    def has_params(self):
        return False

    def forward(self, X):
        # TODO to be done
        expX = np.exp(X - np.max(X))
        s = expX / expX.sum(axis=0, keepdims=True)
        return s

    def delta(self, Y, delta_next):
        # TODO to be done
        #expY = np.exp(Y - np.max(Y))
        #s = expY / expY.sum(axis=0, keepdims=True)
        #si_sj = - s * s.reshape(3, 1)
        #s_der = np.diag(s) + si_sj
        #delta_current = s_der * delta_next
        softmax = np.reshape(Y, (1, -1))
        grad = np.reshape(delta_next, (1, -1))
        d_softmax = (
                softmax * np.identity(softmax.size)
                - softmax.transpose() @ softmax)

        delta_current =  grad @ d_softmax
        return delta_current.reshape(-1,Y.shape[1])
        