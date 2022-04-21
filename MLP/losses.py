import numpy as np
class LossCrossEntropy(object):
    def __init__(self, name):
        super(LossCrossEntropy, self).__init__()
        self.name = name

    def forward(self, X, T):
        """
        Forward message.
        :param X: loss inputs (outputs of the previous layer), shape (n_samples, n_inputs), n_inputs is the same as
        the number of classes
        :param T: one-hot encoded targets, shape (n_samples, n_inputs)
        :return: layer output, shape (n_samples, 1)
        """
        # TODO to be done
        #if X.all() == 1: return -np.log(T)
        #else: return -np.log(1 - T)
        T= np.argmax(T,axis=1)
        log_likelihood = -np.log(X[:, T])
        loss = np.sum(log_likelihood) / len(T)
        return loss

    def delta(self, X, T):
        """
        Computes delta vector for the output layer.
        :param X: loss inputs (outputs of the previous layer), shape (n_samples, n_inputs), n_inputs is the same as
        the number of classes
        :param T: one-hot encoded targets, shape (n_samples, n_inputs)
        :return: delta vector from the loss layer, shape (n_samples, n_inputs)
        """
        # TODO to be done
        #delta_current = X-T
        T= np.argmax(T,axis=1)
        m=len(T)
        grad =np.array(X)
        grad[range(m), T] -= 1
        grad = grad / m
        return grad

class LossCrossEntropyForSoftmaxLogits(object):
    def __init__(self, name):
        super(LossCrossEntropyForSoftmaxLogits, self).__init__()
        self.name = name

    def forward(self, X, T):
        # TODO to be done
        #expX = np.exp(X - np.max(X))
        #s = expX / expX.sum(axis=0, keepdims=True)
        #if X == 1: return -np.log(s)
        #else: return -np.log(1 - s)
        
        expX = np.exp(X - np.max(X))
        s = expX / expX.sum(axis=0, keepdims=True)
        s = s.clip(min=1e-8,max=None)
        return (np.where(T==1,-np.log(s), 0)).sum(axis=1)
        
        
    def delta(self, X, T):
        # TODO to be done
        #expX = np.exp(X - np.max(X))
        #s = expX / expX.sum(axis=0, keepdims=True)
        #si_sj = - s * s.reshape(3, 1)
        #s_der = np.diag(s) + si_sj
        #delta_current = s_der - T
        #return delta_current
        
        expX = np.exp(X - np.max(X))
        s = expX / expX.sum(axis=0, keepdims=True)
        s = s.clip(min=1e-8,max=None)
        delta_current = np.where(T==1,-1/s, 0)
        return delta_current

