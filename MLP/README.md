Multilayer Perceptron (MLP) back-propagation
Implemented LinearLayer, ReLULayer, SoftmaxLayer and LossCrossEntropy classes using numpy only.
The client requested to compute the δl’s (delta methods) directly as the Jacobian matrices representing the backward messages are often sparse.
Similarly, there is no method dedicated to compute the parameter message, instead,
you will find grad method computing the gradient of loss with respect to all layer parameters.
