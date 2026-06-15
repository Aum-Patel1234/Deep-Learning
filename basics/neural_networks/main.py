import numpy as np
from n_layer_nn import NeuralNetwork2L, NeuralNetworkNL

if __name__ == "__main__":
    n2l = NeuralNetwork2L()
    nn = NeuralNetworkNL([5, 4, 3, 1])

    X = np.random.randn(5, 2)

    AL, caches = nn._L_model_forward(X, nn.parameters)

    print(AL.shape)  # (1, 2)
    print(len(caches))
    print(AL)
