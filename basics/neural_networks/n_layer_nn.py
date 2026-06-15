from functools import cache
import numpy as np
from typing import Any, Dict, Tuple, List

SIGMOID_ACTIVATION: str = "sigmoid"
RELU_ACTIVATION: str = "relu"


class BaseClass:
    def _linear_forward(self, A: np.ndarray, W: np.ndarray, b: np.ndarray):
        assert W.shape[1] == A.shape[0]
        Z = W @ A + b
        cache = (A, W, b)
        return Z, cache

    def _linear_activation_forward(
        self, A_prev: np.ndarray, W: np.ndarray, b: np.ndarray, activation: str
    ):
        Z, linear_cache = self._linear_forward(A_prev, W, b)
        if activation == SIGMOID_ACTIVATION:
            A, activation_cache = 1.0 / (1 + np.exp(-Z)), Z
        elif activation == RELU_ACTIVATION:
            A, activation_cache = np.maximum(0, Z), Z
        else:
            raise ValueError(f"Unknown activation: {activation}")

        cache = (linear_cache, activation_cache)
        return A, cache

    def _compute_cost(self, A: np.ndarray, Y: np.ndarray):
        # A -> predicted, Y -> true label
        # row->features, col->data
        assert A.shape == Y.shape
        m = Y.shape[1]
        cost = (-1.0 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
        return np.squeeze(cost)


class NeuralNetwork2L(BaseClass):
    def __init__(self) -> None:
        pass

    def _initialize_parameters(
        self, n_x: int, n_h: int, n_y: int
    ) -> Dict[str, np.ndarray]:
        W1 = np.random.randn(n_x, n_h) * 0.01
        b1 = np.zeros((n_h, 1))
        W2 = np.random.randn(n_h, n_y) * 0.01
        b2 = np.zeros((n_y, 1))

        return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


class NeuralNetworkNL(BaseClass):
    def __init__(self, layer_dims: List[int]) -> None:
        self.layer_dims = layer_dims
        self.parameters = self._initialize_parameters_deep_random(layer_dims)

    # let say layer_dims = [5(in_features),4,3,1]
    # then : A(in_features, m) [ColumnWise]
    # W1 -> (4, 5)
    # b1 -> (4, 1) -> Z1 = W1.A(input) + b1 -> (4,5)*(5,m) -> (4, m)
    #
    # W2 -> (3, 4)
    # b2 -> (3, 1) -> Z2 = W2.A1 + b2 -> (3,4)*(4,m)+(3,1) = (3,m)
    #
    # W3 -> (1, 3)
    # b3 -> (1, 1) -> Z3 = (1,3)*(3, m)+(1,1) -> (1, m)
    def _initialize_parameters_deep_random(
        self, layer_dims: List[int]
    ) -> Dict[str, np.ndarray]:
        L = len(layer_dims)
        parameters: Dict[str, np.ndarray] = {}

        for l in range(1, L):
            parameters["W" + str(l)] = (
                np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
            )
            parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

        return parameters

    def _L_model_forward(self, X: np.ndarray, parameters: Dict[str, np.ndarray]):
        caches = []
        A = X
        L = len(self.layer_dims) - 1

        for l in range(1, L):
            A_prev = A
            A, cache = self._linear_activation_forward(
                A_prev,
                parameters["W" + str(l)],
                parameters["b" + str(l)],
                RELU_ACTIVATION,
            )
            caches.append(cache)

        AL, cache = self._linear_activation_forward(
            A, parameters["W" + str(L)], parameters["b" + str(L)], SIGMOID_ACTIVATION
        )
        caches.append(cache)

        return AL, caches

    # TODO:
    def initialize_parameters_deep_he(self):
        pass
