from functools import cache
import numpy as np
from typing import Any, Dict, Tuple, List
from copy import deepcopy

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

    # TODO: derive them
    def _linear_backward(self, dZ: np.ndarray, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = (1 / m) * dZ @ A_prev.T
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = W.T @ dZ

        assert dW.shape == W.shape
        assert db.shape == b.shape
        assert dA_prev.shape == A_prev.shape

        return dA_prev, dW, db

    def _linear_activation_backward(self, dA: np.ndarray, cache, activation: str):
        linear_cache, activation_cache = cache

        if activation == RELU_ACTIVATION:
            Z = activation_cache
            dZ = np.array(dA, copy=True)
            dZ[Z <= 0] = 0
            dA_prev, dW, db = self._linear_backward(dZ, linear_cache)
        elif activation == SIGMOID_ACTIVATION:
            Z = activation_cache
            s = 1 / (1 + np.exp(-Z))
            dZ = dA * s * (1 - s)
            dA_prev, dW, db = self._linear_backward(dZ, linear_cache)
        else:
            raise ValueError(f"Unknown activation: {activation}")

        return dA_prev, dW, db


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
        # print("layer_dims =", self.layer_dims)
        # print("parameter keys =", parameters.keys())
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

    def _L_model_backward(self, AL: np.ndarray, y: np.ndarray, caches):
        grads = {}
        L = len(self.layer_dims) - 1
        m = AL.shape[1]
        y = y.reshape(AL.shape)
        # IMPORTANT: doing the first derivative by ourself and remaining in chain
        dAL = -(y / AL - (1 - y) / (1 - AL))

        curr_cache = caches[L - 1]
        dA_prev, dW_temp, db_temp = self._linear_activation_backward(
            dAL, curr_cache, SIGMOID_ACTIVATION
        )
        grads["dA" + str(L - 1)] = dA_prev
        grads["dW" + str(L)] = dW_temp
        grads["db" + str(L)] = db_temp

        for l in reversed(range(L - 1)):
            curr_cache = caches[l]

            dA_prev, dW_temp, db_temp = self._linear_activation_backward(
                grads["dA" + str(l + 1)], curr_cache, RELU_ACTIVATION
            )

            grads["dA" + str(l)] = dA_prev
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads

    def _update_parameters_gd(self, grads, lr) -> None:
        params = deepcopy(self.parameters)
        L = len(self.layer_dims) - 1

        for l in range(L):
            params["W" + str(l + 1)] = params["W" + str(l + 1)] - (
                lr * grads["dW" + str(l + 1)]
            )
            params["b" + str(l + 1)] = params["b" + str(l + 1)] - (
                lr * grads["db" + str(l + 1)]
            )

        self.parameters = params

    def _random_mini_batches(self, X: np.ndarray, y: np.ndarray, batch_size=64, seed=0):
        np.random.seed(seed)
        m = X.shape[1]
        assert m == y.shape[1]
        num_batches = m // batch_size
        permutation = list(np.random.permutation(m))

        # 1. Shuffle the data
        X_shuffled = X[:, permutation]
        y_shuffled = y[:, permutation]
        # print(X_shuffled.shape, y_shuffled.shape)

        # 2. make _random_mini_batches
        batches: list[tuple[np.ndarray, np.ndarray]] = []
        for i in range(num_batches):
            start_idx, end_idx = i * batch_size, (i + 1) * batch_size
            mini_batch_X = X_shuffled[:, start_idx:end_idx]
            mini_batch_y = y_shuffled[:, start_idx:end_idx]

            batches.append((mini_batch_X, mini_batch_y))

        if m % batch_size != 0:
            mini_batch_X = X_shuffled[:, batch_size * num_batches : m]
            mini_batch_y = y_shuffled[:, batch_size * num_batches : m]

            batches.append((mini_batch_X, mini_batch_y))

        return batches

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr=0.075,
        iter=3000,
        print_cost=False,
    ):
        for i in range(iter):
            AL, caches = self._L_model_forward(X, self.parameters)
            cost = self._compute_cost(AL, y)
            grads = self._L_model_backward(AL, y, caches)
            self._update_parameters_gd(grads, lr)
            if print_cost and i % 100 == 0 or i == iter - 1:
                print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            self._random_mini_batches()
        return self.parameters

    # TODO:
    def initialize_parameters_deep_he(self):
        pass
