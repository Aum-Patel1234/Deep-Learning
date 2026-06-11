import numpy as np


def zero_pad(X: np.ndarray, pad: int):
    pass


def conv_single_step(a_slice_prev: np.ndarray, W: np.ndarray, b: np.ndarray):
    pass


def conv_forward(A_prev: np.ndarray, W: np.ndarray, b: np.ndarray, hparameters: dict):
    pass


def pool_forward(A_prev: np.ndarray, hparameters: dict, mode="max"):
    pass


# TODO:
def conv_backward(dZ, cache):
    pass


def pool_backward(dA, cache, mode="max"):
    pass


if __name__ == "__main__":
    print("hi")
