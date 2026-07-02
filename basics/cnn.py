import numpy as np
from numpy.testing import assert_array_equal


def zero_pad(X: np.ndarray, pad: int) -> np.ndarray:
    """
    Argument:
    X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions

    Returns:
    X_pad -- padded image of shape (m, n_H + 2 * pad, n_W + 2 * pad, n_C)
    """
    if X.ndim != 4:
        ValueError("X.ndim != 4")

    m, n_h, n_w, n_c = X.shape
    ans = np.zeros((m, n_h + 2 * pad, n_w + 2 * pad, n_c), dtype=X.dtype)
    ans[:, pad : pad + n_h, pad : pad + n_w, :] = X

    return ans


def conv_single_step(a_slice_prev: np.ndarray, W: np.ndarray, b: np.ndarray):
    s = W * a_slice_prev
    Z = np.sum(s)
    Z += b.item()  # confirm to have one scalar val
    return Z


def conv_forward(A_prev: np.ndarray, W: np.ndarray, b: np.ndarray, hparameters: dict):
    # NOTE:
    # W contains the learnable convolution filters (kernels).
    # During training, backpropagation updates these weights to detect useful patterns in the input.
    # A_prev.shape = (m, height, width, input_channels)
    # W.shape      = (f, f, input_channels, num_filters)

    m, n_h, n_w, n_c = A_prev.shape
    f, f, n_c_prev, n_C = W.shape
    assert n_c_prev == n_c
    stride = hparameters["stride"]
    pad = hparameters["pad"]

    n_H = int((n_h - f + 2 * pad) / stride) + 1
    n_W = int((n_w - f + 2 * pad) / stride) + 1
    Z = np.zeros((m, n_H, n_W, n_C))
    A_prev_pad = zero_pad(A_prev, pad)

    for i in range(m):  # training examples
        img = A_prev_pad[i]

        for j in range(n_H):
            nh_start = j * stride
            nh_end = nh_start + f

            for k in range(n_W):
                nw_start = k * stride
                nw_end = nw_start + f

                for c in range(n_C):
                    Z[i, j, k, c] = conv_single_step(
                        img[nh_start:nh_end, nw_start:nw_end, :],
                        W[:, :, :, c],  # convolve with cth kernel
                        b[:, :, :, c],  # add cth bias
                    )

    cache = A_prev, W, b, hparameters
    return Z, cache


def pool_forward(A_prev: np.ndarray, hparameters: dict, mode="max"):
    # hparam -> f which has val of (f*f) window over input and gets single scalar based on mode
    assert A_prev.ndim == 4
    m, n_h_prev, n_w_prev, n_C = A_prev.shape
    f = hparameters["f"]
    stride = hparameters["stride"]

    n_H = int((n_h_prev - f) / stride) + 1
    n_W = int((n_w_prev - f) / stride) + 1

    A = np.zeros((m, n_H, n_W, n_C))

    for i in range(m):
        for h in range(n_H):
            h_start = h * stride
            h_end = h_start + f

            for w in range(n_W):
                w_start = w * stride
                w_end = w_start + f

                for c in range(n_C):
                    a_slice = A_prev[i, h_start:h_end, w_start:w_end, c]

                    if mode == "max":
                        A[i, h, w, c] = np.max(a_slice)
                    else:
                        A[i, h, w, c] = np.mean(a_slice)

    cache = (A_prev, hparameters)
    return A, cache


# TODO:
def conv_backward(dZ, cache):
    pass


def pool_backward(dA, cache, mode="max"):
    pass


def _L_conv_forward(X: np.ndarray, params: dict):
    pass


def _L_conv_backward(AL: np.ndarray, y: np.ndarray, cache):
    pass


def test_conv_forward_values():
    A_prev = np.array([[[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]]])
    expected = np.array([[[[12.0], [16.0]], [[24.0], [28.0]]]])

    W = np.ones((2, 2, 1, 1))
    b = np.zeros((1, 1, 1, 1))

    Z, _ = conv_forward(
        A_prev,
        W,
        b,
        {"stride": 1, "pad": 0},
    )

    assert_array_equal(Z, expected)


def test_pool_forward_max():
    A_prev = np.array(
        [
            [
                [[1], [2], [3], [4]],
                [[5], [6], [7], [8]],
                [[9], [10], [11], [12]],
                [[13], [14], [15], [16]],
            ]
        ]
    )
    A, _ = pool_forward(
        A_prev,
        {"f": 2, "stride": 2},
        mode="max",
    )
    expected = np.array([[[[6], [8]], [[14], [16]]]])
    np.testing.assert_array_equal(A, expected)


if __name__ == "__main__":
    X = np.array([[[[1], [2]], [[3], [4]]]])
    assert_array_equal(zero_pad(X, 0), X)

    # Test 2: pad = 1
    X = np.array([[[[1], [2]], [[3], [4]]]])
    expected = np.array(
        [
            [
                [[0], [0], [0], [0]],
                [[0], [1], [2], [0]],
                [[0], [3], [4], [0]],
                [[0], [0], [0], [0]],
            ]
        ]
    )
    assert expected.shape == (1, 4, 4, 1)
    assert_array_equal(zero_pad(X, 1), expected)
    test_conv_forward_values()
    test_pool_forward_max()
    # Image.fromarray(zero_pad(x, 3)[0].astype(np.uint8)).show()
