import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    assert z.ndim == 2
    _, n = z.shape

    # to avoid inf values
    z = z - np.max(z, axis=0, keepdims=True)
    # print(np.max(z, axis=0, keepdims=True))
    z = np.exp(z)
    for i in range(n):
        col = z[:, i]
        z[:, i] /= np.sum(col)

    return z


def test_softmax():
    # Test 1: Shape
    x = np.array([[2, 1], [1, 3]])
    out = softmax(x)

    assert out.shape == x.shape, "Shape mismatch"

    # Test 2: Columns sum to 1
    assert np.allclose(np.sum(out, axis=0), np.ones(x.shape[1])), (
        "Columns do not sum to 1"
    )

    # Test 3: Probabilities are between 0 and 1
    assert np.all(out >= 0) and np.all(out <= 1), "Invalid probabilities"

    # Test 4: Expected output
    expected = np.array([[0.73105858, 0.11920292], [0.26894142, 0.88079708]])

    assert np.allclose(out, expected, atol=1e-6), "Incorrect softmax output"

    # Test 5: Numerical stability
    x = np.array([[1000, 2000], [1001, 1999]])
    out = softmax(x)

    assert np.all(np.isfinite(out)), "Contains inf or NaN"
    assert np.allclose(np.sum(out, axis=0), np.ones(x.shape[1])), (
        "Columns do not sum to 1 for large inputs"
    )


def initialize_rnn_weights(n_x: int, n_a: int, n_y: int):
    """
    n_x : Number of input features.
    n_a : Number of hidden units.
    n_y : Number of output units/classes.
    """
    params = {}
    params["Wax"] = np.random.randn(n_a, n_x) * 0.01
    params["Waa"] = np.random.randn(n_a, n_a) * 0.01
    params["ba"] = np.zeros((n_a, 1))
    params["Wya"] = np.random.randn(n_y, n_a) * 0.01
    params["by"] = np.zeros((n_y, 1))

    return params


def rnn_cell_forward(xt: np.ndarray, a_prev: np.ndarray, params: dict[str, np.ndarray]):
    """
    xt     : input x at time=t          : shape = (n_x, m)
    a_prev : hidden state at t=t-1      : shape = (n_a, m)
    params : dictionary Containing
                Wax - weight matrix to multiply to input            : shape = (n_a, n_x)
                Waa - weight matrix to multiply hidden state        : shape = (n_a, n_a)
                ba  - bias                                          : shape = (n_a, 1)
                Wya - weight matrix to multiply hidden to output    : shape = (n_y, n_a)
                by  - bias                                          : shape = (n_y, 1)
    """
    Wax = params["Wax"]
    Waa = params["Waa"]
    Wya = params["Wya"]
    ba = params["ba"]
    by = params["by"]

    assert Wax.shape == (n_a, n_x)
    assert Waa.shape == (n_a, n_a)
    assert Wya.shape == (n_y, n_a)

    a_next = np.tanh(Waa @ a_prev + Wax @ xt + ba)
    # (n_a,n_a)*(n_a,m) + (n_a,n_x)*(n_x,m) + (n_a,1) = (n_a,m)

    y_pred = softmax(Wya @ a_next + by)
    # (n_y, n_a)*(n_a,n_a) + (n_y,1) = (n_y, a)

    cache = (a_next, a_prev, xt, params)
    return a_next, y_pred, cache


def rnn_forward(x: np.ndarray, a0: np.ndarray, params: dict[str, np.ndarray]):
    """
    here x in full input vec: shape = (n_x, m, T_x)
    a0: initial hidden state: shape = (n_a, m)
    param - Same as above,

    Returns: a - hidden states for each timestep    : shape = (n_a, m, T_x)
             y_pred - predicitons at each timeste   : shape = (n_y, m, T_x)
             caches - tuple of values for backward pass
    """
    assert x.ndim == 3 and a0.ndim == 2

    caches = []
    n_x, m, T_x = x.shape
    n_y, n_a = params["Wya"].shape

    # initialize a and y_pred with zeros
    a = np.zeros((n_a, m, T_x))
    y_pred = np.zeros((n_y, m, T_x))
    a_next = a0

    for t in range(T_x):
        a_next, yt_pred, cache = rnn_cell_forward(x[:, :, t], a_next, params)
        a[:, :, t] = a_next
        y_pred[:, :, t] = yt_pred
        caches.append(cache)

    caches = (caches, x)
    return a, y_pred, caches


def rnn_cell_backward(da_next, cache):
    """
    Arguments:
    da_next -- Gradient of loss with respect to next hidden state
    cache -- python dictionary containing useful values (output of rnn_cell_forward())
    """
    (a_next, a_prev, xt, parameters) = cache
    Wax = parameters["Wax"]  # (n_a, n_x)
    Waa = parameters["Waa"]  # (n_a, n_a)
    # ba = parameters["ba"] # (n_a, 1)

    dtanh = da_next * (1 - a_next**2)
    # (n_a,n_a)*(n_a,m) + (n_a,n_x)*(n_x,m) + (n_a,1) = (n_a,m)

    dxt = Wax.T @ dtanh  # (n_x, n_a)*(n_a,m) => (n_x,m)
    dWax = dtanh @ xt.T  # (n_a, m)*(m, n_x) => (n_a, n_x)

    da_prev = Waa.T @ dtanh  # (n_a, n_a)*(n_a, m) = (n_a,m)
    dWaa = dtanh @ a_prev.T  # (n_a,m)*(m,n_a) = (n_a, n_a)

    dba = np.sum(dtanh, axis=1, keepdims=True)

    return {"dxt": dxt, "da_prev": da_prev, "dWax": dWax, "dWaa": dWaa, "dba": dba}


# IMPORTANT:
# `da` is NOT computed inside this function.
# It is the gradient of the loss with respect to the hidden states (a)
# coming from the layer after the RNN (e.g., a Linear/Dense layer,
# Softmax, or another RNN layer).
def rnn_backward(da: np.ndarray, caches):
    """
    Implement the backward pass for a RNN over an entire sequence of input data.

    Arguments:
    da -- Upstream gradients of all hidden states, of shape (n_a, m, T_x)
    caches -- tuple containing information from the forward pass (rnn_forward)

    Returns:
    gradients -- python dictionary containing:
                        dx -- Gradient w.r.t. the input data, numpy-array of shape (n_x, m, T_x)
                        da0 -- Gradient w.r.t the initial hidden state, numpy-array of shape (n_a, m)
                        dWax -- Gradient w.r.t the input's weight matrix, numpy-array of shape (n_a, n_x)
                        dWaa -- Gradient w.r.t the hidden state's weight matrix, numpy-arrayof shape (n_a, n_a)
                        dba -- Gradient w.r.t the bias, of shape (n_a, 1)
    """
    (caches, x) = caches
    (a1, a0, x1, parameters) = caches[0]
    n_a, m, T_x = da.shape
    n_x, m = x1.shape

    dx = np.zeros((n_x, m, T_x))
    da0 = np.zeros((n_a, m))
    dWax = np.zeros((n_a, n_x))
    dWaa = np.zeros((n_a, n_a))
    dba = np.zeros((n_a, 1))
    da_prevt = np.zeros((n_a, m))

    for t in reversed(range(T_x)):
        # gradients = rnn_cell_backward(da_prevt, caches[t])
        # we add da[:,:,t] as chain rule from forward layers
        gradients = rnn_cell_backward(da_prevt + da[:, :, t], caches[t])
        dxt, da_prevt, dWaxt, dWaat, dbat = (
            gradients["dxt"],
            gradients["da_prev"],
            gradients["dWax"],
            gradients["dWaa"],
            gradients["dba"],
        )
        dx[:, :, t] = dxt
        dWax += dWaxt
        dWaa += dWaat
        dba += dbat

    da0 = da_prevt

    return {"dx": dx, "da0": da0, "dWax": dWax, "dWaa": dWaa, "dba": dba}


if __name__ == "__main__":
    test_softmax()

    # Vocabulary
    vocab = {
        "I": 0,
        "love": 1,
        "deep": 2,
        "learning": 3,
        "am": 4,
        "RNN": 5,
    }

    sentences = [
        ["I", "love", "deep", "learning"],
        ["I", "am", "learning", "RNN"],
        ["love", "deep", "learning", "RNN"],
    ]

    n_x = len(vocab)  # 6
    n_y = len(vocab)  # 6 (predict one of the 6 words)
    n_a = 3
    m = len(sentences)
    T_x = len(sentences[0])

    x = np.zeros((n_x, m, T_x))

    for example_idx, sentence in enumerate(sentences):
        for t, word in enumerate(sentence):
            x[vocab[word], example_idx, t] = 1
    # print(x)

    # Main training

    params = initialize_rnn_weights(n_x, n_a, n_y)

    learning_rate = 0.01
    epochs = 1000

    # make next word as target
    y_true = np.zeros_like(x)
    y_true[:, :, :-1] = x[:, :, 1:]
    y_true[:, :, -1] = x[:, :, -1]

    for epoch in range(epochs):
        a0 = np.zeros((n_a, m))
        a, y_pred, caches = rnn_forward(x, a0, params)

        # Cross entropy loss
        loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / m
        # Softmax + CrossEntropy derivative
        dy = y_pred - y_true

        da = np.zeros_like(a)
        dWya = np.zeros_like(params["Wya"])
        dby = np.zeros_like(params["by"])

        da = np.zeros_like(a)

        for t in range(T_x):
            dWya += dy[:, :, t] @ a[:, :, t].T
            dby += np.sum(dy[:, :, t], axis=1, keepdims=True)
            # Gradient entering the RNN
            da[:, :, t] = params["Wya"].T @ dy[:, :, t]

        gradients = rnn_backward(da, caches)

        params["Wax"] -= learning_rate * gradients["dWax"]
        params["Waa"] -= learning_rate * gradients["dWaa"]
        params["ba"] -= learning_rate * gradients["dba"]

        params["Wya"] -= learning_rate * dWya
        params["by"] -= learning_rate * dby

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Loss = {loss:.4f}")

    idx_to_word = {v: k for k, v in vocab.items()}

    print("Predictions:\n")
    for example_idx, sentence in enumerate(sentences):
        print(f"\nSentence {example_idx + 1}")

        for t in range(T_x):
            pred_idx = np.argmax(y_pred[:, example_idx, t])

            print(
                f"Input: {sentence[t]:<10} "
                f"Predicted: {idx_to_word[pred_idx]} \t\t\t"
                f"Probabilities: {np.round(y_pred[:, example_idx, t], 3)}"
            )
