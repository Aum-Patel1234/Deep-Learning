import numpy as np
import math
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from n_layer_nn import NeuralNetworkNL

iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

# One-hot encode labels
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y)

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=np.argmax(y, axis=1)
)
print(X_train[:1])
print(y_train[:1], "\n")

# Convert to column-wise format expected by network
X_train = X_train.T
X_test = X_test.T

y_train = y_train.T
y_test = y_test.T

layer_dims = [4, 16, 8, 3]
nn = NeuralNetworkNL(layer_dims)

# parameters = nn.fit(
#     X_train,
#     y_train,
#     lr=0.05,
#     iter=5000,
#     print_cost=True,
# )
#
# # Predict
# AL_test, _ = nn._L_model_forward(X_test, parameters)
#
# preds = np.argmax(AL_test, axis=0)
# true_labels = np.argmax(y_test, axis=0)
#
# accuracy = np.mean(preds == true_labels)
#
# print("\nTest Accuracy:", accuracy)
#
# # Show a few predictions
# print("\nPredictions:", preds[:10])
# print("Ground Truth:", true_labels[:10])

# batches = nn._random_mini_batches(X_train, y_train)
# print(X_train.shape, y_train.shape)
# flag = True
# for batch in batches:
#     print(batch[0].shape, " , ", batch[1].shape)
#     if flag:
#         print("x real - ", X_train[:, :3], "\nx_shuffled_batch - ", batch[0][:, :3])
#         flag = False


def test_random_mini_batches():
    np.random.seed(1)
    mini_batch_size = 64
    nx = 12288
    m = 148
    X = np.array([x for x in range(nx * m)]).reshape((m, nx)).T
    Y = np.random.randn(1, m) < 0.5

    mini_batches = nn._random_mini_batches(X, Y, mini_batch_size)
    n_batches = len(mini_batches)

    assert n_batches == math.ceil(
        m / mini_batch_size
    ), f"Wrong number of mini batches. {n_batches} != {math.ceil(m / mini_batch_size)}"
    for k in range(n_batches - 1):
        assert mini_batches[k][0].shape == (
            nx,
            mini_batch_size,
        ), f"Wrong shape in {k} mini batch for X"
        assert mini_batches[k][1].shape == (
            1,
            mini_batch_size,
        ), f"Wrong shape in {k} mini batch for Y"
        assert np.sum(np.sum(mini_batches[k][0] - mini_batches[k][0][0], axis=0)) == (
            (nx * (nx - 1) / 2) * mini_batch_size
        ), "Wrong values. It happens if the order of X rows(features) changes"
    if m % mini_batch_size > 0:
        assert mini_batches[n_batches - 1][0].shape == (
            nx,
            m % mini_batch_size,
        ), f"Wrong shape in the last minibatch. {mini_batches[n_batches - 1][0].shape} != {(nx, m % mini_batch_size)}"

    assert np.allclose(
        mini_batches[0][0][0][0:3], [294912, 86016, 454656]
    ), "Wrong values. Check the indexes used to form the mini batches"
    assert np.allclose(
        mini_batches[-1][0][-1][0:3], [1425407, 1769471, 897023]
    ), "Wrong values. Check the indexes used to form the mini batches"

    print("\033[92mAll tests passed!")


test_random_mini_batches()
