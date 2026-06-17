import numpy as np
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

parameters = nn.fit(
    X_train,
    y_train,
    lr=0.05,
    iter=5000,
    print_cost=True,
)

# Predict
AL_test, _ = nn._L_model_forward(X_test, parameters)

preds = np.argmax(AL_test, axis=0)
true_labels = np.argmax(y_test, axis=0)

accuracy = np.mean(preds == true_labels)

print("\nTest Accuracy:", accuracy)

# Show a few predictions
print("\nPredictions:", preds[:10])
print("Ground Truth:", true_labels[:10])
