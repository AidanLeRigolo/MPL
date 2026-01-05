import numpy as np
import matplotlib.pyplot as plt

def relu(Z):
    """
    ReLU(Z) = max(0, Z)
    """
    return np.maximum(0, Z)

def relu_derivative(Z):
    """
    dReLU/dZ = 1 si Z > 0, sinon 0
    """
    return (Z > 0).astype(float)

def softmax(Z):
    """
    Softmax stable numériquement
    softmax(z_i) = exp(z_i) / sum_j exp(z_j)
    """
    Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z_shifted)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

def cross_entropy(y_true, y_pred):
    """
    Loss = - sum(y * log(y_hat)) / N
    y_true : one-hot
    y_pred : softmax output
    """
    epsilon = 1e-9  # stabilité numérique
    return -np.mean(np.sum(y_true * np.log(y_pred + epsilon), axis=1))

class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.lr = learning_rate

        # Initialisation aléatoire simple des poids
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b2 = np.zeros((1, hidden_size))

        self.W3 = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b3 = np.zeros((1, hidden_size))

        self.W4 = np.random.randn(hidden_size, output_size) * 0.01
        self.b4 = np.zeros((1, output_size))

    def forward(self, X):
        """
        Calcul des activations couche par couche
        """

        # Couche 1
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = relu(self.Z1)

        # Couche 2
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = relu(self.Z2)

        # Couche 3
        self.Z3 = self.A2 @ self.W3 + self.b3
        self.A3 = relu(self.Z3)

        # Couche de sortie
        self.Z4 = self.A3 @ self.W4 + self.b4
        self.A4 = softmax(self.Z4)

        return self.A4

    def backward(self, X, y):
        """
        Calcul des gradients via la règle de la chaîne
        """

        N = X.shape[0]

        # ---- SORTIE ----
        # dL/dZ4 = A4 - y (propriété softmax + cross-entropy)
        dZ4 = self.A4 - y

        dW4 = self.A3.T @ dZ4 / N
        db4 = np.sum(dZ4, axis=0, keepdims=True) / N

        # ---- COUCHE 3 ----
        dA3 = dZ4 @ self.W4.T
        dZ3 = dA3 * relu_derivative(self.Z3)

        dW3 = self.A2.T @ dZ3 / N
        db3 = np.sum(dZ3, axis=0, keepdims=True) / N

        # ---- COUCHE 2 ----
        dA2 = dZ3 @ self.W3.T
        dZ2 = dA2 * relu_derivative(self.Z2)

        dW2 = self.A1.T @ dZ2 / N
        db2 = np.sum(dZ2, axis=0, keepdims=True) / N

        # ---- COUCHE 1 ----
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * relu_derivative(self.Z1)

        dW1 = X.T @ dZ1 / N
        db1 = np.sum(dZ1, axis=0, keepdims=True) / N

        # ---- MISE À JOUR ----
        self.W4 -= self.lr * dW4
        self.b4 -= self.lr * db4

        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(self, X, y, epochs):
        losses = []

        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = cross_entropy(y, y_pred)
            losses.append(loss)

            self.backward(X, y)
            
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.6f}")

        return losses

def plot_loss(losses):
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Évolution de la loss")
    plt.grid()
    plt.show()

# Données factices
np.random.seed(0)

X = np.random.randn(200, 64)
y_indices = np.random.randint(0, 10, size=200)

# One-hot
y = np.zeros((200, 10))
y[np.arange(200), y_indices] = 1

# Modèle
mlp = MLP(
    input_size=64,
    hidden_size=16,
    output_size=10,
    learning_rate=0.005
)

losses = mlp.train(X, y, epochs=10000)
plot_loss(losses)

