import numpy as np
import matplotlib.pyplot as plt
from data_ia import *


# ---------------- Réseau Neuronal ---------------- #
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(1 / n_inputs)
        self.biases = np.zeros((1, n_neurons))


    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases


    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)




class Activation_ReLu:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)


    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0




class Loss:
    def calcule(self, output, y):
        sample_losses = self.forward(output, y)
        return np.mean(sample_losses)




class Loss_MeanSquaredError(Loss):
    def forward(self, y_pred, y_true):
        return np.mean((y_true - y_pred) ** 2)


    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        self.dinputs = -2 * (y_true - dvalues) / samples




class Optimizer_Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.iterations = 0


    def update_params(self, layer):
        if not hasattr(layer, 'm_w'):
            layer.m_w = np.zeros_like(layer.weights)
            layer.v_w = np.zeros_like(layer.weights)
            layer.m_b = np.zeros_like(layer.biases)
            layer.v_b = np.zeros_like(layer.biases)


        self.iterations += 1


        layer.m_w = self.beta1 * layer.m_w + (1 - self.beta1) * layer.dweights
        layer.v_w = self.beta2 * layer.v_w + (1 - self.beta2) * (layer.dweights ** 2)
        layer.m_b = self.beta1 * layer.m_b + (1 - self.beta1) * layer.dbiases
        layer.v_b = self.beta2 * layer.v_b + (1 - self.beta2) * (layer.dbiases ** 2)


        m_w_corr = layer.m_w / (1 - self.beta1 ** self.iterations)
        v_w_corr = layer.v_w / (1 - self.beta2 ** self.iterations)
        m_b_corr = layer.m_b / (1 - self.beta1 ** self.iterations)
        v_b_corr = layer.v_b / (1 - self.beta2 ** self.iterations)


        layer.weights -= self.learning_rate * m_w_corr / (np.sqrt(v_w_corr) + self.epsilon)
        layer.biases -= self.learning_rate * m_b_corr / (np.sqrt(v_b_corr) + self.epsilon)




class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layers = []
        self.activations = []


        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer_Dense(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                self.activations.append(Activation_ReLu())


    def forward(self, X, y=None):
        A = X
        for i in range(len(self.layers) - 1):
            self.layers[i].forward(A)
            self.activations[i].forward(self.layers[i].output)
            A = self.activations[i].output


        self.layers[-1].forward(A)
        return self.layers[-1].output


    def backward(self, dvalues):
        self.layers[-1].backward(dvalues)
        dvalues = self.layers[-1].dinputs


        for i in reversed(range(len(self.layers) - 1)):
            self.activations[i].backward(dvalues)
            self.layers[i].backward(self.activations[i].dinputs)
            dvalues = self.layers[i].dinputs


    def update(self, optimizer):
        for layer in self.layers:
            optimizer.update_params(layer)




# ---------------- Fonction d'affichage ---------------- #


def predict_grid(nn, width=200, height=200, xlim=(-2, 1), ylim=(-1.5, 1.5)):
    xs = np.linspace(xlim[0], xlim[1], width)
    ys = np.linspace(ylim[0], ylim[1], height)
    image = np.zeros((height, width))


    for i in range(height):
        for j in range(width):
            x = xs[j]
            y = ys[i]
            nn.forward(np.array([[x, y]]))
            image[i, j] = nn.layers[-1].output[0, 0]


    return image




# ---------------- Hyperparametre ---------------- #


width = 100
height = 100
epochs = 10000
learning_rate = 0.005




# ---------------- Data ---------------- #


X, y = generate_mandelbrot_dataset(100, 100, (-2, 1), (-1.5, 1.5))




# ---------------- Création du réseau ---------------- #


layer_sizes = [2, 128, 96, 96, 64, 1]
nn = NeuralNetwork(layer_sizes)


# ---------------- Salle de muscu ---------------- #


loss_function = Loss_MeanSquaredError()
optimizer = Optimizer_Adam(learning_rate)


best_loss = float('inf')
best_epoch = 0
best_accuracy = 0
losses = []


for epoch in range(epochs):
    # forward
    output = nn.forward(X, y)


    loss = loss_function.forward(output, y)
    losses.append(loss)


    predictions = output
    accuracy = 1 - np.mean(np.abs(predictions - y))


    if epoch % 100 == 0:
        best_loss = loss
        best_epoch = epoch
        best_accuracy = accuracy
        best_weights = [np.copy(layer.weights) for layer in nn.layers]
        best_biases = [np.copy(layer.biases) for layer in nn.layers]
        print(f"Epoch {epoch} | Loss: {loss:.4f} | Acc approx: {accuracy:.4f}")


        output_image = predict_grid(nn)
        '''
        plt.imshow(output_image, extent=(-2, 1, -1.5, 1.5), cmap='managua')
        plt.title(f"Epoch {epoch} | Loss: {loss:.4f}")
        plt.pause(0.01)
        plt.clf()
        '''
        output = nn.forward(X, y)


    loss_function.backward(output, y)
    nn.backward(loss_function.dinputs)
    nn.update(optimizer)




# ---------------- Affichage ---------------- #


output_image = predict_grid(nn)
plt.imshow(output_image, extent=(-2, 1, -1.5, 1.5), cmap='managua')
plt.colorbar()
plt.title("Approximation par réseau neuronal")
plt.show()


# courbe de la perte
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("loss")
plt.title("Courbe de perte pendant l'entraînement")
plt.grid(True)
plt.show()

