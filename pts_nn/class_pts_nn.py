import numpy as np
import matplotlib.pyplot as plt
from pts_nn.data_ia import *


# ---------------- Réseau Neuronal ---------------- #


class Layer_Dense:
    """
    creer les couches de neuronnes
   
    Paramètres:
    - n_inputs : int, nombre d'entrées.
    - n_neurons : int, nombre de neurones.
    """
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(1 / n_inputs)
        self.biases = np.zeros((1, n_neurons))


    def forward(self, inputs):
        """calcule la sortie de la couche"""
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases


    def backward(self, dvalues):
        """calcule les gradients des poids et des biais"""
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)




class Activation_ReLu:
    """Rectified Linear Unit"""
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
   
    def backward(self, dvalues):
        """Rétropropagation ReLU"""
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0




class Activation_Softmax:
    """classification pour plusieur classes"""
    def forward(self, inputs):
        exp_val = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        proba = exp_val / np.sum(exp_val, axis=1, keepdims=True)
        self.output = proba




class Loss:
    """Classe pour la perte"""
    def calcule(self, output, y):
        sample_losses = self.forward(output, y)
        return np.mean(sample_losses)




class Loss_Categorical_Cross_Entropy(Loss):
    """Perte d'entropie croisée pour la classification"""
    def forward(self, y_pred, y_true):
        sample = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)


        if len(y_true.shape) == 1:
            correct_confidence = y_pred_clipped[range(sample), y_true]
        elif len(y_true.shape) == 2:
            correct_confidence = np.sum(y_pred_clipped * y_true, axis=1)


        return -np.log(correct_confidence)


    def backward(self, dvalues, y_true):
        samples = len(dvalues)
       
        if len(y_true.shape) == 1:
            y_true = np.eye(dvalues.shape[1])[y_true]


        self.dinputs = (dvalues - y_true) / samples




class Activation_Softmax_Loss_CategoricalCrossentropy:
    """Combinaison de Softmax et de Loss_Categorical_Cross_Entropy"""
    def forward(self, inputs, y_true):
        self.activation = Activation_Softmax()
        self.activation.forward(inputs)
        self.output = self.activation.output
        return Loss_Categorical_Cross_Entropy().forward(self.output, y_true)


    def backward(self, dvalues, y_true):
        samples = len(dvalues)
       
        if len(y_true.shape) == 1:
            y_true = np.eye(dvalues.shape[1])[y_true]


        self.dinputs = (dvalues - y_true) / samples




class Optimizer_Adam:
    """Optimiseur Adam pour l'ajustement des weights et biases"""
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.iterations = 0


    def update_params(self, layer):
        """Met à jour les poids et biais d'une couche selon Adam."""
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




class NoImprovementScheduler: # sert a rien pour l'instant
    def __init__(self, optimizer, patience=30, factor=0.5, min_lr=1e-6, verbose=True):
        self.optimizer = optimizer
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.verbose = verbose


        self.best_loss = float('inf')
        self.wait = 0


    def step(self, current_loss):
        if current_loss + 1e-4 < self.best_loss:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1


        if self.wait >= self.patience:
            old_lr = self.optimizer.learning_rate
            new_lr = max(old_lr * self.factor, self.min_lr)
            if new_lr < old_lr:
                self.optimizer.learning_rate = new_lr
                self.wait = 0
                if self.verbose:
                    print(f"[NoImprovementScheduler] Réduction du LR : {old_lr:.6f} → {new_lr:.6f}")






class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layers = []
        self.activations = []


        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer_Dense(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                self.activations.append(Activation_ReLu())
            else:
                self.activations.append(Activation_Softmax_Loss_CategoricalCrossentropy())


    def forward(self, X, y):
        A = X
        for i in range(len(self.layers) - 1):
            self.layers[i].forward(A)
            self.activations[i].forward(self.layers[i].output)
            A = self.activations[i].output
       
        # Dernière couche (Softmax + Entropie croisée)
        self.layers[-1].forward(A)
        loss = np.mean(self.activations[-1].forward(self.layers[-1].output, y))


        return loss




    def backward(self, y):
        self.activations[-1].backward(self.activations[-1].output, y)
        dvalues = self.activations[-1].dinputs


        for i in reversed(range(len(self.layers))):
            self.layers[i].backward(dvalues)
            if i > 0:
                self.activations[i - 1].backward(self.layers[i].dinputs)
                dvalues = self.activations[i - 1].dinputs


    def update(self, optimizer):
        for layer in self.layers:
            optimizer.update_params(layer)




# ---------------- Hyperparametre ---------------- #


n_forme = square # vortex -- square -- hearth -- triangles -- circles -- spiral -- sinus
nbr_p_class = 700
nbr_sortie = 3
epochs = 5000
learning_rate = 0.005
patience = 500
threshold = 0.001
factor = 0.7
min_lr = 0.00001
t_point = 1
layer_sizes = [2, 40, 35, 30, nbr_sortie]


# ---------------- Data ---------------- #


X, y = n_forme(nbr_p_class, nbr_sortie) # Fonction de génération de données


# Affichage des graphes pour visualisation
print("attention ça arrive !")
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="managua", vmin=0, vmax=np.max(y), s=t_point)
plt.show()




# ---------------- Création du réseau ---------------- #


nn = NeuralNetwork(layer_sizes)




# ---------------- Salle de muscu ---------------- #


optimizer = Optimizer_Adam(learning_rate) # plus learning_rate est bas plus l'ia va etre precise mais apprendra moins vite
#scheduler = NoImprovementScheduler(optimizer, patience, threshold, factor, min_lr) # ajuste le learning_rate automatiquement


best_loss = float('inf')
best_epoch = 0
best_accuracy = 0
losses = []
lr_history = []


for epoch in range(epochs):
    # Forward pass
    loss = nn.forward(X, y)
    losses.append(loss)
    lr_history.append(optimizer.learning_rate)


    # Calcul de la précision
    predictions = np.argmax(nn.activations[-1].output, axis=1)
    accuracy = np.mean(predictions == y) * 100


    # pour que nous, pauvre humain puisse comprendre ce qu'il ce passe et vérification perte et précision meilleures et l'afficher
    if round(loss, 4) < round(best_loss, 4):
        best_loss = loss
        best_epoch = epoch
        best_accuracy = accuracy
        best_weights = [np.copy(layer.weights) for layer in nn.layers]
        best_biases = [np.copy(layer.biases) for layer in nn.layers]
        print(f'Nouvelle meilleure perte : {loss:.4f} | Précision : {accuracy:.2f}% | Itération {epoch}')
       
        # ralentie beaucoup le programme a cause des graphes, on peut l'enlever il aide juste a visualiser
        plt.clf()
        plt.scatter(X[:, 0], X[:, 1], c=predictions, cmap="managua", vmin=0, vmax=np.max(y), s=t_point)
        plt.title(f"Epoch {epoch} - Accuracy: {accuracy:.2f}%")
        plt.pause(0.01)
       
    if accuracy == 100.00 :
        print("Meilleur acr possible sur ce data_set, pour plus general il faut faire descendre la perte")
        break


    # Backpropagation
    nn.backward(y)


    #scheduler.step(loss)


    # Mise à jour des poids
    nn.update(optimizer)




# ---------------- Affichage ---------------- #


for i in range(len(nn.layers)):
    nn.layers[i].weights = best_weights[i]
    nn.layers[i].biases = best_biases[i]


nn.forward(X, y)
predictions = np.argmax(nn.activations[-1].output, axis=1)


plt.figure(figsize=(10, 5))


# graph des vraies classes
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="managua", vmin=0, vmax=np.max(y), s=t_point)
plt.title("Données de référence")


# graph de la prédictions final
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=predictions, cmap="managua", vmin=0, vmax=np.max(y), s=t_point)
plt.title(f"Prédictions final (Epoch {best_epoch})")


plt.show()


plt.figure(figsize=(10, 5))


# courbe learning_rate avec scheduler
plt.plot(losses, label="Loss")
plt.plot(lr_history, label="Learning rate", color='orange')
plt.legend()
plt.title("Courbe de perte et LR")


plt.show()

