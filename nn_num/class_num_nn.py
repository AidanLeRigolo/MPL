import numpy as np
import pickle


# ---------------- Réseau Neuronal ---------------- #


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, l2_lambda=0.0):
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(1 / n_inputs)
        self.biases = np.zeros((1, n_neurons))
        self.l2_lambda = l2_lambda

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues) + 2 * self.l2_lambda * self.weights
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_ReLu:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class Activation_Softmax:
    def forward(self, inputs):
        scaled_inputs = inputs / 1.5
        exp_val = np.exp(scaled_inputs - np.max(scaled_inputs, axis=1, keepdims=True))
        proba = exp_val / np.sum(exp_val, axis=1, keepdims=True)
        self.output = proba


class Loss:
    def calcule(self, output, y):
        sample_losses = self.forward(output, y)
        return np.mean(sample_losses)


class Loss_Categorical_Cross_Entropy(Loss):
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
    def forward(self, inputs, y_true):
        self.activation = Activation_Softmax()
        self.activation.forward(inputs)
        self.output = self.activation.output
        return Loss_Categorical_Cross_Entropy().forward(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = self.output.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples



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
    def __init__(self, layer_sizes, l2_lambda=0.0):
        self.layers = []
        self.activations = []
        self.Softmax = Activation_Softmax()
        self.l2_lambda = l2_lambda

        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer_Dense(layer_sizes[i], layer_sizes[i + 1], l2_lambda=self.l2_lambda))
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

        self.layers[-1].forward(A)
        loss = np.mean(self.activations[-1].forward(self.layers[-1].output, y))

        l2_loss = 0
        for layer in self.layers:
            l2_loss += np.sum(layer.weights ** 2)
        loss += self.l2_lambda * l2_loss

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

    def predict(self, X):
        self.layers[0].forward(X)
        self.activations[0].forward(self.layers[0].output)

        for i in range(1, len(self.layers) - 1):
            self.layers[i].forward(self.activations[i - 1].output)
            self.activations[i].forward(self.layers[i].output)

        self.layers[-1].forward(self.activations[-2].output)
        self.Softmax.forward(self.layers[-1].output)

        return self.Softmax.output

        
    def save(self, filename):
        data = {
            "layers": [
                {"weights": layer.weights, "biases": layer.biases}
                for layer in self.layers
            ]
        }
        with open(filename, "wb") as f:
            pickle.dump(data, f)


    def load(self, filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)

        for layer_data, layer in zip(data["layers"], self.layers):
            layer.weights = layer_data["weights"]
            layer.biases = layer_data["biases"]



# ---------------- Lancement de l'entrainement ----------------


if __name__ == "__main__":


    # ---------------- Data ---------------- #

    
    print("------ Importation des images ------")

    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.metrics import classification_report, confusion_matrix # Affichage
    from scipy.ndimage import rotate, shift, zoom # augment_image()


    digits = load_digits()
    X = digits.data         # shape: (n*1797, 64)
    y = digits.target       # shape: (n*1797,)

    X = X / 16 # nomalise 8x8 = 16, divise pas 16 pour que chaque X entre 0 et 1
    encoder = OneHotEncoder(sparse_output=False)
    y_encoded = encoder.fit_transform(y.reshape(-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2 , random_state=42)

    def augment_image(image):
        image = image.reshape(8, 8)

        # --- Rotation aléatoire ---
        angle = np.random.uniform(-5, 5)
        rotated = rotate(image, angle, reshape=False, mode='nearest')

        # --- Zoom aléatoire ---
        zoom_factor = np.random.uniform(0.9, 1.1)  # entre -10% et +10%
        zoomed = zoom(rotated, zoom=zoom_factor, order=1, mode='nearest')

        if zoom_factor < 1:
            padded = np.zeros((8, 8))
            offset = (8 - zoomed.shape[0]) // 2
            padded[offset:offset+zoomed.shape[0], offset:offset+zoomed.shape[1]] = zoomed
            zoomed = padded
        elif zoom_factor > 1:
            crop_start = (zoomed.shape[0] - 8) // 2
            zoomed = zoomed[crop_start:crop_start+8, crop_start:crop_start+8]

        # --- Décalage aléatoire ---
        dx = np.random.uniform(-1, 1)
        dy = np.random.uniform(-1, 1)
        shifted = shift(zoomed, shift=(dy, dx), mode='nearest')
        
        # --- Bruit gaussien léger ---
        noise = np.random.normal(0, 0.2, size=(8, 8))
        noisy = shifted + noise
        noisy = np.clip(noisy, 0, 16)
        noisy /= 16

        return noisy.flatten()

    X_augmented = []
    y_augmented = []

    print("------ Chargement du dataset augmenter ------")

    for i in range(len(X_train)):
        X_augmented.append(X_train[i])
        y_augmented.append(y_train[i])

        # n augmentations par image originale
        n = 10
        for _ in range(n):
            new_img = augment_image(X_train[i])
            X_augmented.append(new_img)
            y_augmented.append(y_train[i])

    X_train_aug = np.array(X_augmented)
    y_train_aug = np.array(y_augmented)


    # ---------------- Hyperparametre ---------------- #

    taille_input = 8*8
    epochs = 1000
    nbr_sortie = 10
    layer_sizes = [taille_input, 128, 64, 64, 32, 32, nbr_sortie] # le premier doit etre 64, 8*8=64 psk image 8*8


    # ---------------- Entraînement ---------------- #

    nn = NeuralNetwork(layer_sizes, l2_lambda=0.001)
    optimizer = Optimizer_Adam(learning_rate=0.005)

    best_loss = float('inf')
    loss_history = []
    accuracy_history = []

    for epoch in range(epochs):
        loss = nn.forward(X_train_aug, y_train_aug)
        nn.backward(y_train_aug)
        nn.update(optimizer)

        pred_val = nn.predict(X_test)
        val_loss = Loss_Categorical_Cross_Entropy().forward(pred_val, y_test).mean()

        if val_loss < best_loss:
            acc = np.mean(np.argmax(pred_val, axis=1) == np.argmax(y_test, axis=1))
            loss_history.append(loss)
            accuracy_history.append(acc)
            best_loss = val_loss
            best_weights = [np.copy(layer.weights) for layer in nn.layers]
            best_biases = [np.copy(layer.biases) for layer in nn.layers]
            print(f"[Epoch {epoch}] | Train loss : {loss:.6f} | Val loss : {val_loss:.6f} | New best")

    for layer, best_w, best_b in zip(nn.layers, best_weights, best_biases):
        layer.weights = best_w
        layer.biases = best_b
    
    nn.save("path_to_save_model/num_class_nn_model.pkl")
    print("------ Nouveau model sauvegarder ------")


    # ---------------- Affichage ---------------- #


    y_pred = nn.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    if len(y_test.shape) == 2:
        y_test = np.argmax(y_test, axis=1)

    accuracy = np.mean(y_pred_classes == y_test)
    print(f"Précision sur img test : {accuracy * 100:.2f}%")
    print(confusion_matrix(y_test, y_pred_classes))
    print(classification_report(y_test, y_pred_classes))

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(accuracy_history, label='Accuracy', color='pink')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()
