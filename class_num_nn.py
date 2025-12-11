import numpy as np
import pickle


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




class Activation_Softmax:
    def forward(self, inputs):
        exp_val = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
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




class NoImprovementScheduler:
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


    def predict(self, X):
        A = X
        for i in range(len(self.layers) - 1):
            self.layers[i].forward(A)
            self.activations[i].forward(self.layers[i].output)
            A = self.activations[i].output
       
        self.layers[-1].forward(A)
        self.activations[-1].activation.forward(self.layers[-1].output)
        return self.activations[-1].activation.output
   
    def save(self, filename):
        data = [layer.weights for layer in self.layers] + [layer.biases for layer in self.layers]
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
       
    def load(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        num_layers = len(self.layers)
        for i in range(num_layers):
            self.layers[i].weights = data[i]
            self.layers[i].biases = data[i + num_layers]




# ---------------- Lancement de l'entrainement ----------------




if __name__ == "__main__":




    # ---------------- Data ---------------- #




    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.metrics import classification_report, confusion_matrix # Affichage
    from scipy.ndimage import rotate, shift, zoom # augment_image()




    digits = load_digits()
    X = digits.data         # shape: (73677, 64)
    y = digits.target       # shape: (73677,)


    X = X / 16 # nomalise 8x8 = 16, divise pas 16 pour que chaque X entre 0 et 1
    encoder = OneHotEncoder(sparse_output=False)
    y_encoded = encoder.fit_transform(y.reshape(-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2 , random_state=42)


    def augment_image(image):
        image = image.reshape(8, 8)


        # --- Rotation aléatoire ---
        angle = np.random.uniform(-10, 10)
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
        noise = np.random.normal(0, 0.5, size=(8, 8))
        noisy = shifted + noise
        noisy = np.clip(noisy, 0, 16)
        noisy /= 16


        return noisy.flatten()


    X_augmented = []
    y_augmented = []


    print("------ Chargement des images ------")


    for i in range(len(X_train)):
        X_augmented.append(X_train[i])
        y_augmented.append(y_train[i])


        # n augmentations par image originale
        n = 5
        for _ in range(n):
            new_img = augment_image(X_train[i])
            X_augmented.append(new_img)
            y_augmented.append(y_train[i])


    X_train_aug = np.array(X_augmented)
    y_train_aug = np.array(y_augmented)




    # ---------------- Hyperparametre ---------------- #




    epochs = 100
    nbr_sortie = 10
    layer_sizes = [64, 20, 10, nbr_sortie]




    # ---------------- Entraînement ---------------- #




    nn = NeuralNetwork(layer_sizes)
    optimizer = Optimizer_Adam(learning_rate=0.005)


    best_loss = float('inf')


    for epoch in range(epochs):
        loss = nn.forward(X_train_aug, y_train_aug)
        nn.backward(y_train_aug)
        nn.update(optimizer)


        pred_val = nn.predict(X_test)
        val_loss = Loss_Categorical_Cross_Entropy().forward(pred_val, y_test).mean()


    if val_loss < best_loss:
        best_loss = val_loss
        nn.save("Best_mdl_class_num.pkl")
        print(f"[Epoch {epoch}] | Nouveau meilleur modèle (val_loss = {val_loss:.4f})")


    if epoch % 10 == 0:
        print(f"[Epoch {epoch}] | Train loss : {loss:.6f} | Val loss : {val_loss:.6f}")






    # ---------------- Affichage ---------------- #




    y_pred = nn.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    if len(y_test.shape) == 2:
        y_test = np.argmax(y_test, axis=1)


    accuracy = np.mean(y_pred_classes == y_test)
    print(f"Précision sur img test : {accuracy * 100:.2f}%")
    print(confusion_matrix(y_test, y_pred_classes))
    print(classification_report(y_test, y_pred_classes))


    # tous ça ça degage bientot


    import tkinter as tk
    from PIL import Image, ImageDraw


    def create_drawing_interface(nn):
        canvas_size = 512  # Plus grand que 8x8 pour avoir de la marge
        pixel_input_size = 8  # Format attendu par le réseau


        root = tk.Tk()


        canvas = tk.Canvas(root, width=canvas_size, height=canvas_size, bg="black")
        canvas.pack()


        image = Image.new("L", (canvas_size, canvas_size), color=0)
        draw = ImageDraw.Draw(image)


        def paint(event):
            x, y = event.x, event.y
            r = 30
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    canvas.create_oval(x+dx - r, y+dy - r, x+dx + r, y+dy + r, fill="white", outline="white")
                    draw.ellipse([x+dx - r, y+dy - r, x+dx + r, y+dy + r], fill=255)




        canvas.bind("<B1-Motion>", paint)


        def predict_digit():
            # Redimensionne à 8x8 comme le dataset Digits
            img_resized = image.resize((pixel_input_size, pixel_input_size), Image.Resampling.LANCZOS)
            img_array = np.array(img_resized).astype(np.float32)


            # Normalisation
            img_array = (img_array / 255.0) * 16
            img_flat = img_array.flatten().reshape(1, -1)


            prediction = nn.predict(img_flat)
            digit = np.argmax(prediction)
            confidences = prediction[0]


            result_label.config(text=f"Prédiction : {digit}")


            # Tableau de toutes les probabilités
            print("------ Poucentage de confidence de la prediction -------")
            print("\n".join([f"{i} : {confidences[i]*100:.2f}%" for i in range(10)]))
            print()




        def clear_canvas():
            canvas.delete("all")
            draw.rectangle([0, 0, canvas_size, canvas_size], fill=0)
            result_label.config(text="")


        button_frame = tk.Frame(root)
        button_frame.pack()


        predict_btn = tk.Button(button_frame, text="Prédire", command=predict_digit)
        predict_btn.grid(row=0, column=0)


        clear_btn = tk.Button(button_frame, text="Effacer", command=clear_canvas)
        clear_btn.grid(row=0, column=1)


        result_label = tk.Label(root, text="", font=("Arial", 18))
        result_label.pack()


        confidence_label = tk.Label(root, text="", font=("Courier", 12), justify="left")
        confidence_label.pack()




        root.mainloop()


    create_drawing_interface(nn)



