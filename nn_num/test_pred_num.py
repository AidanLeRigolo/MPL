import numpy as np
from nn_num.class_num_nn import NeuralNetwork
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
        print("------ Poucentage de confidence pour chaque chiffre -------")
        print("\n".join([f"{i} : {confidences[i]*100:.2f}%" for i in range(10)]))
        print("Coucou de Flora")
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

if __name__ == "__main__":
    nn = NeuralNetwork(layer_sizes=[64, 128, 64, 64, 32, 32, 10]) # dois avoir le mm nbr de layers et neurones
    nn.load("python/python_ia/MLP_chiffre/Best_mdl_class_num.pkl")


    create_drawing_interface(nn)