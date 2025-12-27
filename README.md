# Neural Network

A Python implementation of deep neural networks with comprehensive examples and experiments focused on digit recognition and pattern classification using NumPy.

## Project Description

This repository contains a lightweight, educational implementation of neural networks from scratch using NumPy. It provides practical examples of building dense layers, activation functions, loss calculations, and backpropagation algorithms. The project includes two main modules and one exemple:

- **nn_num**: Digit recognition neural network trained on numeric datasets
- **pts_nn**: Pattern recognition networks for classifying geometric patterns
- **MLP_base**: simple code to understand with eeez

This project is ideal for learning the fundamentals of neural networks and understanding how everything works under the hood.

## Features

✨ **Core Neural Network Components**
- Dense (fully connected) layers with customizable dimensions
- Activation functions: ReLU and Softmax
- Loss functions: Categorical Cross-Entropy
- L2 regularization support for preventing overfitting
- Full backpropagation implementation

📊 **Data Generation & Utilities**
- Multiple synthetic dataset generators (vortex, square, heart patterns)
- Data normalization and preprocessing utilities
- Support for sklearn datasets (e.g., digit recognition)

🎯 **Interactive Prediction**
- GUI-based digit drawing interface using Tkinter
- Real-time prediction with confidence scores
- Visual feedback for predictions

🔬 **Pattern Classification**
- Support for multi-class classification
- Examples for learning on synthetic geometric patterns

## Project Structure

```
Neural_network/
├── MLP_base.py                  # Simple MLP implementation for learning fundamentals
├── nn_num/
│   ├── class_num_nn.py          # Digit recognition network implementation
│   └── test_pred_num.py         # Interactive digit drawing and prediction
├── pts_nn/
│   ├── class_pts_nn.py          # Pattern classification network
│   └── data_ia.py               # Synthetic data generation (vortex, square, heart, ...)
├── README.md                     # This file
└── requirements.txt              # Python dependencies
```

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Setup

1. **Clone or download the repository**
   ```bash
   cd Neural_network
   ```

2. **Create a virtual environment** (recommended)

   On Windows PowerShell:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   On macOS/Linux:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**

   Option A - Using requirements.txt:
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

   Option B - Install packages directly:
   ```bash
   pip install numpy scikit-learn scipy pillow matplotlib
   ```

## Usage

### Basic MLP Implementation (Learning Fundamentals)

Start with `MLP_base.py` to understand the core concepts of neural networks:

```bash
python MLP_base.py
```

This simple implementation demonstrates:
- How layers work with forward and backward propagation
- Activation functions in action
- Loss calculation and optimization
- Perfect for beginners to understand MLP fundamentals

For now this code have a random data set, it's not learning anything useful.

### Digit Recognition with Interactive GUI

Run the interactive digit drawing application:

```bash
python nn_num/test_pred_num.py
```

This launches a window where you can:
- Draw digits (0-9) with your mouse
- Get instant predictions from a pre-trained network
- View confidence scores for each digit

### Pattern Classification

Use the pts_nn module for classifying geometric patterns:

```python
from class_pts_nn import NeuralNetwork
from data_ia import *

# Generate synthetic data
X, y = vortex(points=100, classes=3) 
# You can see the dataset you want to generate before training directly in data_ia.py

# Create and train network
nn = NeuralNetwork([2, 64, 32, 3]) # Last diggits always = classes
nn.train(X, y, epochs=1000, learning_rate=0.1)

# Make predictions
predictions = nn.predict(X)
```

### Import and Use Components

```python
from nn_num.class_num_nn import Layer_Dense, Activation_ReLu, NeuralNetwork

# Create a dense layer
layer = Layer_Dense(n_inputs=784, n_neurons=128)

# Forward pass
layer.forward(input_data)

# Backward pass (gradient computation)
layer.backward(gradients)
```

## Technologies Used

- **NumPy** (1.20+) - Numerical computing and linear algebra
- **scikit-learn** - Dataset loading and preprocessing utilities
- **SciPy** - Advanced scientific computing functions
- **Pillow (PIL)** - Image processing and manipulation
- **Matplotlib** - Data visualization
- **Tkinter** - GUI for interactive predictions

## Future Improvements

🚀 **Planned Enhancements**

- **Convolutional Neural Networks (CNN)** - Add CNN layers for better image classification
- **Advanced Optimizers** - Implement Adam, RMSprop, and other modern optimizers
- **Batch Normalization** - Improve training stability and speed
- **Dropout Regularization** - Additional regularization technique
- **Model Serialization** - Save and load trained models efficiently
- **Unit Tests** - Comprehensive test suite for all components
- **Documentation** - Detailed docstrings and tutorial notebooks
- **GPU Support** - Optional GPU acceleration using CuPy or similar
- **Additional Activation Functions** - Sigmoid, Tanh, ELU, etc.
- **Hyperparameter Tuning** - Tools for automated parameter optimization

## Contributing

Contributions are welcome! Feel free to:
- Report issues or bugs
- Suggest new features
- Submit pull requests with improvements
- Share feedback and suggestions

## License

This project is provided as-is for educational purposes.

---

**Last Updated:** December 2025