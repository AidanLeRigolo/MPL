import numpy as np
import matplotlib.pyplot as plt


# ---------------- vortex Function ---------------- #
def vortex(points, classes, noise=0.04):
    X = np.zeros((points * classes, 2))
    y = np.zeros(points * classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points * class_number, points * (class_number + 1))
        r = np.linspace(0.0, 1, points)
        t = np.linspace(class_number * 4, (class_number + 1) * 4, points) + np.random.randn(points) * noise
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        X[ix] += np.random.normal(0, noise, X[ix].shape)
        y[ix] = class_number
    return X, y


# ---------------- square Function ---------------- #
def square(points, classes, noise=0.02):
    grid_size = int(np.sqrt(classes))
    X = np.zeros((points * classes, 2))
    y = np.repeat(np.arange(classes), points)
    for class_number in range(classes):
        row, col = divmod(class_number, grid_size)
        x_min, x_max = col / grid_size, (col + 1) / grid_size
        y_min, y_max = row / grid_size, (row + 1) / grid_size
        ix = slice(points * class_number, points * (class_number + 1))
        X[ix, 0] = np.random.rand(points) * (x_max - x_min) + x_min
        X[ix, 1] = np.random.rand(points) * (y_max - y_min) + y_min
        X[ix] += np.random.normal(0, noise, X[ix].shape)
    return X, y


# ---------------- hearth Function ---------------- #
def hearth(points, classes, noise=0.001):
    X = np.zeros((points * classes, 2))
    y = np.zeros(points * classes, dtype='uint8')
    for class_number in range(classes):
        ix = slice(points * class_number, points * (class_number + 1))
        t = np.linspace(0, 2 * np.pi, points)
        size = 1 - class_number * 0.03
        x = size * (16 * np.sin(t)**3)
        y_ = size * (13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t))
        X[ix] = np.stack((x, y_), axis=1)
        X[ix] += np.random.normal(0, noise, X[ix].shape)
        y[ix] = class_number
    return X, y


# ---------------- triangles Function ---------------- #
def triangles(points, classes, noise=0.001):
    X = np.zeros((points * classes, 2))
    y = np.zeros(points * classes, dtype='uint8')
    for class_number in range(classes):
        ix = slice(points * class_number, points * (class_number + 1))
        size = 1 - class_number * 0.05
        vertices = np.array([
            [0, size],
            [-size * np.sqrt(3)/2, -size/2],
            [size * np.sqrt(3)/2, -size/2]])
        t = np.random.rand(points)
        sides = np.random.randint(0, 3, points)
        X[ix] = (1 - t[:, None]) * vertices[sides] + t[:, None] * vertices[(sides + 1) % 3]
        X[ix] += np.random.normal(0, noise, X[ix].shape)
        y[ix] = class_number
    return X, y


# ---------------- circle Function ---------------- #
def circles(points, classes, noise=0.05):
    X = np.zeros((points * classes, 2))
    y = np.zeros(points * classes, dtype='uint8')
    for class_number in range(classes):
        ix = slice(points * class_number, points * (class_number + 1))
        radius = 1 + class_number * 0.3
        t = np.linspace(0, 2*np.pi, points)
        x = radius * np.cos(t)
        y_ = radius * np.sin(t)
        X[ix] = np.stack((x, y_), axis=1)
        X[ix] += np.random.normal(0, noise, X[ix].shape)
        y[ix] = class_number
    return X, y


# ---------------- spiral Function ---------------- #
def spiral(points, classes, noise=0.04):
    X = np.zeros((points * classes, 2))
    y = np.zeros(points * classes, dtype='uint8')
    for class_number in range(classes):
        ix = slice(points * class_number, points * (class_number + 1))
        r = np.linspace(0.2, 1, points)
        theta = np.linspace(class_number * 4, (class_number + 1) * 4, points)
        X[ix, 0] = r * np.cos(theta)
        X[ix, 1] = r * np.sin(theta)
        X[ix] += np.random.normal(0, noise, X[ix].shape)
        y[ix] = class_number
    return X, y


# ---------------- sinus Function ---------------- #
def sinus(points, classes, noise=0.05):
    X = np.zeros((points * classes, 2))
    y = np.zeros(points * classes, dtype='uint8')
    for class_number in range(classes):
        ix = slice(points * class_number, points * (class_number + 1))
        x = np.linspace(-2*np.pi, 2*np.pi, points)
        y_ = np.sin(x + class_number)
        X[ix] = np.stack((x, y_), axis=1)
        X[ix] += np.random.normal(0, noise, X[ix].shape)
        y[ix] = class_number
    return X, y


# ---------------- Mandelbrot Function ---------------- #
def mandelbrot(x, y, max_depth=100):
    a = x + 1j * y
    z = 0
    for i in range(max_depth):
        z = z**2 + a
        if abs(z) > 2:
            return 1 - (1 / ((i + 1) / 50) + 1)
    return 1.0


def generate_mandelbrot_dataset(width, height, xlim, ylim):
    xs = np.linspace(xlim[0], xlim[1], width)
    ys = np.linspace(ylim[0], ylim[1], height)
    X = []
    y = []
    for i in range(width):
        for j in range(height):
            x_val = xs[i]
            y_val = ys[j]
            X.append([x_val, y_val])
            y.append([mandelbrot(x_val, y_val)])
    return np.array(X), np.array(y)


# ---------------- Exécution d'un exemple ---------------- #
if __name__ == "__main__":
    X ,y = np.zeros((1, 2)), np.zeros((1, 2))
    print("Génération d'un dataset de classification")
    #X, y = vortex(500, 5, noise=0.04)
    #X, y = square(500, 9, noise=0.02)
    #X, y = hearth(700, 20, noise=0.001)
    #X, y = triangles(500, 15, noise=0.001)
    #X, y = circles(500, 25,  noise=0.05)
    #X, y = spiral(500, 11, noise=0.04)
    #X, y = sinus(500, 5, noise=0.05)
   
    if X.any():
        plt.figure(figsize=(6, 6))
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap="managua", s=1)
        plt.title("Données simulées")
        plt.axis("equal")
        plt.grid(True)
        plt.show()
    else:        
        X, y = generate_mandelbrot_dataset(300, 300, (-2.0, 1.0), (-1.5, 1.5))
        image = y.reshape((300, 300))
        plt.imshow(image.T, extent=(*(-2.0, 1.0), *(-1.5, 1.5)), cmap='managua', origin='lower')
        plt.show()
       

