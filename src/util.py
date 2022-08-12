import matplotlib.pyplot as plt
import numpy as np


def add_intercept(x):
    """Agrega término independiente a la matriz x.

    Args:
        x: 2D NumPy array.

    Returns:
        Nueva matriz igual que x con 1's agregados como columna 0.
    """
    new_x = np.zeros((x.shape[0], x.shape[1] + 1), dtype=x.dtype)
    new_x[:, 0] = 1
    new_x[:, 1:] = x

    return new_x


def load_dataset(csv_path, label_col='y', add_intercept=False):
    """Carga dataset desde un CSV.

    Args:
         csv_path: directorio al CSV conteniendo el dataset.
         label_col: nombre de la columna a usar como labels (debería ser'y' o 'l').
         add_intercept: agregra término independiente a las x

    Returns:
        xs: Numpy array de x-valores (entradas).
        ys: Numpy array de y-valores (salidas).
    """

    def add_intercept_fn(x):
        global add_intercept
        return add_intercept(x)

    # Validate label_col argument
    allowed_label_cols = ('y', 't')
    if label_col not in allowed_label_cols:
        raise ValueError('label_col invalida: {} (se espera {})'
                         .format(label_col, allowed_label_cols))

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    l_cols = [i for i in range(len(headers)) if headers[i] == label_col]
    inputs = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols)
    labels = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=l_cols)

    if inputs.ndim == 1:
        inputs = np.expand_dims(inputs, -1)

    if add_intercept:
        inputs = add_intercept_fn(inputs)

    return inputs, labels


def plot(x, y, theta, save_path=None, correction=1.0):
    """Grafica dataset de acuerdo a parámetros entrenados por una regresión logística.
    Args:
        x: matriz de ejemplos train, uno por fila.
        y: vector de etiquetas en {0, 1}.
        theta: vector de parámetros del modelo de regresión logística
        save_path: directorio para guardar el plot.
        correction: factor de correción (Problema 2(e) solamente).
    """
    # Plot dataset
    plt.figure()
    plt.plot(x[y == 1, -2], x[y == 1, -1], 'bx', linewidth=2)
    plt.plot(x[y == 0, -2], x[y == 0, -1], 'go', linewidth=2)

    # Plot límite de decisión (encontrato por resolver para theta^t x = 0)
    margin1 = (max(x[:, -2]) - min(x[:, -2]))*0.2
    margin2 = (max(x[:, -1]) - min(x[:, -1]))*0.2
    x1 = np.arange(min(x[:, -2])-margin1, max(x[:, -2])+margin1, 0.01)
    x2 = -(theta[0] / theta[2] * correction + theta[1] / theta[2] * x1)
    plt.plot(x1, x2, c='red', linewidth=2)
    plt.xlim(x[:, -2].min()-margin1, x[:, -2].max()+margin1)
    plt.ylim(x[:, -1].min()-margin2, x[:, -1].max()+margin2)

    # Agregar etiquetas y guardar a disco
    plt.xlabel('x1')
    plt.ylabel('x2')
    if save_path is not None:
        plt.savefig(save_path)
