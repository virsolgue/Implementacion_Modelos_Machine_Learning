import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def p05b(tau, train_path, eval_path, save_path):
    """Problema 5(b): regresión lineal pesada (LWR)

    Args:
        tau: parámetro de ancho de banda para LWR.
        train_path: directorio al CSV conteniendo el archivo de entrenamiento.
        eval_path: directorio al CSV conteniendo el archivo de evaluación.
    """
    # Cargar el dataset de entrenamiento
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** EMPEZAR CÓDIGO AQUÍ ***

    # Entrenar un modelo LWR
    LWR = LocallyWeightedLinearRegression(tau)
    LWR.fit(x_train, y_train)

    # Obtener el MSE para el conjunto de validación
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = LWR.predict(x_eval)
    mse = ((y_pred - y_eval)**2).mean()
    print(mse)

    # Graficar las predicciones de validación sobre el conjunto de entrenamiento
    # No hace falta guardar las predicciones
    # Graficar los datos

    plt.figure()
    plt.plot(x_train, y_train, 'bx')
    plt.plot(x_eval, y_pred, 'ro')
    # plt.show()
    plt.savefig(save_path)

    # *** TERMINAR CÓDIGO AQUÍ ***


class LocallyWeightedLinearRegression(LinearModel):
    """regresión lineal pesada (LWR).

    Ejemplo de uso:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Entrenar LWR simplemente guardando el conjunto de entrenamiento.

        """
        # *** EMPEZAR CÓDIGO AQUÍ ***
        self.x = x
        self.y = y
        # *** TERMINAR CÓDIGO AQUÍ ***

    def predict(self, x):
        """Hace una predicción sobre x nuevos.

        Args:
            x: entradas de tamaño (m, n).

        Returns:
            Salidas de tamaño (m,).
        """
        # *** EMPEZAR CÓDIGO AQUÍ ***

        m, n = x.shape
        g = lambda x: np.exp(-(x**2)/(2*self.tau**2))
        
        # Calcula w para cada x
        from numpy.linalg import inv, norm
        w = g(norm(self.x[None]-x[:,None], axis=2))
        y_pred = np.zeros(m)  
        for i, W in enumerate(w):
            W = np.diag(W)
            theta = inv(self.x.T.dot(W).dot(self.x)).dot(self.x.T).dot(W).dot(self.y)           
            # make prediction
            y_pred[i] = x[i].dot(theta)
        return (y_pred)

        # *** TERMINAR CÓDIGO AQUÍ ***
