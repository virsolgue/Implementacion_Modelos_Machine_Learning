import numpy as np
import util as util

import matplotlib.pyplot as plt
from linear_model import LinearModel


def p03d(lr, train_path, eval_path, pred_path, save_path):
    """Problema 3(d): regresión Poisson con ascenso por gradiente.

    Args:
        lr: tasa de aprendizaje para el ascenso por gradiente.
        train_path: directorio al CSV conteniendo el archivo de entrenamiento.
        eval_path: directorio al CSV conteniendo el archivo de evaluación.
        pred_path: direcotrio para guardar las predicciones.
    """
    # Cargar dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** EMPEZAR EL CÓDIGO AQUÍ ***

    # Entrenar una regresión poisson
    Poisson = PoissonRegression(step_size=lr)
    Poisson.fit(x_train, y_train)

    # Correr en el conjunto de validación, y usar  np.savetxt para guardar las salidas en pred_path.
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)

    p_eval = Poisson.predict(x_eval)
    np.savetxt(pred_path, p_eval)
    plt.figure()
    plt.scatter(y_eval, p_eval, alpha=0.4, c='red', label='Datos reales vs Predicciones')
    plt.xlabel('Datos reales')
    plt.ylabel('Predicciones')
    plt.legend()
    plt.savefig(save_path)    
    # *** TERMINAR CÓDIGO AQUÍ


class PoissonRegression(LinearModel):
    """Regresión poisson.

    Ejemplo de uso:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Corre ascenso por gradiente para maximizar la verosimilitud de una regresión poisson.

        Args:
            x: ejemplos de entrenamiento (features solamente). Tamaño (m, n).
            y: etiquetas de ejemplos de entrenamiento. Tamaño (m,).
        """
        # *** EMPEZAR EL CÓDIGO AQUÍ ***

        m, n = x.shape
        if self.theta is None:
            self.theta = np.zeros(n, dtype=np.float32)

        prev_theta = None
        i = 0
        while i < self.max_iter \
                and (prev_theta is None
                     or np.sum(np.abs(self.theta - prev_theta)) > self.eps):
            i += 1
            prev_theta = np.copy(self.theta)
            self._step(x, y)
            print('Theta:', self.theta, '\nIteración: ', i, '\nMax Iter: ', self.max_iter)

        # *** TERMINAR CÓDIGO AQUÍ

    def predict(self, x):
        """Hace una predicción sobre x nuevos.

        Args:
            x: entradas de tamaño (m, n).

        Returns:
            Predicción en punto flotante para cada entrada. Tamaño (m,).
        """
        # *** EMPEZAR EL CÓDIGO AQUÍ ***

        y_hat = np.exp(x.dot(self.theta))
        
        print(y_hat)
        return y_hat

    def _step(self, x, y):
        """Ascenso por gradiente para maximizar el logaritmo de
        la verosimilitud de theta.
        """
        grad = np.expand_dims(y - np.exp(x.dot(self.theta)), 1) * x
        self.theta = self.theta + self.step_size * np.sum(grad, axis=0)

        # *** TERMINAR CÓDIGO AQUÍ
