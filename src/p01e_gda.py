import numpy as np
import util as util

from linear_model import LinearModel


def p01e(train_path, eval_path, pred_path, save_path):
    """Problema 1(e): análisis de discriminante gaussiano (GDA)

    Args:
        train_path: directorio al CSV conteniendo el archivo de entrenamiento.
        eval_path: directorio al CSV conteniendo el archivo de evaluación.
        pred_path: directorio para guardar las predicciones.
    """
    # Cargar dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** EMPEZAR CÓDIGO AQUÍ ***
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False)
    gda = GDA()
    gda.fit(x_train, y_train)
    util.plot(x_eval, y_eval, gda.theta, save_path)
    np.savetxt(pred_path, gda.predict(x_eval))
    # *** TERMINAR CÓDIGO AQUÍ ***


class GDA(LinearModel):
    """Análisis de discriminante gaussiano.

    Ejemplo de uso:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Entrena un modelo GDA.

        Args:
            x: ejemplos de entrenamiento (features solamente). Tamaño (m, n).
            y: etiquetas de ejemplos de entrenamiento. Tamaño (m,).

        Returns:
            theta: parámetros del modelo GDA.
        """
        # *** EMPEZAR CÓDIGO AQUÍ ***
        m, n = x.shape
        print(x.shape)

        # Find phi, mu_0, mu_1, and sigma
        phi = 1 / m * np.sum(y == 1)
        mu_0 = (y == 0).dot(x) / np.sum(y == 0)
        mu_1 = (y == 1).dot(x) / np.sum(y == 1)
        mu_yi = np.where(np.expand_dims(y == 0, -1),
                         np.expand_dims(mu_0, 0),
                         np.expand_dims(mu_1, 0))
        sigma = 1 / m * (x - mu_yi).T.dot(x - mu_yi)

        # Write theta in terms of the parameters
        self.theta = np.zeros(n + 1)
        sigma_inv = np.linalg.inv(sigma)
        mu_diff = mu_0.T.dot(sigma_inv).dot(mu_0) \
            - mu_1.T.dot(sigma_inv).dot(mu_1)
        self.theta[0] = 1 / 2 * mu_diff - np.log((1 - phi) / phi)
        self.theta[1:] = -sigma_inv.dot(mu_0 - mu_1)
        # *** TERMINAR CÓDIGO AQUÍ ***

    def predict(self, x):
        """Hace una predicción sobre x nuevos.

        Args:
            x: entradas de tamaño (m, n).

        Returns:
            Salidas de tamaño (m,).
        """
        # *** EMPEZAR CÓDIGO AQUÍ ***
        decision = self.theta[0] + self.theta[1:] @ x.T
        
        return (decision > 0).astype('int')
        # *** TERMINAR CÓDIGO AQUÍ ***
