import numpy as np
import util as util

from linear_model import LinearModel


def p01b(train_path, eval_path, pred_path, save_path):
    """Problema 1(b): Regresión Logística con el método de Newton.

    Args:
        train_path: directorio al CSV conteniendo el archivo de entrenamiento.
        eval_path: directorio al CSV conteniendo el archivo de evaluación.
        pred_path: directorio para guardar las predicciones.
        save_path: directorio para guardar las imagenes.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** EMPEZAR CÓDIGO AQUÍ ***
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    reglog = LogisticRegression()
    reglog.fit(x_train, y_train)
    util.plot(x_eval, y_eval, reglog.theta, save_path)
    np.savetxt(pred_path, reglog.predict(x_eval))
    # *** TERMINAR CÓDIGO AQUÍ ***


class LogisticRegression(LinearModel):
    """Regresión Logística con Newton como solver.

    Ejemplo de uso:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Corre el método de Newton para minimizar J(tita) para regresión logística.

        Args:
            x: ejemplos de entrenamiento (features solamente). Tamaño (m, n).
            y: etiquetas de ejemplos de entrenamiento. Tamaño (m,).
        """
        # *** EMPEZAR CÓDIGO AQUÍ ***
        self.theta = np.zeros(x.shape[1])
        while True:
            y_pred = self.predict(x)
            grad = ((y_pred - y) * x.T).mean(axis=1)
            hess = ((y_pred * (1 - y_pred)) * x.T) @ x / x.shape[1]
            diff = grad @ np.linalg.inv(hess.T)
            self.theta = self.theta - diff
            if np.abs(diff).sum() < self.eps:
                return self
        # *** TERMINAR CÓDIGO AQUÍ ***

    def predict(self, x):
        """Hace una predicción sobre x nuevos.

        Args:
            x: entradas de tamaño (m, n).

        Returns:
            Salidas de tamaño (m,).
        """
        # *** EMPEZAR CÓDIGO AQUÍ ***
        z = self.theta @ x.T
        return 1 / (1 + np.exp(-z))
        # pred = la cuenta de la sigmoide con el x

        # *** TERMINAR CÓDIGO AQUÍ ***
