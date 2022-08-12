class LinearModel(object):
    """Clase base para modelos lineales."""

    def __init__(self, step_size=0.2, max_iter=10000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: tamaño del paso para cada iteración.
            max_iter: cantidad máxima de iteraciones.
            eps: umbral para convergencia.
            theta_0: valor inicial de tita. Si None, use el vector cero.
            verbose: imprimir valores de pérdida durante el entrenamiento.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Entrenar el modelo lineal.

        Args:
            x: ejemplos de entrenmiento (solamente features). Tamaño (m, n).
            y: etiquetas de los ejemplos de entrenamiento. Tamaño (m,).
        """
        raise NotImplementedError('Subclase de LinearModel debe implementar el método fit.')

    def predict(self, x):
        """Hace una presicción dado un nuevo x.

        Args:
            x: entradas (solamente features) de tamaño (m, n).

        Devuelve:
            Predicciones de tamaño (m,).
        """
        raise NotImplementedError('Subclase de LinearModel debe implementar el método predict.')
