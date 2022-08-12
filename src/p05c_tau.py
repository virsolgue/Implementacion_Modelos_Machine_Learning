import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def p05c(tau_values, train_path, valid_path, test_path, pred_path):
    """Problema 5(b): ajustar el parámetro tau para LWR.

    Args:
        tau_values: lista de valores de tau a probar.
        train_path: directorio al CSV conteniendo el archivo de entrenamiento.
        valid_path: directorio al CSV conteniendo el archivo de validación.
        test_path: directorio al CSV conteniendo el archivo de test.
        pred_path: directorio para guardar las predicciones.
    """
    # Cargar el dataset de train
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** EMPEZAR CÓDIGO AQUÍ ***

    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)

    # Buscar entre tau_values por el mejor tau (MSE más bajo en validación)
    best_tau = None
    best_mse = float('inf')
    for tau in tau_values:
        LWR = LocallyWeightedLinearRegression(tau)
        LWR.fit(x_train, y_train)
        y_pred = LWR.predict(x_eval)
        mse = ((y_eval - y_pred) ** 2).mean()
        # Grafica para cada valor de tau
        plt.figure()
        plt.title('$tau = {}$'.format(tau))
        plt.plot(x_train, y_train, 'bx')
        plt.plot(x_eval, y_pred, 'ro')
        plt.show()
        if mse < best_mse:
            best_mse = mse
            best_tau = tau
    
    # Entrenar un modelo LWR con el mejor tau.
    print('Train: best tau={}, best mse={}'.format(best_tau, best_mse))
    LWR = LocallyWeightedLinearRegression(best_tau)
    LWR.fit(x_train, y_train)

    # Correr en test para obtener el MSE
    y_pred = LWR.predict(x_test)
    mse = ((y_test - y_pred) ** 2).mean()
    print('Test: mse={}'.format(mse))

    # Guardar predicciones en pred_path
    np.savetxt(pred_path, y_pred)
    
    # *** TERMINAR CÓDIGO AQUÍ ***
