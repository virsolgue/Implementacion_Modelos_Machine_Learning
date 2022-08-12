import numpy as np
import util as util

from p01b_logreg import LogisticRegression

# Caracter a reemplazar con el sub problema correspondiente.`
WILDCARD = 'X'


def p02cde(train_path, valid_path, test_path, pred_path, save_path):
    """Problema 2: regresión logística para positivos incompletos.

    Correr bajo las siguientes condiciones:
        1. en y-labels,
        2. en l-labels,
        3. en l-labels con el factor de correción alfa.

    Args:
        train_path: directorio al CSV conteniendo el archivo de entrenamiento.
        valid_path: directorio al CSV conteniendo el archivo de validación.
        test_path: directorio al CSV conteniendo el archivo de test.
        pred_path: direcotrio para guardar las predicciones.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** EMPEZAR EL CÓDIGO AQUÍ ***

    pred_path_c_plot = save_path.replace(WILDCARD, 'c')
    pred_path_d_plot = save_path.replace(WILDCARD, 'd')
    pred_path_e_plot = save_path.replace(WILDCARD, 'e')

    # Parte (c): Train y test en labels verdaderos.
    # Asegurarse de guardar las salidas en pred_path_c

    x_train, t_train = util.load_dataset(train_path, label_col='t', add_intercept=True)
    reglog = LogisticRegression()
    reglog.fit(x_train, t_train)

    x_test, t_test = util.load_dataset(test_path, label_col='t', add_intercept=True)
    
    p_test = reglog.predict(x_test)
    
    np.savetxt(pred_path_c, p_test)
    util.plot(x_test, t_test, reglog.theta, pred_path_c_plot)

    # Part (d): Train en y-labels y test en labels verdaderos.
    # Asegurarse de guardar las salidas en pred_path_d

    x_train, y_train = util.load_dataset(train_path, label_col='y', add_intercept=True)
    reglog = LogisticRegression()
    reglog.fit(x_train, y_train)
    x_test, t_test = util.load_dataset(test_path, label_col='t',
                                       add_intercept=True)
    p_test = reglog.predict(x_test)
    np.savetxt(pred_path_d, p_test)
    util.plot(x_test, t_test, reglog.theta, pred_path_d_plot)

    # Part (e): aplicar el factor de correción usando el conjunto de validación, y test en labels verdaderos.
    # Plot y usar np.savetxt para guardar las salidas en  pred_path_e

    x_valid, y_valid = util.load_dataset(valid_path, label_col='y')
    x_valid = x_valid[y_valid == 1, :]  # Restrict to just the labeled examples
    x_valid = util.add_intercept(x_valid)
    y_pred = reglog.predict(x_valid)
    alpha = np.mean(y_pred)
    x_test, t_test = util.load_dataset(test_path, label_col='t', add_intercept=True)

    # Plot and use np.savetxt to save outputs to output_path_adjusted
    np.savetxt(pred_path_e, p_test / alpha)
    util.plot(x_test, t_test, reglog.theta, pred_path_e_plot, correction=alpha)

    # *** TERMINAR CÓDIGO AQUÍ
