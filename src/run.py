from p01b_logreg import p01b as p01b
from p01e_gda import p01e as p01e
from p02cde_posonly import p02cde as p02
from p03d_poisson import p03d as p03d
from p05b_lwr import p05b as p05b
from p05c_tau import p05c as p05c


correr = 1  #CAMBIAR POR número de problema a correr. 0 corre todos.

# Problema 1
if correr == 0 or correr == 1:
    p01b(train_path='../data/ds1_train.csv',
         eval_path='../data/ds1_valid.csv',
         pred_path='../outputs/p01b_pred_1.txt',
         save_path='../outputs/p01b_pred_1.png')

    p01b(train_path='../data/ds2_train.csv',
         eval_path='../data/ds2_valid.csv',
         pred_path='../outputs/p01b_pred_2.txt',
         save_path='../outputs/p01b_pred_2.png')

    p01e(train_path='../data/ds1_train.csv',
         eval_path='../data/ds1_valid.csv',
         pred_path='../outputs/p01e_pred_1.txt',
         save_path='../outputs/p01e_pred_1.png')

    p01e(train_path='../data/ds2_train.csv',
         eval_path='../data/ds2_valid.csv',
         pred_path='../outputs/p01e_pred_2.txt',
         save_path='../outputs/p01e_pred_2.png')

# Problema 2
if correr == 0 or correr == 2:
    p02(train_path='../data/ds3_train.csv',
        valid_path='../data/ds3_valid.csv',
        test_path='../data/ds3_test.csv',
        pred_path='../outputs/p02X_pred.txt',
        save_path='../outputs/p02X.png')

# Problema 3
if correr == 0 or correr == 3:
    p03d(lr=1e-9,
        train_path='../data/ds4_train.csv',
        eval_path='../data/ds4_valid.csv',
        pred_path='../outputs/p03d_pred.txt',
        save_path='../outputs/p03d.png')

# Problema 5
if correr == 0 or correr == 5:
    p05b(tau=5e-1,
         train_path='../data/ds5_train.csv',
         eval_path='../data/ds5_valid.csv',
         save_path='../outputs/p05b.png')

    p05c(tau_values=[3e-2, 5e-2, 1e-1, 5e-1, 1e0, 1e1],
          train_path='../data/ds5_train.csv',
          valid_path='../data/ds5_valid.csv',
          test_path='../data/ds5_test.csv',
          pred_path='../outputs/p05c_pred.txt')
