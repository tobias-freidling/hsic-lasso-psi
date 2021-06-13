import numpy as np
import pandas as pd
from psi.poly_multi import Poly_Multi_HSIC
from psi.hsic_lasso_psi import Split_HSIC_Lasso

"""Evaluation of the Turkish student dataset"""


# Reading in data
df_student = pd.read_csv('Datasets/turkiye-student-evaluation_generic.csv', sep = ',')
Y_student = df_student.loc[:,'difficulty'].values
Y_student = Y_student.astype(float)
X_student = df_student.loc[:,'Q1':].values
X_student = X_student.astype(float)

# Models
targets = ['partial', 'full', 'carved', 'H']
alpha = 0.05
split_student = Split_HSIC_Lasso(targets, split_ratio = 0.2, n_screen = 28, adaptive_lasso = False,
                                 cv = True, cov_mode = 'oas', M_estimator = 'unbiased',
                                 H_estimator = 'block', H_B = 10, discrete_output = False)
multi_student = Poly_Multi_HSIC(n_select = 10, poly = False, estimator = 'block', B = 10)

# Inference
H0 = np.ones(28)
M0 = np.eye(28)
res_split_student = split_student.sel_inf(X_student, Y_student, 'test', alpha, None, H0, M0)
res_multi_student = multi_student.sel_inf(X_student, Y_student, 'test', alpha, niv = 0)

res_split_student.print_summary()
res_multi_student.print_summary()