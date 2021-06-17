import numpy as np
import pandas as pd
from psi.poly_multi import Poly_Multi_HSIC
from psi.hsic_lasso_psi import Split_HSIC_Lasso

"""Evaluation of the Divorce predictors dataset"""


# Reading in data
df_divorce = pd.read_csv('Datasets/divorce.csv', sep = ';')
df_divorce = df_divorce.sample(frac=1).reset_index(drop=True)
Y_divorce = df_divorce.iloc[:,-1].values
Y_divorce = Y_divorce.astype(float)
X_divorce = df_divorce.iloc[:,:-1].values
X_divorce = X_divorce.astype(float)

# Preparation for binary data-kernel
n = Y_divorce.shape[0]
freq_dict = {0 : n, 1:n}

# Models
targets = ['partial', 'full', 'carved', 'H']
alpha = 0.05
split_inc_divorce = Split_HSIC_Lasso(targets, split_ratio = 0.2, n_screen = 54, adaptive_lasso = False,
                                     cv = True, cov_mode = 'oas', M_estimator = 'unbiased', H_estimator = 'inc',
                                     H_l = 15, discrete_output = (freq_dict, freq_dict))
split_block_divorce = Split_HSIC_Lasso(targets, split_ratio = 0.2, n_screen = 54, adaptive_lasso = False,
                                       cv = True, cov_mode = 'oas', M_estimator = 'unbiased', H_estimator = 'block',
                                       H_B = 5, discrete_output = (freq_dict, freq_dict))
multi_inc_divorce = Poly_Multi_HSIC(n_select = 15, poly = False, estimator = 'inc',
                                    l = 15, discrete_output = (freq_dict, freq_dict))
multi_block_divorce = Poly_Multi_HSIC(n_select = 15, poly = False, estimator = 'block',
                                    B = 5, discrete_output = (freq_dict, freq_dict))

# Inference
H0 = np.ones(54)
M0 = np.eye(54)
res_split_inc = split_inc_divorce.sel_inf(X_divorce, Y_divorce, 'test', alpha, None, H0, M0)
res_split_block= split_block_divorce.sel_inf(X_divorce, Y_divorce, 'test', alpha, None, H0, M0)
res_multi_inc = multi_inc_divorce.sel_inf(X_divorce, Y_divorce, 'test', alpha, niv = 0)
res_block_inc = multi_block_divorce.sel_inf(X_divorce, Y_divorce, 'test', alpha, niv = 0)

res_split_inc.print_summary()
res_split_block.print_summary()
res_multi_inc.print_summary()
res_block_inc.print_summary()