import sys, os
sys.path.append(os.path.join(sys.path[0], 'paper_version', 'psi'))

from hsic_lasso_psi import Split_HSIC_Lasso
import numpy as np
from sklearn.utils import shuffle

# Loading data (Villani et. al, 2017)
featurenames = np.load('Datasets/scrna-seq/featnames.npy')
# features selected by Climente-Gonzalez et al., 2019
features_hl = np.load('Datasets/scrna-seq/features_hl.npy')
x_imputed = np.load('Datasets/scrna-seq/x_imputed.npy')
y = np.load('Datasets/scrna-seq/y.npy')

# Deleting data of unclassified cells
del_ind = np.where(y == -1)
y = np.delete(y, del_ind)
x_imputed = np.delete(x_imputed, del_ind, axis = 0)

# Shuffling
x_imputed_sh, y_sh = shuffle(x_imputed, y)

# Setup for inference
targets = ['partial', 'H']
alpha = 0.05
split = Split_HSIC_Lasso(targets, split_ratio = 0.5, n_screen = 1000, adaptive_lasso = False,
                         cv = True, cov_mode = 'oas', M_estimator = 'unbiased',
                         H_estimator = 'inc', H_l = 20, discrete_output = True)

# Inference
H0 = np.ones(26593) # dummy variable
M0 = np.eye(26593) # dummy variable
res = split.sel_inf(x_imputed_sh, y_sh, 'test', alpha, None, H0, M0,
                    unbiased_parallel = True, n_jobs = 30)

# Displaying results
res.print_summary()

for t in targets:
    print('p-values of {} target:'.format(t))
    print(res.p_values[t][(res.ind_sel).astype(bool)])

print('Feature names of selected indices:')
print(featurenames[(res.ind_sel).astype(bool)])
