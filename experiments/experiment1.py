from psi.simulation import Data_Generator, Evaluator, Visualisation
from psi.hsic_lasso_psi import Split_HSIC_Lasso

"""
Comparison of different methods for hyperparameter choice
"""

dg_id = Data_Generator(p = 500, experiment = 'non_linear', rho = 0, decay = False,
                       customize = False, noise_signal_ratio = 0.2)
dg_decay = Data_Generator(p = 500, experiment = 'non_linear', rho = 0.3, decay = True,
                       customize = False, noise_signal_ratio = 0.2)
dg_const = Data_Generator(p = 500, experiment = 'non_linear', rho = 0.1, decay = False,
                       customize = False, noise_signal_ratio = 0.2)

targets = ['partial', 'full', 'carved', 'H']
m_cv_nada = Split_HSIC_Lasso(targets, split_ratio = 0.25, n_screen = 50,
                             adaptive_lasso = False, cv = True)
m_cv_ada = Split_HSIC_Lasso(targets, split_ratio = 0.25, n_screen = 50,
                            adaptive_lasso = True, gamma = 1.5, cv = True)
m_ic_nada = Split_HSIC_Lasso(targets, split_ratio = 0.25, n_screen = 50,
                             adaptive_lasso = False, cv = False, criterion = 'aic')
m_ic_ada = Split_HSIC_Lasso(targets, split_ratio = 0.25, n_screen = 50,
                            adaptive_lasso = True, gamma = 1.5, cv = False, criterion = 'aic')

models = [m_cv_nada, m_cv_ada, m_ic_nada, m_ic_ada]
names = ['m_cv_nada', 'm_cv_ada', 'm_ic_nada', 'm_ic_ada']
sample_sizes = [250, 500, 1000, 1500, 2000]
eval_id = Evaluator(models, names, rep = 200, dg = dg_id, n_record_variables = 4,
                    sample_sizes = sample_sizes, alpha = 0.05, start_seed = 2020)
eval_decay = Evaluator(models, names, rep = 200, dg = dg_decay, n_record_variables = 4,
                    sample_sizes = sample_sizes, alpha = 0.05, start_seed = 2020)
eval_const = Evaluator(models, names, rep = 200, dg = dg_const, n_record_variables = 4,
                    sample_sizes = sample_sizes, alpha = 0.05, start_seed = 2020)

eval_id.simulation_parallel()
eval_decay.simulation_parallel()
eval_const.simulation_parallel()


vis = Visualisation({'eval_id': eval_id, 'eval_decay': eval_decay, 'eval_const': eval_const})

# Visualise screening
v_dict1 = {'eval_id': ['m_cv_nada'],
           'eval_decay': ['m_cv_nada'],
           'eval_const': ['m_cv_nada']}
label_dict1 = {'eval_id': ['identity'],
               'eval_decay': ['decaying corr.'],
               'eval_const': ['const corr.']}
vis.visualise_screening(v_dict1, None, label_dict1, width = 6, height = 4,
                        var_ex = 1, legend_position = 'lower right')

# Visualise selection
v_dict2 = {'eval_id': ['m_cv_nada', 'm_cv_ada', 'm_ic_nada', 'm_ic_ada'],
           'eval_decay': ['m_cv_nada', 'm_cv_ada', 'm_ic_nada', 'm_ic_ada'],
           'eval_const': ['m_cv_nada', 'm_cv_ada', 'm_ic_nada', 'm_ic_ada']}
label_dict2 = {'eval_id': ['CV - non-adaptive', 'CV - adaptive', 'AIC - non-adaptive', 'AIC - adaptive'],
               'eval_decay': ['CV - non-adaptive', 'CV - adaptive', 'AIC - non-adaptive', 'AIC - adaptive'],
               'eval_const': ['CV - non-adaptive', 'CV - adaptive', 'AIC - non-adaptive', 'AIC - adaptive']}
vis.visualise_selection(v_dict2, ['Identity', 'Decaying correlation', 'Constant correlation'],
                        label_dict2, width = 12, height = 4)