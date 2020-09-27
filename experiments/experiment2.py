from psi.simulation import Data_Generator, Evaluator, Visualisation
from psi.hsic_lasso_psi import Split_HSIC_Lasso

"""
Comparison of different estimators
"""

dg_id = Data_Generator(p = 500, experiment = 'non_linear', rho = 0, decay = False,
                       customize = False, noise_signal_ratio = 0.2)
dg_decay = Data_Generator(p = 500, experiment = 'non_linear', rho = 0.3, decay = True,
                       customize = False, noise_signal_ratio = 0.2)
dg_const = Data_Generator(p = 500, experiment = 'non_linear', rho = 0.1, decay = False,
                       customize = False, noise_signal_ratio = 0.2)


targets = ['partial', 'full', 'carved', 'H']
m_b5 = Split_HSIC_Lasso(targets, split_ratio = 0.25, n_screen = 50,
                        H_estimator = 'block', H_B = 5)
m_b10 = Split_HSIC_Lasso(targets, split_ratio = 0.25, n_screen = 50,
                        H_estimator = 'block', H_B = 10)
m_inc1 = Split_HSIC_Lasso(targets, split_ratio = 0.25, n_screen = 50,
                          H_estimator = 'inc', H_l = 1)
m_inc5 = Split_HSIC_Lasso(targets, split_ratio = 0.25, n_screen = 50,
                          H_estimator = 'inc', H_l = 5)


models = [m_b5, m_b10, m_inc1, m_inc5]
names = ['m_b5', 'm_b10', 'm_inc1', 'm_inc5']
sample_sizes = [250, 500, 1000, 1500, 2000]
eval_id2 = Evaluator(models, names, rep = 200, dg = dg_id, n_record_variables = 4,
                    sample_sizes = sample_sizes, alpha = 0.05, start_seed = 2020)
eval_decay2 = Evaluator(models, names, rep = 200, dg = dg_decay, n_record_variables = 4,
                    sample_sizes = sample_sizes, alpha = 0.05, start_seed = 2020)
eval_const2 = Evaluator(models, names, rep = 200, dg = dg_const, n_record_variables = 4,
                    sample_sizes = sample_sizes, alpha = 0.05, start_seed = 2020)

eval_id2.simulation_parallel()
eval_decay2.simulation_parallel()
eval_const2.simulation_parallel()


vis2 = Visualisation({'eval_id2': eval_id2, 'eval_decay2': eval_decay2, 'eval_const2': eval_const2})

sub_dict_partial = {'m_b5': ['partial'], 'm_b10': ['partial'],
                    'm_inc1': ['partial'], 'm_inc5': ['partial']}
v_dict3 = {'eval_id2': sub_dict_partial,
           'eval_decay2': sub_dict_partial,
           'eval_const2': sub_dict_partial}
label_dict3 = {'m_b5': ['Block, B=5'], 'm_b10': ['Block, B=10'],
               'm_inc1': ['incomplete, l=1'], 'm_inc5': ['incomplete, l=5']}
titles = ['Identity', 'Decaying correlation', 'Constant correlation']

# Visualise FPR - partial target
vis2.visualise_rates('fpr', v_dict3, titles, label_dict3, width = 12, height = 4)
# Visualise TPR - partial target
vis2.visualise_rates('tpr', v_dict3, titles, label_dict3, width = 12, height = 4)


sub_dict_H = {'m_b5': ['carved'], 'm_b10': ['carved'], 'm_inc1': ['carved'], 'm_inc5': ['carved']}
v_dict4 = {'eval_id2': sub_dict_H,
           'eval_decay2': sub_dict_H,
           'eval_const2': sub_dict_H}
# Visualise FPR - HSIC-target
vis2.visualise_rates('fpr', v_dict4, titles, label_dict3, width = 12, height = 4)
# Visualise TPR - HSIC-target
vis2.visualise_rates('tpr', v_dict4, titles, label_dict3, width = 12, height = 4)

