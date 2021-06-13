from psi.simulation import Data_Generator, Evaluator, Visualisation
from psi.linear import Linear_Model
from psi.poly_multi import Poly_Multi_HSIC
from psi.hsic_lasso_psi import Split_HSIC_Lasso

"""
Comparison with other method for (model-free) post-selection inference
"""

dg_id = Data_Generator(p = 500, experiment = 'non_linear', rho = 0, decay = False,
                       customize = False, noise_signal_ratio = 0.2)
dg_const = Data_Generator(p = 500, experiment = 'non_linear', rho = 0.1, decay = False,
                          customize = False, noise_signal_ratio = 0.2)
dg_logistic = Data_Generator(p = 500, experiment = 'logistic', rho = 0, decay = False)
dg_linear = Data_Generator(p = 500, experiment = 'linear', rho = 0, decay = False, customize = False,
                           noise_signal_ratio = 0.3)

targets = ['partial', 'full', 'carved', 'H']
split_b10 = Split_HSIC_Lasso(targets, split_ratio = 0.25, n_screen = 50, H_estimator = 'block', H_B = 10)
split_inc1 = Split_HSIC_Lasso(targets, split_ratio = 0.25, n_screen = 50, H_estimator = 'inc', H_l = 1)
multi_b10 = Poly_Multi_HSIC(30, poly = False, estimator = 'block', B = 10)
multi_inc1 = Poly_Multi_HSIC(30, poly = False, estimator = 'inc', l = 1)
linear = Linear_Model(sigma = 5 * 0.3, reg_factor = 3)

models1 = [split_b10, split_inc1, multi_b10, multi_inc1]
names1 = ['split_b10', 'split_inc1', 'multi_b10', 'multi_inc1']
models2 = [split_b10, split_inc1, multi_b10, multi_inc1, linear]
names2 = ['split_b10', 'split_inc1', 'multi_b10', 'multi_inc1', 'linear']

sample_sizes = [250, 500, 1000, 1500, 2000]
eval_id = Evaluator(models1, names1, rep = 100, dg = dg_id, n_record_variables = 4,
                    sample_sizes = sample_sizes, alpha = 0.05, start_seed = 2020)
eval_const = Evaluator(models1, names1, rep = 100, dg = dg_const, n_record_variables = 4,
                       sample_sizes = sample_sizes, alpha = 0.05, start_seed = 2020)
eval_logistic = Evaluator(models1, names1, rep = 100, dg = dg_logistic, n_record_variables = 5,
                          sample_sizes = sample_sizes, alpha = 0.05, start_seed = 2020)
eval_linear = Evaluator(models2, names2, rep = 100, dg = dg_linear, n_record_variables = 5,
                        sample_sizes = sample_sizes, alpha = 0.05, start_seed = 2020)

eval_id.simulation_parallel()
eval_const.simulation_parallel()
eval_logistic.simulation_parallel()
eval_linear.simulation_parallel()


vis4 = Visualisation({'eval_id': eval_id, 'eval_const': eval_const,
                      'eval_logistic': eval_logistic, 'eval_linear': eval_linear})

# Visualisation of TPR and FPR for HSIC-target
labels1 = ['Proposal, Block, B=10', 'Proposal, inc., l=1',
           'Multi, Block, B=10', 'Multi, inc., l=1']
labels2 = ['Proposal, Block, B=10', 'Proposal, inc., l=1',
           'Multi, Block, B=10', 'Multi, inc., l=1', 'Linear']
titles = ['Non-linear, Identity', 'Non-linear, Const. corr.', 'Logistic', 'Linear']
subdict_model_comp1 = {'split_b10': ['H'], 'split_inc1': ['H'],
                       'multi_b10': ['H'], 'multi_inc1': ['H']}
v_dict12 = {'eval_id': subdict_model_comp1,
            'eval_const': subdict_model_comp1,
            'eval_logistic': subdict_model_comp1,
            'eval_linear': subdict_model_comp1}
label_dict12 = {'split_b10': ['Proposal, Block, B=10'], 'split_inc1': ['Proposal, inc., l=1'],
               'multi_b10': ['Multi, Block, B=10'], 'multi_inc1': ['Multi, inc., l=1']}
vis4.visualise_rates('fpr', v_dict12, titles, label_dict12, width = 12, height = 3)
vis4.visualise_rates('tpr', v_dict12, titles, label_dict12, width = 12, height = 3)

# Visualisation of TPR and FPR for other targets
subdict_model_comp3 = {'split_b10': ['partial', 'full', 'carved'],
                       'split_inc1': ['partial', 'full', 'carved'],
                       'linear': ['beta']}
subdict_model_comp2 = {'split_b10': ['partial', 'full', 'carved'],
                       'split_inc1': ['partial', 'full', 'carved']}
v_dict13 = {'eval_id': subdict_model_comp2,
            'eval_const': subdict_model_comp2,
            'eval_logistic': subdict_model_comp2,
            'eval_linear': subdict_model_comp3}
label_dict13 = {'split_b10': ['Proposal, B=10, partial', 'Proposal, B=10, full', 'Proposal, B=10, carved'],
                'split_inc1': ['Proposal, l=1, partial', 'Proposal, l=1, full', 'Proposal, l=1, carved'],
                'linear': ['Linear, partial']}
vis4.visualise_rates('fpr', v_dict13, titles, label_dict13, width = 12, height = 3)
vis4.visualise_rates('tpr', v_dict13, titles, label_dict13, width = 12, height = 3)