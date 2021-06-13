from psi.simulation import Data_Generator, Evaluator, Visualisation
from psi.hsic_lasso_psi import Split_HSIC_Lasso

"""
Investigation on the influence of strong correlation among covariates
"""

dg_id = Data_Generator(p = 500, experiment = 'non_linear', rho = 0, decay = False,
                       customize = False, noise_signal_ratio = 0.2)
dg_id12 = Data_Generator(p = 500, experiment = 'non_linear', rho = 0, decay = False,
                         customize = True, combination = '12', rho_comb = 0.7, noise_signal_ratio = 0.2)
dg_id23 = Data_Generator(p = 500, experiment = 'non_linear', rho = 0, decay = False,
                         customize = True, combination = '23', rho_comb = 0.7, noise_signal_ratio = 0.2)
dg_id15 = Data_Generator(p = 500, experiment = 'non_linear', rho = 0, decay = False,
                         customize = True, combination = '15', rho_comb = 0.7, noise_signal_ratio = 0.2)
dg_id45 = Data_Generator(p = 500, experiment = 'non_linear', rho = 0, decay = False,
                         customize = True, combination = '45', rho_comb = 0.7, noise_signal_ratio = 0.2)

dg_const = Data_Generator(p = 500, experiment = 'non_linear', rho = 0.1, decay = False,
                          customize = False, noise_signal_ratio = 0.2)
dg_const12 = Data_Generator(p = 500, experiment = 'non_linear', rho = 0.1, decay = False,
                          customize = True, combination = '12', rho_comb = 0.7, noise_signal_ratio = 0.2)
dg_const23 = Data_Generator(p = 500, experiment = 'non_linear', rho = 0.1, decay = False,
                          customize = True, combination = '23', rho_comb = 0.7, noise_signal_ratio = 0.2)
dg_const15 = Data_Generator(p = 500, experiment = 'non_linear', rho = 0.1, decay = False,
                          customize = True, combination = '15', rho_comb = 0.7, noise_signal_ratio = 0.2)
dg_const45 = Data_Generator(p = 500, experiment = 'non_linear', rho = 0.1, decay = False,
                          customize = True, combination = '45', rho_comb = 0.7, noise_signal_ratio = 0.2)

targets = ['partial', 'full', 'carved', 'H']
model = Split_HSIC_Lasso(targets, split_ratio = 0.25, n_screen = 50, H_estimator = 'block', H_B = 10)
models = [model]
names = ['model']
sample_sizes = [250, 500, 1000, 1500, 2000]

eval_id = Evaluator(models, names, rep = 200, dg = dg_id, n_record_variables = 5,
                    sample_sizes = sample_sizes, alpha = 0.05, start_seed = 2020)
eval_id12 = Evaluator(models, names, rep = 200, dg = dg_id12, n_record_variables = 5,
                      sample_sizes = sample_sizes, alpha = 0.05, start_seed = 2020)
eval_id23 = Evaluator(models, names, rep = 200, dg = dg_id23, n_record_variables = 5,
                      sample_sizes = sample_sizes, alpha = 0.05, start_seed = 2020)
eval_id15 = Evaluator(models, names, rep = 200, dg = dg_id15, n_record_variables = 5,
                      sample_sizes = sample_sizes, alpha = 0.05, start_seed = 2020)
eval_id45 = Evaluator(models, names, rep = 200, dg = dg_id45, n_record_variables = 5,
                      sample_sizes = sample_sizes, alpha = 0.05, start_seed = 2020)
eval_const = Evaluator(models, names, rep = 200, dg = dg_const, n_record_variables = 5,
                       sample_sizes = sample_sizes, alpha = 0.05, start_seed = 2020)
eval_const12 = Evaluator(models, names, rep = 200, dg = dg_const12, n_record_variables = 5,
                         sample_sizes = sample_sizes, alpha = 0.05, start_seed = 2020)
eval_const23 = Evaluator(models, names, rep = 200, dg = dg_const23, n_record_variables = 5,
                         sample_sizes = sample_sizes, alpha = 0.05, start_seed = 2020)
eval_const15 = Evaluator(models, names, rep = 200, dg = dg_const15, n_record_variables = 5,
                         sample_sizes = sample_sizes, alpha = 0.05, start_seed = 2020)
eval_const45 = Evaluator(models, names, rep = 200, dg = dg_const45, n_record_variables = 5,
                         sample_sizes = sample_sizes, alpha = 0.05, start_seed = 2020)
evaluators = [eval_id, eval_id12, eval_id23, eval_id15, eval_id45,
              eval_const, eval_const12, eval_const23, eval_const15, eval_const45]

for e in evaluators:
    e.simulation_parallel()

evaluator_names = ['eval_id', 'eval_id12', 'eval_id23', 'eval_id15', 'eval_id45',
                   'eval_const', 'eval_const12', 'eval_const23', 'eval_const15', 'eval_const45']
vis3 = Visualisation(dict(zip(evaluator_names, evaluators)))

# Visualisation of selection rate for Xi = Id
model_subdict = {'model': [0, 1, 2, 3, 4]}
v_dict5 = {'eval_id': model_subdict, 'eval_id12': model_subdict,
           'eval_id23': model_subdict, 'eval_id15': model_subdict, 'eval_id45': model_subdict}
labels = ['$X_1$', '$X_2$', '$X_3$', '$X_4$', '$X_5$']
titles = ['No correlation', '$X_1$ and $X_2$ correlated', '$X_2$ and $X_3$ correlated',
          '$X_1$ and $X_5$ correlated', '$X_4$ and $X_5$ correlated']
vis3.visualise_selection_covariates5(v_dict5, titles, labels, width = 12, height = 6)

# Visualisation of selection rate for Xi_{ij} = 0.1 + 0.9*delta_{ij}
v_dict6 = {'eval_const': model_subdict, 'eval_const12': model_subdict,
           'eval_const23': model_subdict, 'eval_const15': model_subdict, 'eval_const45': model_subdict}
vis3.visualise_selection_covariates5(v_dict6, titles, labels, width = 12, height = 6)

# Visualisation of H_0 rejection rate for HSIC-target (Xi = Id)
model_H_subdict = {'model': {'H': [0, 1, 2, 3, 4]}}
model_partial_subdict = {'model': {'partial': [0, 1, 2, 3, 4]}}
model_full_subdict = {'model': {'full': [0, 1, 2, 3, 4]}}
model_carved_subdict = {'model': {'carved': [0, 1, 2, 3, 4]}}
v_dict7 = {'eval_id': model_H_subdict, 'eval_id12': model_H_subdict,
           'eval_id23': model_H_subdict, 'eval_id15': model_H_subdict, 'eval_id45': model_H_subdict}
labels_H = [r'$H_1$', r'$H_2$', r'$H_3$', r'$H_4$', r'$H_5$',]
labels_partial = [r'$\beta_{1,S}^{par}$', r'$\beta_{2,S}^{par}$', r'$\beta_{3,S}^{par}$',
              r'$\beta_{4,S}^{par}$', r'$\beta_{5,S}^{par}$']
labels_full = [r'$\beta_{1}^{full}$', r'$\beta_{2}^{full}$', r'$\beta_{3}^{full}$',
              r'$\beta_{4}^{full}$', r'$\beta_{5}^{full}$']
labels_carved = [r'$\beta_{1,I}^{car}$', r'$\beta_{2,I}^{car}$', r'$\beta_{3,I}^{car}$',
                 r'$\beta_{4,I}^{car}$', r'$\beta_{5,I}^{car}$']
vis3.visualise_acc_covariates5(v_dict7, titles, labels_H, width = 12, height = 6)

# Visualisation of H_0 rejection rate for HSIC-target for partial target (Xi = Id)
v_dict8 = {'eval_id': model_partial_subdict, 'eval_id12': model_partial_subdict,
           'eval_id23': model_partial_subdict, 'eval_id15': model_partial_subdict,
           'eval_id45': model_partial_subdict}
vis3.visualise_acc_covariates5(v_dict8, titles, labels_partial, width = 12, height = 6)