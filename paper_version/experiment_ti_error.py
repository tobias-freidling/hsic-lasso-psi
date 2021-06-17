import sys, os
sys.path.append(os.path.join(sys.path[0], 'paper_version', 'psi'))

from simulation import Data_Generator, Evaluator, Visualisation
from linear import Linear_Model
from poly_multi import Poly_Multi_HSIC
from hsic_lasso_psi import Split_HSIC_Lasso

"""
Evaluation of empirical type-I error for different data
generating processes, HSIC-estimators and targets.
"""
# 4 different data generating processes
n_jobs = 30
dg_log_id = Data_Generator(50, experiment = 'logistic', rho = 0,
                           decay = False, error_size = None,
                           noise_signal_ratio = None, n_jobs = n_jobs)
dg_log_corr = Data_Generator(50, experiment = 'logistic', rho = 0.5,
                             decay = True, error_size = None,
                             noise_signal_ratio = None, n_jobs = n_jobs)
dg_nonlin_id = Data_Generator(50, experiment = 'non_linear', rho = 0,
                              decay = False, error_size = None,
                              noise_signal_ratio = 0.2, n_jobs = n_jobs)
dg_nonlin_corr = Data_Generator(50, experiment = 'non_linear', rho = 0.5,
                                decay = True, error_size = None,
                                noise_signal_ratio = 0.2, n_jobs = n_jobs)

# 3 different HSIC-estimators: block (B=5), block (B=10), incomplete (l=1)
# for discrete and continuous reponse
targets = ['partial', 'H']
model_B5_disc = Split_HSIC_Lasso(targets, split_ratio = 0.25, n_screen = 50,
                                 adaptive_lasso = False, cv = True, M_B = 10,
                                 H_B = 5, discrete_output = True)
model_B10_disc = Split_HSIC_Lasso(targets, split_ratio = 0.25, n_screen = 50,
                                  adaptive_lasso = False, cv = True, M_B = 10,
                                  H_B = 10, discrete_output = True)
model_inc1_disc = Split_HSIC_Lasso(targets, split_ratio = 0.25, n_screen = 50,
                                   adaptive_lasso = False, cv = True,
                                   H_estimator = 'inc', H_l = 1, discrete_output = True)
model_B5_cont = Split_HSIC_Lasso(targets, split_ratio = 0.25, n_screen = 50,
                                 adaptive_lasso = False, cv = True, M_B = 10,
                                 H_B = 5, discrete_output = False)
model_B10_cont = Split_HSIC_Lasso(targets, split_ratio = 0.25, n_screen = 50,
                                  adaptive_lasso = False, cv = True, M_B = 10,
                                  H_B = 10, discrete_output = False)
model_inc1_cont = Split_HSIC_Lasso(targets, split_ratio = 0.25, n_screen = 50,
                                   adaptive_lasso = False, cv = True,
                                   H_estimator = 'inc', H_l = 1, discrete_output = False)


# Set-up for repeated simulation with Evaluator class
sample_sizes = [400, 800, 1200, 1600]
models_disc = [model_B5_disc, model_B10_disc, model_inc1_disc]
models_cont = [model_B5_cont, model_B10_cont, model_inc1_cont]
names_disc = ['model_B5_disc', 'model_B10_disc', 'model_inc1_disc']
names_cont = ['model_B5_cont', 'model_B10_cont', 'model_inc1_cont']

eval_log_id = Evaluator(models_disc, names_disc, rep = 100, dg = dg_log_id,
                        n_record_variables = 10, sample_sizes = sample_sizes,
                        alpha = 0.05, start_seed = 2021)
eval_log_corr = Evaluator(models_disc, names_disc, rep = 100, dg = dg_log_corr,
                          n_record_variables = 10, sample_sizes = sample_sizes,
                          alpha = 0.05, start_seed = 2021)
eval_nonlin_id = Evaluator(models_cont, names_cont, rep = 100, dg = dg_nonlin_id,
                           n_record_variables = 10, sample_sizes = sample_sizes,
                           alpha = 0.05, start_seed = 2021)
eval_nonlin_corr = Evaluator(models_cont, names_cont, rep = 100, dg = dg_nonlin_corr,
                             n_record_variables = 10, sample_sizes = sample_sizes,
                             alpha = 0.05, start_seed = 2021)


# Running simulation
n_jobs = 30

eval_log_id.simulation_parallel(n_jobs)
res_log_id = eval_log_id.model_results

eval_nonlin_id.simulation_parallel(n_jobs)
res_nonlin_id = eval_nonlin_id.model_results

eval_log_corr.simulation_parallel(n_jobs)
res_log_corr = eval_log_corr.model_results

eval_nonlin_corr.simulation_parallel(n_jobs)
res_nonlin_corr = eval_nonlin_corr.model_results

# Printing results
# evals = [eval_log_id, eval_log_corr, eval_nonlin_id, eval_nonlin_corr]
# for e in evals:
#     print(e.ti_error)
#     print('#################################')


# Visualisation of results
vis = Visualisation({'eval_log_id': eval_log_id, 'eval_log_corr': eval_log_corr,
                     'eval_nonlin_id': eval_nonlin_id, 'eval_nonlin_corr': eval_nonlin_corr})

sub_dict_disc = {'model_B5_disc': ['H'], 'model_B10_disc': ['H'],'model_inc1_disc': ['H']}
sub_dict_cont = {'model_B5_cont': ['H'], 'model_B10_cont': ['H'], 'model_inc1_cont': ['H']}
v_dict = {'eval_log_id': sub_dict_disc, 'eval_log_corr': sub_dict_disc,
          'eval_nonlin_id': sub_dict_cont, 'eval_nonlin_corr': sub_dict_cont}
label_dict = {'model_B5_disc': ['Block, B=5'], 'model_B10_disc': ['Block, B=10'],
              'model_inc1_disc': ['incomplete, l=1'], 'model_B5_cont': ['Block, B=5'],
              'model_B10_cont': ['Block, B=10'], 'model_inc1_cont': ['incomplete, l=1']}
titles = ['Logistic, Identity', 'Logistic, Decaying corr.', 'Mult., Identity', 'Mult., Decaying corr.']
vis.visualise_rates('ti-error', v_dict, titles, label_dict, width = 12, height = 3, display_titles = False)
