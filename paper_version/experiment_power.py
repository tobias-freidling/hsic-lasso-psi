import sys, os
sys.path.append(os.path.join(sys.path[0], 'paper_version', 'psi'))

from simulation import Data_Generator, Evaluator, Visualisation
from linear import Linear_Model
from poly_multi import Poly_Multi_HSIC
from hsic_lasso_psi import Split_HSIC_Lasso

"""
Evaluation of empirical power for different data
generating processes and HSIC-estimators and comparison with
Multi and PSI for a linear model.
"""
# Creating objects for data generation
thetas = [0, 0.33, 0.67, 1, 1.33, 1.67, 2, 2.33]
n_jobs = 30

def create_dg(thetas, experiment, noise_signal_ratio):
    dg = []
    for theta in thetas:
        dg.append(Data_Generator(50, experiment = experiment, rho = 0, decay = False,
                                  error_size = None, noise_signal_ratio = noise_signal_ratio,
                                  theta = theta, n_jobs = n_jobs))
    return dg

dg_log = create_dg(thetas, 'logistic', None)
dg_lin = create_dg(thetas, 'linear', 0.2)
dg_linmod = create_dg(thetas, 'non_linear2', 0.2) # linmod: linear modified


# Proposed method and Multi for two different estimators with discrete and
# continuous reponse variable and linear model
targets = ['H']
model_B10_disc = Split_HSIC_Lasso(targets, split_ratio = 0.25, n_screen = 50,
                                  adaptive_lasso = False, cv = True, M_B = 10,
                                  H_B = 10, discrete_output = True)
multi_B10_disc = Poly_Multi_HSIC(15, poly = False, estimator = 'block', B = 10,
                                 discrete_output = True, only_evaluate_first = True)
model_inc1_disc = Split_HSIC_Lasso(targets, split_ratio = 0.25, n_screen = 50,
                                   adaptive_lasso = False, cv = True, H_estimator = 'inc',
                                   H_l = 1, discrete_output = True)
multi_inc1_disc = Poly_Multi_HSIC(15, poly = False, estimator = 'inc', l = 1,
                                 discrete_output = True, only_evaluate_first = True)
model_B10_cont = Split_HSIC_Lasso(targets, split_ratio = 0.25, n_screen = 50,
                                  adaptive_lasso = False, cv = True, M_B = 10,
                                  H_B = 10, discrete_output = False)
multi_B10_cont = Poly_Multi_HSIC(15, poly = False, estimator = 'block', B = 10,
                                 discrete_output = False, only_evaluate_first = True)
model_inc1_cont = Split_HSIC_Lasso(targets, split_ratio = 0.25, n_screen = 50,
                                   adaptive_lasso = False, cv = True, H_estimator = 'inc',
                                   H_l = 1, discrete_output = False)
multi_inc1_cont = Poly_Multi_HSIC(15, poly = False, estimator = 'inc', l = 1,
                                  discrete_output = False, only_evaluate_first = True)
linear = Linear_Model(sigma = 10 * 0.2, reg_factor = 3)


# Set-up for repeated simulation with Evaluator class
sample_sizes = [400, 800, 1200, 1600]
models_disc = [model_B10_disc, model_inc1_disc, multi_B10_disc, multi_inc1_disc]
models_cont = [model_B10_cont, model_inc1_cont, multi_B10_cont, multi_inc1_cont, linear]
names_disc = ['model_B10_disc', 'model_inc1_disc', 'multi_B10_disc', 'multi_inc1_disc']
names_cont = ['model_B10_cont', 'model_inc1_cont', 'multi_B10_cont', 'multi_inc1_cont', 'linear']

def create_eval(dg, models, names):
    evals = []
    for d, t in zip(dg, thetas):
        # reduction of the size of repetitions with increasing theta to save computation time:
        # the larger theta is, the more likely it is that the first feature is selected and
        # we conduct a hypothesis test
        e = Evaluator(models, names, rep = 200 - int(50*t), dg = d, n_record_variables = 10,
                      sample_sizes = sample_sizes, alpha = 0.05, start_seed = 2021)
        evals.append(e)
    return evals

eval_log = create_eval(dg_log, models_disc, names_disc)
eval_lin = create_eval(dg_lin, models_cont, names_cont)
eval_linmod = create_eval(dg_linmod, models_cont, names_cont)


# Running simulation
n_jobs = 30
for e in eval_log:
    e.simulation_parallel(n_jobs)
for e in eval_lin:
    e.simulation_parallel(n_jobs)
for e in eval_linmod:
    e.simulation_parallel(n_jobs)

# Printing results
for e in eval_log:
    print(e.power)
for e in eval_lin:
    print(e.power)
for e in eval_linmod:
    print(e.power)


# Visualisation of results
tn = [int(t*100) for t in thetas]
names_log = []
for t in tn:
    names_log.append("eval_log_" + str(t))
names_lin = []
for t in tn:
    names_lin.append("eval_lin_" + str(t))
names_linmod = []
for t in tn:
    names_linmod.append("eval_linmod_" + str(t))
eval_comp = eval_log + eval_lin + eval_linmod
names_comp = names_log + names_lin + names_linmod

vis = Visualisation(dict(zip(names_comp, eval_comp)))

p_dict = {'plot_log': eval_log, 'plot_lin': eval_lin, 'plot_linmod': eval_linmod}

dict_log = dict(zip(names_disc, ['H' for i in range(4)]))
dict_lin = dict(zip(names_cont, ['H' for i in range(4)] + ['beta']))
dict_linmod = dict_lin
n_dict = {'plot_log': dict_log, 'plot_lin': dict_lin, 'plot_linmod': dict_linmod}

titles = ['(M1\')', '(M3)', '(M4)']

labels = ['Proposal, Block, B=10', 'Proposal, inc., l=1',
          'Multi, Block, B=10', 'Multi, inc., l=1']
label_dict = dict(zip(names_disc + names_cont, labels + labels + ['Linear']))

vis.visualise_power(thetas, p_dict, n_dict, titles, label_dict, width = 12,
                    height = 3.3, display_titles = False)
