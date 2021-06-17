import sys, os
sys.path.append(os.path.join(sys.path[0], 'paper_version', 'psi'))

import numpy as np
import helper
from joblib import Parallel, delayed
import time
import matplotlib.pyplot as plt
import matplotlib

class Inference_Result:
    """Collection of the results of the post-selection inference procedure on one dataset"""

    def __init__(self, p, ind_sc, ind_sel, ind_h0_rej, ind_h0_rej_true, p_values, conf_int):
        self.p = p # number of features
        self.ind_sc = ind_sc # screened indices
        self.ind_sel = ind_sel # selected indices
        self.ind_h0_rej = ind_h0_rej # (dict) indices with rejected null hypothesis
        self.ind_h0_rej_true = ind_h0_rej_true # (dict) indices with rightfully rejected null hypothesis
        self.p_values = p_values # (dict) p-values of inference targets
        self.conf_int = conf_int # (dict) confidence intervals for inference targets
        self.nrv = None # for the first nrv variables results are reported feature by feature

    def set_record_variables(self, nrv):
        self.nrv = nrv

    def tpr(self):
        """Calculation of true positive rate for experiment"""
        tpr = dict()
        for (t, h0_rej), (_, h0_rej_true)\
                in zip(self.ind_h0_rej.items(), self.ind_h0_rej_true.items()):
            N = np.maximum(np.sum(h0_rej_true), 1)
            Z = np.sum(np.minimum(h0_rej_true, h0_rej))
            tpr[t] = Z / N
        return tpr

    def fpr(self):
        """Calculation of false positive rate (FPR) for experiment"""
        fpr = dict()
        for (t, h0_rej), (_, h0_rej_true)\
                in zip(self.ind_h0_rej.items(), self.ind_h0_rej_true.items()):

            N = np.maximum(np.sum(np.minimum(1 - h0_rej_true, self.ind_sel)), 1)
            Z = np.sum(np.minimum(1 - h0_rej_true, h0_rej))
            fpr[t] = Z / N
        return fpr

    def fpr_Z(self):
        """Numerator(s) of FPR for experiment"""
        fpr_Z = dict()
        for (t, h0_rej), (_, h0_rej_true)\
                in zip(self.ind_h0_rej.items(), self.ind_h0_rej_true.items()):
            fpr_Z[t] = np.sum(np.minimum(1 - h0_rej_true, h0_rej))
        return fpr_Z

    def fpr_N(self):
        """Denominator(s) of FPR for experiment"""
        fpr_N = dict()
        for (t, h0_rej), (_, h0_rej_true)\
                in zip(self.ind_h0_rej.items(), self.ind_h0_rej_true.items()):
            fpr_N[t] = np.maximum(np.sum(np.minimum(1 - h0_rej_true, self.ind_sel)), 1)
        return fpr_N

    def first_Z(self):
        """Determining whether first feature was selected and found significant;
        Used for simulation of empirical power
        """
        first_Z = dict()
        for (t, h0_rej), (_, h0_rej_true)\
                in zip(self.ind_h0_rej.items(), self.ind_h0_rej_true.items()):
            first_Z[t] = np.minimum(self.ind_sel[0], h0_rej[0])
        return first_Z

    def first_N(self):
        """Determining whether first feature was selected;
        Used for simulation of empirical power
        """
        first_N = dict()
        for (t, h0_rej), (_, h0_rej_true)\
                in zip(self.ind_h0_rej.items(), self.ind_h0_rej_true.items()):
            first_N[t] = self.ind_sel[0]
        return first_N

    def screening_summary(self):
        assert self.nrv is not None
        assert self.ind_sc is not None
        return np.append(self.ind_sc[:self.nrv], np.sum(self.ind_sc[self.nrv:]))

    def selection_summary(self):
        assert self.nrv is not None
        return np.append(self.ind_sel[:self.nrv], np.sum(self.ind_sel[self.nrv:]))

    def acceptance_summary(self):
        assert self.nrv is not None
        acc_dict = dict()
        for t, h0_rej in self.ind_h0_rej.items():
            acc_dict[t] = np.append(h0_rej[:self.nrv], np.sum(h0_rej[self.nrv:]))
        return acc_dict

    def print_summary(self):
        """Printing the screened and selected indices as well as
        outcome of hypothesis tests.
        """
        print("p: ", self.p)
        if self.ind_sc is not None:
            print("screened indices: ", np.where(self.ind_sc == 1)[0])
        print("selected indices: ", np.where(self.ind_sel == 1)[0])
        for t, l in self.ind_h0_rej.items():
            print("target: {}, H_0-rejection indices: {}" \
                  .format(t, np.where(l==1)[0]))
        for t, l in self.p_values.items():
            print("target: {}, p-values: {}".format(t, l))



class Data_Generator:
    """Artificial data generation for various settings"""

    def __init__(self, p, experiment, rho, decay, error_size = None,
                 noise_signal_ratio = None, theta = 1, n_jobs = 20):
        assert experiment in ['logistic', 'linear',
                              'non_linear', 'non_linear2']
        self.p = p # number of covariates
        self.experiment = experiment # experiment to conduct
        self.rho = rho # strength of correlation
        self.decay = decay # (bool) use of decaying correlation
        self.niv = 10 # number of influential variables (first niv features)
        self.theta = theta # degree of influence of the first feature in power simulation
        self.error_size = error_size
        self.noise_signal_ratio = noise_signal_ratio
        # number of parallel processes for estimating H0 and M0
        self.n_jobs = n_jobs
        self.create_cov() # covariance generation
        self.set_H0_M0() # set true values of H and M


    def number_inf_var(self):
        return self.niv


    def set_H0_M0(self):
        """Calculation of the true H-vector and M-matrix, denoted by H0 and M0,
        by simulating a sample of size 10000"""
        self.H0 = np.zeros(self.p)
        self.M0 = np.zeros((self.p, self.p))
        X, Y = self.generate(2000, seed = 1)
        self.H0[:self.niv] = np.maximum(helper\
                                        .estimate_H_unbiased_parallel(X[:,:self.niv], Y,
                                                                      n_jobs = self.n_jobs),
                                        np.zeros(self.niv))
        if self.decay:
            eps = 1e-8
            self.M0 = np.zeros((self.p, self.p))
            # row estimation
            row = np.zeros(self.p)
            i, cont = 0, True
            while i < self.p and cont:
                m = helper.estimate_H_unbiased(X[:,0,np.newaxis], X[:,i])
                if m > eps:
                    row[i] = m
                    i += 1
                else:
                    cont = False
            for i in range(self.p):
                for j in range(self.p):
                    self.M0[i, j] = row[int(abs(i-j))]
        else:
            d = np.maximum(helper.estimate_H_unbiased(X[:,0, np.newaxis], X[:,0]), 0)
            if self.rho > 0:
                b = np.maximum(helper.estimate_H_unbiased(X[:,0, np.newaxis], X[:,1]), 0)
            else:
                b = 0
            self.M0 = b * np.ones((self.p, self.p))
            np.fill_diagonal(self.M0, d)
        self.M0 = helper.nearestPD(self.M0)


    def create_cov(self):
        if self.decay:
            def f(i, j):
                return (self.rho)**int(np.abs(i-j))
            l = [f(i, j) for i in range(self.p) for j in range(self.p)]
            self.cov = np.array(l).reshape(self.p, self.p)
        else:
            self.cov = (1-self.rho) * np.eye(self.p) + self.rho * np.ones((self.p, self.p))


    def generate(self, n, seed, p = None):
        """Generation of artifical data
        :param n: sample size
        :param seed: seed for random data generation
        """
        if p is None:
            p = self.p
        mean = np.zeros(p)
        np.random.seed(seed)
        X = np.random.multivariate_normal(mean, self.cov[:,:p][:p,:], n)

        if self.experiment == 'logistic':
            x = self.theta * X[:,0] + np.sum(X[:,1:self.niv], axis = 1)
            p = np.exp(x) / (1 + np.exp(x))
            Y = np.random.binomial(1, p)

        elif self.experiment == 'non_linear':
            if self.error_size is None and self.noise_signal_ratio is not None:
                error = np.sqrt(self.noise_signal_ratio) * 2.23 * np.random.randn(n)
            elif self.error_size is not None:
                error = self.error_size * np.random.randn(n)
            else:
                assert 0 == 1, "Specify size of error!"
            temp = np.zeros(n)
            for i in range(5):
                temp += X[:,i] * X[:,i+5]
            Y = temp + error

        elif self.experiment == 'linear':
            if self.error_size is None and self.noise_signal_ratio is not None:
                error = np.sqrt(self.noise_signal_ratio * self.niv) * np.random.randn(n)
            elif self.error_size is not None:
                error = self.error_size * np.random.randn(n)
            else:
                assert 0 == 1, "Specify size of error!"
            Y = self.theta * X[:,0] + np.sum(X[:,1:self.niv], axis = 1) + error

        elif self.experiment == 'non_linear2':
            if self.error_size is None and self.noise_signal_ratio is not None:
                error = np.sqrt(self.noise_signal_ratio * self.niv) * np.random.randn(n)
            elif self.error_size is not None:
                error = self.error_size * np.random.randn(n)
            else:
                assert 0 == 1, "Specify size of error!"
            Y = self.theta * (X[:,0] - X[:,0]**3) + np.sum(X[:,1:self.niv], axis = 1) + error

        return X, Y



# Method outside of class needed for Parallel-framework
def one_simulation_outside(self, r, i, n):
    return Evaluator.one_simulation(self, r, i, n)

class Evaluator:
    """Conducting and evaluating experiments for different settings on artifical data"""

    def __init__(self, models, names, rep, dg, n_record_variables,
                 sample_sizes, alpha, start_seed):
        self.models = models # list of models, e.g. Split_HSIC_lasso
        self.names = names # names of models
        self.rep = rep # number of simulated datasets for each sample size
        self.dg = dg # data generator
        self.nrv = n_record_variables # the first nrv variables are recorded one by one
        self.sample_sizes = sample_sizes # list of sample_sizes
        self.alpha = alpha
        self.start_seed = start_seed

    def one_simulation(self, r, i, n):
        # Data generation
        X, Y = self.dg.generate(n, self.start_seed + i * self.rep + r)
        niv = self.dg.number_inf_var()
        H0, M0 = self.dg.H0, self.dg.M0
        results = dict()
        for model, name in zip(self.models, self.names):
            inf_res = model.sel_inf(X, Y, 'test', self.alpha, niv, H0, M0, i)
            inf_res.set_record_variables(self.nrv)
            results[name] = inf_res
        return results

    def simulation_parallel(self, n_jobs = 20):
        """Parallel simulation and evaluation of multiple datasets """
        tic = time.time()
        # For debugging
        # parallel = Parallel(n_jobs = n_jobs, prefer = 'threads')
        parallel = Parallel(n_jobs = n_jobs)
        res = parallel(delayed(one_simulation_outside)(self, r, i, n)\
                       for (i, n) in enumerate(self.sample_sizes) for r in range(self.rep))
        toc = time.time()
        self.overall_time = toc - tic
        # Reporting
        self.model_results  = dict()
        for name in self.names:
            self.model_results[name] = [res_dict[name] for res_dict in res]
        self.calc_summaries()

    def calc_summaries(self):
        """Calculation of summaries for screening, selection, H_0 rejection,
        TPR, FPR, power and type-I error
        Note: FPR is estimated by averaging the false positive rate over all simulations;
        the type-I error is the fraction of erroneously rejected null hypotheses
        in all tests and simulations"""
        self.tpr, self.fpr = dict(), dict()
        self.ti_error, self.power = dict(), dict()
        self.sel_summ, self.sc_summ, self.acc_summ = dict(), dict(), dict()
        for name in self.names:
            # selection
            sel_summ_model = [r_obj.selection_summary() for r_obj in self.model_results[name]]
            temp = np.array(sel_summ_model).reshape(len(self.sample_sizes), self.rep, -1)
            self.sel_summ[name] = np.mean(temp, axis = 1) # size: (ss, variables)

            # screening
            if self.model_results[name][0].ind_sc is None:
                self.sc_summ[name] = None
            else:
                sc_summ_model = [r_obj.screening_summary() for r_obj in self.model_results[name]]
                temp = np.array(sc_summ_model).reshape(len(self.sample_sizes), self.rep, -1)
                self.sc_summ[name] = np.mean(temp, axis = 1) # size: (ss, variables)

            # target-dependent metrics
            self.tpr[name], self.fpr[name] = dict(), dict()
            self.ti_error[name], self.power[name] = dict(), dict()
            self.acc_summ[name] = dict() # or rather H_0 rejection summary
            tpr_model = [r_obj.tpr() for r_obj in self.model_results[name]]
            fpr_model = [r_obj.fpr() for r_obj in self.model_results[name]]
            acc_model = [r_obj.acceptance_summary() for r_obj in self.model_results[name]]
            ti_N = [r_obj.fpr_N() for r_obj in self.model_results[name]]
            ti_Z = [r_obj.fpr_Z() for r_obj in self.model_results[name]]
            first_N = [r_obj.first_N() for r_obj in self.model_results[name]]
            first_Z = [r_obj.first_Z() for r_obj in self.model_results[name]]

            for t in tpr_model[0].keys():
                # TPR
                tpr_model_t = [tpr_dict[t] for tpr_dict in tpr_model]
                temp = np.array(tpr_model_t).reshape(len(self.sample_sizes), self.rep)
                self.tpr[name][t] = np.mean(temp, axis = 1) # size: (ss,)
                # FPR
                fpr_model_t = [fpr_dict[t] for fpr_dict in fpr_model]
                temp = np.array(fpr_model_t).reshape(len(self.sample_sizes), self.rep)
                self.fpr[name][t] = np.mean(temp, axis = 1) # size: (ss,)
                # type-I error
                ti_N_t = [ti_N_dict[t] for ti_N_dict in ti_N]
                ti_Z_t = [ti_Z_dict[t] for ti_Z_dict in ti_Z]
                temp_N = np.array(ti_N_t).reshape(len(self.sample_sizes), self.rep)
                temp_Z = np.array(ti_Z_t).reshape(len(self.sample_sizes), self.rep)
                self.ti_error[name][t] = np.sum(temp_Z, axis = 1) / np.sum(temp_N, axis = 1)

                # power
                power_N_t = [first_N_dict[t] for first_N_dict in first_N]
                power_Z_t = [first_Z_dict[t] for first_Z_dict in first_Z]
                temp_N = np.array(power_N_t).reshape(len(self.sample_sizes), self.rep)
                temp_Z = np.array(power_Z_t).reshape(len(self.sample_sizes), self.rep)
                self.power[name][t] = np.sum(temp_Z, axis = 1) / np.sum(temp_N, axis = 1)

                # acceptance summary
                acc_model_t = [acc_dict[t] for acc_dict in acc_model]
                temp = np.array(acc_model_t).reshape(len(self.sample_sizes), self.rep, -1)
                self.acc_summ[name][t] = np.mean(temp, axis = 1) # size: (ss, variables)


    def print_comp_time(self):
        print('Sample sizes:')
        print(self.sample_sizes)
        print('----------------')
        print('Overall computation time: ', self.overall_time)


    def print_rates(self):
        for name in self.names:
            print(name)
            print('True positive rate:')
            for t, tpr in self.tpr[name].items():
                print("{}: {}".format(t, tpr))
            print('-------------------------')
            print('False positive rate:')
            for t, fpr in self.fpr[name].items():
                print("{}: {}".format(t, fpr))
            print('-------------------------')
            print('Type-I error:')
            for t, ti in self.ti_error[name].items():
                print("{}: {}".format(t, ti))
            print('-------------------------')
            print('Power:')
            for t, po in self.power[name].items():
                print("{}: {}".format(t, po))
            print('#################################')

    def print_summaries(self):
        for name in self.names:
            print(name)
            if self.sc_summ[name] is not None:
                print('Screening summary:')
                s = self.sc_summ[name].shape[1]
                for j in range(s-1):
                    print("X{}: {}".format(j+1, self.sc_summ[name][:,j]))
                print("X_rest: {}".format(self.sc_summ[name][:,s-1]))
                print('-------------------------')
            print('Selection summary:')
            s = self.sel_summ[name].shape[1]
            for j in range(s-1):
                print("X{}: {}".format(j+1, self.sel_summ[name][:,j]))
            print("X_rest: {}".format(self.sel_summ[name][:,s-1]))
            print('-------------------------')
            print('Acceptance summary:')
            acc_dict = self.acc_summ[name]
            for t, val in acc_dict.items():
                print('***Target: {}***'.format(t))
                s = val.shape[1]
                for j in range(s-1):
                    print("X{}: {}".format(j+1, val[:,j]))
                print("X_rest: {}".format(val[:,s-1]))
            print('#################################')



class Visualisation:
    """Visualisation of power and type-I error of one or multiple evaluators"""

    def __init__(self, evaluators):
        self.eval = evaluators # dictionary

    def visualise_rates(self, rate, v_dict, titles, label_dict,
                        width = 10, height = 4, display_titles = True):
        """Visualisation of TPR, FPR and type-I error
        for different inference targets, models and datasets
        """
        l = len(v_dict)
        #assert l > 1
        assert len(titles) == l
        fig, axes = plt.subplots(nrows = 1, ncols = l, sharey = True,
                                 **{'figsize': (width, height)})
        for i, (e, dict_m) in enumerate(v_dict.items()):
            assert e in self.eval.keys()
            ax = axes if l == 1 else axes[i]
            for name, targets in dict_m.items():
                assert name in self.eval[e].names

                for j, t in enumerate(targets):
                    if rate == 'tpr':
                        ax.plot(self.eval[e].sample_sizes, self.eval[e].tpr[name][t],
                                lw = 2, marker = '.', label = label_dict[name][j])
                    elif rate == 'fpr':
                        ax.plot(self.eval[e].sample_sizes, self.eval[e].fpr[name][t],
                                lw = 2, marker = '.', label = label_dict[name][j])
                    elif rate == 'ti-error':
                        ax.plot(self.eval[e].sample_sizes, self.eval[e].ti_error[name][t],
                                lw = 2, marker = '.', label = label_dict[name][j])
            if i == l-1:
                ax.legend(loc = 'upper left', bbox_to_anchor = (1.02, 1),
                          borderaxespad = 0., fontsize = 'large')
            if display_titles:
                ax.set_title(titles[i], {'fontsize' : 'large'})
            if i == 0:
                if rate == 'tpr':
                    ax.set_ylabel("TPR", fontsize = 'x-large')
                elif rate == 'fpr':
                    ax.set_ylabel("FPR", fontsize = 'x-large')
                elif rate == 'ti-error':
                    ax.set_ylabel("type-I error", fontsize = 'x-large')
            ax.set_xlabel("sample size", fontsize = 'large')
            ax.set_ylim(bottom = 0)
        fig.tight_layout()
        fig.savefig("figure.eps")
        fig.savefig("figure.pdf")
        fig.show()

    def visualise_power(self, thetas, p_dict, n_dict, titles, label_dict,
                         width = 10, height = 4, display_titles = True):
        """Visualisation of power of detecting the first feature
        for different thetas
        """
        l = len(p_dict)
        assert len(titles) == l
        fig, axes = plt.subplots(nrows = 1, ncols = l, sharey = True,
                                 **{'figsize': (width, height)})
        for i, (p, evaluators) in enumerate(p_dict.items()):
            ax = axes if l == 1 else axes[i]
            dict_m = n_dict[p]
            for name, target in dict_m.items():
                power_values = [e.power[name][target] for e in evaluators]
                ax.plot(thetas, power_values,
                        lw = 2, marker = '.', label = label_dict[name])
            if i == l-1:
                ax.legend(loc = 'upper left', bbox_to_anchor = (1.02, 1),
                          borderaxespad = 0., fontsize = 'large')
            if display_titles:
                ax.set_title(titles[i], {'fontsize' : 'large'})
            if i == 0:
                ax.set_ylabel("power", fontsize = 'x-large')
            ax.set_xlabel("theta", fontsize = 'large')
            ax.set_ylim(bottom = -0.05)
        fig.tight_layout()
        fig.savefig("figure-power.eps")
        fig.savefig("figure-power.pdf")
        fig.show()
