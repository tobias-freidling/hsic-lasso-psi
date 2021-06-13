import numpy as np
from psi import helper
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
        """Calculation of true positive rate"""
        tpr = dict()
        for (t, h0_rej), (_, h0_rej_true)\
                in zip(self.ind_h0_rej.items(), self.ind_h0_rej_true.items()):
            N = np.sum(h0_rej_true)
            Z = np.sum(np.minimum(h0_rej_true, h0_rej))
            tpr[t] = Z / N
        return tpr    
    
    def fpr(self):
        """Calculation of false positive rate"""
        fpr = dict()
        for (t, h0_rej), (_, h0_rej_true)\
                in zip(self.ind_h0_rej.items(), self.ind_h0_rej_true.items()):
            
            N = np.maximum(np.sum(np.minimum(1 - h0_rej_true, self.ind_sel)), 1)
            Z = np.sum(np.minimum(1 - h0_rej_true, h0_rej))
            fpr[t] = Z / N
        return fpr

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
        print("p: ", self.p)
        if self.ind_sc is not None:
            print("screened indices: ", np.where(self.ind_sc == 1)[0])
        print("selected indices: ", np.where(self.ind_sel == 1)[0])
        for t, l in self.ind_h0_rej.items():
            print("target: {}, H_0-rejection indices: {}".format(t, np.where(l==1)[0]))
        for t, l in self.p_values.items():
            print("target: {}, p-values: {}".format(t, l))



class Data_Generator:
    """Artificial data generation for various settings"""
    
    def __init__(self, p, experiment, rho, decay, customize = False, combination = '12',
                 rho_comb = 0, error_size = None, noise_signal_ratio = None):
        assert experiment in ['logistic', 'linear', 'non_linear', 'tanh']
        self.p = p # number of covariates
        self.experiment = experiment # experiment to conduct
        self.rho = rho # strength of correlation
        self.decay = decay # (bool) use of decaying correlation
        self.customize = customize # (bool) use of correlated pair
        assert combination in ['12', '23', '15', '45']
        self.combination = combination # pair to correlate
        self.rho_comb = rho_comb # strength of pair correlation
        self.error_size = error_size
        self.noise_signal_ratio = noise_signal_ratio
        self.create_cov() # covariance generation
        self.set_H0_M0() # set true values of H and M 
    
    def number_inf_var(self):
        """Getting number of influential variables"""
        niv = 4 if self.experiment == 'non_linear' else 5
        return niv
    
    def set_H0_M0(self):
        """Calculation of the true H-vector and M-matrix, denoted by H0 and M0,
        by simulating a sample of size 20000"""
        self.H0 = np.zeros(self.p)
        self.M0 = np.zeros((self.p, self.p))
        niv = self.number_inf_var()
        X, Y = self.generate(20000, seed = 1)
        self.H0[:niv] = np.maximum(helper.estimate_H_unbiased_parallel(X[:,:niv], Y),
                                   np.zeros(niv))
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
        
        if self.customize:
            if self.combination == '12':
                c = np.maximum(helper.estimate_H_unbiased(X[:,0, np.newaxis], X[:,1]), 0)
                self.M0[0, 1] = self.M0[1, 0] = c
            elif self.combination == '23':
                c = np.maximum(helper.estimate_H_unbiased(X[:,1, np.newaxis], X[:,2]), 0)
                self.M0[1, 2] = self.M0[2, 1] = c
            elif self.combination == '15':
                c = np.maximum(helper.estimate_H_unbiased(X[:,0, np.newaxis], X[:,4]), 0)
                self.M0[0, 4] = self.M0[4, 0] = c
            elif self.combination == '45':
                c = np.maximum(helper.estimate_H_unbiased(X[:,3, np.newaxis], X[:,4]), 0)
                self.M0[3, 4] = self.M0[4, 3] = c
        self.M0 = helper.nearestPD(self.M0)
    
    def create_cov(self):
        if self.decay:
            def f(i, j):
                return (self.rho)**int(np.abs(i-j))
            l = [f(i, j) for i in range(self.p) for j in range(self.p)]
            self.cov = np.array(l).reshape(self.p, self.p)
        else:
            self.cov = (1-self.rho) * np.eye(self.p) + self.rho * np.ones((self.p, self.p))
        if self.customize:
            if self.combination == '12':
                self.cov[0, 1] = self.cov[1, 0] = self.rho_comb
            elif self.combination == '23':
                self.cov[1, 2] = self.cov[2, 1] = self.rho_comb
            elif self.combination == '15':
                self.cov[0, 4] = self.cov[4, 0] = self.rho_comb
            elif self.combination == '45':
                self.cov[3, 4] = self.cov[4, 3] = self.rho_comb
    
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
            x = np.sum(X[:,0:5], axis = 1)
            p = np.exp(x) / (1 + np.exp(x))
            Y = np.random.binomial(1, p)
        
        elif self.experiment == 'tanh': 
            assert self.error_size is not None
            Y = np.tanh(2 * np.sum(X[:,0:5], axis = 1)) \
                + self.error_size * np.random.randn(n)
            
        elif self.experiment == 'non_linear':
            if self.error_size is None and self.noise_signal_ratio is not None:
                error = np.sqrt(self.noise_signal_ratio) * 1.4 * np.random.randn(n)
            elif self.error_size is not None:
                error = self.error_size * np.random.randn(n)
            else:
                assert 0 == 1, "Specify size of error!"
            Y = (X[:,0]-1) * np.tanh(X[:,1] + X[:,2] + 1) + np.sign(X[:,3]) + error
            
        elif self.experiment == 'linear':
            if self.error_size is None and self.noise_signal_ratio is not None:
                error = np.sqrt(self.noise_signal_ratio * 5) * np.random.randn(n)
            elif self.error_size is not None:
                error = self.error_size * np.random.randn(n)
            else:
                assert 0 == 1, "Specify size of error!"
            Y = np.sum(X[:,:5], axis = 1) + error
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
        self.p = self.dg.p # number of features
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
    
    def simulation_parallel(self):   
        """Parallel simulation and evaluation of multiple datasets """
        tic = time.time()
        # parallel = Parallel(n_jobs = -1, prefer = 'threads')
        parallel = Parallel(n_jobs = -1)
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
        """Calculation of summaries for screening, selection, H_0 rejection, TPR and FPR"""
        self.tpr, self.fpr = dict(), dict()
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
            self.acc_summ[name] = dict() # or rather H_0 rejection summary
            tpr_model = [r_obj.tpr() for r_obj in self.model_results[name]]
            fpr_model = [r_obj.fpr() for r_obj in self.model_results[name]]
            acc_model = [r_obj.acceptance_summary() for r_obj in self.model_results[name]]
            
            for t in tpr_model[0].keys():
                # TPR
                tpr_model_t = [tpr_dict[t] for tpr_dict in tpr_model]
                temp = np.array(tpr_model_t).reshape(len(self.sample_sizes), self.rep)
                self.tpr[name][t] = np.mean(temp, axis = 1) # size: (ss,)
                # FPR
                fpr_model_t = [fpr_dict[t] for fpr_dict in fpr_model]
                temp = np.array(fpr_model_t).reshape(len(self.sample_sizes), self.rep)
                self.fpr[name][t] = np.mean(temp, axis = 1) # size: (ss,)
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
    """Visualisation of the results of one or multiple evaluators"""
    
    def __init__(self, evaluators):
        self.eval = evaluators # dictionary

    def visualise_selection(self, v_dict, titles, label_dict, width = 6, height = 4):
        """Visualising number of selected variables for different models and datasets"""
        l = len(v_dict)
        assert len(titles) == l
        fig, axes = plt.subplots(nrows = 1, ncols = l, sharey = True,
                                 **{'figsize': (width, height)})
        for i, (e, models) in enumerate(v_dict.items()):
            assert e in self.eval.keys()
            ax = axes if l == 1 else axes[i]
            
            for label, name in zip(label_dict[e], models):
                assert name in self.eval[e].names
                values = np.sum(self.eval[e].sel_summ[name], axis = 1)
                ax.plot(self.eval[e].sample_sizes, values, lw = 2,
                        marker = '.', label = label)
            if i == l-1:
                ax.legend(loc = 'upper left', bbox_to_anchor = (1.02, 1),
                          borderaxespad = 0., fontsize = 'large')
            ax.set_title(titles[i], {'fontsize' : 'large'})
            if i == 0:
                ax.set_ylabel("Number of selected variables", fontsize = 'large')
            ax.set_xlabel("sample size", fontsize = 'large')
            ax.set_ylim(bottom = 0)
        fig.tight_layout()
        fig.savefig("figure.eps")
        fig.show()
    
    
    def visualise_screening(self, v_dict, title, label_dict, width = 6, height = 4,
                            var_ex = 1, legend_position = 'lower right'):
        """Visualising number of selected variables for different models and datasets"""
        fig, ax = plt.subplots(figsize = (width, height))
        for e, models in v_dict.items():
            assert e in self.eval.keys()
            
            for label, name in zip(label_dict[e], models):
                assert name in self.eval[e].names
                values = np.mean(self.eval[e].sc_summ[name][:, :-var_ex], axis = 1)
                ax.plot(self.eval[e].sample_sizes, values, lw = 2,
                        marker = '.', label = label)
        ax.legend(loc = legend_position, fontsize = 'large')
        ax.set_title(title, {'fontsize' : 'large'})
        ax.set_xlabel("sample size", fontsize = 'large')
        ax.set_ylabel("True positive screening rate", fontsize = 'large')
        ax.set_ylim(bottom = 0)
        fig.savefig("figure.eps")
        fig.show()
    
    def visualise_selection_covariates5(self, v_dict, titles, labels, width = 10, height = 5):
        """Visualising the selection rate of single covariates for different models and 5 datasets"""
        l = len(v_dict)
        assert len(titles) == l
        fig, axes = plt.subplots(nrows = 2, ncols = 3, sharey = True,
                                 **{'figsize': (width, height)})
        cmap = plt.get_cmap("tab10")
        for i, (e, dict_m) in enumerate(v_dict.items()):
            assert e in self.eval.keys()
            row = i//3
            col = i%3
            ax = axes if l == 1 else axes[row, col]
            for name, covariates in dict_m.items():
                assert name in self.eval[e].names
                
                for j, c in enumerate(covariates):
                    ax.plot(self.eval[e].sample_sizes, self.eval[e].sel_summ[name][:, c],
                            lw = 2, marker = '.', color = cmap(j))
            ax.set_title(titles[i], {'fontsize' : 'large'})
            ax.set_ylim(bottom = 0)
        
        lines = [matplotlib.lines.Line2D([], [], color = cmap(i), marker = '.', lw = 2, label = l)\
                 for i, l in enumerate(labels)]
        axes[1,2].legend(handles = lines, loc = 'upper left', fontsize = 'large')
        axes[1,2].set_axis_off()
        fig.text(0.5, 0.05, 'sample size', ha = 'center', size = 'x-large')
        fig.text(0.05, 0.5, 'Selection rate', va = 'center',
                 rotation='vertical', size = 'x-large')
        fig.tight_layout()
        plt.subplots_adjust(left = 0.1, bottom = 0.13, right = 0.9)
        fig.savefig("figure.eps")
        fig.show()
    
    def visualise_rates(self, rate, v_dict, titles, label_dict, width = 10, height = 4):
        """Visualisation of true positive and true negative rates for different inference targets,
        models and datasets"""
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
            if i == l-1:
                ax.legend(loc = 'upper left', bbox_to_anchor = (1.02, 1),
                          borderaxespad = 0., fontsize = 'large')
            ax.set_title(titles[i], {'fontsize' : 'large'})
            if i == 0:
                if rate == 'tpr':
                    ax.set_ylabel("TPR", fontsize = 'x-large')
                elif rate == 'fpr':
                    ax.set_ylabel("FPR", fontsize = 'x-large')
            ax.set_xlabel("sample size", fontsize = 'large')
            ax.set_xlim(left = 0)
            ax.set_ylim(bottom = 0, top = 1.02)
        fig.tight_layout()
        fig.savefig("figure.eps")
        fig.show()    
    
    def visualise_acc_covariates5(self, v_dict, titles, labels, width = 10, height = 5):
        """Visualising the H_0 rejection rate of single covariates for different models and 5 datasets"""
        l = len(v_dict)
        assert len(titles) == l
        fig, axes = plt.subplots(nrows = 2, ncols = 3, sharey = True,
                                 **{'figsize': (width, height)})
        cmap = plt.get_cmap("tab10")
        
        for i, (e, dict_m_t) in enumerate(v_dict.items()):
            assert e in self.eval.keys()
            row = i//3
            col = i%3
            ax = axes if l == 1 else axes[row, col]
            for name, targets in dict_m_t.items():
                assert name in self.eval[e].names
                for t, covariates in targets.items():
                    for j, c in enumerate(covariates):
                        ax.plot(self.eval[e].sample_sizes, self.eval[e].acc_summ[name][t][:, c],
                                lw = 2, marker = '.', color = cmap(j))        
            
            ax.set_title(titles[i], {'fontsize' : 'large'})
            ax.set_ylim(bottom = 0)
        
        lines = [matplotlib.lines.Line2D([], [], color = cmap(i), marker = '.', lw = 2, label = l)\
                 for i, l in enumerate(labels)]
        axes[1,2].legend(handles = lines, loc = 'upper left', fontsize = 'large')
        axes[1,2].set_axis_off()
        fig.text(0.5, 0.05, 'sample size', ha = 'center', size = 'x-large')
        fig.text(0.05, 0.5, 'Acceptance rate', va = 'center',
                 rotation='vertical', size = 'x-large')
        fig.tight_layout()
        plt.subplots_adjust(left = 0.1, bottom = 0.13, right = 0.9)
        fig.savefig("figure.eps")
        fig.show()