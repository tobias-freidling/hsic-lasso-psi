import sys, os
sys.path.append(os.path.join(sys.path[0], 'paper_version', 'psi'))

import helper
import simulation as sim
import numpy as np
from scipy.stats import truncnorm
import bisect
from sklearn.linear_model import LassoCV, LassoLarsIC



class Screening:
    """Screening procedure on first fold"""

    def __init__(self, X, Y, estimator, B, ratio, n_screen, freq_dict, unbiased_parallel, n_jobs):
        self.X = X # covariate data
        self.p = self.X.shape[1]
        self.Y = Y # response data
        self.estimator = estimator # 'block' or  'inc'
        self.B = B # Block size
        self.ratio = ratio # size of inc. U-stats estimator
        self.n_screen = n_screen # number of screened variables
        self.freq_dict = freq_dict
        self.unbiased_parallel = unbiased_parallel # parallelised computation of unbiased estimates
        self.n_jobs = n_jobs # number of parallel processes
        self.ind_sc = None # indices of screened variables
        self.ind_nsc = None # indices of non-screened variables
        self.Hs = None # H with screened features
        self.M = None
        self.w = None # weights
        self.lam = None # regularisation parameter

    def calc_param(self, gamma, adaptive_lasso, cv, criterion):
        """Calculation of hyperparameters
        :param gamma: exponent for Adaptive Lasso
        :param adaptive_lasso: (bool) use of Adaptive Lasso
        :param cv: (bool) use of cross validation
        :param criterion: if cross validation not used, information criterion to be applied
        """
        # Screening
        if self.estimator == 'unbiased' and self.unbiased_parallel:
            H = helper.estimate_H_unbiased_parallel(self.X, self.Y, self.n_jobs, self.freq_dict)
        elif self.estimator == 'unbiased' and not self.unbiased_parallel:
            H = helper.estimate_H_unbiased(self.X, self.Y, self.freq_dict)
        else:
            _, H, _ = helper.estimate_H(self.X, self.Y, self.estimator, self.B, self.ratio, self.freq_dict)
        self.ind_sc = np.sort((np.argsort(H))[-self.n_screen:])
        self.ind_nsc = np.array([i for i in range(self.X.shape[1]) if i not in self.ind_sc])
        self.Hs = H[self.ind_sc]

        # Calculation of M
        if self.unbiased_parallel:
            _, self.M = helper.estimate_M_unbiased_parallel(self.X[:, self.ind_sc], self.n_jobs)
        else:
            _, self.M = helper.estimate_M(self.X[:,self.ind_sc], self.estimator, self.B, self.ratio)

        # Weights
        if adaptive_lasso:
            w = np.matmul(np.linalg.pinv(self.M), self.Hs)
            self.w = 1.0 / (np.abs(w)**gamma)
        else:
            self.w = np.ones(self.ind_sc.shape[0])

        # Decomposition: M = L*L^T; H = L * Y
        # Lasso problem: 0.5*||Y - L.T * beta||_2^2 + lambda ||beta||_1
        L = np.linalg.cholesky(self.M)
        Y = np.linalg.solve(L, self.Hs)
        # Calculation of lambda
        if cv:
            reg = LassoCV(fit_intercept = False, cv = 10, positive = True)
        else:
            reg = LassoLarsIC(criterion = criterion, fit_intercept = False, positive = True)
        reg.fit((L.T) / self.w, Y)
        self.lam = reg.alpha_ * self.Hs.shape[0]



class PSI:
    """Post-selection inference procedure on second fold"""

    def __init__(self, H, M, ind_sel, ind_nsel, beta, lam, w, cov):
        self.H = H
        self.p = self.H.shape[0]
        self.M = M
        self.ind_sel = ind_sel
        self.ind_nsel = ind_nsel
        self.beta = beta
        self.lam = lam
        self.w = w
        self.cov = cov # covariance matrix of H


    def selection_model(self):
        """ Selection event characterising the entire model {M = M_hat}
        """
        l = self.ind_sel.shape[0]
        k = self.ind_nsel.shape[0]
        M_1_inv = np.linalg.pinv(self.M[self.ind_sel,:][:,self.ind_sel])
        M_21 = np.matmul(self.M[self.ind_nsel,:][:,self.ind_sel], M_1_inv)

        A_1 = np.zeros((l, l+k))
        A_1[:, self.ind_sel] = M_1_inv
        A_1 = - A_1 / self.lam
        b_1 = - np.matmul(M_1_inv, self.w[self.ind_sel])

        A_2 = np.zeros((k, l+k))
        A_2[:, self.ind_nsel] = np.eye(k)
        A_2[:, self.ind_sel] = - M_21
        A_2 = A_2 / self.lam
        b_2 = self.w[self.ind_nsel] - np.matmul(M_21, self.w[self.ind_sel])

        A = np.vstack((A_1, A_2))
        b = np.hstack((b_1, b_2))
        return A, b


    def threshold_model(self, A, b, eta):
        """Calculation of truncation points for selection event {AY<=b}
        """
        Sigma_eta = self.cov.dot(eta)
        deno = Sigma_eta.dot(eta)
        alpha = A.dot(Sigma_eta) / deno
        assert(np.shape(A)[0] == np.shape(alpha)[0])
        pos_alpha_ind = np.argwhere(alpha > 0).flatten()
        neg_alpha_ind = np.argwhere(alpha < 0).flatten()
        acc = (b - np.matmul(A, self.H)) / alpha + np.matmul(eta, self.H)

        if (np.shape(neg_alpha_ind)[0] > 0):
            l_thres = np.max(acc[neg_alpha_ind])
        else:
            l_thres = -10.0**10

        if (np.shape(pos_alpha_ind)[0] > 0):
            u_thres = np.min(acc[pos_alpha_ind])
        else:
            u_thres = 10**10
        return l_thres, u_thres


    def threshold_variable(self, index, eta):
        """Calculation of truncation points for selection event {j in S}
        """
        beta_star = self.beta
        beta_star[index] = 0
        deno = np.dot(np.matmul(self.cov, eta), eta)
        C = np.matmul(self.cov, eta) / deno
        Z = np.matmul(np.eye(self.p) - np.outer(C, eta), self.H)
        l_thres = (np.matmul(self.M, beta_star)[index] + self.lam * self.w[index] \
                  - Z[index]) / C[index]
        u_thres = 10**10
        return l_thres, u_thres


    def get_p_val(self, stat, eta, l_thres, u_thres, mu = 0):
        """Calculation of one p-value for one-sided hypothesis test"""
        sigma = np.dot(eta, np.dot(self.cov, eta))
        scale = np.sqrt(sigma)
        p_val = truncnorm.sf(stat, (l_thres-mu) / scale, (u_thres-mu) / scale,
                             loc = mu, scale = scale)
        return p_val


    def get_ci(self, stat, eta, l_thres, u_thres, alpha):
        """Calculation of one two-sided confidence interval"""
        sigma = np.dot(eta, np.dot(self.cov, eta))
        scale = np.sqrt(sigma)
        pivot = lambda mu : truncnorm.cdf(stat, (l_thres-mu)/scale,
                                          (u_thres-mu)/scale, loc = mu, scale = scale)
        lb = stat - 20. * scale # lower bound
        ub = stat + 20. * scale # upper bound
        ci_l = helper.find_root(pivot, 1 - alpha/2, lb, ub)
        ci_u = helper.find_root(pivot, alpha/2, lb, ub)
        return np.array([ci_l, ci_u])


    def conf_int_partial(self, alpha):
        """Calculation of confidence intervals for partial targets."""
        conf_int = np.zeros((self.p, 2))
        A, b = self.selection_model()
        M_one = self.M[self.ind_sel,:][:,self.ind_sel]
        etas = np.linalg.pinv(M_one)
        for i, index in enumerate(self.ind_sel):
            eta = np.zeros(self.p)
            eta[self.ind_sel] = etas[i,:]
            stat = np.dot(eta, self.H)
            l_thres, u_thres = self.threshold_model(A, b, eta)
            conf_int[index, :] = self.get_ci(stat, eta, l_thres, u_thres, alpha)
        return conf_int

    def conf_int_H(self, alpha):
        """Calculation of confidence intervals for HSIC-targets."""
        conf_int = np.zeros((self.p, 2))
        for i, index in enumerate(self.ind_sel):
            eta = np.zeros(self.p)
            eta[index] = 1
            stat = self.H[index]
            l_thres, u_thres = self.threshold_variable(index, eta)
            conf_int[index, :] = self.get_ci(stat, eta, l_thres, u_thres, alpha)
        return conf_int

    def test_partial(self, alpha, H0, M0):
        """Hypothesis testing for partial targets
        :param alpha: level
        :param H0: true H-values
        :param M0: true M-values
        :returns: (nd-array) ind_h0_rej: indices where H_0 was rejected
                      (nd-array) ind_h0_rej_true: indices where H_0 was rightfully rejected
                      (nd-array) p_values: p-values of the tests
        """
        eps = 5e-7
        p_values = -np.ones(self.p) # -1 signifies that p-value was not calculated
        ind_h0_rej = np.zeros(self.p)
        ind_h0_rej_true = np.zeros(self.p)
        A, b = self.selection_model()
        M_one = self.M[self.ind_sel,:][:,self.ind_sel]
        etas = np.linalg.pinv(M_one)
        M_one0 = M0[self.ind_sel,:][:,self.ind_sel]
        etas0 = np.linalg.pinv(M_one0)
        for i, index in enumerate(self.ind_sel):
            eta = np.zeros(self.p)
            eta[self.ind_sel] = etas[i,:]
            stat = np.dot(eta, self.H)
            l_thres, u_thres = self.threshold_model(A, b, eta)
            p_val = self.get_p_val(stat, eta, l_thres, u_thres)
            p_values[index] = p_val
            # p_val small -> reject null -> accept index
            ind_h0_rej[index] = 1 if p_val < alpha else 0
            eta0 = np.zeros(self.p)
            eta0[self.ind_sel] = etas0[i,:]
            stat_true = np.dot(eta0, H0)
            ind_h0_rej_true[index] = 1 if stat_true > eps else 0
        return ind_h0_rej, ind_h0_rej_true, p_values


    def test_H(self, alpha, H0, M0):
        """Hypothesis testing for HSIC-targets"""
        eps = 5e-7
        p_values = -np.ones(self.p)
        ind_h0_rej = np.zeros(self.p)
        ind_h0_rej_true = np.zeros(self.p)
        for i, index in enumerate(self.ind_sel):
            eta = np.zeros(self.p)
            eta[index] = 1
            stat = self.H[index]
            l_thres, u_thres = self.threshold_variable(index, eta)
            p_val = self.get_p_val(stat, eta, l_thres, u_thres)
            p_values[index] = p_val
            # p_val small -> reject null -> accept index
            ind_h0_rej[index] = 1 if p_val < alpha else 0
            stat_true = H0[index]
            ind_h0_rej_true[index] = 1 if stat_true > eps else 0
        return ind_h0_rej, ind_h0_rej_true, p_values



class Split_HSIC_Lasso:
    """Post-selection inference for HSIC-Lasso by splitting the data into two folds"""

    def __init__(self, targets, split_ratio, n_screen, adaptive_lasso = False, cv = True,
                 gamma = 1.5, criterion = 'aic', cov_mode = 'oas',
                 screen_estimator = 'unbiased', screen_B = 10, screen_l = 1,
                 M_estimator = 'block', M_B = 10, M_l = 1,
                 H_estimator = 'block', H_B = 10, H_l = 1, discrete_output = False):
        self.targets = targets # list of inference targets
        self.split_ratio = split_ratio # ratio of data used for first fold
        self.n_screen = n_screen # number of screened variables
        self.adaptive_lasso = adaptive_lasso
        self.cv = cv
        self.gamma = gamma
        self.criterion = criterion
        self.cov_mode = cov_mode # covariance calculation method
        self.screen_estimator = screen_estimator # estimator used for screening
        self.screen_B = screen_B
        self.screen_l = screen_l
        self.M_estimator = M_estimator # estimator used for the calculation of M
        self.M_B = M_B
        self.M_l = M_l
        self.H_estimator = H_estimator # estimator used for the calculation of M
        self.H_B = H_B
        self.H_l = H_l
        self.discrete_output = discrete_output # use of delta kernel for output


    def sel_inf(self, X, Y, inf_type, alpha, niv, H0, M0, i = None, unbiased_parallel = False, n_jobs = 20):
        """Post-selection inference
        :param X, Y: covariate and response data
        :param inf_type: one-sided hypothesis testing or two-sided confidence interval calculation
        :param alpha: level 1-alpha
        :param niv: number of important variables, used for reporting results
        :param H0: true value of H-vector (only known for artifical data)
        :param M0: true value of M-vector (only known for artifical data)
        :param i: not used here
        :param unbiased_parallel: parallelised computation of unbiased estimates
        :param n_jobs: number of parallel processes
        """
        assert inf_type in ['test', 'ci']
        p = X.shape[1]

        if self.discrete_output:
            # Determine number of occurrences of different values
            # to define the Delta kernel
            values, counts = np.unique(Y, return_counts = True)
            freq_dict = dict(zip(values, counts))
        else:
            freq_dict = None

        # Splitting
        split_point = int(np.floor(X.shape[0] * self.split_ratio))
        ind1 = np.arange(0, split_point)
        ind2 = np.arange(split_point, X.shape[0])

        # Get parameters from screening on first fold
        screen = Screening(X[ind1, :], Y[ind1], self.screen_estimator, self.screen_B,
                           self.screen_l, self.n_screen, freq_dict, unbiased_parallel, n_jobs)
        screen.calc_param(self.gamma, self.adaptive_lasso, self.cv, self.criterion)
        ind_sc = screen.ind_sc
        w = screen.w
        lam = screen.lam * np.sqrt((1-self.split_ratio) / self.split_ratio)

        # Calculate H, M, covariance matrix and betas
        H_estimates, H, m = helper.estimate_H(X[ind2,:][:, ind_sc], Y[ind2], self.H_estimator,
                                              self.H_B, self.H_l, freq_dict)
        if self.M_estimator == 'unbiased' and unbiased_parallel:
            _, M = helper.estimate_M_unbiased_parallel(X[ind2,:][:, ind_sc], n_jobs)
        elif self.M_estimator =='unbiased' and not unbiased_parallel:
            _, M = helper.estimate_M_unbiased(X[ind2,:][:, ind_sc])
        else:
            _, M = helper.estimate_M(X[ind2,:][:, ind_sc], self.M_estimator, self.M_B, self.M_l)
        cov = helper.covariance(H_estimates, m, self.cov_mode)
        beta, ind_sel, ind_nsel = helper.lasso_sol(M, H, lam, w)

        # Reporting
        ind_sc_np = np.zeros(p)
        ind_sc_np[ind_sc] = 1
        ind_sel_np = np.zeros(p)
        ind_sel_np[ind_sc[ind_sel]] = 1

        # PSI on second fold
        psi = PSI(H, M, ind_sel, ind_nsel, beta, lam, w, cov)
        pot_targets = ['partial', 'H']
        if inf_type == 'test':
            ind_h0_rej_d = dict()
            ind_h0_rej_true_d = dict()
            p_values_d = dict()
            test_methods = [psi.test_partial, psi.test_H]
            for t, test in zip(pot_targets, test_methods):
                ind_h0_rej, ind_h0_rej_true, p_values = test(alpha, H0[ind_sc], M0[ind_sc,:][:,ind_sc])
                # results into dictionaries
                ind_h0_rej_full = np.zeros(p)
                ind_h0_rej_full[ind_sc] = ind_h0_rej
                ind_h0_rej_d[t] = ind_h0_rej_full

                ind_h0_rej_true_full = np.zeros(p)
                ind_h0_rej_true_full[ind_sc] = ind_h0_rej_true
                ind_h0_rej_true_d[t] = ind_h0_rej_true_full

                p_values_full = np.ones(p)
                p_values_full[ind_sc] = p_values
                p_values_d[t] = p_values_full
            return sim.Inference_Result(p, ind_sc_np, ind_sel_np, ind_h0_rej_d,
                                        ind_h0_rej_true_d, p_values_d, None)
        else:
            conf_int_d = dict()
            ci_methods = [psi.conf_int_partial, psi.conf_int_H]
            for t, conf_int in zip(pot_targets, ci_methods):
                conf_int = conf_int(alpha)
                # result into dictionary - confidence intervals
                conf_int_full = np.zeros((p, 2))
                conf_int_full[ind_sc, :] = conf_int
                conf_int_d[t] = conf_int_full
            return sim.Inference_Result(p, ind_sc_np, ind_sel_np, None, None, None, conf_int_d)
