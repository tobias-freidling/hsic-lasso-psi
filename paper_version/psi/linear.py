import sys, os
sys.path.append(os.path.join(sys.path[0], 'paper_version', 'psi'))

import numpy as np
import simulation as sim
from scipy.stats import truncnorm
from sklearn.linear_model import Lasso



class Linear_Model:
    """Post-selection inference with the Polyhedral Lemma for the partial regression
     coefficients in a linear model."""

    def __init__(self, sigma, reg_factor = 2):
        self.sigma = sigma # variance of response: sigma*Id
        self.reg_factor = reg_factor # used for calculation of regularisation parameter

    def selection_model(self, X, s, ind_sel, ind_nsel, lam):
        """ Selection event characterising the entire model {M = M_hat}
        :param s: sign-vector of the solution beta
        """
        Xs = X[:, ind_sel]
        Xns = X[:, ind_nsel]
        P = np.matmul(Xs, np.matmul(np.linalg.pinv(np.matmul(Xs.T, Xs)), Xs.T))
        C = np.matmul(Xs, np.linalg.pinv(np.matmul(Xs.T, Xs)))
        ones = np.ones(ind_nsel.shape[0])

        A_0 = np.matmul(Xns.T, (np.eye(X.shape[0]) - P)) / lam
        A_1 = - np.matmul(np.diag(s), np.matmul(np.linalg.pinv(np.matmul(Xs.T, Xs)), Xs.T))
        b_0 = np.matmul(Xns.T, np.matmul(C, s))
        b_1 = - lam * np.matmul(np.diag(s), np.matmul(np.linalg.pinv(np.matmul(Xs.T, Xs)), s))
        A = np.vstack((A_0, -A_0, A_1))
        b = np.hstack((ones - b_0, ones + b_0, b_1))
        return A, b


    def threshold_model(self, Y, A, b, eta):
        """Calculation of truncation points for selection event {AY<=b}
        """
        deno = self.sigma * eta.dot(eta)
        alpha = self.sigma * A.dot(eta) / deno
        assert(np.shape(A)[0] == np.shape(alpha)[0])
        pos_alpha_ind = np.argwhere(alpha > 0).flatten()
        neg_alpha_ind = np.argwhere(alpha < 0).flatten()
        acc = (b - np.matmul(A, Y)) / alpha + np.matmul(eta, Y)
        if (np.shape(neg_alpha_ind)[0] > 0):
            l_thres = np.max(acc[neg_alpha_ind])
        else:
            l_thres = -10.0**10
        if (np.shape(pos_alpha_ind)[0] > 0):
            u_thres = np.min(acc[pos_alpha_ind])
        else:
            u_thres = 10**10
        return l_thres, u_thres


    def get_p_val(self, stat, eta, l_thres, u_thres, mu = 0):
        """Calculation of one p-value for one-sided hypothesis test"""
        scale = np.sqrt(self.sigma * np.dot(eta, eta))
        p_val = truncnorm.sf(stat, (l_thres-mu) / scale, (u_thres-mu) / scale,
                             loc = mu, scale = scale)
        return p_val

    def sel_inf(self, X, Y, inf_type, alpha, niv, H0 = None, M0 = None, i = None):
        """Post-selection inference
        :param X, Y: covariate and response data
        :param inf_type: one-sided hypothesis testing or two-sided confidence interval calculation
        :param alpha: level 1-alpha
        :param niv: number of important variables, used for reporting results
        H0, M0 and i are not used
        """
        assert inf_type == 'test'
        n, p = X.shape

        eps = np.random.standard_normal((n, 10000)) * np.sqrt(self.sigma)
        # Negahban et al. 2012, optimal inference paper
        lam = self.reg_factor * np.mean(np.fabs((X.T).dot(eps)))
        # other method:
        #lam = np.sqrt(self.sigma) * np.sqrt(2 * np.log(p) / n)

        # Lasso optimisation
        reg = Lasso(alpha = lam / n, fit_intercept = False)
        reg.fit(X, Y)
        beta = reg.coef_
        ind_sel = np.where(np.abs(beta) > 1e-10)[0]
        ind_nsel = np.where(np.abs(beta) <= 1e-10)[0]
        # signs of selected variables
        s = np.sign(beta[ind_sel])

        # PSI
        p_values = np.ones(p)
        ind_h0_rej = np.zeros(p)
        A, b = self.selection_model(X, s, ind_sel, ind_nsel, lam)
        Xs = X[:, ind_sel]
        etas = np.matmul(np.linalg.pinv(np.matmul(Xs.T, Xs)), Xs.T)
        for i, index in enumerate(ind_sel):
            eta = etas[i,:]
            stat = np.dot(eta, Y)
            l_thres, u_thres = self.threshold_model(Y, A, b, eta)
            p_val = self.get_p_val(stat, eta, l_thres, u_thres)
            p_values[index] = p_val
            # p_val small -> reject null -> accept index
            ind_h0_rej[index] = 1 if p_val < alpha else 0

        # Reporting
        ind_sc_np = None
        ind_sel_np = np.zeros(p)
        ind_sel_np[ind_sel] = 1
        ind_h0_rej = {'beta' : ind_h0_rej}
        ind_h0_rej_true = np.zeros(p)
        ind_h0_rej_true[ind_sel] = 1
        ind_h0_rej_true[niv:] = 0
        ind_h0_rej_true = {'beta' : ind_h0_rej_true}
        p_values = {'beta' : p_values}

        inf_res = sim.Inference_Result(p, ind_sc_np, ind_sel_np, ind_h0_rej,
                                       ind_h0_rej_true, p_values, None)
        return inf_res
