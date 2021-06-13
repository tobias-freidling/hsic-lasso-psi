import numpy as np
from psi import KDiscrete
from psi import simulation as sim
from mskernel import hsic, util, kernel
from mskernel.featsel import MultiSel, PolySel



class Poly_Multi_HSIC:
    """Post-selection inference with the Polyhedral Lemma and Multiscale bootstrapping
    for HSIC-ordering selection; Wrapper for implementation in mskernel """
    
    def __init__(self, n_select, poly, estimator = 'block', B = 10, l = 1, discrete_output = False):
        self.n_select = n_select # number of selected features
        self.poly = poly # (bool) use of Polyhedral Lemma framework
        self.estimator = estimator
        self.B = B
        self.l = l
        self.discrete_output = discrete_output
    
    
    def sel_inf(self, X, Y, inf_type, alpha, niv, H0 = None, M0 = None, i = None):
        """Post-selection inference
        :param X, Y: covariate and response data
        :param inf_type: one-sided hypothesis testing or two-sided confidence interval calculation
        :param alpha: level 1-alpha
        :param niv: number of important variables, used for reporting results
        H0, M0 and i are not used
        """
        assert inf_type == 'test'
        p = X.shape[1]
        # Initialising kernels
        x_bw = util.meddistance(X, subsample = 1000)**2
        kx = kernel.KGauss(x_bw)
        if self.discrete_output:
            ky = KDiscrete()
        else:
            y_bw = util.meddistance(Y[:, np.newaxis], subsample = 1000)**2
            ky = kernel.KGauss(y_bw)
        
        if self.estimator == 'inc':
            hsic_H = hsic.HSIC_Inc(kx, ky, ratio = self.l)
        else: # 'block'
            hsic_H = hsic.HSIC_Block(kx, ky, bsize = self.B)
        
        if self.poly:
            feat_select = PolySel(hsic_H)
        else: # multi
            feat_select = MultiSel(hsic_H)
        results = feat_select.test(X, Y[:, np.newaxis], args = self.n_select, alpha = alpha)
        
        # Reporting
        sel_vars = results['sel_vars']
        h0_rejs = results['h0_rejs']
        
        ind_sc_np = None
        ind_sel_np = np.zeros(p)
        ind_sel_np[sel_vars] = 1
        
        ind_h0_rej = np.zeros(p)
        ind_h0_rej[sel_vars[h0_rejs]] = 1        
        ind_h0_rej = {'H' : ind_h0_rej}
        
        ind_h0_rej_true = np.zeros(p)
        ind_h0_rej_true[sel_vars] = 1
        ind_h0_rej_true[niv:] = 0
        ind_h0_rej_true = {'H' : ind_h0_rej_true}
        
        if self.poly:
            # p-values not provided for Poly
            p_values = None
        else:
            p_values = -np.ones(p)
            p_values[sel_vars] = results['pvals']
            p_values = {'H': p_values}
        return sim.Inference_Result(p, ind_sc_np, ind_sel_np, ind_h0_rej,
                                       ind_h0_rej_true, p_values, None)