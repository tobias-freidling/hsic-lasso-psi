from mskernel import hsic, util, kernel
import numpy as np
from sklearn.covariance import ledoit_wolf, oas
from joblib import Parallel, delayed 
import covar


class KDiscrete(kernel.Kernel):
    
    def eval(self, X1, X2):
        """
        X1 : n1 x d   nd-array
        X2 : n2 x d   nd-array
        
        return: n1 x n2 Gram matrix
        """
        (n1, d1) = X1.shape
        (n2, d2) = X2.shape
        assert d1 == d2, 'Dimensions of the two inputs must be the same'
        K = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                if X1[i,:] == X2[j,:]:
                    K[i, j] = 1
        K = K / np.sqrt(n1 * n2)
        return K
    
    def pair_eval(self, X, Y):
        """only used for MMD"""
        pass
    
    def __str__(self):
        return "Discrete Kernel"


def nearestPD(A):
    """Find the nearest positive-definite matrix to input
    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].
    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False

# From Taylor
def find_root(f, y, lb, ub, tol=1e-6):
    """
    searches for solution to f(x) = y in (lb, ub), where 
    f is a monotone decreasing function
    """       
    
    # make sure solution is in range
    a, b   = lb, ub
    fa, fb = f(a), f(b)
    
    # assume a < b
    if fa > y and fb > y:
        while fb > y : 
            b, fb = b + (b-a), f(b + (b-a))
    elif fa < y and fb < y:
        while fa < y : 
            a, fa = a - (b-a), f(a - (b-a))
    
    # determine the necessary number of iterations
    max_iter = int( np.ceil( ( np.log(tol) - np.log(b-a) ) / np.log(0.5) ) )

    # bisect (slow but sure) until solution is obtained
    for _ in range(max_iter):
        c, fc  = (a+b)/2, f((a+b)/2)
        if fc > y: a = c
        elif fc < y: b = c
    
    return c

def estimate_H_unbiased(X, Y, discrete_output = False):
    assert Y.shape[0] == X.shape[0]
    n = X.shape[0]
    p = X.shape[1]  
    x_bw = util.meddistance(X, subsample = 1000)**2
    kx = kernel.KGauss(x_bw)
    if discrete_output:
        ky = KDiscrete()
    else:
        y_bw = util.meddistance(Y[:, np.newaxis], subsample = 1000)**2
        ky = kernel.KGauss(y_bw)
    
    hsic_H = hsic.HSIC_U(kx, ky)
    H = np.zeros(p)
    for i in range(p):
        H[i] = hsic_H.compute(X[:,i,np.newaxis], Y[:,np.newaxis])
    return H

def estimate_H_unbiased_parallel(X, Y, discrete_output = False):
    assert Y.shape[0] == X.shape[0]
    n = X.shape[0]
    p = X.shape[1]  
    x_bw = util.meddistance(X, subsample = 1000)**2
    kx = kernel.KGauss(x_bw)
    if discrete_output:
        ky = KDiscrete()
    else:
        y_bw = util.meddistance(Y[:, np.newaxis], subsample = 1000)**2
        ky = kernel.KGauss(y_bw)
    
    hsic_H = hsic.HSIC_U(kx, ky)    
    def one_calc(i):
        return hsic_H.compute(X[:,i,np.newaxis], Y[:,np.newaxis])
    
    par = Parallel(n_jobs = -1)
    res = par(delayed(one_calc)(i) for i in range(p))
    return np.array(res)

def estimate_H(X, Y, estimator, B, ratio, discrete_output = False):
    assert Y.shape[0] == X.shape[0]
    n = X.shape[0]
    p = X.shape[1]  
    x_bw = util.meddistance(X, subsample = 1000)**2
    kx = kernel.KGauss(x_bw)
    if discrete_output:
        ky = KDiscrete()
    else:
        y_bw = util.meddistance(Y[:, np.newaxis], subsample = 1000)**2
        ky = kernel.KGauss(y_bw)
    
    if estimator == 'inc':
        hsic_H = hsic.HSIC_Inc(kx, ky, ratio = ratio)
        m = int(n * ratio)
    else: # 'block'
        hsic_H = hsic.HSIC_Block(kx, ky, bsize = B)
        m = int(np.floor(n / B))

    H_estimates = np.zeros((p, m))
    for i in range(p):
        H_estimates[i, :] = np.reshape(hsic_H.estimates(X[:, i, np.newaxis],
                                                        Y[:, np.newaxis]), -1)
    H = np.mean(H_estimates, axis = 1)
    return H_estimates, H, m


def estimate_M_unbiased(X):
    n = X.shape[0]
    p = X.shape[1]
    x_bw = util.meddistance(X, subsample = 1000)**2
    kx = kernel.KGauss(x_bw)
    hsic_M = hsic.HSIC_U(kx, kx)
    M_true = np.zeros((p, p))
    for i in range(p):
        for j in range(i+1):
            M_true[i, j] = hsic_M.compute(X[:,i,np.newaxis], X[:, j, np.newaxis])
            M_true[j, i] = M_true[i, j]
    M = nearestPD(M_true)
    return M_true, M

def estimate_M_unbiased_parallel(X):
    n = X.shape[0]
    p = X.shape[1]
    x_bw = util.meddistance(X, subsample = 1000)**2
    kx = kernel.KGauss(x_bw)
    hsic_M = hsic.HSIC_U(kx, kx)
    M_true = np.zeros((p, p))
    
    def one_calc(i, j):
        return hsic_M.compute(X[:,i,np.newaxis], X[:, j, np.newaxis])
    
    par = Parallel(n_jobs = -1)
    res = par(delayed(one_calc)(i, j) for i in range(p) for j in range(i+1))
    sp = 0
    for i in range(p):
        for j in range(i+1):
            M_true[i, j] = M_true[j, i] = res[sp + j]
        sp += i+1
    M = nearestPD(M_true)
    return M_true, M
    

def estimate_M(X, estimator, B, ratio):
    n = X.shape[0]
    p = X.shape[1]
    x_bw = util.meddistance(X, subsample = 1000)**2
    kx = kernel.KGauss(x_bw)
    if estimator == 'inc':
        hsic_M = hsic.HSIC_Inc(kx, kx, ratio = ratio)
    else: # 'block'
        hsic_M = hsic.HSIC_Block(kx, kx, bsize = B)

    M_true = np.zeros((p, p))
    for i in range(p):
        for j in range(i+1):
            M_true[i, j] = np.mean(hsic_M.estimates(X[:, i, np.newaxis], X[:, j, np.newaxis]))
            M_true[j, i] = M_true[i, j]
    M = nearestPD(M_true)
    return M_true, M

def lasso_sol_helper(M, H, lam, w = None):
    # Decomposition: M = L*L^T; H = L * Y
    if w is None:
        w = np.ones(H.shape[0])
    L = np.linalg.cholesky(M)
    #L = np.linalg.cholesky(self.M).real #lower triangular matrix
    Y = np.linalg.solve(L, H)
    reg = Lasso(alpha = lam / H.shape[0], fit_intercept = False, positive = True)
    reg.fit(L.T / w, Y)
    #lam = reg.alpha_
    beta = reg.coef_
    ind_sel = np.where(beta > 1e-10)[0]
    ind_nsel = np.where(beta <= 1e-10)[0]
    return beta, ind_sel, ind_nsel

def covariance(H_estimates, m, cov_mode):
    if cov_mode == 'ledoit_wolf':
        cov, _ = ledoit_wolf(H_estimates.T)
    elif cov_mode == 'empirical':
        cov = np.cov(H_estimates)
    elif cov_mode == 'shrink_ss':
        cov, _ = covar.cov_shrink_ss(H_estimates.T)
    elif cov_mode == "shrink_rblw":
        S = np.cov(H_estimates)
        cov, _ = covar.cov_shrink_rblw(S, H_estimates.shape[1])
    else: # default: 'oas'
        cov, _ = oas(H_estimates.T)
    cov = cov / m
    return cov