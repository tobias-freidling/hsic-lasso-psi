

class Inference_Result:
    
    def __init__(self, p, ind_sc, ind_sel, ind_h0_rej, ind_h0_rej_true, p_values, conf_int):
        self.p = p
        self.ind_sc = ind_sc
        self.ind_sel = ind_sel
        self.ind_h0_rej = ind_h0_rej # dicitionary
        self.ind_h0_rej_true = ind_h0_rej_true # dictionary
        self.p_values = p_values
        self.conf_int = conf_int # dictionary
        self.nrv = None
        
    def set_record_variables(self, nrv):
        self.nrv = nrv
    
    def tpr(self):
        tpr = dict()
        for (t, h0_rej), (_, h0_rej_true)\
                in zip(self.ind_h0_rej.items(), self.ind_h0_rej_true.items()):
            N = np.sum(h0_rej_true)
            Z = np.sum(np.minimum(h0_rej_true, h0_rej))
            tpr[t] = Z / N
        return tpr    
    
    def fpr(self):
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
        """
        self.p = p
        self.ind_sc = ind_sc
        self.ind_sel = ind_sel
        self.ind_h0_rej = ind_h0_rej # dicitionary
        self.ind_h0_rej_true = ind_h0_rej_true # dictionary
        self.p_values = p_values
        self.conf_int = conf_int # dictionary
        self.nrv = None
        """
        print("p: ", self.p)
        if self.ind_sc is not None:
            print("screened indices: ", np.where(self.ind_sc == 1)[0])
        print("selected indices: ", np.where(self.ind_sel == 1)[0])
        for t, l in self.ind_h0_rej.items():
            print("target: {}, H_0-rejection indices: {}".format(t, np.where(l==1)[0]))
        for t, l in self.p_values.items():
            print("target: {}, p-values: {}".format(t, l))