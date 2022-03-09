import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import normalize
from poibin.poibin import PoiBin

class rankDice(object):
    """rankDice for detection based on a given probability

    Parameters
    ----------

    """
    def __init__(self, shape, normal_approx=True, return_vol=False):
        self.normal_approx = normal_approx
        self.shape = shape
        self.d = np.prod(shape)
        self.return_vol = return_vol

    def vol_est(self, p, rank_ind):
        d = self.d
        score = np.zeros(d)
        weight_old = np.zeros(d - 1)
        for tau_tmp in range(1, d):
            # the index need to be removed
            ind_tmp = rank_ind[tau_tmp - 1]
            # compute pmf of PB
            p_tmp = np.delete(p, ind_tmp)
            pb_tmp = PoiBin(p_tmp)
            # compute increasement
            inc_tmp = p[ind_tmp] * np.array([pb_tmp.pmf(l) for l in range(1, d)])
            weight_new = weight_old + inc_tmp
            # compute the final score for tau_tmp
            discount = np.array([ 1. / (tau_tmp + l +1) for l in range(1, d)])
            score[tau_tmp] = np.sum(discount * weight_new)
            # update weight new
            weight_old = weight_new.copy()
        opt_vol = np.argmax(score)
        return opt_vol
        
    def predict(self, p):
        n, d = len(p), self.d
        p_vec = p.reshape(n, d)
        opt_vol_lst, pred_label = [], np.zeros((n,d), dtype=bool)
        for i in range(n):
            p_tmp = p_vec[i]
            # ranking
            rank_ind = np.argsort(-p_tmp)
            # volumn estimation
            opt_vol_tmp = self.vol_est(p=p_tmp, rank_ind=rank_ind)
            pred_label[i][rank_ind[:opt_vol_tmp]] = 1
            opt_vol_lst.append(opt_vol_tmp)
        opt_vol_lst = np.array(opt_vol_lst)
        pred_label = pred_label.reshape(p.shape)
        if self.return_vol:
            return pred_label, opt_vol_lst
        else:
            return pred_label








