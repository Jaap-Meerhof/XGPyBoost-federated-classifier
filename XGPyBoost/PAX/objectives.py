import numpy as np

def softprob(y_true, y_pred):
    grad = np.zeros((y_pred.shape[0], y_pred.shape[1]), dtype=float) # for multi-class
    hess = np.zeros((y_pred.shape[0], y_pred.shape[1]), dtype=float) # for multi-class
    for rowid in range(y_pred.shape[0]):
        wmax = max(y_pred[rowid]) # line 100 multiclass_obj.cu
        wsum =0.0
        for i in y_pred[rowid] : wsum +=  np.exp(i - wmax)
        for c in range(y_pred.shape[1]):
            p = np.exp(y_pred[rowid][c]- wmax) / wsum
            target = y_true[rowid]
            g = p - 1.0 if c == target else p
            h = max((2.0 * p * (1.0 - p)).item(), 1e-6)
            grad[rowid][c] = g
            hess[rowid][c] = h
    return grad, hess