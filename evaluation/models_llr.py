import numpy 
import utilities
import scipy.optimize
def SVM_RBF(DTR, LTR, DTE, C, K, gamma):
    
    # compute the zi for each train sample, z = 1 if class is 1, -1 if class is 0
    Z = numpy.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1
    
    # Compute H_hat directly on training data, no expansion needed. 
    # and compute H using kernel function instead of dot product
    H = numpy.zeros((DTR.shape[1], DTR.shape[1]))
    for i in range(DTR.shape[1]):
        for j in range(DTR.shape[1]):
            H[i, j] = numpy.exp(-gamma * (numpy.linalg.norm(DTR[:, i] - DTR[:, j]) ** 2)) + K * K
    H = utilities.colFromRow(Z) * utilities.rowFromCol(Z) * H
    
    def JDual(alpha):
        Ha = numpy.dot(H,utilities.colFromRow(alpha))
        aHa = numpy.dot(utilities.rowFromCol(alpha),Ha)
        a1 = alpha.sum()
        return -0.5 * aHa.ravel() + a1, -Ha.ravel() + numpy.ones(alpha.size)
    
    def LDual(alpha):
        loss, grad = JDual(alpha)
        return -loss, -grad
    

    alphaStar , _x, _y = scipy.optimize.fmin_l_bfgs_b(
    LDual,
    numpy.zeros(DTR.shape[1]),
    bounds = [(0,C)] * DTR.shape[1],
    factr = 1.0,
    maxiter = 100000,
    maxfun = 100000,
    )
    
    wStar = numpy.dot(DTR, utilities.colFromRow(alphaStar) * utilities.colFromRow(Z))
    
    # compute kernel on train and TEST set
    kernel = numpy.zeros((DTR.shape[1], DTE.shape[1]))
    for i in range(DTR.shape[1]):
        for j in range(DTE.shape[1]):
            kernel[i, j] = numpy.exp(-gamma * (numpy.linalg.norm(DTR[:, i] - DTE[:, j]) ** 2)) + K * K
    
    scores = numpy.sum(numpy.dot(alphaStar * utilities.rowFromCol(Z), kernel), axis=0)
    return scores.ravel()
    