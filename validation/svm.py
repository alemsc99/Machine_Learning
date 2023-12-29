
import numpy
import scipy.optimize
import utilities
import featureAnalysis
import ev
from prettytable import PrettyTable
import plots

def compute_lagrangian_wrapper(H):

    def compute_lagrangian(alpha):

        elle = numpy.ones(alpha.size)

        L_hat_D = 0.5 * \
            (numpy.linalg.multi_dot([alpha.T, H, alpha])
             ) - numpy.dot(alpha.T, utilities.colFromRow(elle))  # 1x1
        L_hat_D_gradient = numpy.dot(H, alpha)-elle  # 66x1

        return L_hat_D, L_hat_D_gradient

    return compute_lagrangian



def compute_primal_objective(C, Z, w_hat_star, Dhat):
    w_hat_star = utilities.colFromRow(w_hat_star)
    Z = utilities.rowFromCol(Z)

    f1 = Z*numpy.dot(w_hat_star.T, Dhat)
    f2 = 1-f1
    zeros = numpy.zeros(f2.shape)
    sommatoria = numpy.maximum(zeros, f2)
    f4 = numpy.sum(sommatoria)
    f5 = C*f4
    f6 = 0.5*(w_hat_star*w_hat_star).sum()
    return f6+f5


def kFoldSvm(D, L):
    K_Fold=5
    N = int(D.shape[1]/K_Fold)
    numpy.random.seed(0)
    indexes = numpy.random.permutation(D.shape[1])
    pi_values = [0.1, 0.5]
    Cfp=Cfn=1
    results = PrettyTable()
    results.align = "c"
    results.field_names = ["K", "C", "PCA", "minDCF (\pi = 0.1)","minDCF (\pi = 0.5)", "Primary Metric"]

    
    K_values = [0.01, 0.1, 1.0]

    C_values = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]

    PCA_values = [0, 2, 3, 4, 5]

    for k in K_values:
        primSlist = {
             0:numpy.array([]),
             2: numpy.array([]),
             3:numpy.array([]),
             4:numpy.array([]),
             5:numpy.array([])
        }
        for p in PCA_values:
            
            for c in C_values:
                
                labels = numpy.array([])
                scores = numpy.array([])
                
                for i in range(K_Fold):
                    idxTest = indexes[i*N:(i+1)*N]
                    if i > 0:
                        idxTrainLeft = indexes[0:i*N]
                    elif (i+1) < K_Fold:
                        idxTrainRight = indexes[(i+1)*N:]
                    if i == 0:
                        idxTrain = idxTrainRight
                    elif i == K_Fold-1:
                        idxTrain = idxTrainLeft
                    else:
                        idxTrain = numpy.hstack(
                        [idxTrainLeft, idxTrainRight])
                        
                    DTR = D[:, idxTrain]
                    if p != 0:
                        DTR, P = featureAnalysis.PCA(DTR, p)
                        DTE = numpy.dot(P.T, D[:, idxTest] )
                    else:
                        DTE = D[:, idxTest]

                    LTR = L[idxTrain]
                    LTE = L[idxTest]
                    
                    
                    
                    Ks = numpy.ones((1, DTR.shape[1]))*k
                    Dhat = numpy.vstack((DTR, Ks))
                    Ghat = numpy.dot(Dhat.T, Dhat)

                    Z = numpy.copy(LTR)
                    Z[Z == 0] = -1
                    Z = utilities.colFromRow(Z)

                    Hhat = Z*Z.T*Ghat

                    lagr = compute_lagrangian_wrapper(Hhat)

                   
                    x0 = numpy.zeros(LTR.size)  # alpha
                    bound_list = [(0, c)]*LTR.size
                    (x, f, d) = scipy.optimize.fmin_l_bfgs_b(
                        lagr, approx_grad=False,  x0=x0, bounds=bound_list, factr=1.0)

                    # Recover the primal solution
                    sommatoria = utilities.colFromRow(x)*utilities.colFromRow(Z)*Dhat.T

                    w_hat_star = numpy.sum(sommatoria, axis=0)

                    w_star = w_hat_star[0:-1]
                    b_star = w_hat_star[-1]
                    srs = numpy.dot((w_star).T, DTE)+b_star
                   
                    
                    labels = numpy.hstack((labels, LTE))
                    
                    scores = numpy.hstack((scores, srs))
                    print("computing...")
                    
                print(f"Computed K={k} C={c} PCA={p}")
                minDCFS = numpy.zeros(2)
                for i, pi in enumerate(pi_values):
                    minDCFS[i] = numpy.round(ev.evalF(pi, Cfp, Cfn, scores, labels), 3)

                PrimS = (minDCFS[0] + minDCFS[1]) / 2
                print(PrimS)
                primSlist[p] = numpy.hstack((primSlist[p],PrimS))
                results.add_row([k, c, p, minDCFS[0], minDCFS[1], PrimS])
        
        
        plots.plot_svm(C_values, k, primSlist)
    
    
    output = str(results)
    # save output txt file containing results for each model
    with open('validationResults/SVM/results.txt', 'a') as file:
        
        file.write("Results with Linear Support Vector Machine" + '\n' + str(output) + '\n')
                    
                    

                    
                    
                    

                    


    


