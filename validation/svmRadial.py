# import numpy
# import scipy.optimize
# import utilities
# import featureAnalysis
# import ev
# from prettytable import PrettyTable
# import plots

# def compute_lagrangian_wrapper(H):

#     def compute_lagrangian(alpha):

#         elle = numpy.ones(alpha.size)

#         L_hat_D = 0.5 * \
#             (numpy.linalg.multi_dot([alpha.T, H, alpha])
#               ) - numpy.dot(alpha.T, utilities.colFromRow(elle))  # 1x1
#         L_hat_D_gradient = numpy.dot(H, alpha)-elle  # 66x1

#         return L_hat_D, L_hat_D_gradient

#     return compute_lagrangian



# def compute_primal_objective(C, Z, w_hat_star, Dhat):
#     w_hat_star = utilities.colFromRow(w_hat_star)
#     Z = utilities.rowFromCol(Z)

#     f1 = Z*numpy.dot(w_hat_star.T, Dhat)
#     f2 = 1-f1
#     zeros = numpy.zeros(f2.shape)
#     sommatoria = numpy.maximum(zeros, f2)
#     f4 = numpy.sum(sommatoria)
#     f5 = C*f4
#     f6 = 0.5*(w_hat_star*w_hat_star).sum()
#     return f6+f5

# def compute_scores(DTR,DTE,LTR, gamma, k, c):
#     Z = numpy.zeros(LTR.shape)
#     # Z[LTR == 0] = 1
#     # Z[LTR == 1] = -1 0.8365
#     Z[LTR == 0] = 1
#     Z[LTR == 1] = 0
#     #Z = utilities.colFromRow(Z)
    
    
#     H = numpy.zeros((DTR.shape[1], DTR.shape[1]))
    
    
#     for i in range(DTR.shape[1]):
#         for j in range(DTR.shape[1]):
#             H[i, j] = numpy.exp(-gamma * (numpy.linalg.norm(DTR[:, i] - DTR[:, j]) ** 2)) + k * k
    
#     #kernel=numpy.exp(-gamma* Ghat)+ k*k                       

#     H = utilities.colFromRow(Z)*utilities.rowFromCol(Z)*H
   
#     #print(f"{H.shape}")
#     def JDual(alpha):
#         Ha = numpy.dot(H,utilities.colFromRow(alpha))
#         aHa = numpy.dot(utilities.rowFromCol(alpha),Ha)
#         a1 = alpha.sum()
#         return -0.5 * aHa.ravel() + a1, -Ha.ravel() + numpy.ones(alpha.size)

#     def LDual(alpha):
#         loss, grad = JDual(alpha)
#         return -loss, -grad


  
   
    
#     alphaStar, _f, _dd = scipy.optimize.fmin_l_bfgs_b(
#         LDual,   numpy.zeros(DTR.shape[1]),
#     bounds = [(0,c)] * DTR.shape[1],
#     factr = 1.0,
#     maxiter = 100000,
#     maxfun = 100000)
    
#     wStar = numpy.dot(DTR, utilities.colFromRow(alphaStar) * utilities.colFromRow(Z))
    
#     kernel = numpy.zeros((DTR.shape[1], DTE.shape[1]))
#     for i in range(DTR.shape[1]):
#         for j in range(DTE.shape[1]):
#             kernel[i, j] = numpy.exp(-gamma * (numpy.linalg.norm(DTR[:, i] - DTE[:, j]) ** 2)) + k * k
            
    
    

#     scores= numpy.sum(numpy.dot(alphaStar * utilities.colFromRow(Z), kernel), axis=0)
    
#     return scores.ravel()
    

    

# def kFoldSvmR(D, L):
#     K_Fold=5
#     N = int(D.shape[1]/K_Fold)
#     numpy.random.seed(0)
#     indexes = numpy.random.permutation(D.shape[1])
#     pi_values = [0.1, 0.5]
#     Cfp=Cfn=1
#     results = PrettyTable()
#     results.align = "c"
#     results.field_names = ["K", " γ", "PCA", "minDCF (\pi = 0.1)","minDCF (\pi = 0.5)", "Primary Metric"]
#     nf = D[:,L==0].shape[1]
#     nt = D[:,L==1].shape[1]
#     emp_prior_f = nf/ D.shape[1]
#     emp_prior_t = nt/ D.shape[1]
    
#     K_values = [0.01]#, 0.1, 1.0]

#     C_values = [10]#, 1e-3, 1e-2, 1e-1, 1.0, 1e-4]
    
#     gamma_values=[0.1]#, 1.0]
#     d=2

#     #PCA_values = [5, 2, 3, 4, 0]
#     PCA_values = [0]

#     for k in K_values:
#         for gamma in gamma_values:
#             primSlist = {
#                   0:numpy.array([]),
#                   2: numpy.array([]),
#                   3:numpy.array([]),
#                   4:numpy.array([]),
#                   5:numpy.array([])
#             }
#             for p in PCA_values:
                
#                 for c in C_values:
                                  
#                     labels = numpy.array([])
#                     scores = numpy.array([])
                    
                    
#                     for i in range(K_Fold):
#                         idxTest = indexes[i*N:(i+1)*N]
#                         if i > 0:
#                             idxTrainLeft = indexes[0:i*N]
#                         elif (i+1) < K_Fold:
#                             idxTrainRight = indexes[(i+1)*N:]
#                         if i == 0:
#                             idxTrain = idxTrainRight
#                         elif i == K_Fold-1:
#                             idxTrain = idxTrainLeft
#                         else:
#                             idxTrain = numpy.hstack(
#                             [idxTrainLeft, idxTrainRight])
                            
#                         DTR = D[:, idxTrain]
#                         if p != 0:
#                             DTR, P = featureAnalysis.PCA(DTR, p)
#                             DTE = numpy.dot(P.T, D[:, idxTest] )
#                         else:
#                             DTE = D[:, idxTest]
                            
                    
                        
#                         LTR = L[idxTrain]
#                         LTE = L[idxTest]
                        
#                         labels = numpy.hstack((labels, LTE))
                        
    
#                         scores = numpy.hstack((scores, compute_scores(DTR, DTE, LTR, gamma, k, c)))                   
                        
#                         #scores = numpy.hstack((scores, srs))
                   
#                     print(DTR.shape)   
#                     print(f"Computed K={k} C={c} PCA={p} gamma={gamma}")
#                     minDCFS = numpy.zeros(2)
#                     for i, pi in enumerate(pi_values):
#                         Cfp = (c * pi) / emp_prior_t
#                         Cfn = (c * pi) / emp_prior_f
#                         minDCFS[i] = numpy.round(ev.evalF(pi, Cfn, Cfp, scores, labels), 3)
                        
#                     PrimS = (minDCFS[0] + minDCFS[1]) / 2
#                     print(PrimS)
                   
#                     primSlist[p] = numpy.hstack((primSlist[p],PrimS))
#                     results.add_row([k, gamma, p, minDCFS[0], minDCFS[1], PrimS])
        
        
#             plots.plot_svmR(C_values, k, gamma, primSlist)
    
    
#     output = str(results)
#     # save output txt file containing results for each model
#     with open('validationResults/SVMRadial/results.txt', 'a') as file:
        
#         file.write("Results with Polinomial Support Vector Machines" + '\n' + str(output) + '\n')
                    
                    

                    



import utilities 
import ev
import numpy
import featureAnalysis
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import datetime
import models_llr
import logisticRegression

# D, L = utilities.load_dataset("Train.txt")

# # number of folds for K-fold
# folds = 5 # can't use K for K-fold because in SVM K is already used
# N = int(D.shape[1]/folds)
# PCA = [5,4,3,2,None] #number of dimension to keep in PCA

# numpy.random.seed(0)
# indexes = numpy.random.permutation(D.shape[1])  

# C_list = numpy.logspace(-4,1,6).tolist() 
# K_list = numpy.logspace(-2,0,3).tolist() 
# gamma_list = numpy.logspace(-2,0,3).tolist() 
# # working points
# Cfn = 1
# Cfp = 1
# pi_list = [0.1, 0.5]

# results = PrettyTable()
# results.align = "c"
# results.field_names = ["K", "C", "PCA", "Kernel", "minDCF (pi = 0.1)", "minDCF (pi = 0.5)", "Cprim"]


# # produce a graph for each K,c,d: on x plot different C used for training, on Y plot relative Cprim obtained
# for K_num, K in enumerate(K_list):
    
#     for PCA_m in PCA:
    
#         st = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
#         print(f"### {st}: Starting SVM RBF with K = {K}, PCA = {PCA_m}") #feedback print
        
#         #set graph
#         fig, ax = plt.subplots() 
#         ax.set_xscale('log')
#         ax.set(xlabel='C', ylabel='Cprim', title=f'SVM K={K} - RBF Kernel')
#         plt.grid(True)
#         plt.xticks(C_list)
    
#         for gamma in gamma_list:    
#             Cprim_list = numpy.array([])
#             for C in C_list:
#             #for each C compute minDCF after K-fold
                        
#                 scores_pool = numpy.array([])
#                 labels_pool = numpy.array([])
                
#                 for i in range(folds):
                
#                     idxTest = indexes[i*N:(i+1)*N]
                
#                     if i > 0:
#                         idxTrainLeft = indexes[0:i*N]
#                     elif (i+1) < folds:
#                         idxTrainRight = indexes[(i+1)*N:]
                
#                     if i == 0:
#                         idxTrain = idxTrainRight
#                     elif i == folds-1:
#                         idxTrain = idxTrainLeft
#                     else:
#                         idxTrain = numpy.hstack([idxTrainLeft, idxTrainRight])
                
#                     DTR = D[:, idxTrain]
#                     if PCA_m != None:
#                         DTR,P = featureAnalysis.PCA(DTR,PCA_m) # fit PCA to training set
#                     LTR = L[idxTrain]
                    
                    
                    
#                     if PCA_m != None:
#                         DTE = numpy.dot(P.T,D[:, idxTest]) # transform test samples according to P from PCA on dataset
#                     else:
#                         DTE = D[:,idxTest]
#                     LTE = L[idxTest]
                    
#                     #DTR, DTE = utilities.apply_Z_Norm(DTR,DTE)
#                     #pool test scores and test labels in order to compute the minDCF on complete pool set
#                     labels_pool = numpy.hstack((labels_pool,LTE))
#                     scores_pool = numpy.hstack((scores_pool,models_llr.SVM_RBF(DTR, LTR, DTE, C, K, gamma)))
                 
#                 #compute minDCF for the current SVM with/without PCA for the 2 working points  
#                 minDCF = numpy.zeros(2)
#                 for i, pi in enumerate(pi_list):
#                     minDCF[i] = ev.compute_min_DCF(scores_pool, labels_pool, pi, Cfn, Cfp)
#                 #compute Cprim (the 2 working points average minDCF) and save it in a list for plotting after
#                 Cprim = numpy.round((minDCF[0] + minDCF[1])/ 2 , 3)
#                 print(f"Cprim={Cprim}")
#                 Cprim_list = numpy.hstack((Cprim_list,Cprim))
#                 # add current result to table
#                 results.add_row([K, C, PCA_m, f"RBF(γ = {gamma}) ", numpy.round(minDCF[0],3), numpy.round(minDCF[1],3), Cprim])
#                 print(f"\t...computed C={C}, γ={gamma}") #feedback print
                
#             #plot the graph
#             ax.plot(C_list, Cprim_list, label =f'K={K} (PCA 2)')
#             print(f"\tCprim values for γ={gamma}: {Cprim_list}") #feedback print         
    
#         print(f'Completed SVM RBF with K = {K} and PCA = {PCA_m} ###') #feedback print
#         plt.legend()
#         fig.savefig(f"C:\\Users\\masci\\Desktop\\PoliTo\\Secondo_semestre\\Machine_Learning\\exam_project\\validationResults\\SVMRadial\\K{K}_PCA{PCA_m}.png", dpi=200)
#         plt.show()
    
# # print and save as txt the final results table
# print(results)
# data = results.get_string()

# with open('C:\\Users\\masci\\Desktop\\PCA2.txt', 'w') as file:
#    file.write(data)

    
def score_calib(D,L):
        # number of folds for K-fold
    folds = 5 # can't use K for K-fold because in SVM K is already used
    N = int(D.shape[1]/folds)
    PCA = 5 #number of dimension to keep in PCA
    
    numpy.random.seed(0)
    indexes = numpy.random.permutation(D.shape[1])  
    
    #best RBF SVM configuration in validation phase
    C = 10 
    K = 0.01 
    gamma = 0.1 
    
    # working points
    Cfn = 1
    Cfp = 1
    pi_list = [0.1, 0.5]
    
    results = PrettyTable()
    results.align = "c"
    results.field_names = ["Pi Calibration","minDCF (pi = 0.1)", "minDCF (pi = 0.5)", "min Cprim", "actDCF (pi = 0.1)", "actDCF (pi = 0.5)", "Cprim"]
    
         
    scores_pool = numpy.array([])
    labels_pool = numpy.array([])
    
    for i in range(folds):
    
        idxTest = indexes[i*N:(i+1)*N]
    
        if i > 0:
            idxTrainLeft = indexes[0:i*N]
        elif (i+1) < folds:
            idxTrainRight = indexes[(i+1)*N:]
    
        if i == 0:
            idxTrain = idxTrainRight
        elif i == folds-1:
            idxTrain = idxTrainLeft
        else:
            idxTrain = numpy.hstack([idxTrainLeft, idxTrainRight])
    
        DTR = D[:, idxTrain]
        if PCA != None:
            DTR,P = featureAnalysis.PCA(DTR,PCA) # fit PCA to training set
        LTR = L[idxTrain]
        if PCA != None:
            DTE = numpy.dot(P.T,D[:, idxTest]) # transform test samples according to P from PCA on dataset
        else:
            DTE = D[:,idxTest]
        LTE = L[idxTest]
        
        #pool test scores and test labels in order to compute the minDCF on complete pool set
        labels_pool = numpy.hstack((labels_pool,LTE))
        scores_pool = numpy.hstack((scores_pool,models_llr.SVM_RBF(DTR, LTR, DTE, C, K, gamma)))
    
    #calibrate scores using single split approach#
    #reshuffle scores and relative labels
    p = numpy.random.permutation(scores_pool.shape[0])
    scores_pool = scores_pool[p]
    labels_pool = labels_pool[p]
    
    #split calibration set in training set (80%) and validation set (20%)
    C_DTR = utilities.rowFromCol(scores_pool[:int((scores_pool.shape[0]*80)/100)])
    C_LTR = labels_pool[:int((scores_pool.shape[0]*80)/100)]
    C_DTE = utilities.rowFromCol(scores_pool[int((scores_pool.shape[0]*80)/100):])
    C_LTE = labels_pool[int((scores_pool.shape[0]*80)/100):]
    
    #train a weigthed Linear LogReg with lambda set to 0 (unregularized)
    prior_list = [0.1, 0.2, 0.5] # calibrate for different priors
    out = "Score calibration - SVM RBF"
    for prior in prior_list:
        calibrated_scores=logisticRegression.BinaryLogisticRegression(C_DTR, C_LTR,C_DTE, C_LTE, 0, prior)
        #calibrated_scores = m.logistic_regression(C_DTR, C_LTR, 0, C_DTE, C_LTE, prior, cal = True)
        #compute minDCF and actDCF on calibrated vlaidation set
        plt = utilities.bayes_error_plot(calibrated_scores, C_LTE, "")
        plt.ylim([0, 1])
        plt.xlim([-3, 3])
        plt.savefig(f"C:\\Users\\masci\\Desktop\\BayesPlot_Best_SVM_RBF_calibrated_{prior}.png", dpi = 200)
        plt.show()
        #compute minDCF and actualDCF for the working points on calibration validation set  
        minDCF = numpy.zeros(2)
        actDCF = numpy.zeros(2)
        for i, pi in enumerate(pi_list):
            actDCF[i] = ev.compute_act_DCF(calibrated_scores, C_LTE, pi, Cfn, Cfp)
            minDCF[i] = ev.compute_min_DCF(calibrated_scores, C_LTE, pi, Cfn, Cfp)
        Cprim = numpy.round((minDCF[0] + minDCF[1])/ 2 , 3)
        act_Cprim = numpy.round((actDCF[0] + actDCF[1])/ 2 , 3)
        results.add_row([prior,numpy.round(minDCF[0],3), numpy.round(minDCF[1],3), Cprim, numpy.round(actDCF[0],3), numpy.round(actDCF[1],3), act_Cprim ])
        
    with open('C:\\Users\\masci\\Desktop\\SVM_RBF_Calibration_Results.txt', 'w') as file:
        file.write(out+results.get_string()+"\nLatex version\n"+results.get_latex_string())
    print("Done")
    
        
        


