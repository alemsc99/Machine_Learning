import numpy
from prettytable import PrettyTable 
import featureAnalysis 
import matplotlib.pyplot as plt
import ev
import GMMEval

def colFromRow(v): #this function turns a row into a column
    return v.reshape((v.size, 1))

def rowFromCol(v): #this function turns a column into a row
    return v.reshape((1, v.size))


def load_dataset(filename):
    features=[]
    labels=[]
    with open(filename, 'r') as f:
        for line in f:
            feats= [float(fe) for fe in line.split(',')[0:6]]
            
            features.append(colFromRow(numpy.array(feats)))
            
            label=line.split(',')[6].strip()
            labels.append(label)
                
    L=numpy.array(labels, dtype=numpy.int32) #1-d array from a list
    D=numpy.hstack(features) 
    #each column of D is a sample, each sample has 6 rows and each row represents a feature
    #print(D.shape) #6x2371
    #print(L.shape) #2371
    return D,L

def logpdf_GAU_ND(x, mu, C):
    M = x.shape[0]
    _, det = numpy.linalg.slogdet(C)
    det = numpy.log(numpy.linalg.det(C))
    inv = numpy.linalg.inv(C)    
    res = []
    x_centered = x - mu
    for x_col in x_centered.T:
        res.append(numpy.dot(x_col.T, numpy.dot(inv, x_col)))

    return -M/2*numpy.log(2*numpy.pi) - 1/2*det - 1/2*numpy.hstack(res).flatten()

def apply_Z_Norm(DTR, DTE):
    mu = colFromRow(DTR.mean(1))
    std = colFromRow(DTR.std(1))

    DTR_znorm = (DTR - mu) / std
    DTE_znorm = (DTE - mu) / std
    return DTR_znorm, DTE_znorm

def bayes_error_plot(scores,labels, title):
    # compute the p-tilde values
    effPriorLogOdds = numpy.linspace(-3, 3, 21)
    
    # initialize DCF and minDCF vectors that will be plotted
    DCF = numpy.zeros(effPriorLogOdds.size)
    minDCF = numpy.zeros(effPriorLogOdds.size)
    
    # set Cfn and Cfp to 1
    Cfn = 1
    Cfp = 1
    
    #compute DCF and minDCF for each value of p-tilde considered (21 in total)
    for idx, p_tilde in enumerate(effPriorLogOdds):
        
        #compute prior π
        pi_tilde = 1 / (1 + numpy.exp(-p_tilde))
        
        # compute DCF
        DCF[idx] = ev.compute_act_DCF(scores, labels, pi_tilde, Cfn, Cfp)
        
        # compute minDCF
        minDCF[idx] = ev.evalF(pi_tilde, Cfn, Cfp, scores, labels)
            
    plt.plot(effPriorLogOdds, DCF, label='DCF', color='r')
    plt.plot(effPriorLogOdds, minDCF, label='minDCF', color='b')
    plt.xlabel(r'$log\frac{\pi}{1-\pi}$')
    plt.ylabel("DCF")
    plt.legend()
    plt.grid()
    plt.title(title)
    
    # the following will be done in scripts that use this function:
    # plt.ylim([0, 0.4])
    # plt.xlim([-3, 3])
    # plt.savefig("filename", dpi = 200)
    # plt.show()
   
    return plt

def bayes_plot(D, L):
    # number of folds for K-fold
    K_Fold = 5 # can't use K for K-fold because in SVM K is already used
    N = int(D.shape[1]/K_Fold)
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
    results.field_names = ["minDCF (pi = 0.1)", "minDCF (pi = 0.5)", "min Cprim", "actDCF (pi = 0.1)", "actDCF (pi = 0.5)", "Cprim"]


     
    scores_pool = numpy.array([])
    labels_pool = numpy.array([])

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
        if PCA != 0:
            DTR, P = featureAnalysis.PCA(DTR, PCA)
            DTE = numpy.dot(P.T, D[:, idxTest] )
        else:
            DTE = D[:, idxTest]       
        

        LTR = L[idxTrain]
        LTE = L[idxTest]
    
        #pool test scores and test labels in order to compute the minDCF on complete pool set
        labels_pool = numpy.hstack((labels_pool,LTE))
        scores_pool = numpy.hstack((scores_pool,GMM.GMM_scores(DTR, LTR, DTE, 6, "Diagonal", 0, "Diagonal") ))

    plt = bayes_error_plot(scores_pool, labels_pool, "GMM Bayes plot")
    plt.ylim([0, 0.4])
    plt.xlim([-3, 3])
    plt.savefig("C:\\Users\\masci\\Desktop\\BayesPlot_Best_GMM.png", dpi = 200)
    plt.show()
    #compute minDCF and actualDCF for the working points  
    minDCF = numpy.zeros(2)
    actDCF = numpy.zeros(2)
    for i, pi in enumerate(pi_list):
        actDCF[i] = ev.compute_act_DCF(scores_pool, labels_pool, pi, Cfn, Cfp)
        minDCF[i] = ev.compute_min_DCF(scores_pool, labels_pool, pi, Cfn, Cfp)
    Cprim = numpy.round((minDCF[0] + minDCF[1])/ 2 , 3)
    act_Cprim = numpy.round((actDCF[0] + actDCF[1])/ 2 , 3)
    results.add_row([numpy.round(minDCF[0],3), numpy.round(minDCF[1],3), Cprim, numpy.round(actDCF[0],3), numpy.round(actDCF[1],3), act_Cprim ])
    
    # print and save as txt the final results table
    print(results)
    with open('C:\\Users\\masci\\Desktop\\GMM_actDCF.txt', 'w') as file:
        file.write(results.get_string())
        

   