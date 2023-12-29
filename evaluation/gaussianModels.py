import utilities
import numpy
import scipy
import ev
import featureAnalysis
from prettytable import PrettyTable

def mean_cov_estimate(DCLS):
    mu = utilities.colFromRow(DCLS.mean(1))
    DCLS_C = DCLS-mu
    C = numpy.dot(DCLS_C, DCLS_C.T)/(DCLS_C.shape[1])
    return mu, C


def mean_cov_bayesian_estimate(DCLS):
    mu = utilities.colFromRow(DCLS.mean(1))
    DCLS_C = DCLS-mu
    C = numpy.dot(DCLS_C, DCLS_C.T)/(DCLS_C.shape[1])
    # uso solo gli elementi sulla diagonale principale della matrice di covarianza e azzero tutti gli altri
    C = numpy.dot(C, numpy.eye(C.shape[0], C.shape[1]))
    return mu, C


def mean_cov_tied_estimate(D, L):
    N = D.shape[1]
    NC = 2  # number of classes in the dataset
    mu = utilities.colFromRow((D.mean(1)))
    # will contain the samples of i-th class (0,1) in the i-th position
    samples = []
    muC = []  # will contain the mean of i-th class in the i-th position as column vector
    for i in range(NC):
        samples.append(D[:, L == i])  # get samples for each class
        # compute mean for each class
        muC.append(utilities.colFromRow(D[:, L == i].mean(1)))

    # compute SW
    SWC = []  # will contain SW,c for each class where SW,c is the covariance matrix of each class
    for i in range(NC):
        # CC is for Centered Class, the samples of a specific class centered (subtractaction of class mean from all class samples)
        CC = samples[i] - muC[i]
        # compute SW for the i-th class
        SWC.append(numpy.dot(CC, CC.T)/samples[i].shape[1])

    # will contain sum of (SW,c * Nc) where Nc is the number of samples for a class
    s = 0
    for i in range(NC):
        s += SWC[i] * samples[i].shape[1]
    SW = s/N  # compute the SW matrix
    return muC, SW


def MultivariateGaussianC(DTR, LTR, DTE, LTE):
    # MULTIVARIATE GAUSSIAN CLASSIFIER
    # DTR e LTR sono i dati per il training con le rispettive etichette, DTE e LTE sono i dati per la validation
    # 100 samples per il training e 50 samples per la validation

    evalSamples = LTE.shape[0]
    # MULTIVARIATE GAUSSIAN CLASSIFIER
    # per ogni classe calcolo la media e la covarianza e le salvo come tupla in un dizionario
    hCls = {}

    for lab in [0, 1, 2]:
        DCLS = DTR[:, LTR == lab]
        hCls[lab] = mean_cov_estimate(DCLS)

    logprior = utilities.colFromRow(numpy.ones(3)/3.0)

    # Calcolo le likelihoods per ogni test sample e per ogni classe
    S = []

    for hyp in [0, 1, 2]:
        mu, C = hCls[hyp]

        fcond = utilities.logpdf_GAU_ND(DTE, mu, C)
        S.append(utilities.rowFromCol(fcond))

    S = numpy.vstack(S)

    S = numpy.exp(S)  # applico l'esponenziale
    # calcolo la probabilità congiunta moltiplicando per la probabilità a priori
    SJoint = S*logprior
    resSJoint = numpy.load("C:\\Users\\masci\\Downloads\\SJoint_MVG.npy")

    # Calcolo le probabilità a posteriori
    # prima calcolo il denominatore
    SMarginal = utilities.rowFromCol(SJoint.sum(0))
    # print(SMarginal.shape)
    SPost = SJoint/SMarginal

    predicted_labels = numpy.argmax(SPost, axis=0)
    # print(predicted_labels)
    # conto il numero di label assegnate correttamente
    correct_predictions = sum(LTE == predicted_labels)
    print("accuracy={:.2f}%".format(correct_predictions/evalSamples*100))
    print("error rate={:.2f}%".format((1-correct_predictions/evalSamples)*100))
    return correct_predictions, DTE.shape[1]


def LogMultivariateGaussianC(DTR, LTR, DTE, LTE):
    # LOG-MULTIVARIATE GAUSSIAN CLASSIFIER
    # DTR e LTR sono i dati per il training con le rispettive etichette, DTE e LTE sono i dati per la validation
    # 100 samples per il training e 50 samples per la validation

    evalSamples = LTE.shape[0]
    # MULTIVARIATE GAUSSIAN CLASSIFIER
    # per ogni classe calcolo la media e la covarianza e le salvo come tupla in un dizionario
    hCls = {}

    for lab in [0, 1]:
        DCLS = DTR[:, LTR == lab]
        hCls[lab] = mean_cov_estimate(DCLS)

    # Calcolo le likelihoods per ogni test sample e per ogni classe
    S = []

    for hyp in [0, 1]:
        mu, C = hCls[hyp]
        fcond = utilities.logpdf_GAU_ND(DTE, mu, C)
        S.append(utilities.rowFromCol(fcond))

    S = numpy.vstack(S)
    return S[1, :]-S[0, :]


def NaiveBayesGaussianC(DTR, LTR, DTE, LTE):
    DTR_c = [DTR[:, LTR == i] for i in range(2)]
    
    m_c = []
    s_c = []
    for d in DTR_c:
        m_c.append(d.mean(1))
        s_c.append(numpy.cov(d, bias=True)*numpy.identity(d.shape[0]))

    S = []

    for i in [0, 1]:

        fcond = utilities.logpdf_GAU_ND(
            DTE, utilities.colFromRow(m_c[i]), s_c[i])
        S.append(utilities.rowFromCol(fcond))

    S = numpy.vstack(S)

    return S[1, :]-S[0, :]


def TiedCovarianceGaussianC(DTR, LTR, DTE, LTE):

    

    mu, C = mean_cov_tied_estimate(DTR, LTR)
   

    # Calcolo le likelihoods per ogni test sample e per ogni classe
    S = []
    for hyp in [0, 1]:
        #mu, C=hCls[hyp]
        fcond = utilities.logpdf_GAU_ND(DTE, mu[hyp], C)
        S.append(utilities.rowFromCol(fcond))

    S = numpy.vstack(S)

    return S[1, :]-S[0, :]


def kFold(D, L):
    K = 5
    N = int(D.shape[1]/K)
    
    
    
    
     
    classifiers = [ (TiedCovarianceGaussianC, "Tied Covariance Gaussian Classifier")]

    PCA_valules = [0, 2, 3, 4, 5, 6]
    pi_values=[0.1, 0.5]
    Cfp=Cfn=1
    

    for j, (c, cstring) in enumerate(classifiers):
        numpy.random.seed(j)
        indexes = numpy.random.permutation(D.shape[1])
        
        
        results = PrettyTable()
        results.align = "c"
        results.field_names = ["PCA","minDCF (\pi = 0.1)","minDCF (\pi = 0.5)", "Primary Metric"]
        
        for p in PCA_valules:
            labels=numpy.array([])
            scores=numpy.array([])            
            
           
            for i in range(K):
                idxTest = indexes[i*N:(i+1)*N]
                if i > 0:
                    idxTrainLeft = indexes[0:i*N]
                elif (i+1) < K:
                    idxTrainRight = indexes[(i+1)*N:]
                if i == 0:
                    idxTrain = idxTrainRight
                elif i == K-1:
                    idxTrain = idxTrainLeft
                else:
                    idxTrain = numpy.hstack([idxTrainLeft, idxTrainRight])
                    
                DTR = D[:, idxTrain]
                if p != 0:
                    DTR, P = featureAnalysis.PCA(DTR, p)
                    DTE=numpy.dot(P.T, D[:, idxTest] )
                else:
                    DTE = D[:, idxTest]
                    
                    
                LTR = L[idxTrain]                
                LTE = L[idxTest]
                llrs = c(DTR, LTR, DTE, LTE)
                labels = numpy.hstack((labels, LTE))
                scores = numpy.hstack((scores, llrs))
            
            
            minDCFS=numpy.zeros(2)
            for i,pi in enumerate(pi_values):
                minDCFS[i]= numpy.round(ev.evalF(pi, Cfp, Cfn, scores, labels), 3)
            
            primS = (minDCFS[0] + minDCFS[1])/2
            results.add_row([p, minDCFS[0], minDCFS[1], primS])
        
        
        output =  str(results)
        # save output txt file containing results for each model
    with open('validationResults/gaussianModels/results.txt', 'r+') as file:
            
        file.write("Results with "+ cstring+ '\n'+ str(output)+ '\n')
            
        
        
    
        

    return minDCFS




