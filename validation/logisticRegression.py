import sklearn.datasets
import numpy
import scipy
from prettytable import PrettyTable
import featureAnalysis
import ev
import plots


class logRegClass:
    def __init__(self, DTR, LTR, l, pi):
        self.DTR = DTR
        self.LTR = LTR
        self.l = l
        self.pi=pi
        
    def logreg_obj(self, v):
        if self.pi==None:     
            
            Z = self.LTR * 2.0 - 1.0
            M=self.DTR.shape[0]
            
            w= v[0:M]
            b= v[-1]
            first_term= (self.l/2)*numpy.linalg.norm(w)**2
            S = numpy.dot(w.T, self.DTR) + b
            cxe = numpy.logaddexp(0, -S*Z).mean() #calcola log exp(0)=1 + log exp (-S*Z) per ogni elemento
            return first_term+cxe
        else:
           
           
            w=v[0:-1]
            b=v[-1]
            first_term= (self.l/2)*numpy.linalg.norm(w)**2
            nT=self.DTR[:, self.LTR==1].shape[1]
            nF=self.DTR[:, self.LTR==0].shape[1]
            
            Z=self.LTR*2.0-1.0
            
            St=self.DTR[:, Z==1.0]
            Sf=self.DTR[:, Z==-1.0]
            
            
            Sp=numpy.dot(w.T, St)+b
            Sff=numpy.dot(w.T, Sf)+b
            
            
            
            cxt=numpy.logaddexp(0, -1.0*(Sp)).mean()
            cxf=numpy.logaddexp(0, 1.0*(Sff)).mean()
            
            return first_term + (self.pi/nT)*cxt + ((1-self.pi)/nF)*cxf
            
            
            
    
 
    
def BinaryLogisticRegression(DTR,LTR, DTE, LTE, lamb, pi):
    logRegObj = logRegClass(DTR, LTR, lamb, pi)
    _v, _J, _d = scipy.optimize.fmin_l_bfgs_b(logRegObj.logreg_obj, numpy.zeros(DTR.shape[0]+1), approx_grad=True)
    _w = _v[0:DTR.shape[0]]
    _b = _v[-1]
    #calcolo la posterior log-likelihood ratio semplicemente calcolando, per ogni campione xt, il punteggio S(xt)
    S = numpy.dot(_w.T, DTE) + _b 
    #calibrate scores in order to obtain llr in output
    c = numpy.log(pi / (1 - pi))
    S = S - c
    return S


def kFoldLR(D, L):
    K = 5
    N = int(D.shape[1]/K) 
     
   
    datasetP=numpy.sum(L)/L.shape[0]
    PCA_valules = [0, 2, 3, 4, 5]
    prior_values=[None, datasetP , 0.1, 0.5]
    lamb_values=[1e-6, 1e-3, 0.1, 1.0, 100, 1000]
    Cfp=Cfn=1
    pi_values=[0.1, 0.5]
    results = PrettyTable()
    results.align = "c"
    results.field_names = ["Lambda", "Prior", "PCA","minDCF (\pi = 0.1)","minDCF (\pi = 0.5)", "Primary Metric"]

   
    numpy.random.seed(0)
    indexes = numpy.random.permutation(D.shape[1])        
    
    for p in PCA_valules:
        primSlist={
            None:numpy.array([]),
            datasetP:numpy.array([]),
            0.1: numpy.array([]),
            0.5:numpy.array([])            
            }
        for prior in prior_values:
            
            for lamb in lamb_values:
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
                    
                    
                    llrs = BinaryLogisticRegression(DTR, LTR, DTE, LTE, lamb, prior)
                    
                    labels = numpy.hstack((labels, LTE))
                    
                    scores = numpy.hstack((scores, llrs))
                
                print(f"Computed PCA={p}, Lambda={lamb}, Prior {prior}")       
                minDCFS=numpy.zeros(2)
                for i,pi in enumerate(pi_values):
                    minDCFS[i]= numpy.round(ev.evalF(pi, Cfp, Cfn, scores, labels), 3)
            
                PrimS = (minDCFS[0] + minDCFS[1])/ 2
                primSlist[prior]= numpy.hstack((primSlist[prior],PrimS))
                results.add_row([lamb,prior, p, minDCFS[0], minDCFS[1], PrimS])
                
        
        #plots.plot_lr(lamb_values, p, primSlist)
        
        
            
        
    output =  str(results)
    # save output txt file containing results for each model
    with open('validationResults/logisticRegression/results.txt', 'r+') as file:
        file.truncate(0)
        file.write("Results with Linear Logistic Regression"+ '\n'+ str(output)+ '\n')
            
        
        
    
        

    return minDCFS

        
    


    
    
    
    
    
    
    
    