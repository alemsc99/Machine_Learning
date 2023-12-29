import numpy
import matplotlib.pyplot as plt
import utilities
def compute_act_DCF(scores, labels, pi, Cfn, Cfp, th=None):
    """Compute the actual DCF, which is basically the normalized bayes risk"""
    #compute opt bayes decisions
    if th == None:
        th = -numpy.log((pi*Cfn)/((1-pi)*Cfp))
    OptDecisions = numpy.array([scores > th])
    pred= numpy.int32(OptDecisions)
    
    
    classes = numpy.unique(labels)
    # initialize the confusion matrix
    CM = numpy.zeros((len(classes), len(classes)))
    # loop across the different combinations of actual / predicted classes
    for i in range(len(classes)):
        for j in range(len(classes)):
           # count the number of instances in each combination of actual / predicted classes
           CM[i, j] = numpy.sum((labels == classes[i]) & (pred == classes[j]))
           
    
    CM=CM.T.astype(int)
    FNR = CM[0,1]/(CM[0,1] + CM[1,1])
    FPR = CM[1,0]/(CM[0,0] + CM[1,0])
    empBayes=pi*Cfn*FNR + (1-pi)*Cfp*FPR
    B_dummy = numpy.array([pi*Cfn, (1-pi)*Cfp]).min()
    return empBayes / B_dummy
    


def computeConfM(predicted_labels, LTE):
    Conf_Matrix=numpy.zeros((2,2))
    
    #print(Conf_Matrix.shape)
    for i,p in enumerate(predicted_labels): 
        
        Conf_Matrix[int(p)][int(LTE[i])]+=1 
        
    return Conf_Matrix


def computePredictions(piT, C, llrs, th):
    predicted_labels=numpy.zeros(llrs.shape[0])
    #t=-numpy.log((piT*C[0][1])/((1-piT)*C[1][0]))
    for i,v in enumerate(llrs):
        if v>th:
            predicted_labels[i]=1
        else:
            predicted_labels[i]=0
    
    return predicted_labels
            

def compute_normalized_DCF(Cfn, Cfp, piT, llrs, labels, th):
    
    C=[[0, Cfn], [Cfp, 0]]
    
    predicted_labels=computePredictions(piT, C, llrs, th)
    
    conf_M=computeConfM(predicted_labels, labels)
    
    
    #BINARY TASK:EVALUATION
    #Empirical Bayes Risk
    FNR=conf_M[0][1]/(conf_M[0][1]+conf_M[1][1])
    FPR=conf_M[1][0]/(conf_M[0][0]+conf_M[1][0])
    
    EBR=piT*Cfn*FNR+(1-piT)*Cfp*FPR
    
    #Normalized Empirical Bayes Risk
    Bdummy=min(piT*Cfn, (1-piT)*Cfp)
    normEBR=EBR/Bdummy
    
    
    return normEBR


def compute_min_DCF(llrs, labels, piT, Cfn, Cfp):
    
    t = numpy.array(llrs)
    
    t.sort()
    t=numpy.concatenate([numpy.array([-numpy.inf]), t, numpy.array([numpy.inf])])
    dcfList = []
    for _th in t:
        dcfList.append(compute_normalized_DCF(Cfn, Cfp, piT,llrs,labels,_th))
    return numpy.array(dcfList).min()
    
    
def evalF(piT, Cfn, Cfp, llrs, labels):
   
    #BINARY TASK OPTIMAL DECISIONS
    #Ht=Target Language Hf=Non-target Language
    
    
    
    minDcf=compute_min_DCF(llrs, labels, piT, Cfn, Cfp);
    return minDcf
    
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    