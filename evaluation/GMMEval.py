import numpy
import scipy.optimize
import utilities
import featureAnalysis
import ev
from prettytable import PrettyTable
import gaussianModels




def logpdfGMM(X, gmm):
    
    SJ = numpy.zeros((len(gmm),X.shape[1]))
    
    for g, (w, mu, C) in enumerate(gmm):
        SJ[g,:] = utilities.logpdf_GAU_ND(X, mu, C) + numpy.log(w)

    SM = scipy.special.logsumexp(SJ, axis=0)
    
    return SJ, SM #Note: use SM to compute then llr -> SM class 1 - SM class 0, we use logpdf!

def GMM_EM(X, gmm):
    '''
    EM algorithm for GMM full covariance
    It estimates the parameters of a GMM that maximize the ll for
    a training set X
    '''
    llNew = None
    llOld = None
    G = len(gmm)
    N = X.shape[1]
    
    psi = 0.01
    while llOld is None or llNew - llOld > 1e-6:
        llOld = llNew
        SJ, SM = logpdfGMM(X,gmm)
        llNew = SM.sum()/N
        P = numpy.exp(SJ-SM)
        gmmNew = []
        for g in range(G):
            gamma = P[g, :]
            Z = gamma.sum()
            F = (utilities.rowFromCol(gamma)*X).sum(1)
            S = numpy.dot(X, (utilities.rowFromCol(gamma)*X).T)
            w = Z/N
            mu = utilities.colFromRow(F/Z)
            Sigma = S/Z - numpy.dot(mu, mu.T)
            U, s, _ = numpy.linalg.svd(Sigma)
            s[s<psi] = psi
            Sigma = numpy.dot(U, utilities.colFromRow(s)*U.T)
            gmmNew.append((w, mu, Sigma))
        gmm = gmmNew
        
    return gmm

def GMM_EM_diag(X, gmm):
    '''
    EM algorithm for GMM diagonal covariance
    It estimates the parameters of a GMM that maximize the ll for
    a training set X
    '''
    llNew = None
    llOld = None
    G = len(gmm)
    N = X.shape[1]
    psi = 0.01
    while llOld is None or llNew - llOld > 1e-6:
        llOld = llNew
        SJ, SM = logpdfGMM(X,gmm)
        llNew = SM.sum()/N
        P = numpy.exp(SJ-SM)
        gmmNew = []
        
        for g in range(G):
            gamma = P[g, :]
            Z = gamma.sum()
            F = (utilities.rowFromCol(gamma)*X).sum(1)
            S = numpy.dot(X, (utilities.rowFromCol(gamma)*X).T)
            w = Z/N
            mu = utilities.colFromRow(F/Z)
            Sigma = S/Z - numpy.dot(mu, mu.T)
            #diag
            Sigma = Sigma * numpy.eye(Sigma.shape[0])
            U, s, _ = numpy.linalg.svd(Sigma)
            s[s<psi] = psi
            sigma = numpy.dot(U, utilities.colFromRow(s)*U.T)
            gmmNew.append((w, mu, sigma))
        gmm = gmmNew
       
    return gmm

def GMM_EM_tied(X, gmm):
    '''
    EM algorithm for GMM tied covariance
    It estimates the parameters of a GMM that maximize the ll for
    a training set X
    '''
    llNew = None
    llOld = None
    G = len(gmm)
    N = X.shape[1]
    #sigma_list = []
    psi = 0.01
    while llOld is None or llNew - llOld > 1e-6:
        llOld = llNew
        SJ, SM = logpdfGMM(X,gmm)
        llNew = SM.sum()/N
        P = numpy.exp(SJ-SM)
        gmmNew = []
        
        sigmaTied = numpy.zeros((X.shape[0],X.shape[0]))
        for g in range(G):
            gamma = P[g, :]
            Z = gamma.sum()
            F = (utilities.rowFromCol(gamma)*X).sum(1)
            S = numpy.dot(X, (utilities.rowFromCol(gamma)*X).T)
            w = Z/N
            mu = utilities.colFromRow(F/Z)
            Sigma = S/Z - numpy.dot(mu, mu.T)
            sigmaTied += Z*Sigma
            gmmNew.append((w, mu))
        #get tied covariance
        gmm = gmmNew
        sigmaTied = sigmaTied/N
        U,s,_ = numpy.linalg.svd(sigmaTied)
        s[s<psi]=psi 
        sigmaTied = numpy.dot(U, utilities.colFromRow(s)*U.T)
        
        gmmNew=[]
        for g in range(G):
            (w,mu)=gmm[g]
            gmmNew.append((w,mu,sigmaTied))
        gmm=gmmNew
        
        #print(llNew)
    #print(llNew-llOld)
    return gmm

def GMM_LBG(X, doub, version):
   
    init_mu, init_sigma = gaussianModels.mean_cov_estimate(X)
    gmm = [(1.0, init_mu, init_sigma)]
    
    for i in range(doub):
        doubled_gmm = []
        
        for component in gmm: 
            w = component[0]
            mu = component[1]
            sigma = component[2]
            U, s, Vh = numpy.linalg.svd(sigma)
            d = U[:, 0:1] * s[0]**0.5 * 0.1 # 0.1 is alpha
            component1 = (w/2, mu+d, sigma)
            component2 = (w/2, mu-d, sigma)
            doubled_gmm.append(component1)
            doubled_gmm.append(component2)
            if version == "Full":
                gmm = GMM_EM(X, doubled_gmm)
            elif version == "Diagonal":
                gmm = GMM_EM_diag(X, doubled_gmm)
            elif version == "Tied":
                gmm = GMM_EM_tied(X, doubled_gmm)
            
    return gmm


def kFoldGMM(DTR, LTR, DTE, LTE):
    
   
    numpy.random.seed(0)
    indexTR=numpy.random.permutation(DTR.shape[1])
    DTR = DTR[:,indexTR]
    LTR = LTR[indexTR]
    indexTE=numpy.random.permutation(DTE.shape[1])
    DTE = DTE[:,indexTE]
    LTE = LTE[indexTE]
    pi_values = [0.1, 0.5]
    Cfp=Cfn=1
    
    results = PrettyTable()
    results.align = "c"
    results.field_names = ["Target", " Non-Target", "PCA", "minDCF (\pi = 0.1)","minDCF (\pi = 0.5)", "Primary Metric"]

    PCA_values = [0]
    #PCA_values=[5,4,3,2,None]
    
    
    non_target_gmm=["Full","Diagonal", "Tied"]
    target_gmm=["Diagonal"]
    
    doub_fact_target=[1]
    doub_fact_nontarget=[3, 4, 5, 6]
    
    
    for p in PCA_values:
        if p!= 0:
            DTR,P = featureAnalysis.PCA(DTR,p) # fit PCA to training set
            DTE = numpy.dot(P.T,DTE) # transform test samples according to P from PCA on dataset                
        
        print("computing...")
        scores = numpy.array([])
        labels = numpy.array([])
        
            
        DTR0=DTR[:, LTR==0] #Training samples of class 0
        DTR1=DTR[:, LTR==1] #Training samples of class 1

       
            
        for dft in doub_fact_target:              
            for tgmm in target_gmm:
                for dfnt in doub_fact_nontarget:
                    for ntgmm in non_target_gmm:

                       
                            
                            
                        print(f"Computing dft={2**dft} tgmm={tgmm} dfnt={2**dfnt} ntgmm={ntgmm} PCA={p}")
                        gmm_class0=GMM_LBG(DTR0, dfnt, ntgmm)
                        _, SM0=logpdfGMM(DTE, gmm_class0)
                           
                           
                        gmm_class1=GMM_LBG(DTR1, dft, tgmm)
                        _, SM1=logpdfGMM(DTE, gmm_class1)
                            
                        srs=SM1-SM0     
                          
                        labels = numpy.hstack((labels, LTE))
                            
                        scores = numpy.hstack((scores, srs))
                            
                           
                        minDCFS = numpy.zeros(2)
                        for i, pi in enumerate(pi_values):
                            minDCFS[i] = numpy.round(ev.evalF(pi, Cfp, Cfn, scores, labels), 3)
        
                        PrimS = numpy.round((minDCFS[0] + minDCFS[1]) / 2, 3)
                        print(f"PrimS={PrimS} pi(0.1)={minDCFS[0]} pi(0.5)={minDCFS[1]}");
                        
                        results.add_row([str(2**dft)+'-'+tgmm, str(2**dfnt)+'-'+ntgmm, p, minDCFS[0], minDCFS[1], PrimS])
        
        
       
    
                        print("finito");
                        output = str(results)
                        # # save output txt file containing results for each model
                        with open('/content/drive/MyDrive/MachineLearningResults/results.txt', 'a') as file:
                            file.write(f"dft={dft} tgmm={tgmm} dfnt={dfnt} ntgmm={ntgmm} PCA={p} pi(0.1)={minDCFS[0]} pi(0.5)={minDCFS[1]} PrimS={PrimS}" + '\n')
                                
                                
                           
                    
                    
def GMM_scores(DTR, LTR, DTE, dfnt, ntgmm, dft, tgmm ):
    scores = numpy.array([])
    DTR0=DTR[:, LTR==0] #Training samples of class 0
    DTR1=DTR[:, LTR==1] #Training samples of class 1
    gmm_class0=GMM_LBG(DTR0, dfnt, ntgmm)
    _, SM0=logpdfGMM(DTE, gmm_class0)
    
   
    gmm_class1=GMM_LBG(DTR1, dft, tgmm)
    _, SM1=logpdfGMM(DTE, gmm_class1)
    
    srs=SM1-SM0     
    
    
    scores = numpy.hstack((scores, srs))
    return scores
                    
                    
                    

                    


    


