import utilities
import svmPolyEval
# import svmRadial
import GMMEval


if __name__=="__main__":
    #Loading the evaluation dataset
    DTR,LTR=utilities.load_dataset("/content/Machine_Learning/Train.txt")
    DTE,LTE=utilities.load_dataset("/content/Machine_Learning/Test.txt")
    
    
   #SVMpoly
    #svmPolyEval.kFoldSvmP(DTR, LTR, DTE, LTE)
    
    #SVMRadial    
    #svmRadial.kFoldSvmR(DTR, LTR)
    
    #GMM
    GMMEval.kFoldGMM(DTR, LTR, DTE, LTE)
    
    #Score calibration
     #utilities.bayes_plot(DTR, LTR)
    #svmRadial.score_calib(DTR, LTR)
