import utilities 
import ev
import numpy
import featureAnalysis
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import datetime
import models_llr


DTR, LTR = utilities.load_dataset("../Train.txt")
DTE, LTE = utilities.load_dataset("../Test.txt")


PCA = [5,None] #number of dimension to keep in PCA




C_list = numpy.logspace(-4,1,6).tolist() 
K_list = numpy.logspace(-2,0,3).tolist() 
gamma_list = numpy.logspace(-2,0,3).tolist() 
# working points
Cfn = 1
Cfp = 1
pi_list = [0.1, 0.5]

results = PrettyTable()
results.align = "c"
results.field_names = ["K", "C", "PCA", "Kernel", "minDCF (pi = 0.1)", "minDCF (pi = 0.5)", "Cprim"]

numpy.random.seed(0)
indexTR=numpy.random.permutation(DTR.shape[1])
DTR = DTR[:,indexTR]
LTR = LTR[indexTR]
indexTE=numpy.random.permutation(DTE.shape[1])
DTE = DTE[:,indexTE]
LTE = LTE[indexTE]
# produce a graph for each K,c,d: on x plot different C used for training, on Y plot relative Cprim obtained
for K_num, K in enumerate(K_list):
    
    for PCA_m in PCA:
        if PCA_m!= 0:
            DTR,P = featureAnalysis.PCA(DTR,PCA_m) # fit PCA to training set
            DTE = numpy.dot(P.T,DTE) # transform test samples according to P from PCA on dataset                
        
    
        st = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        print(f"### {st}: Starting SVM RBF with K = {K}, PCA = {PCA_m}") #feedback print
        
        #set graph
        fig, ax = plt.subplots() 
        ax.set_xscale('log')
        ax.set(xlabel='C', ylabel='Cprim', title=f'SVM K={K} - RBF Kernel')
        plt.grid(True)
        plt.xticks(C_list)
    
        for gamma in gamma_list:    
            Cprim_list = numpy.array([])
            for C in C_list:
            #for each C compute minDCF after K-fold
                        
                scores_pool = numpy.array([])
                labels_pool = numpy.array([])
                
                
                    
                    #DTR, DTE = utilities.apply_Z_Norm(DTR,DTE)
                    #pool test scores and test labels in order to compute the minDCF on complete pool set
                labels_pool = numpy.hstack((labels_pool,LTE))
                scores_pool = numpy.hstack((scores_pool,models_llr.SVM_RBF(DTR, LTR, DTE, C, K, gamma)))
                 
                #compute minDCF for the current SVM with/without PCA for the 2 working points  
                minDCF = numpy.zeros(2)
                for i, pi in enumerate(pi_list):
                    minDCF[i] = ev.compute_min_DCF(scores_pool, labels_pool, pi, Cfn, Cfp)
                #compute Cprim (the 2 working points average minDCF) and save it in a list for plotting after
                Cprim = numpy.round((minDCF[0] + minDCF[1])/ 2 , 3)
                print(f"Cprim={Cprim}")
                Cprim_list = numpy.hstack((Cprim_list,Cprim))
                # add current result to table
                results.add_row([K, C, PCA_m, f"RBF(γ = {gamma}) ", numpy.round(minDCF[0],3), numpy.round(minDCF[1],3), Cprim])
                print(f"\t...computed C={C}, γ={gamma}") #feedback print
                
            #plot the graph
            ax.plot(C_list, Cprim_list, label =f'K={K} (PCA 2)')
            print(f"\tCprim values for γ={gamma}: {Cprim_list}") #feedback print         
    
        print(f'Completed SVM RBF with K = {K} and PCA = {PCA_m} ###') #feedback print
        plt.legend()
        fig.savefig(f"C:\\Users\\masci\\Desktop\\PoliTo\\Secondo_semestre\\Machine_Learning\\exam_project\\evaluation\\evaluationResults\\SVMRadial\\K{K}_PCA{PCA_m}.png", dpi=200)
        plt.show()
    
# print and save as txt the final results table
print(results)
data = results.get_string()

with open('C:\\Users\\masci\\Desktop\\resultsEvalSVMRBF.txt', 'w') as file:
    file.write(data)
