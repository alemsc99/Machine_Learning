import matplotlib.pyplot as plt


def plot_scatter(D, L):
    D0 = D[:, L == 0]  # considering only samples belonging to class 0
    D1 = D[:, L == 1]  # considering only samples belonging to class 1

    hFea = {
        0: 'Feature 1',
        1: 'Feature 2',
        2: 'Feature 3',
        3: 'Feature 4',
        4: 'Feature 5',
        5: 'Feature 6',
        }
    N = len(hFea.keys())
    
    for idx in range(N):
        for ids in range(N):
            if(ids != idx):
                plt.figure()
                plt.xlabel(hFea[idx])
                plt.ylabel(hFea[ids])
                plt.scatter(D0[idx, :], D0[ids, :], linestyle='',
                            marker='.', color="red", alpha=0.5, label="Non-Target Language")
                plt.scatter(D1[idx, :], D1[ids, :], linestyle='',
                            marker='.', color="green", alpha=0.5, label='Target Language')
                plt.legend()
                plt.tight_layout()
                plt.show()


def plot_hist(D, L):
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]

    hFea={
       0:'Feature 1',
       1:'Feature 2',
       2:'Feature 3', 
       3:'Feature 4',
       4:'Feature 5', 
       5:'Feature 6',
       }
    N=len(hFea.keys())
    
    
    for idx in range(N):
        plt.figure()
        plt.xlabel(hFea[idx])
        plt.hist(D0[idx, :], bins=20, density=True, color="red", alpha=0.5, edgecolor="k", label='Non-Target Language') 
        # prendo solo la idx-th riga tra le colonne con label uguale 0
        plt.hist(D1[idx, :], bins=20, density=True, color="green", alpha=0.5,edgecolor="k", label='Target Language')       
        plt.legend()
        plt.tight_layout()
    plt.show()
    
    
def plot_scatter_PCA(D, L):
    
    D0 = D[:, L == 0]  # considering only samples belonging to class 0
    D1 = D[:, L == 1]  # considering only samples belonging to class 1
    
    plt.figure()
    plt.scatter(D0[0, :], D0[1, :], color="red", alpha=0.5, label='Non-Target Language')
    plt.scatter(D1[0, :], D1[1, :], color="green", alpha=0.5,label='Target Language')

    
    #plt.title(title)
    plt.legend()
    plt.show()
    
    
def plot_hist_LDA(D, L):
    D0 = D[:, L == 0]  # considering only samples belonging to class 0
    D1 = D[:, L == 1]  # considering only samples belonging to class 1
    
    plt.figure()
    plt.hist(D0[0, :], bins=100, color="red", alpha=0.5, label='Non-Target Language')
    plt.hist(D1[0, :], bins=100, color="green", alpha=0.5,label='Target Language')

    
    #plt.title(title)
    plt.legend()
    plt.show()
    
    
def plot_lr(l_list, p, primSlist):
    fig,ax = plt.subplots()
    
    
    for key in primSlist:
        if key==None:
            strK="None"
        elif (key != 0.1 and key != 0.5):
            strK="Dataset"
        else:
            strK=key
            
        ax.plot(l_list, primSlist[key],label=strK)
        
        
    ax.set_xscale('log')
    ax.set(xlabel='Lambda', ylabel='Primary Metric', title=f'PCA={p}')
    ax.legend()
    plt.xticks(l_list)
    #plt.yticks(Cprim_list)
    plt.grid(True)
    fig.savefig(f"validationResults/logisticRegression/PCA{p}.png", dpi=200)
    
    plt.show()
    
def plot_qlr(l_list, p, primSlist):
    fig,ax = plt.subplots()
    
    
    for key in primSlist:
        if key==None:
            strK="None"
        elif (key != 0.1 and key != 0.5):
            strK="Dataset"
        else:
            strK=key
            
        ax.plot(l_list, primSlist[key],label=strK)
        
        
    ax.set_xscale('log')
    ax.set(xlabel='Lambda', ylabel='Primary Metric', title=f'PCA={p}')
    ax.legend()
    plt.xticks(l_list)
    #plt.yticks(Cprim_list)
    plt.grid(True)
    fig.savefig(f"validationResults/quadraticLogisticRegression/PCA{p}.png", dpi=200)
    
    plt.show()
    
    
    
    
    
def plot_svm(c_list, k, primSlist):
    fig,ax = plt.subplots()
    
    
    for key in primSlist:
        
        if key==0:
            strK="PCA None"
        else:
            strK="PCA "+ str(key)
            
        ax.plot(c_list, primSlist[key],label=strK)
        
        
    ax.set_xscale('log')
    ax.set(xlabel='C', ylabel='Primary Metric', title=f'K={k}')
    ax.legend()
    plt.xticks(c_list)
    #plt.yticks(Cprim_list)
    plt.grid(True)
    fig.savefig(f"validationResults/SVM/K{k}.png", dpi=200)
    
    plt.show()
    
    
    
    
       
def plot_svmP(c_list, k, coeff,  primSlist):
    fig,ax = plt.subplots()
    
    
    for key in primSlist:
        
        if key==0:
            strK="PCA None"
        else:
            strK="PCA "+ str(key)
            
        ax.plot(c_list, primSlist[key],label=strK)
        
        
    ax.set_xscale('log')
    ax.set(xlabel='C', ylabel='Primary Metric', title=f'K={k} c={coeff}')
    ax.legend()
    plt.xticks(c_list)
    #plt.yticks(Cprim_list)
    plt.grid(True)
    fig.savefig(f"validationResults/SVMPoly/K{k}_c{coeff}.png", dpi=200)
    
    plt.show()
     
    
    
    
       
def plot_svmR(c_list, k, gamma,  primSlist):
    fig,ax = plt.subplots()
    
    
    for key in primSlist:
        
        if key==0:
            strK="PCA None"
        else:
            strK="PCA "+ str(key)
            
        ax.plot(c_list, primSlist[key],label=strK)
        
        
    ax.set_xscale('log')
    ax.set(xlabel='C', ylabel='Primary Metric', title=f'K={k}  γ={gamma}')
    ax.legend()
    plt.xticks(c_list)
    #plt.yticks(Cprim_list)
    plt.grid(True)
    fig.savefig(f"validationResults/SVMRadial/K{k}_gamma{gamma}.png", dpi=200)
    
    plt.show()
         

    