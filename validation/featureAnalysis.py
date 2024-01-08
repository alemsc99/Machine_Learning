import numpy
import utilities
import scipy.linalg
import matplotlib.pyplot as plt
def cumulative_explained_variance(D):
    N = D.shape[1]  # number of samples

    # Calcola i valori propri della matrice di covarianza
    mu = D.mean(1)
    mu = utilities.colFromRow(mu)
    DC = D-mu
    
    C = numpy.dot(DC, DC.T)/N #Covariance matrix
    eigenvalues, _ = numpy.linalg.eigh(C)
    
    # Ordina i valori propri in ordine decrescente
    eigenvalues = eigenvalues[::-1]
    
    # Calcola la varianza spiegata per ogni componente principale
    explained_variance = eigenvalues / numpy.sum(eigenvalues)
    y_min, y_max = plt.ylim()
    y_values = numpy.linspace(y_min, y_max, 20)
    plt.yticks(y_values)
    plt.xlim(right=5)
    # Creare un grafico della varianza spiegata
    plt.plot(numpy.cumsum(explained_variance))
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.grid()
    plt.show()
    
def PCA(D, m):
    N = D.shape[1]  # number of samples

    # centering data
    mu = D.mean(1)
    mu = utilities.colFromRow(mu)
    DC = D-mu
    
    C = numpy.dot(DC, DC.T)/N #Covariance matrix
    
    s, U = numpy.linalg.eigh(C)  # dato che C è simmetrica possiamo usarla
    
    P = U[:, ::-1][:, 0:m]
    
    DP = numpy.dot(P.T, D) #projected dataset
    return DP, P


def LDA(D,L, m):
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
    # compute SB
    summ = 0
    for i in range(NC):
        temp = muC[i] - mu
        summ += numpy.dot(temp, temp.T) * samples[i].shape[1]
    SB = summ/N

    # Generalized eigenvalue problem
    s, U = scipy.linalg.eigh(SB, SW)

    W = U[:, ::-1][:, 0:m]

    return numpy.dot(W.T, D) #projected dataset

    