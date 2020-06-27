import scipy as scipy
import scipy.special
import numpy as numpy 

def getepsilon_nonconvex(k,N,beta): 
    """Compute upper bound for non-convex scenario programs. 
    Inputs
    ----------
    k: Number of support scenarios
    N: Number of scenarios in the dataset
    beta: a small confidence parameter (beta=10^-8 means high confidence level)  
    Returns
    -------
    epsil : upper bound on the violation probability
    """ 
    if k < N:
    #  this is numerically for high N and k
    #  epsilon=1-(beta./(N.*nchoosek(N,k))).^(1/(N-k));
    #  This expansion is slower but guarantees stability (and numerical accuracy)
            E = (beta*k/N**2)**(1/(N-k))
            for l in range(k-1):
                Temp = (1/((N-l)/(k-l)))**(1/(N-k))
                E = E*Temp    
                epsil = 1-E 
    elif k == N:
         epsil=1
    elif k > N: 
        epsil= 'ERROR: numbber of support constraints k should be k<=N' 
    return epsil  

def getepsilon_relaxedConstraints(k,N,bet):  
    """Compute upper bound for convex scenario programs wit soft-constraints. 
    Inputs
    ----------
    k: Number of support scenarios
    N: Number of scenarios in the dataset
    beta: a small confidence parameter (beta=10**-8 means high confidence level)  
    Returns
    -------
    epsL,epsU : lower and upper bound on the violation probability
    """ 
    alphaL = scipy.special.betaincinv(k,N-k+1,bet)
    alphaU = 1- scipy.special.betaincinv(N-k+1,k,bet) 
    Temp=numpy.log(range(k,N+1))
    Temp[0]=0
    #Temp=numpy.sort(Temp)[::-1]
    CUMSUM=numpy.cumsum(numpy.sort(Temp)[::-1])
    CUMSUM[-1]=0
    aux1=numpy.sort(CUMSUM)[::-1]# auxiliary variables 1
    # auxiliary variables 2 
    Temp=numpy.log(range(1,N+2-k))
    Temp[-1]=0
    #Temp=numpy.sort(Temp)[::-1]
    CUMSUM=numpy.cumsum(numpy.sort(Temp)[::-1])
    CUMSUM[-1]=0
    aux2=numpy.sort(CUMSUM)[::-1]
    Temp=numpy.sort(numpy.log(range(N+1,4*N+1)))  
    aux3=numpy.sort(numpy.cumsum(Temp))  # auxiliary variables 3 
    Temp=numpy.sort(numpy.log(range(N+1,4*N+1)))  
    aux3=numpy.sort(numpy.cumsum(Temp))
    Temp=numpy.sort(numpy.log(range(N+1-k,4*N+1-k)))  
    aux4=numpy.sort(numpy.cumsum(Temp))  # auxiliary variables 4 
    coeffs1 = aux2-aux1 
    coeffs2= aux3-aux4 
    t1 = 1-alphaL 
    t2 = 1   
    M1=numpy.sort(range(0,N-k+1))[::-1]
    M2=range(1,3*N+1)
    poly1 = 1+bet/(2*N)-bet/(2*N)*numpy.sum(numpy.exp(coeffs1-M1*numpy.log(t1))) -bet/(6*N)*numpy.sum(numpy.exp(coeffs2 +M2*numpy.log(t1))) 
    poly2 = 1+bet/(2*N)-bet/(2*N)*numpy.sum(numpy.exp(coeffs1-M1*numpy.log(t2))) -bet/(6*N)*numpy.sum(numpy.exp(coeffs2 +M2*numpy.log(t2)))
    # Now it calculates the lower and upper bounds
    if ((poly1*poly2) > 0):
        epsL = 0;
    else:
        while t2-t1 > 1e-10:
            t = (t1+t2)/2;
            polyt = 1+bet/(2*N)-bet/(2*N)*numpy.sum(numpy.exp(coeffs1 - M1*numpy.log(t)))  -bet/(6*N)*numpy.sum(numpy.exp(coeffs2 + M2*numpy.log(t)))  
            if polyt > 0:
                t1=t
            else:
                t2=t 
        epsL = 1-t2
    t1 = 0
    t2 = 1-alphaU
    poly2 = 1+bet/(2*N)-bet/(2*N)*numpy.sum(numpy.exp(coeffs1-M1*numpy.log(t2))) -bet/(6*N)*numpy.sum(numpy.exp(coeffs2 + M2*numpy.log(t2))) 

    while t2-t1 > 1e-10:
            t = (t1+t2)/2  
            polyt = 1+bet/(2*N)-bet/(2*N)*numpy.sum(numpy.exp(coeffs1 - M1*numpy.log(t)))  -bet/(6*N)*numpy.sum(numpy.exp(coeffs2 + M2*numpy.log(t)))
            if polyt > 0:
                t2=t
            else:
                t1=t  
            epsU = 1-t1
    #epsilon=(epsL,epsU) # resulting lower and upper bound on the violation probability
    return (epsL,epsU)

### Sample-and-discard bounds
def getepsilon_apriori_convex_discard(k,N,beta,Nd): 
    """Plot the decision boundaries for a classifier. 
    Parameters
    ----------
     k: Number of samples removed from the data set
     N: Number of scenarios in the dataset
     beta: s small confidence parameter (beta=10^-8 means high confidence level)   
     Nd: Nuber of dimensions of the decision variables"""
    
    EPSILON= numpy.linspace(0.0001,0.9999,10**4)
    BETA=0
    for j in range(0,k+Nd):
        BETA=BETA+scipy.special.binom(N,j)*EPSILON**j*(1-EPSILON)**(N-j)
    
    BETA=BETA*scipy.special.binom(k+Nd-1,k);
    Temp=EPSILON[(BETA<=beta)]
    epsil=Temp[0] 
    
    return epsil
