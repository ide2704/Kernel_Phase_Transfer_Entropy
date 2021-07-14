# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 21:06:13 2020

@author: ide2704

Kernel-based phase transfer entropy

Ivan De La Pava Panche, Automatics Research Group
Universidad Tecnologica de Pereira, Pereira - Colombia
email: ide@utp.edu.co
"""
# Import the necessary libraries 
import numpy as np
import scipy.spatial as sp_spatial

# =============================================================================
# Transfer entropy 
# =============================================================================

def embeddingX(x,tau,dim,u):
    """
    Time delay embbeding source time series x
    INPUT = x in R^{T}
          = tau in R (time delay)
          = dim in R (dimensions)
    OUTPUT = X_emb in R^{Lxdim}
    """
    T = np.size(x)
    L = T -(dim-1)*tau
    firstP = T - L
    X_emb = np.zeros((L,dim))
    for i in range(L):
      for j in range(dim):
        X_emb[i,j] = x[i+firstP-(j*tau)]
    
    X_emb = X_emb[0:-u,:]
    return X_emb

def embeddingY(y,tau,dim,u):
    """
    Time delay embbeding target time series y 
    INPUT = y in R^{T}
          = tau in R (time delay)
          = dim in R (dimensions)
    OUTPUT = Y_emb in R^{Lxdim}
    """
    T = np.size(y)
    L = T -(dim-1)*tau
    firstP = T - L
    Y_emb = np.zeros((L,dim))
    
    for i in range(L):
      for j in range(dim):
        Y_emb[i,j] = y[i+firstP-(j*tau)]
    
    y_t = y[firstP+u::] 
    y_t = y_t.reshape(y_t.shape[0],1)
    Y_emb = Y_emb[u-1:-1,:]
    return Y_emb,y_t

def GaussianKernel(X):
    """
    Compute Gaussian Kernel
    INPUT: X in R^{Lxdim}
    OUTPUT: A in R^{LxL}
    """
    utri_ind =  np.triu_indices(X.shape[0], 1)
    dist = sp_spatial.distance.cdist(X,X,'euclidean')
    sigma = np.median(dist[utri_ind])
    K = np.exp(-1*(dist**2)/(2*sigma**2))
    return K

def kernelRenyiEntropy(K_lst,alpha):
    """
    INPUT: A in R^{LxL}
    OUTPUT: h in R
    """
    if len(K_lst) == 1:
      K = K_lst[0]
    elif len(K_lst) == 2:
      K = K_lst[0]*K_lst[1]
    else:
      K = K_lst[0]*K_lst[1]*K_lst[2]     
    K = K/np.trace(K)  
    h = np.real((1/(1-alpha))*np.log2(np.trace(np.linalg.matrix_power(K,alpha))))
    return h

def kernelTransferEntropy(x,y,dim,tau,u,alpha): 
    """
    Compute transfer entropy from channel x to channel y
    INPUT: X in R^{Lxdim}
    OUTPUT: A in R^{LxL}
    """
    dim = int(dim)
    tau = int(tau)
    u = int(u)
    
    X_emb = embeddingX(x,tau,dim,u)
    Y_emb, y_t = embeddingY(y,tau,dim,u)
    
    K_X_emb = GaussianKernel(X_emb)
    K_Y_emb = GaussianKernel(Y_emb)
    K_y_t = GaussianKernel(y_t)    
    
    h1 = kernelRenyiEntropy([K_X_emb,K_Y_emb],alpha)
    h2 = kernelRenyiEntropy([K_X_emb,K_Y_emb,K_y_t],alpha)
    h3 = kernelRenyiEntropy([K_Y_emb,K_y_t],alpha)
    h4 = kernelRenyiEntropy([K_Y_emb],alpha)
    
    TE = h1 - h2 + h3 - h4
    return TE

def kernelTransferEntropy_AllCh(X,Dim,Tau,u,alpha): 
    """
    Compute transfer entropy among all channels in X 
    INPUT: X in R^{Lxdim}
    OUTPUT: A in R^{LxL}
    """
    num_ch = X.shape[0]
    TE = np.zeros([num_ch,num_ch])
    for j in range(num_ch):
        # Target channel  
        y = X[j,:]  
        # Embedding parameters 
        tau = int(Tau[j])
        dim = int(Dim[j])  
        u = int(u)
        # Time embeddings for y 
        Y_emb, y_t = embeddingY(y,tau,dim,u)
        # Kernels for y's time embeddings 
        K_Y_emb = GaussianKernel(Y_emb)
        K_y_t = GaussianKernel(y_t)   
        # Entropies 
        h3 = kernelRenyiEntropy([K_Y_emb,K_y_t],alpha)
        h4 = kernelRenyiEntropy([K_Y_emb],alpha)
        for i in range(num_ch):
            if i!=j:
                # Source channel 
                x = X[i,:]
                # Time embedding for x 
                X_emb = embeddingX(x,tau,dim,u)
                # Kernels for x's time embedding
                K_X_emb = GaussianKernel(X_emb)
                # Entropies 
                h1 = kernelRenyiEntropy([K_X_emb,K_Y_emb],alpha)
                h2 = kernelRenyiEntropy([K_X_emb,K_Y_emb,K_y_t],alpha)
                # Transfer entropy
                TE[i,j] =  h1 - h2 + h3 - h4
    return TE

def kernelTransferEntropy_AllCh_freq(X,Dim,Tau,u_matrix,alpha,freq,time,component='phase'): 
    """
    Compute transfer entropy among all channels in X 
    INPUT: X in R^{Lxdim}
    OUTPUT: A in R^{LxL}
    """
                
    num_ch = X.shape[0]
    TE_ph = np.zeros([num_ch,num_ch])
    
    # Wavelet decomposition 
    X_ph = Wavelet_Trial_Dec(X,time,freq,component)
    
    for j in range(num_ch):
        # Embedding parameters 
        tau = int(Tau[j])
        dim = int(Dim[j])  
        for i in range(num_ch):
            if i!=j:
                # Delay time 
                u = int(u_matrix[i,j])
            
                # Target channel  
                y = X_ph[j,:]  
                # Source channel 
                x = X_ph[i,:]
                
                # Time embeddings for y 
                Y_emb, y_t = embeddingY(y,tau,dim,u)
                # Kernels for y's time embeddings 
                K_Y_emb = GaussianKernel(Y_emb)
                K_y_t = GaussianKernel(y_t)   
                # Entropies 
                h3 = kernelRenyiEntropy([K_Y_emb,K_y_t],alpha)
                h4 = kernelRenyiEntropy([K_Y_emb],alpha)
        
                # Time embedding for x 
                X_emb = embeddingX(x,tau,dim,u)
                # Kernels for x's time embedding
                K_X_emb = GaussianKernel(X_emb)
                # Entropies 
                h1 = kernelRenyiEntropy([K_X_emb,K_Y_emb],alpha)
                h2 = kernelRenyiEntropy([K_X_emb,K_Y_emb,K_y_t],alpha)
               
                # Transfer entropy
                TE_ph[i,j] =  h1 - h2 + h3 - h4
    return TE_ph
            
# =============================================================================
# Embedding functions 
# =============================================================================

def autocorrelation(x):
    """
    Autocorrelation of time series
    INPUT  = x in R^{T}
    OUTPUT =
    """
    xp = (x - np.mean(x))/np.std(x)
    result = np.correlate(xp, xp, mode='full')
    return result[int(result.size/2):]/len(xp)

def autocorr_decay_time(x,maxlag):
    """ 
    Autocorrelation decay time (embedding time)
    INPUT = x in R^{T}
          = maxlag in R (maximum delay time)
    OUTPU = act in R (time delay for embedding)
    """
    autocorr = autocorrelation(x)
    thresh = np.exp(-1)
    aux = autocorr[0:maxlag];
    aux_lag = np.arange(0,maxlag)
    if len(aux_lag[aux<thresh]) == 0:
        act = maxlag
    else:
        act = np.min(aux_lag[aux<thresh])
    return act

def cao_criterion(x,d_max,tau):
    """ Cao criterion (embedding dimension)
    """
    tau = int(tau)
    N = len(x)
    d_max = int(d_max)+1 
    x_emb_lst = []
    
    for d in range(d_max):
        # Time embedding 
        T = np.size(x)
        L = T-(d*tau)
        if L>0:
            FirstP = T-L
            x_emb = np.zeros((L,d+1))
            for ii in range(0,L):
                for jj in range(0,d+1): 
                    x_emb[ii,jj] = x[ii+FirstP-(jj*tau)]
            x_emb_lst.append(x_emb)
    
    d_aux = len(x_emb_lst)
    E = np.zeros(d_aux-1)
    for d in range(d_aux-1):
        emb_len = N-((d+1)*tau)
        a = np.zeros(emb_len)
        for i in range(emb_len): 
            var_den = x_emb_lst[d][i,:]-x_emb_lst[d][0:emb_len,:]
            inf_norm_den = np.linalg.norm(var_den,np.inf,axis=1)
            inf_norm_den[inf_norm_den==0] = np.inf
            den = np.min(inf_norm_den)
            ind = np.argmin(inf_norm_den)
            num = np.linalg.norm(x_emb_lst[d+1][i,:]-x_emb_lst[d+1][ind,:],np.inf)
            a[i] = num/den
        E[d] = np.sum(a)/emb_len
    
    E1 = np.roll(E,-1)  # circular shift
    E1 = E1[:-1]/E[:-1]
    
    dim_aux = np.zeros([1,len(E1)-1])
    
    for j in range(1,len(E1)-1):
        dim_aux[0,j] = E1[j-1]+E1[j+1]-2*E1[j]
    dim_aux[dim_aux==0] = np.inf
    dim = np.argmin(dim_aux)+1

    return dim

# =============================================================================
# Wavelet transform
# =============================================================================

def Morlet_Wavelet(data,time,freq):
    """
    Morlet wavelet decomposition 
    
    Parameters
    ----------
    data : input signal (Nx1) 
    time : time vector (must be sampled at the sampling frequency o data, best practice is to have time=0 at the center of the wavelet)
    freq : frequencies to evaluate in Hz

    Returns
    -------
    dataW : dictionary containing the Morlet wavelet decomposition 

    """
    # =============================================================================
    # Create the Morlet wavelets
    # =============================================================================
    num_freq = len(freq); 
    cmw = np.zeros([data.shape[0],num_freq],dtype = 'complex_')
    
    # Number of cycles in the wavelets 
    range_cycles = [3,10]
    max_freq = 60 
    freq_vec = np.arange(1,max_freq+1)
    nCycles_aux = np.logspace(np.log10(range_cycles[0]),np.log10(range_cycles[-1]),len(freq_vec))
    nCycles = np.array([nCycles_aux[np.argmin(np.abs(freq_vec - freq[i]))] 
                        for i in range(num_freq)])
    
    for ii in range(num_freq): 
        # create complex sine wave
        sine_wave = np.exp(1j*2*np.pi*freq[ii]*time)
        
        # create Gaussian window
        s = nCycles[ii]/(2*np.pi*freq[ii]) #this is the standard deviation of the gaussian
        gaus_win  = np.exp((-time**2)/(2*s**2))
        
        # now create Morlet wavelet
        cmw[:,ii] = sine_wave*gaus_win
        
    # =============================================================================
    # Convolution 
    # =============================================================================
    
    # Define convolution parameters 
    nData = len(data)
    nKern = cmw.shape[0]
    nConv = nData + nKern - 1
    half_wav = int(np.floor(cmw.shape[0]/2)+1)
    
    # FFTs
    
    # Note that the "N" parameter is the length of convolution, NOT the length
    # of the original signals! Super-important!
    
    # FFT of wavelet, and amplitude-normalize in the frequency domain
    cmwX = np.fft.fft(cmw,nConv,axis=0)
    cmwX = cmwX/np.max(cmwX,axis=0)
        
    # FFT of data
    dataX = np.fft.fft(data,nConv)
    dataX = np.repeat(dataX.reshape([-1,1]),num_freq,axis=1)
    
    # Now for convolution...
    data_wav = np.fft.ifft(dataX*cmwX,axis=0)
    
    # Cut 1/2 of the length of the wavelet from the beginning and from the end
    data_wav = data_wav[half_wav-2:-half_wav,:]
    
    # Extract filtered data, amplitude and phase 
    data_wav = data_wav.T
    dataW = {}
    dataW['filt'] = np.real(data_wav)
    dataW['amp'] = np.abs(data_wav)
    dataW['phase'] = np.angle(data_wav)
    dataW['f'] = freq 
    
    return dataW

def Wavelet_Trial_Dec(trial_data,time,freq_range,component ='phase'):
    
    if np.size(freq_range) == 1:
        freq_range = [freq_range]

    # Time centering 
    t = (time.flatten())/1000 # ms to s
    t = t - t[0]
    t = t-(t[-1]/2) # best practice is to have time=0 at the center of the wavelet

    if (trial_data.shape[1] % 2) == 0:
        wav_phase_dec = np.zeros([trial_data.shape[0],trial_data.shape[1]-1,len(freq_range)])
    else:
        wav_phase_dec = np.zeros([trial_data.shape[0],trial_data.shape[1],len(freq_range)])

    for ch in range(trial_data.shape[0]):
       
        # Data detrending
        ch_data = trial_data[ch,:]
        ch_data = ch_data - np.mean(ch_data)
        
        # Data decomposition 
        dataW = Morlet_Wavelet(ch_data,t,freq_range)
        wav_phase_dec[ch,:,:] = dataW[component].T
    
    return wav_phase_dec

# =============================================================================
# Permutation test 
# =============================================================================

def permutation_test(Te,Te_sh,alpha):
    '''
    input
    TE: Nx2 connectivity matrix, where N is the number of trials. The rows
         of TE hold effective connectivity values between 2 channels x and y.
         For instance, the first row of TE holds elements of the form
         [TE(x1->y1),TE(y1->x1)], where x1 and y1 stand for channels x and y
         of trial 1.
    TE_sh : Nx2 shuffled connectivity matrix, where N is the number of trials. The rows
          of TE_sh hold effective connectivity values between 2 channels x and y.
          For instance, the first row of TE_sh holds elements of the form
          [TE(x1->y2),TE(y1->x2)], where x1 and y1 stand for channels x and y
          of trial 1, and x2 and y2 stand for channels x and y of trial 2.  
    alpha : alpha level of the permutation test (usually 0.05 or
          0.01).
   
    Output:
    TEpermvalues: Value of the permutation test
    signigicance: Statistical significance (based on the value of the
                   permutation test and the stablished alpha level)          
    '''
    mean_TE = np.mean(Te,axis=0)
    mean_TE_sh = np.mean(Te_sh,axis=0)
    Testatistic = np.abs(mean_TE-mean_TE_sh)

    dTE = Te[:,0]-Te[:,1]
    mean_dTE = np.mean(dTE)
    dTE_sh = Te_sh[:,0] - Te_sh[:,1]
    mean_dTE_sh = np.mean(dTE_sh)
    dTestatistic =  np.abs(mean_dTE-mean_dTE_sh)
    
    # Permutation #
    #nrcmc = 2
    n = Te.shape[0]
    m = Te_sh.shape[0]
    numperm = 10000
    dist_1 = np.zeros((numperm,Te.shape[1]))
    dist_2 = np.zeros((numperm,Te_sh.shape[1]))
    
    ddist_1 = np.zeros(numperm)
    ddist_2 = np.zeros(numperm)
    
    for l in range(numperm):
        data_pool = np.concatenate((Te,Te_sh),axis=0)
        # permuting indexes from 1 to n+m
        ind = np.argsort(np.random.rand(1,n+m))
        ind_1 = ind[0,:n]
        data_1 = data_pool[ind_1,:]
        ind_2 = ind[0,n:n+m]
        data_2 = data_pool[ind_2,:]
        dist_1[l,:] = np.mean(data_1,axis=0)
        dist_2[l,:] = np.mean(data_2,axis=0)
        
        data_pool_d = np.concatenate((dTE,dTE_sh))
        ddist_1[l] = np.mean(data_pool_d[ind_1])
        ddist_2[l] = np.mean(data_pool_d[ind_2])
        
    Tepermdist = np.abs(dist_1-dist_2) 
    dTepermdist = np.abs(ddist_1-ddist_2)
    
    Tepermvalues = np.zeros(2)
    significance = np.zeros(2)
    for i in range(2):
        Tepermvalues[i] = len(np.where(Tepermdist[:,i]>Testatistic[i])[0])/numperm
        significance[i] = Tepermvalues[i]<=alpha
    
    dTepermvalues = len(np.where(dTepermdist>dTestatistic)[0])/numperm
    dsignificance = float(dTepermvalues<=alpha)
        
    return Tepermvalues,significance,dTepermvalues,dsignificance