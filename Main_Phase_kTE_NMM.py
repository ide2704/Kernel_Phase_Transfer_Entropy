# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 19:53:07 2021

@author: ide2704

Kernel-based phase transfer entropy

This script reproduces the results for the kernel-based phase transfer 
entropy method shown in Figure 5 (column A, second topmost row, for a coupling 
strength of 0.5) of the paper ""Kernel-based phase transfer entropy with 
enhanced feature relevance analysis for brain computer interfaces"". 

Ivan De La Pava Panche, Automatics Research Group
Universidad Tecnologica de Pereira, Pereira - Colombia
email: ide@utp.edu.co
"""

import os
import sys
import numpy as np
import scipy.io as sio
from joblib import Parallel,delayed
import TransferEntropy as TE

# Add current working directory to sys path 
sys.path.append(os.getcwd())

# Load simulated data 

NMM_data = sio.loadmat('NMM_data.mat')
f_sample = NMM_data['fs'][0].astype('float')  # Sampling frequency 
t = NMM_data['t'].flatten()
data = NMM_data['NMM_data']

num_trials = data.shape[0]
coupling = [0,0.2,0.5,0.8]
coupling_ind = [2]#[0,1,2,3]
 
freq_range = np.arange(2,62,2) # frequency of wavelet, in Hz (2 to 60 Hz)
num_freq = len(freq_range)

# Creating shuffled data for statistical testing 
ind_shuffle = np.roll(np.arange(0,num_trials),-1)
data_sh = data[ind_shuffle,:]

# Time limits
t_lim = [0.5,2.5] #The simulated signals will be segmented between t_lim[0] y t_lim[-1] seconds

tau_lst = []
dim_lst = []
u_lst = []
kTE_u_lst = []
kTE_lst = []
kTE_sh_lst = []

# =============================================================================
# %% Phase kernel transfer entropy estimation
# =============================================================================
for j in coupling_ind: 
  
    u_vec = np.arange(1,11)
    tau_matrix = np.zeros((num_trials,2))
    dim_matrix = np.zeros((num_trials,2))
    u_matrix = np.zeros((num_trials,2))
    kTE_u_matrix = np.zeros((num_trials,len(u_vec),2))
    kTE_matrix = np.zeros((num_trials,num_freq,2))
    kTE_sh_matrix = np.zeros((num_trials,num_freq,2))
    
    for k in range(num_trials):
        print('Coupling {}, trial {}/{}'.format(coupling[j],k+1,num_trials))
        
        # Data downsampling 
        t_ind = (t>=t_lim[0])&(t<=t_lim[1])
        X = data[k,j][:,t_ind]
        X_sh = data_sh[k,j][:,t_ind]
        
        # Data downsampling (fs: 1000 Hz -> 250 Hz)
        n = 4
        X = X[:,::n]
        X_sh = X_sh[:,::n]
        t_vec = 1000*t[t_ind]
        t_vec = t_vec[::n]
        fsample = f_sample/n
        
        # kTE parameters 
        
        alpha = 2
        
        print('Estimating embedding parameters...')          
        # Embedding time (autocorrelation decay time)
        maxlag = 20
        tau_matrix[k,0] = TE.autocorr_decay_time(X[0,:],maxlag)
        tau_matrix[k,1] = TE.autocorr_decay_time(X[1,:],maxlag) 
        
        # Embedding dimension (obtained using the cao criterion) 
        d_max = 10
        dim_matrix[k,0] = TE.cao_criterion(X[0,:],d_max,tau_matrix[k,0])
        dim_matrix[k,1] = TE.cao_criterion(X[1,:],d_max,tau_matrix[k,1]) 

        # Estimating interaction time 
        print('Computing kernel TE (to estimate u)...')
        TE_aux = Parallel(n_jobs=-1,verbose=0)(delayed(TE.kernelTransferEntropy_AllCh) 
                                        (X,dim_matrix[k,:],tau_matrix[k,:],u,alpha) for u in u_vec)
        
        TE_aux = np.transpose(np.array(TE_aux), (1,2,0))
        kTE_u_matrix[k,:,0] = TE_aux[0,1,:]
        kTE_u_matrix[k,:,1] = TE_aux[1,0,:]
        u_matrix[k,:] = u_vec[np.argmax(kTE_u_matrix[k,:,:],axis=0)]
        
        # kTE
        print('Computing phase kernel TE...')
        u_trial = np.array([[0,u_matrix[k,0]],[u_matrix[k,1],0]])
        TE_aux = Parallel(n_jobs=-1,verbose=0)(delayed(TE.kernelTransferEntropy_AllCh_freq) 
                                        (X,dim_matrix[k,:],tau_matrix[k,:],u_trial,alpha,f,t_vec,component='phase') for f in freq_range)

        TE_aux = np.transpose(np.array(TE_aux), (1,2,0))
        kTE_matrix[k,:,0] = TE_aux[0,1,:]
        kTE_matrix[k,:,1] = TE_aux[1,0,:]
        
        # kTE shuffled data 
        # x -> y_sh
        TE_sh_aux = Parallel(n_jobs=-1,verbose=0)(delayed(TE.kernelTransferEntropy_AllCh_freq) 
                                (np.vstack((X[0,:],X_sh[1,:])),dim_matrix[k,:],tau_matrix[k,:],u_trial,alpha,f,t_vec,component='phase') for f in freq_range)

        TE_sh_aux = np.transpose(np.array(TE_sh_aux), (1,2,0))
        kTE_sh_matrix[k,:,0] = TE_sh_aux[0,1,:]
        
        # y -> x_sh
        TE_sh_aux = Parallel(n_jobs=-1,verbose=0)(delayed(TE.kernelTransferEntropy_AllCh_freq) 
                                (np.vstack((X_sh[0,:],X[1,:])),dim_matrix[k,:],tau_matrix[k,:],u_trial,alpha,f,t_vec,component='phase') for f in freq_range)

        TE_sh_aux = np.transpose(np.array(TE_sh_aux), (1,2,0))
        kTE_sh_matrix[k,:,1] = TE_sh_aux[1,0,:]
        
    tau_lst.append(tau_matrix)
    dim_lst.append(dim_matrix)
    u_lst.append(u_matrix)
    kTE_u_lst.append(kTE_u_matrix)
    kTE_lst.append(kTE_matrix)
    kTE_sh_lst.append(kTE_sh_matrix)
        
# =============================================================================
# %% Permutation tests
# =============================================================================

print('\nRunning permutation tests ...')

alpha_level = 0.01  #Significance level for the test 
nr2cmc = num_freq   #number of tests (For Bonferroni correction)

kTEpermval_lst = []
kTEsignif_lst = []
dkTEpermval_lst = []
dkTEsignif_lst = []

for n,m in enumerate(coupling_ind):
    
    print('Coupling {}'.format(coupling[m]))
    
    kTEpermval = np.zeros((num_freq,2))
    kTEsignif = np.zeros((num_freq,2)) 
    dkTEpermval = np.zeros(num_freq) 
    dkTEsignif = np.zeros(num_freq) 
    
    kTE_matrix = kTE_lst[n] 
    kTE_sh_matrix = kTE_sh_lst[n]
    
    for nn in range(num_freq):
        kTE_aux = kTE_matrix[:,nn,:]
        kTE_sh_aux = kTE_sh_matrix[:,nn,:]

        # Permutation test
        kTEpermval[nn,:],kTEsignif[nn,:],dkTEpermval[nn],dkTEsignif[nn] = TE.permutation_test(kTE_aux,kTE_sh_aux,alpha_level/nr2cmc)
    
    kTEpermval_lst.append(kTEpermval)
    kTEsignif_lst.append(kTEsignif) 
    dkTEpermval_lst.append(dkTEpermval)
    dkTEsignif_lst.append(dkTEsignif)
    
# =============================================================================
# %% Data saving    
# =============================================================================

kTE_dict = {} 
kTE_dict['tau'] = tau_lst
kTE_dict['dim'] = dim_lst
kTE_dict['u'] = u_lst
kTE_dict['kTE_u'] = kTE_u_lst
kTE_dict['kTE'] = kTE_lst
kTE_dict['kTE_sh'] = kTE_sh_lst 
kTE_dict['kTEpermval'] = kTEpermval_lst
kTE_dict['kTEsignif'] = kTEsignif_lst
kTE_dict['dkTEpermval'] = dkTEpermval_lst
kTE_dict['dkTEsignif'] = dkTEsignif_lst
sio.savemat('results_Phase_kTE_NMM.mat',kTE_dict) 

# =============================================================================
# %% Results plots
# =============================================================================
        
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
plt.close('all')

cmap = cm.cool
norm = Normalize(vmin=coupling_ind[0], vmax=coupling_ind[-1])
color_palette = cmap(norm(coupling_ind))

plt.figure()
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlim(freq_range[0],freq_range[-1])
plt.xlabel('Frequency (Hz)',fontsize=14)
plt.ylabel(r'$\Delta$TE$_{\kappa\alpha}^{\theta}$',fontsize=16)
for i,ii in enumerate(coupling_ind): 
    kTE_avg = np.mean(kTE_lst[i][:,:,0]-kTE_lst[i][:,:,1],axis=0)
    kTE_std = np.std(kTE_lst[i][:,:,0]-kTE_lst[i][:,:,1],axis=0)
    plt.fill_between(freq_range,kTE_avg-kTE_std,kTE_avg+kTE_std,facecolor=color_palette[i,:],alpha=0.2)
    plt.plot(freq_range,kTE_avg,color=color_palette[i,:],label=str(coupling[ii]),linewidth=3)
    signif_ind = dkTEsignif_lst[i].astype('bool')
    plt.plot(freq_range[signif_ind],kTE_avg[signif_ind],'ok',markersize=8,markerfacecolor='none')

plt.plot(freq_range,np.zeros(len(freq_range)),'--k',linewidth=3,alpha=0.5)
plt.xticks(ticks=[10,20,30,40,50,60],labels=[10,20,30,40,50,60])
plt.yticks(ticks=[0,0.2])
plt.ticklabel_format(axis='y', style='sci',scilimits=(0,0))
ax.yaxis.get_offset_text().set_fontsize(12)
ax.tick_params(axis='y', labelsize=14)
ax.tick_params(axis='x', labelsize=14)

plt.legend(fontsize=14,loc='best',frameon=False)
plt.savefig('Phase_kTE_NMM.png')  
