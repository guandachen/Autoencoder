# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 15:25:27 2022

@author: hunter
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gennorm, norm, laplace


epochs = 1000
with open('gradient_decoder.pickle', 'rb') as f:
    gradient_decoder = pickle.load(f)
    
with open('gradient_encoder.pickle', 'rb') as f:
    gradient_encoder = pickle.load(f)
    
def W(p, u, v):
    assert len(u) == len(v)
    return np.mean(np.abs(np.sort(u)[1:u.size-1]-np.sort(v)[1:v.size-1])**p)**(1/p)

# distribution
distance_norm = []
distance_gennorm = []
distance_laplace = []


for epoch in range(epochs):
    
    #---------------------------------------------------------#
    data_decoder = np.array(gradient_decoder[epoch])
    # data_encoder = np.array(gradient_encoder[epoch])
    #---------------------------------------------------------#
    # normalize
    data_decoder_n = (data_decoder-data_decoder.mean())/data_decoder.std()
    # data_decoder_n = data_decoder/data_decoder.std()
    # generate the fit distribution to calculate the wasserstein_distance 
    
    # [mean_fit,std_fit] = norm.fit(data_decoder_n, floc=0)  # estimate norm distribution
    # distance1 =W(2, norm.ppf(np.linspace(0,1,data_decoder_n.size), loc=mean_fit, scale=std_fit), np.sort(data_decoder_n))
    # distance_norm.append(distance1)
    
    
    # [arg1,arg2] = laplace.fit(data_decoder_n, floc=0)  # estimate laplace distribution
    # distance2 = W(2, laplace.ppf(np.linspace(0,1,data_decoder_n.size), loc=arg1, scale=arg2), np.sort(data_decoder_n))
    # distance_laplace.append(distance2)
    
    # [arg4, arg5, arg6] = gennorm.fit(data_decoder_n, floc=0)  # estimate gennorm distribution
    # distance3 = W(2, gennorm.ppf(np.linspace(0,1,data_decoder_n.size), beta=arg4, loc =arg5, scale=arg6), np.sort(data_decoder_n))
    # distance_gennorm.append(distance3)
    [mean_fit,std_fit] = norm.fit(data_decoder_n, floc=0)  # estimate norm distribution
    v_norm = norm.rvs(mean_fit,std_fit, size=data_decoder_n.size) # generate
    
    [arg1,arg2] = laplace.fit(data_decoder_n, floc=0)  # estimate laplace distribution
    v_laplace = laplace.rvs(arg1,arg2, size=data_decoder_n.size) # generate   
    
    [arg4, arg5, arg6] = gennorm.fit(data_decoder_n, floc=0)  # estimate gennorm distribution
    v_gennorm = gennorm.rvs(arg4, arg5, arg6, size=data_decoder_n.size) # generate
    distance_norm.append(stats.wasserstein_distance(data_decoder_n, v_norm))
    distance_laplace.append(stats.wasserstein_distance(data_decoder_n, v_laplace ))
    distance_gennorm.append(scipy.stats.wasserstein_distance(data_decoder_n, v_gennorm))
    print(epoch)
    

plt.plot(distance_norm,color='red',label='norm')
plt.plot(distance_norm,color='blue',label='laplace')
plt.plot(distance_norm,color='yellow',label='gennorm')
plt.xlabel('training epochs')
plt.ylabel('wasserstein_distance')
plt.legend()
plt.grid()
plt.show()
