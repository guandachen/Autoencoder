# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 12:45:47 2022

@author: hunter
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gennorm, norm, laplace
import pickle
import time
import scipy
from scipy import stats

# Load
with open('gradient_decoder.pickle', 'rb') as f:
    gradient_decoder = pickle.load(f)
with open('gradient_encoder.pickle', 'rb') as f:
    gradient_encoder = pickle.load(f)

epochs = 1000



# gennorm parameter
alpha_decoder = []
beta_decoder = []
alpha_encoder = []
beta_encoder = []
start = time.time()

for epoch in range(epochs):
    
    #---------------------------------------------------------#
    data_decoder = np.array(gradient_decoder[epoch])
    data_encoder = np.array(gradient_encoder[epoch])
    #---------------------------------------------------------#
    # decoder & encoder 
    de = data_decoder/data_decoder.std()
    en = data_encoder/data_decoder.std()
    # fit the distribution
    [arg4, arg5, arg6] = gennorm.fit(de, floc=0)  # estimate gennorm distribution
    beta_decoder.append(arg4)
    alpha_decoder.append(arg6)
    
    [arg7, arg8, arg9] = gennorm.fit(en, floc=0)  # estimate gennorm distribution
    beta_encoder.append(arg7)
    alpha_encoder.append(arg9)
    print(epoch) #
end = time.time()
with open('alpha_decoder.pickle', 'wb') as f:
    pickle.dump(alpha_decoder, f)
with open('beta_decoder.pickle', 'wb') as f:
    pickle.dump(beta_decoder, f)
with open('alpha_encoder.pickle', 'wb') as f:
    pickle.dump(alpha_encoder, f)
with open('beta_encoder.pickle', 'wb') as f:
    pickle.dump(beta_encoder, f)    
print(end - start)
plt.figure(1)
plt.plot(beta_decoder,color='red',label='beta')
plt.xlabel('training epochs')
plt.title('decoder')
plt.ylabel('beta')

plt.figure(2)
plt.plot(alpha_decoder,color='blue',label='alpha')
plt.xlabel('training epochs')
plt.ylabel('alpha')
plt.title('decoder')

plt.figure(3)
plt.plot(beta_encoder,color='red',label='beta')
plt.xlabel('training epochs')
plt.title('encoder')
plt.ylabel('beta')

plt.figure(4)
plt.plot(alpha_encoder,color='blue',label='alpha')
plt.xlabel('training epochs')
plt.ylabel('alpha')
plt.title('encoder')

plt.legend()
plt.grid()
plt.show()