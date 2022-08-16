# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 13:14:51 2022

@author: hunter
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import scipy
from scipy.interpolate import interp1d
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import gennorm, norm, laplace



### note on 8/11 try the encoder for 3 layers can convergence but more than 3 layer will convergence.

# parameter for Network
epochs = 1000
lr = 0.01

# node for autoencoder C
batch_size = 100
node0_C = 1000
node1_C = 500
node2_C = 200
node3_C = 100

# construct encoder network train compared with input & output with the MMSE without bias
class Network_C(tf.keras.Model):
    def __init__(self):
        super().__init__()
        
        self.encoder = tf.keras.Sequential([
          tf.keras.layers.Dense(units = node1_C, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=1),
                       use_bias=False, activation='sigmoid'),
          tf.keras.layers.Dense(units = node2_C, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=1),
                       use_bias=False, activation='sigmoid'),
           tf.keras.layers.Dense(units = node3_C, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=1),
                        use_bias=False),
        ])
        
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(units = node2_C, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=1),
                          use_bias=False, activation='sigmoid'),
            tf.keras.layers.Dense(units = node1_C, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=1),
                         use_bias=False, activation='sigmoid'),
            tf.keras.layers.Dense(units = node0_C, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=1),
                         use_bias=False),
        ])

    def call(self, input):
        encoded = self.encoder(input)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, input):
        return self.encoder(input)
    
    def decode(self, encoded):
        return self.decoder(encoded)
        
## tool
# change tf to numpy    
def change_type(grads):
    for idx in range(len(grads)):
           grads[idx] = grads[idx].numpy().flatten()
    return np.array(grads,dtype=object)

model = Network_C()
optimizer = tf.keras.optimizers.SGD(learning_rate=lr) 

loss_fun = tf.keras.losses.MeanSquaredError() # loss
losses = []
empty = []

# train 
for epoch in range(epochs):
    'The input needs to be changing each epoch'
    X = np.random.normal(0, 1, size=(batch_size, node0_C)).astype(np.float32)  # input 
    # record the gradient and updated the weight by the MMSE
    with tf.GradientTape() as tape: # record the forward prop
        y_pred = model(X)
        loss = loss_fun(y_pred, X) # use MMSE should start from construct the instance
        losses.append(loss)
        print("epoch %d: loss %f" % (epoch+1, loss.numpy()))
    # calculate the backward prop
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
    
    # if epoch % 25 == 0:
    #     new = change_type(grads)
    #     # sort in an array
    #     d = new.flatten()
    #     empty.append(d[1])

plt.close('all')
plt.figure()
plt.plot(losses,color='red', label='loss')
plt.xlabel('training epochs')
plt.ylabel('training loss')
plt.legend()
plt.grid()
plt.show()