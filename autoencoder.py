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
epochs = 100
lr =0.0001

# node for autoencoder C
node1_C = 1000
node2_C = 500
node3_C = 100

# construct encoder network train compared with input & output with the MMSE without bias
class Network_C(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1= tf.keras.layers.Dense(units = node1_C,
                                            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=1),
                                            use_bias=False,
                                            )
        self.dense2 = tf.keras.layers.Dense(units = node2_C,
                                            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=1),
                                            use_bias=False,
                                            )
        self.dense3 = tf.keras.layers.Dense(units = node3_C,
                                            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=1),
                                            use_bias=False,
                                            )

        self.dense4 = tf.keras.layers.Dense(units = node2_C,
                                            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=1),
                                            use_bias=False,
                                            )
        self.dense5 = tf.keras.layers.Dense(units = node1_C,
                                            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=1),
                                            use_bias=False,
                                            )
    def call(self, input):
        hidden1 = self.dense1(input)
        hidden2 = self.dense2(hidden1)
        hidden3 = self.dense3(hidden2)
        hidden4 = self.dense4(hidden3)
        output = self.dense5(hidden4)
        return output
        
## tool
# change tf to numpy    
def change_type(grads):
    for idx in range(len(grads)):
           grads[idx] = grads[idx].numpy().flatten()
    return np.array(grads,dtype=object)

model = Network_C()
optimizer = tf.keras.optimizers.SGD(learning_rate=lr) 
X = np.random.normal(0, 1, size=(1000, 1)).astype(np.float32)  # input 
loss_fun = tf.keras.losses.MeanSquaredError() # loss
losses = []
empty = []

# train 
for epoch in range(epochs):
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
    #    new = change_type(grads)
    #    # sort in an array
    #    d = new.flatten()
    #    empty.append(d[1])
       
plt.plot(losses,color='red', label='loss')
plt.xlabel('training epochs')
plt.ylabel('training loss')
plt.legend()
plt.grid()
plt.show()





