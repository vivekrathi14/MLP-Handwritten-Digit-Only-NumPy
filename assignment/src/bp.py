#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: bp.py

import numpy as np
from src.activation import sigmoid, sigmoid_prime

def backprop(x, y, biases, weightsT, cost, num_layers):
    """ function of backpropagation
        Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient of all biases and weights.

        Args:
            x, y: input image x and label y
            biases, weights (list): list of biases and transposed weights of entire network
            cost (CrossEntropyCost): object of cost computation
            num_layers (int): number of layers of the network

        Returns:
            (nabla_b, nabla_wT): tuple containing the gradient for all the biases
                and weightsT. nabla_b and nabla_wT should be the same shape as 
                input biases and weightsT
    """
    # initial zero list for store gradient of biases and weights
    nabla_b = [np.zeros(b.shape) for b in biases]
    nabla_wT = [np.zeros(wT.shape) for wT in weightsT]

    ### Implement here
    # feedforward
    # Here you need to store all the activations of all the units
    # by feedforward pass
    ###
    h_k = x
    h_ks = [h_k]
    a_ks = []
    for b, wT in zip(biases,weightsT):
        a_k = np.dot(wT,h_k) + b
        a_ks.append(a_k)
        h_k = sigmoid(a_k)
        h_ks.append(h_k)

    # compute the gradient of error respect to output
    # activations[-1] is the list of activations of the output layer
    #delta = (cost).df_wrt_a(h_ks[-1], y)
    delta = (cost).df_wrt_a(h_ks[-1], y) * sigmoid_prime(a_ks[-1])
    nabla_b[-1] = delta 
    nabla_wT[-1] = np.dot(delta,h_ks[-2].T)
    

        
    ### Implement here
    # backward pass
    # Here you need to implement the backward pass to compute the
    # gradient for each weight and bias
    ###
    for i in range(2,num_layers):
        delta = np.dot(weightsT[-i+1].T,delta) * sigmoid_prime(a_ks[-i])
        nabla_b[-i] = delta
        nabla_wT[-i] = np.dot(delta,h_ks[-i-1].T)
    
#    for i in range(num_layers-2,-1,-1):
#        delta = np.multiply(delta,sigmoid_prime(a_ks[i]))
#        nabla_b[i] = delta
#        nabla_wT[i] = (np.dot(h_ks[i],delta.T)).T
#        delta = np.dot(weightsT[i].T,delta)

    return (nabla_b, nabla_wT)

