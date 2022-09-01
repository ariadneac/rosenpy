# -*- coding: utf-8 -*-
"""
/*---------------------------------------------------------------------------*\
  RosenPy: An Open Source Python Framework for Complex-Valued Neural Networks
  Copyright © A. A. Cruz, K. S. Mayer, D. S. Arantes
-------------------------------------------------------------------------------
License
    This file is part of RosenPy.
    RosenPy is an open source framework distributed under the terms of the GNU
    General Public License, as published by the Free Software Foundation, either
    version 3 of the License, or (at your option) any later version. For additional
    information on license terms, please open the Readme.md file.

    RosenPy is distributed in the hope that it will be useful to every user, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
    See the GNU General Public License for more details. 

    You should have received a copy of the GNU General Public License
    along with RosenPy.  If not, see <http://www.gnu.org/licenses/>.
    
"""
from rp_utils import regFunc, initFunc, actFunc
from rosenpymodel import rplayer, rpnn
import numpy as np


class FCRBFNN(rpnn.NeuralNetwork):
    def feedforward(self, x):
        self.layers[0].input = np.array(x, ndmin=2)
        
        for current_layer, next_layer in zip(self.layers, self.layers[1:] + [rplayer.Layer(0,0, status=3)]):
  
            current_layer.seuc = current_layer.input - current_layer.gamma
            current_layer.kern =  np.array(np.sum(np.multiply(current_layer.sigma, current_layer.seuc), axis=1), ndmin=2).T   
            current_layer.phi = actFunc.sech(current_layer.kern)
     
            current_layer._activ_out = np.dot(current_layer.phi.T, current_layer.weights.T) + current_layer.biases
           
            next_layer.input = current_layer._activ_out
            
        return np.array(self.layers[-1]._activ_out, ndmin=2)   
    
    def backprop(self, y, y_pred, epoch):
        y = np.array(y, ndmin=2).T
        
        error = y - y_pred.T
        
        for layer in reversed(self.layers):
            phi_l = actFunc.sech(layer.kern, derivative=True)
        
            a = np.multiply((np.dot(error.T, np.conj(layer.weights)).T), np.conj(phi_l))
            
            layer._dweights = np.dot(error, np.conj(layer.phi.T)) - (regFunc.l2_regularization(layer.lambda_init, layer.reg_strength, epoch)if layer.reg_strength else 0)*layer.weights
            layer._prev_dweights = layer._dweights*self.learning_rate + self.momentum*layer._prev_dweights 
            layer.weights = layer.weights + layer._prev_dweights

            layer._dbiases = error.T - (regFunc.l2_regularization(layer.lambda_init, layer.reg_strength, epoch)if layer.reg_strength else 0)*layer.biases
            layer._prev_dbiases = layer._dbiases*self.learning_rate + self.momentum*layer._prev_dbiases
            layer.biases = layer.biases + layer._prev_dbiases

            layer._dgamma = layer._prev_dgamma
            layer._prev_dgamma = np.multiply(a,np.conj(layer.sigma))  - (regFunc.l2_regularization(layer.lambda_init, layer.reg_strength, epoch)if layer.reg_strength else 0)*layer.gamma
            layer.gamma = layer.gamma - layer._prev_dgamma*layer.gamma_rate + self.momentum*layer._dgamma
    
            
            layer._dsigma = layer._prev_dsigma
            layer._prev_dsigma =  np.multiply(a,np.conj(layer.seuc)) - (regFunc.l2_regularization(layer.lambda_init, layer.reg_strength, epoch)if layer.reg_strength else 0)*layer.sigma
            layer.sigma = layer.sigma + layer._prev_dsigma*layer.sigma_rate +  self.momentum*layer._dsigma

                
    def predict(self, x_train):
        s = []
        for x in x_train:
            y_pred = self.feedforward(x).T
            s.append(y_pred.flatten())
        
        return np.array(s)
    
    
    def addLayer(self, ishape, neurons, oshape, weights_initializer=initFunc.random_normal, bias_initializer=initFunc.ones, reg_strength=0.0, lambda_init=0.1, gamma_rate=0.01, sigma_rate=0.01):
        self.layers.append(rplayer.Layer(ishape, neurons, oshape, 
                                          weights_initializer=weights_initializer, 
                                          bias_initializer=bias_initializer, 
                                          reg_strength=reg_strength, 
                                          lambda_init=lambda_init, 
                                          sigma_rate=sigma_rate,
                                          gamma_rate=gamma_rate,
                                          status=3))
               