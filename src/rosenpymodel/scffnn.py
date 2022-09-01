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
import numpy as np
from rosenpymodel import rplayer, rpnn

class SCFFNN(rpnn.NeuralNetwork):   
    def feedforward(self, x):
        
        self.layers[0].input = np.array(x, ndmin=2)
        
        for current_layer, next_layer in zip(self.layers, self.layers[1:] + [rplayer.Layer(0,0, status=1)]):
            y = np.dot(current_layer.weights, current_layer.input.T) + current_layer.biases.T
            current_layer._activ_in = y
            current_layer._activ_out = actFunc.splitComplex(y, current_layer.activation, derivative=False)
            next_layer.input = current_layer._activ_out.T
           
        return self.layers[-1]._activ_out
    
    
    def backprop(self, y, y_pred, epoch):
        y=np.array(y, ndmin=2).T
        
        e = y - y_pred
        auxW = 0
        last = True
        for layer in reversed(self.layers):
            d = actFunc.splitComplex(layer._activ_in, layer.activation, derivative=True)
           
           
            if last==True: 
                deltaDir = np.multiply(np.real(d),np.real(e)) + 1j*np.multiply(np.imag(d),np.imag(e))  
                last = False  
            else:
                p_real = np.multiply(np.real(d),np.dot((np.conj(auxW.T)),np.real(deltaDir))) 
                p_imag = 1j*np.multiply(np.imag(d),np.dot((np.conj(auxW.T)),np.imag(deltaDir))) 
                
                deltaDir = p_real + p_imag
               
            auxW = layer.weights
             
            layer._dweights = np.dot(deltaDir,np.conj(layer.input)) - (regFunc.l2_regularization(layer.lambda_init, layer.reg_strength, epoch)*layer.reg_strength)*layer.weights
            layer._prev_dweights = layer._dweights*self.learning_rate + self.momentum*layer._prev_dweights
            layer.weights = layer.weights + layer._prev_dweights
            
            layer._dbiases =  deltaDir.T - (regFunc.l2_regularization(layer.lambda_init, layer.reg_strength, epoch)*layer.reg_strength)*layer.biases
            layer._prev_dbiases = layer._dbiases*self.learning_rate + self.momentum*layer._prev_dbiases
            layer.biases = layer.biases + layer._prev_dbiases
  
    def addLayer(self, neurons, ishape=0, weights_initializer=initFunc.random_normal, bias_initializer=initFunc.random_normal, activation=actFunc.tanh, reg_strength=0.0, lambda_init=0.1):
        self.layers.append(rplayer.Layer(ishape if not len(self.layers) else self.layers[-1].neurons, neurons, 
                                              weights_initializer=weights_initializer, 
                                              bias_initializer=bias_initializer, 
                                              activation=activation, 
                                              reg_strength=reg_strength, 
                                              lambda_init=lambda_init, 
                                              status=1))