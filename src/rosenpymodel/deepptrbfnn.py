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
from rp_utils import regFunc, initFunc
from rosenpymodel import rplayer, rpnn
import numpy as np


class DeepPTRBFNN(rpnn.NeuralNetwork):   
    def feedforward(self, x):
        self.layers[0].input = np.array(x, ndmin=2)
        
        for current_layer, next_layer in zip(self.layers, self.layers[1:] + [rplayer.Layer(0,0,0, status=4)]):
            current_layer.kern = current_layer.input - current_layer.gamma
                
            seuc_r  = np.divide(np.array(np.sum(np.square(current_layer.kern.real), axis=1), ndmin=2).T, current_layer.sigma.real)
            seuc_i  = np.divide(np.array(np.sum(np.square(current_layer.kern.imag), axis=1), ndmin=2).T, current_layer.sigma.imag)
            current_layer.seuc = seuc_r + 1j*seuc_i
            current_layer.phi = np.exp(-seuc_r) + 1j*(np.exp(-seuc_i))
           
            current_layer._activ_out = (np.dot(current_layer.weights, current_layer.phi) + current_layer.biases)
           
            next_layer.input = current_layer._activ_out.T
            
        return np.array(self.layers[-1]._activ_out, ndmin=2)
        
    def backprop(self, y, y_pred, epoch):
        y = np.array(y, ndmin=2).T
        
        error = y - y_pred
        
        last = True
        auxK = auxN = delta = 0
        for layer in reversed(self.layers):
            if last==True:
                psi = error
                last = False
            else:  
                psi = np.dot(np.dot(auxK.T.real,delta.real) + 1j*(np.dot(auxK.T.imag,delta.imag)), np.ones((auxN, 1)))
               
            auxK = layer.kern
            auxN = layer.neurons
            
            epsilon = np.dot(np.conj(layer.weights.T),psi)
            beta_r = np.divide(layer.phi.real, layer.sigma.real)
            beta_i = np.divide(layer.phi.imag, layer.sigma.imag)
    
            delta = -np.diag(np.multiply(epsilon.real, beta_r).T[0]) - 1j*np.diag(np.multiply(epsilon.imag, beta_i).T[0])
           
            layer._dweights = np.dot(psi, np.conj(layer.phi.T)) - (regFunc.l2_regularization(layer.lambda_init, layer.reg_strength, epoch)if layer.reg_strength else 0)*layer.weights
            layer._prev_dweights = layer._dweights*self.learning_rate + self.momentum*layer._prev_dweights 
            layer.weights = layer.weights + layer._prev_dweights

            layer._dbiases = psi - (regFunc.l2_regularization(layer.lambda_init, layer.reg_strength, epoch)if layer.reg_strength else 0)*layer.biases
            layer._prev_dbiases = layer._dbiases*self.learning_rate + self.momentum*layer._prev_dbiases
            layer.biases = layer.biases + layer._prev_dbiases
            
            layer._dsigma = layer._prev_dsigma           
            layer._prev_dsigma = (-np.dot(delta.real,layer.seuc.real) - 1j*np.dot(delta.imag,layer.seuc.imag))  - (regFunc.l2_regularization(layer.lambda_init, layer.reg_strength, epoch)if layer.reg_strength else 0)*layer.sigma
            layer.sigma = layer.sigma + layer._prev_dsigma*layer.sigma_rate +  self.momentum*layer._dsigma        
           
            layer._dgamma = layer._prev_dgamma
            layer._prev_dgamma = (-np.dot(delta.real,layer.kern.real) - 1j*np.dot(delta.imag,layer.kern.imag))  - (regFunc.l2_regularization(layer.lambda_init, layer.reg_strength, epoch)if layer.reg_strength else 0)*layer.gamma
            layer.gamma = layer.gamma + layer._prev_dgamma*layer.gamma_rate + self.momentum*layer._dgamma
 
            layer.sigma = np.where(layer.sigma.real>0.0001, layer.sigma.real, 0.0001) + 1j*np.where(layer.sigma.imag>0.0001, layer.sigma.imag, 0.0001)


   
    
    def predict(self, x_train):   
        s = []
        for x in x_train:
            y_pred = self.feedforward(x).T
            s.append(y_pred.flatten())
        
        return np.array(s)
    
    def addLayer(self, neurons, ishape=0, oshape=0, weights_initializer=initFunc.random_normal, bias_initializer=initFunc.ones, reg_strength=0.0, lambda_init=0.1, gamma_rate=0.01, sigma_rate=0.01):
        self.layers.append(rplayer.Layer(ishape if not len(self.layers) else self.layers[-1].oshape, neurons, neurons if not oshape else oshape, 
                                          weights_initializer=weights_initializer, 
                                          bias_initializer=bias_initializer, 
                                          reg_strength=reg_strength, 
                                          lambda_init=lambda_init, 
                                          sigma_rate=sigma_rate,
                                          gamma_rate=gamma_rate,
                                          status=4))
                