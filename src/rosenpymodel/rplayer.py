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
from rp_utils import actFunc, initFunc
import numpy as np

class Layer():
    def __init__(self, ishape, neurons, oshape=0, weights_initializer=initFunc.random_normal, bias_initializer=initFunc.random_normal, activation=actFunc.tanh, reg_strength=0.0, lambda_init=0.1, gamma_rate=0.0, sigma_rate=0.0, status=1):
        self.input = None
        self.reg_strength = reg_strength
        self.lambda_init = lambda_init
     
        self._activ_inp, self._activ_out = None, None
        
        self.gamma_rate = gamma_rate
        self.sigma_rate = sigma_rate
        self.neurons = neurons
        self.oshape = oshape
        self.seuc = None
        self.phi = None
        self.kern = None
        
        ## Parameters for FF Networks
        if status==1:
            self.weights = weights_initializer(neurons, ishape)
            self.biases = bias_initializer(1, neurons)
            self.activation = activation
            self._dweights = self._prev_dweights = initFunc.zeros(neurons, ishape)
            self._dbiases = self._prev_dbiases = initFunc.zeros(1, neurons)
            
        ## Parameters for CVRBF Networks
        elif status==2:
            self.weights = weights_initializer(oshape, neurons)
            self.biases = bias_initializer(oshape, 1)
            
            self._dweights = self._prev_dweights = initFunc.zeros(oshape, neurons)
            self._dbiases = self._prev_dbiases = initFunc.zeros(oshape, 1)
            
            self.gamma = np.random.randint(2, size=[neurons,ishape])*0.7 + 1j*(np.random.randint(2, size=[neurons,ishape])*2-1)*0.7
            self.sigma = np.ones((neurons,1)) 
             
            self._prev_dgamma = self._dgamma = initFunc.zeros(neurons, ishape)
            self._prev_dsigma = self._dsigma = initFunc.zeros(neurons,1)
        ## Parameters for FCRBF Networks   
        elif status==3:
            self.weights = weights_initializer(oshape, neurons)
            self.biases = bias_initializer(1, oshape)
        
            self._dweights = self._prev_dweights = initFunc.zeros(oshape, neurons)
            self._dbiases = self._prev_dbiases = initFunc.zeros(1, oshape)
            
            self.gamma = np.random.randint(2, size=[neurons,ishape])*0.7 + 1j*(np.random.randint(2, size=[neurons,ishape])*2-1)*0.7
            self.sigma = np.random.randint(2, size=[neurons,ishape])*0.7 + 1j*(np.random.randint(2, size=[neurons,ishape])*2-1)*0.7
        
            self._prev_dgamma = self._dgamma = initFunc.zeros(neurons, ishape)
            self._prev_dsigma = self._dsigma = initFunc.zeros(neurons, ishape)
            
        ## Parameters for Deep PTRBF Networks    
        elif status==4:
            self.weights = weights_initializer(oshape, neurons)
            self.biases = bias_initializer(oshape, 1)
            self.gamma =  np.random.randint(2, size=[neurons,ishape])*0.7 + 1j*(np.random.randint(2, size=[neurons,ishape])*2-1)*0.7
            self.sigma = initFunc.ones(neurons,1)
            
            self._ddweights = self._dweights = self._prev_dweights = initFunc.zeros(oshape, neurons)
            self._ddbiases = self._dbiases = self._prev_dbiases = initFunc.zeros(oshape, 1)
            
            self._prev_dgamma = self._dgamma = initFunc.zeros(neurons, ishape)
            self._prev_dsigma = self._dsigma = initFunc.zeros(neurons, 1)
            
    
       
       