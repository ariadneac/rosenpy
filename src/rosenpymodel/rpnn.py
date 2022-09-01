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

import numpy as np
from rp_utils import costFunc, decayFunc, batchGenFunc
        
class NeuralNetwork():
    def __init__(self, cost_func=costFunc.mse, learning_rate=1e-3, lr_decay_method=decayFunc.none_decay,  lr_decay_rate=0.0, lr_decay_steps=1, momentum=0.0, patience=np.inf):
        self.layers = []
        self.cost_func = cost_func
        self.momentum = momentum
        self.learning_rate = self.lr_initial = learning_rate
       
        self.lr_decay_method = lr_decay_method
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_steps = lr_decay_steps
        
        self.patience, self.waiting = patience, 0
        
        self._best_model, self._best_loss = self.layers, np.inf
        self._history = {'epochs': [], 'loss': [], 'loss_val': []}
    
    def fit(self, x_train, y_train, x_val=None, y_val=None, epochs=100, verbose=10, batch_gen=batchGenFunc.batch_sequential, batch_size=None):
  
        x_val, y_val = (x_train, y_train) if (x_val is None or y_val is None) else (x_val, y_val)
   
        for epoch in range(epochs+1):
            self.learning_rate = self.lr_decay_method(self.lr_initial, epochs, self.lr_decay_rate, self.lr_decay_steps)
        
            for x_batch, y_batch in batch_gen(x_train, y_train, batch_size):
                for x_batch1, y_batch1 in zip(x_batch, y_batch):
                    y_pred = self.feedforward(x_batch1) 
                    self.backprop(y_batch1, y_pred, epoch) 
            
            loss_val = self.cost_func(y_val, self.predict(x_val))
            
            if self.patience != np.inf:
                if loss_val < self._best_loss:
                    self._best_model, self._best_loss = self.layers, loss_val
                    self.waiting = 0
                else: 
                    self.waiting +=1
                    print("not improving: [{}] current loss val: {} best: {}".format(self.waiting, loss_val, self._best_loss))
                    if self.waiting >= self.patience:
                        self.layers = self._best_model
                        print("early stopping at epoch ", epoch)
                        return
            
            if epoch % verbose == 0:
                loss_train = self.cost_func(y_train, self.predict(x_train))
                self._history['epochs'].append(epoch)
                self._history['loss'].append(loss_train)
                self._history['loss_val'].append(loss_val)
                print("epoch: {0:=4}/{1} loss_train: {2:.8f} loss_val: {3:.8f}".format(epoch, epochs, loss_train, loss_val))
                
        return self._history    
                
    def predict(self, x): 
        return self.feedforward(x).T
    
    def addLayer(self): 
        pass
    
    def getHistory(self):
        return self._history