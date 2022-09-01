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

def split_set(x, y, train_size=0.7, random_state=None):
    if random_state:
        rand_state = np.random.RandomState(random_state)
        rand_state.shuffle(x)
        rand_state.seed(random_state)
        rand_state.shuffle(y)
    
    split = int(train_size*x.shape[0])
   
    x_Train = x[0:split]
    y_Train = y[0:split]
    x_Test =  x[split:x.shape[0]]
    y_Test = y[split:x.shape[0]]
    
    return x_Train, y_Train, x_Test, y_Test

def accuracy(y, y_pred):
        return 100*(1-np.mean(np.abs((y-y_pred))))