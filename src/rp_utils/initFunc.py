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

def zeros(rows, cols):
    w = np.zeros((rows, cols))
    w = np.array(w, ndmin=2, dtype='complex128')
    w += 1j*np.zeros((rows, cols))
    return w

def ones(rows, cols):
    w = np.ones((rows, cols))
    w = np.array(w, ndmin=2, dtype='complex128')
    w += 1j*np.ones((rows, cols))
    return w

def random_normal(rows, cols):
    w = (np.random.randn(rows, cols)-0.5)/10
    w = np.array(w, ndmin=2, dtype='complex128')
    w += (1j*np.random.randn(rows, cols)-0.5)/10
    return w

def random_uniform(rows, cols):
    w = np.random.rand(rows, cols)/10
    w = np.array(w, ndmin=2, dtype='complex128')
    w += 1j*np.random.rand(rows, cols)/10
    
    return w

def glorot_normal(rows, cols):
    std_dev = np.sqrt(2.0/(rows+cols))/10
    w = std_dev*np.random.randn(rows, cols)
    w = np.array(w, ndmin=2, dtype='complex128')
    w += 1j*std_dev*np.random.randn(rows, cols)/10
    return w

def glorot_uniform(rows, cols):
    std_dev = np.sqrt(6.0/(rows+cols))/10
    w = 2*std_dev*np.random.randn(rows, cols)-std_dev
    w = np.array(w, ndmin=2, dtype='complex128')
    w += 1j*(std_dev*np.random.randn(rows, cols)-std_dev)/5
    return w
    
    
    #return 2*std_dev*np.random.rand(rows, cols)-std_dev


