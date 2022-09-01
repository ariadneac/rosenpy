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

def tanh(x, derivative=False):
    if derivative:
        return 1/np.square(np.cosh(x))
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

def sinh(x, derivative=False):
    if derivative:
        return np.cosh(x)
    return np.sinh(x)

def atanh(x, derivative=False):
    if derivative:
        return 1/(1-np.square(x))
    return np.arctanh(x)

def asinh(x, derivative=False):
    if derivative:
        return 1/(1+np.square(x))
    return np.arcsinh(x)

def tan(x, derivative=False):
    if derivative:
        return 1/(np.square(np.cos(x)))
    return np.tan(x)

def sin(x, derivative=False):
    if derivative:
        return np.cos(x)
    return np.sin(x)

def atan(x, derivative=False):
    if derivative:
        return 1/(1+np.square(x))
    return np.arctan(x)

def asin(x, derivative=False):
    if derivative:
        return 1/np.sqrt((1-np.square(x)))
    return np.arcsin(x)

def acos(x, derivative=False):
    if derivative:
        return 1/np.sqrt((np.square(x)-1))
    return np.arccos(x)

def sech(x, derivative=False):
    if derivative:
        return -(2/(np.exp(x) + np.exp(-x)))*(np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
    return 2/(np.exp(x) + np.exp(-x))

def linear(x, derivative=False):
    return np.ones_like(x) if derivative else x

def splitComplex(y, act, derivative=False):
    if (derivative):
        return act(np.real(y), derivative=True) + 1.0j*act(np.imag(y), derivative=True)
    return act(y.real) + 1.0j*act(y.imag)