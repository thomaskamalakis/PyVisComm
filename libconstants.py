#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 09:25:26 2019

@author: thomas
"""

import numpy as np
    
# Constants 
h=6.62607004e-34                  # Planck's constant
kB=1.38064852e-23                 # Boltzmann's constant
c=3.0e8                           # Speed of light
b=2.8977729e-3                    # Wien's constant
Dl_visible=360e-9                 # Visible spectrum width
q=1.602e-19                       # electron charge
lcentral=(380.0+740.0)/2.0*1e-9   # central wavelength of visible spectrum

# Orientations
orient= {
    '+z': np.array([0.0,0.0,1.0]),
    '-z': np.array([0.0,0.0,-1.0]),
    '+y': np.array([0.0,1.0,0.0]),
    '-y': np.array([0.0,-1.0,0.0]),
    '+x': np.array([1.0,0.0,0.0]),
    '-x': np.array([-1.0,0.0,0.0]),
    }


