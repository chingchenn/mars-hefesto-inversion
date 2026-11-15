#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 21:31:37 2026

@author: chingchen
"""

import numpy as np 

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
colors = ['#282130','#849DAB','#CD5C5C','#35838D',
          '#97795D','#414F67','#4198B9','#2F4F4F']

pressure  = np.linspace(0,26,50)

fig3, ax3 = plt.subplots(figsize=(5,7))
for P in pressure:
    if P <=10:
        T_solidus = -4.877*P**2 + 120.2*P + 1088   
    elif (P > 10) and (P <= 23):    
        T_solidus = -1.323*(P-10)**2 + 38.18*(P-10) + 1802   
    else:
        T_solidus = 77.75*(P-23) + 2075   
    ax3.scatter(T_solidus, P,color=colors[5])


ax3.set_ylim(26,0)
