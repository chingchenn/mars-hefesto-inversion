#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 14:14:21 2026

@author: chingchen
"""
import glob
import numpy as np
import matplotlib.pyplot as plt

labelsize=12
bwith=2
path='/Users/chingchen/Desktop/HeFESTo/python/adin_files/'
file = glob.glob(path + "ad*")


radius  = 3,389.5 
g_mars  = 3.72
rho = 3500  # kg/m^3
g   = 3.72  # m/s^2
R   = 3389.5  # km

colors = ['#282130', '#849DAB', '#CD5C5C', '#35838D',
          '#97795D', '#414F67', '#4198B9', '#2F4F4F']
    
    
print(file)
fig1, ax1 = plt.subplots(figsize=(5,7))
fig2, ax2 = plt.subplots(figsize=(10,7))
for kk in file:
    pressure,_,temp = np.loadtxt(kk).T
    ax1.plot(temp, pressure,label =kk )
    depth = (pressure * 1e9) / (rho * g) / 1000  # km
    radius = R - depth
    ax2.plot(radius,temp,label =kk,color=colors[6])

ax1.set_xlabel("temperature", fontsize=labelsize)
ax1.set_ylabel("pressure", fontsize=labelsize)
ax2.set_ylabel("temperature", fontsize=labelsize)
ax2.set_xlabel("radius", fontsize=labelsize)
ax1.set_xlim(0,2500)
ax1.set_ylim(25,0)
ax2.set_xlim(R,0)
for ax in [ax1, ax2]:
    
    ax.set_title('geotherom', fontsize=labelsize)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.tick_params(labelsize=labelsize)
  
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(bwith)
