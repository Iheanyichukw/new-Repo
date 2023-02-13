# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 18:46:54 2023

@author: user
"""

import numpy as np 
import matplotlib.pyplot as plt

x = np.linspace (-3,5, 1000)
sinh = np.sinh(x)
cosh = np.cosh(x)

plt.figure ()
plt.plot(x,sinh, label = "sinh")
plt.plot(x,cosh, label = "cosh")
plt.legend()
plt.xlim(-3,5)
plt.xlabel("x")
plt.ylabel("y")
plt.show()

def trigon1 (t,a,b,j):
    x = np.cos (a*t) - np.cos (b*t)**j
    return x 

t = np.linspace (0,2*np.pi, 1000)
a = 1
b = 60
j = 3

plt.figure()
plt.plot(t,trigon1(t,a,b,j), label = "trigon1")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

def trigon2(t,c,d,k):
    x = np.sin(c * t)- np.sin(d*t)**k
    return x 

c=1
d=120
k=4
plt.figure(figsize =(6,6))
plt.plot(trigon1(t,a,b,j),trigon2(t,c,d,k))
plt.show()
plt.savefig("picture.png")
