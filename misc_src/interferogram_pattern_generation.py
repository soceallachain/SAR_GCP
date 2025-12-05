#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 14:39:19 2025

@author: joe
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.special import jv  # Bessel function of the first kind
plt.style.use('dark_background')

phi = np.linspace(- 5*np.pi, 5*np.pi, 2000)
chi = phi
x = 3.141519/6
y = x/2

Phi, Chi = np.meshgrid(phi, chi)

X = np.sin(x - Phi) / (x - Phi)
Y = np.sin(y - Chi) / (y - Chi)

P = ((X**2) * (Y**2))

plt.figure(figsize=(8,6))
#plt.pcolormesh(Phi, Chi, P, shading='auto', cmap='viridis', norm=LogNorm())
plt.pcolormesh(Phi, Chi, np.sqrt(P), shading='auto', cmap='viridis')
plt.colorbar(label='P')
plt.xlabel('phi')
plt.ylabel('chi')
plt.title('2D Plot of P(phi, chi)')
plt.show()


plt.figure(figsize=(8,6))

contour = plt.contourf(Phi, Chi, np.sqrt(P), levels=50, cmap='viridis')
#contour = plt.contourf(Phi, Chi, np.sqrt(P), levels=10, cmap='viridis', norm=LogNorm())
plt.colorbar(contour, label='P value')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Contour Plot of 2D Sinc^2 ')
plt.axis('equal')  # Equal scaling by x and y
plt.show()



from scipy.special import j0  # Bessel function of the first kind, order 0

# Create a 2D grid of points
x = np.linspace(-20, 20, 400)
y = np.linspace(-20, 20, 400)
X, Y = np.meshgrid(x, y)

# Compute radius values from origin
R = np.sqrt(X**2 + Y**2)

# Evaluate Bessel function on the 2D grid
Z = j0(R)

# Plot as a heatmap
plt.figure(figsize=(8, 6))
plt.pcolormesh(X, Y, Z, shading='auto', cmap='viridis')
plt.colorbar(label=r'$J_0(r)$')
plt.xlabel('x')
plt.ylabel('y')
plt.title('2D Bessel Function $J_0(r)$')
plt.show()



# Plot contour
plt.figure(figsize=(8, 6))
contours = plt.contourf(X, Y, Z, levels=50, cmap='viridis')  # filled contours
plt.colorbar(contours, label=r'$J_0(r)$')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Contour Plot of 2D Bessel Function $J_0(r)$')
plt.axis('equal')  # Equal scaling by x and y
plt.show()


phi = np.linspace(- 5*np.pi, 5*np.pi, 2000)
chi = phi
x = 3.141519/6
y = x/2

Phi, Chi = np.meshgrid(phi, chi)

X = np.sin(x - Phi) / (x - Phi)
Y = np.sin(y - Chi) / (y - Chi)

P = ((X**2) * (Y**2))

plt.figure(figsize=(8,6))
#plt.pcolormesh(Phi, Chi, P, shading='auto', cmap='viridis', norm=LogNorm())
plt.pcolormesh(Phi, Chi, np.sqrt(P), shading='auto', cmap='viridis')
plt.colorbar(label='P')
plt.xlabel('phi')
plt.ylabel('chi')
plt.title('2D Plot of P(phi, chi)')
plt.show()



# Define range of x values
x = np.linspace(0, 20, 1000)

# Plot Bessel functions for different orders n
plt.figure(figsize=(8, 6))
for n in range(4):
    y = jv(n, x)
    plt.plot(x, y, label=f'$J_{n}(x)$')

plt.xlabel('x')
plt.ylabel('Bessel function value')
plt.title('Bessel Functions of the First Kind')
plt.legend()
plt.grid(True)
plt.show()



phi = np.linspace(0,2*3.141519, 1000)
chi = phi

x=3.141519/6;
y=x/2;


X=np.sin(x-phi)/(x-phi)
Y=np.sin(y-phi)/(y-phi)


P=np.square(X) * np.square(Y)



