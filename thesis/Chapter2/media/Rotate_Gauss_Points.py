# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 23:46:50 2021

@author: Kiffer
"""

import os
import numpy as np
from sympy import *

from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from matplotlib.animation import FuncAnimation
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm, animation

import pdb

# In[gp]
# Linear Gauss Quadrature
c0, c1, c2, c3, x, w1, w2, xi1, xi2 = symbols('c_0 c_1 c_2 c_3 x W_1 W_2 xi_1 xi_2')
# General Shape Function
integrand = c0 + c1*x + c2*x**2 + c3*x**3
I = integrate(integrand,(x,-1,1))
display(I)
GLQ = w1*integrand.subs(x,xi1) + w2*integrand.subs(x,xi2)
display(GLQ)
factor(GLQ)

eq1 = Eq(GLQ.subs([(c0,1),(c1,0),(c2,0),(c3,0)]) - I.subs([(c0,1),(c2,0)]),0)
eq2 = Eq(GLQ.subs([(c0,0),(c1,1),(c2,0),(c3,0)]),0)
eq3 = Eq(GLQ.subs([(c0,0),(c1,0),(c2,1),(c3,0)]) - I.subs([(c0,0),(c2,1)]),0)
eq4 = Eq(GLQ.subs([(c0,0),(c1,0),(c2,0),(c3,1)]),0)

eq = [eq1,eq2,eq3,eq4]
display(Matrix(eq))
ans = solve((eq1,eq2,eq3,eq4), (w1,w2,xi1,xi2))
print('Gauss Weights & Points')
display(Matrix(ans))

import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
    

# In[Plots]
def model(fig, ax):

    def plot_cube(cube_definition):
        cube_definition_array = [np.array(list(item)) for item in cube_definition]

        points = []
        points += cube_definition_array
        vectors = [
            cube_definition_array[1] - cube_definition_array[0],
            cube_definition_array[2] - cube_definition_array[0],
            cube_definition_array[3] - cube_definition_array[0]
        ]

        points += [cube_definition_array[0] + vectors[0] + vectors[1]]
        points += [cube_definition_array[0] + vectors[0] + vectors[2]]
        points += [cube_definition_array[0] + vectors[1] + vectors[2]]
        points += [cube_definition_array[0] + vectors[0] + vectors[1] + vectors[2]]

        points = np.array(points)

        edges = [
            [points[0], points[3], points[5], points[1]],
            [points[1], points[5], points[7], points[4]],
            [points[4], points[2], points[6], points[7]],
            [points[2], points[6], points[3], points[0]],
            [points[0], points[2], points[4], points[1]],
            [points[3], points[6], points[7], points[5]]
        ]

        #fig = plt.figure()
        #ax = fig.add_subplot(111, projection='3d')

        faces = Poly3DCollection(edges, linewidths=1, edgecolors='k')
        faces.set_facecolor((0,0,1,0.1))

        ax.add_collection3d(faces)

        # Plot the points themselves to force the scaling of the axes
        ax.scatter(points[:,0], points[:,1], points[:,2], s=0)

        #ax.set_aspect('equal')


    cube_definition = [(-1,-1,-1), (1,-1,-1), (-1,1,-1), (-1,-1,1)]
    plot_cube(cube_definition)

    # Draw the axis
    x, y, z = np.array([[-1,0,0],[0,-1,0],[0,0,-1]])
    u, v, w = np.array([[2,0,0],[0,2,0],[0,0,2]])
    ax.quiver(x,y,z,u,v,w,arrow_length_ratio=0.05, color='b')

    def make_dashedLines(x,y,z,ax):
        for i in range(0, len(x)):
            x_val, y_val, z_val = x[i],y[i],z[i]
            ax.plot([0,x_val],[y_val,y_val],zs=[0,0], linestyle="dashed",color="black")
            ax.plot([x_val,x_val],[0,y_val],zs=[0,0], linestyle="dashed",color="black")
            ax.plot([x_val,x_val],[y_val,y_val],zs=[0,z_val], linestyle="dashed",color="black")

    num = 1
    gps = 2
    for i in range(gps):
        for j in range(gps):
            for k in range(gps):
                x = ans[0][i + gps]
                y = ans[1][j + gps]
                z = ans[1][k + gps]
                ax.scatter(float(x), float(y), float(z), c='r', marker='o', s=10)
                ax.text(float(x)+0.1, float(y)+0.1, float(z)+0.1,  '%s' % (str(num)), size=8, zorder=1, color='k') 
                num = num + 1
    plt.show()     
    codeDir = os.getcwd()
    masterPath = os.path.split(codeDir)
    mediaPath = os.path.join(masterPath[0], 'media', '3D_Linear_Gauss_Quad_Points.pdf')
    #fig.savefig(mediaPath, bbox_inches='tight')
    return fig

def animate_data_and_model():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel(r'$\xi$')
    ax.set_xlim(-1, 1)
    ax.xaxis.set_major_locator(LinearLocator(5))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_ylabel(r'$\eta$')
    ax.set_ylim(-1, 1)
    ax.yaxis.set_major_locator(LinearLocator(5))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_zlabel(r'$\gamma$')
    ax.set_zlim(-1, 1)
    ax.zaxis.set_major_locator(LinearLocator(5))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.01f'))
    
    init = model(fig, ax)
    # pdb.set_trace()
    def animate(i):
        print(i)
        ax.view_init(elev=30.0, azim=i)
        return fig,
    
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=25)

    # # save a gif (needed for ppt)
    # anim.save("example_plot.gif", fps=5, writer='imagemagick')

    # save a stack of png files (needed for beamer)
    subdir = os.path.join(os.getcwd(), 'pngs')
    os.mkdir(subdir)
    os.chdir(subdir)
    anim.save("anim.png", writer="imagemagick")

    return None

if __name__ == '__main__':

    animate_data_and_model()