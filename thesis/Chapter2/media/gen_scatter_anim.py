import os
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from matplotlib.animation import FuncAnimation
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm, animation


def f(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))


def get_data(mins, maxs, npts):

    x = np.linspace(mins[0], maxs[0], npts)
    y = np.linspace(mins[1], maxs[1], npts)
    
    X, Y = np.meshgrid(x, y)
    return X, Y, f(X, Y)


def animate_data_and_model():

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plt.tight_layout()

    # define evaluation points for model contour
    X, Y, Z = get_data((-6, -6), (6, 6), 30)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    def init():
        ax.contour3D(X, Y, Z, 50, cmap='hsv')
        return fig,

    def animate(i):
        print(i)
        ax.view_init(elev=30.0, azim=i)
        return fig,
    
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=25)

    # save a gif (needed for ppt)
    anim.save("example_plot.gif", fps=5, writer='imagemagick')

    # save a stack of png files (needed for beamer)
    subdir = os.path.join(os.getcwd(), 'pngs')
    os.mkdir(subdir)
    os.chdir(subdir)
    anim.save("anim.png", writer="imagemagick")

    return None


if __name__ == '__main__':

    animate_data_and_model()
