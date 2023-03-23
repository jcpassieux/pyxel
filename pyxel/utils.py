import numpy as np
#from numba import njit
import matplotlib.pyplot as plt

import timeit as time
class Timer():
    def __init__(self):
        self.tstart = time.default_timer()
    def stop(self):
        dt = time.default_timer() - self.tstart
        print('Elapsed: %f' % dt)
        return dt

#@njit(cache=True)
def isInBox(b, x, y, z=None):
    """Find whether set of points of coords x, y
    is in the box b = [[xmin, ymin, zmin],
                       [xmax, ymax, zmax]]"""
    if len(b) != 2:
        print("the box not correct")
    e = 1e-6 * np.max(np.abs(b.ravel())) + 1e-6 * np.std(b.ravel())
    if z is None:
        return (
            ((b[0, 0] - e) < x)
            * ((b[0, 1] - e) < y)
            * (x < (b[1, 0] + e))
            * (y < (b[1, 1] + e))
        )
    else:
        return (
            ((b[0, 0] - e) < x)
            * ((b[0, 1] - e) < y)
            * ((b[0, 2] - e) < z)
            * (x < (b[1, 0] + e))
            * (y < (b[1, 1] + e))
            * (z < (b[1, 2] + e))
        )


#@njit(cache=True)
def meshgrid(a, b):
    A = a.repeat(len(b)).reshape((-1, len(b))).T
    B = b.repeat(len(a)).reshape((-1, len(a)))
    return A, B

def PlotMeshImage(f, m, cam, U=None, plot='mesh'):
    """Plotting the mesh over the image. 

    Parameters
    ----------
    f : pyxel.Image
        The image over which to plot the mesh
    m : pyxel.Mesh
        The mesh
    cam : pyxel.Camera
        The camera model
    U : Numpy array
        A displacement dof vector (OPTIONNAL) to warp the mesh.
    plot : String (OPTIONNAL)
        'mesh': plot the mesh in yellow (DEFAULT)
        'displ': plot contour displacement field
        'strain': plot contour strain field

    """
    n = m.n.copy()
    if U is not None:
        n += U[m.conn]
    plt.figure()
    f.Plot()
    u, v = cam.P(n[:, 0], n[:, 1])
    if plot == 'mesh':
        m.Plot(n=np.c_[v, u], edgecolor="y", alpha=0.6)
    elif plot == 'strain':
        m.PlotContourStrain(U, n=np.c_[v, u], alpha=0.8, stype='maxpcp', newfig=False)
    elif plot == 'displ':
        m.PlotContourDispl(U, n=np.c_[v, u], alpha=0.8, stype='mag', newfig=False)
    else:
        print('Unknown plot type in PlotMeshImage')
    # plt.xlim([0,f.pix.shape[1]])
    # plt.ylim([f.pix.shape[0],0])
    # plt.axis("on")