# -*- coding: utf-8 -*-
"""
Finite Element Digital Image Correlation method 

@author: JC Passieux, INSA Toulouse, 2021

pyxel

PYthon library for eXperimental mechanics using Finite ELements

"""
import os
import numpy as np
import scipy as sp
import scipy.sparse.linalg as splalg
import scipy.interpolate as spi
import matplotlib.pyplot as plt
import PIL.Image as image
import matplotlib.collections as cols
from numba import njit

import vtktools as vtk

#import pdb
#pdb.set_trace()

import timeit as time
class Timer():
    def __init__(self):
        self.tstart = time.default_timer()
    def stop(self):
        dt = time.default_timer() - self.tstart
        print('Elapsed: %f' % dt)
        return dt

#%% Mesh Utils

class Elem:
    """ Class Element """
    def __init__(self):
        self.pgx = []
        self.pgy = []
        self.phi = []
        self.dphidx = []
        self.dphidy = []

def IsPointInElem2d(xn, yn, x, y):
    """Find if a point (XP, YP) belong to an element with vertices (XN, YN)"""
    yes = np.ones(len(x))
    cx = np.mean(xn)
    cy = np.mean(yn)
    for jn in range(0, len(xn)):
        if jn == (len(xn) - 1):
            i1 = jn
            i2 = 0
        else:
            i1 = jn
            i2 = jn + 1
        x1 = xn[i1]
        y1 = yn[i1]
        dpx = xn[i2] - x1
        dpy = yn[i2] - y1
        a = dpy * (x - x1) - dpx * (y - y1)
        b = dpy * (cx - x1) - dpx * (cy - y1)
        yes = yes * (a * b >= 0)
    return yes


@njit(cache=True)
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

def GetPixelsQua(xn, yn, xpix, ypix):
    """Finds the pixels that belong to a quadrilateral element and"""
    """inverse the mapping to know their corresponding position in """
    """the parent element."""
    wg = IsPointInElem2d(xn, yn, xpix, ypix)
    ind = np.where(wg)
    xpix = xpix[ind]
    ypix = ypix[ind]
    xg = 0 * xpix
    yg = 0 * ypix
    res = 1
    for k in range(7):
        N = np.array([0.25 * (1 - xg) * (1 - yg),
                      0.25 * (1 + xg) * (1 - yg),
                      0.25 * (1 + xg) * (1 + yg),
                      0.25 * (1 - xg) * (1 + yg)]).T
        N_r = np.array([-0.25 * (1 - yg), 0.25 * (1 - yg), 0.25 * (1 + yg), -0.25 * (1 + yg)]).T
        N_s = np.array([-0.25 * (1 - xg), -0.25 * (1 + xg), 0.25 * (1 + xg), 0.25 * (1 - xg)]).T
        dxdr = np.dot(N_r, xn)
        dydr = np.dot(N_r, yn)
        dxds = np.dot(N_s, xn)
        dyds = np.dot(N_s, yn)
        detJ = dxdr * dyds - dydr * dxds
        invJ = np.array([dyds / detJ, -dxds / detJ, -dydr / detJ, dxdr / detJ]).T
        xp = np.dot(N, xn)
        yp = np.dot(N, yn)
        dxg = invJ[:, 0] * (xpix - xp) + invJ[:, 1] * (ypix - yp)
        dyg = invJ[:, 2] * (xpix - xp) + invJ[:, 3] * (ypix - yp)
        res = np.dot(dxg, dxg) + np.dot(dyg, dyg)
        xg = xg + dxg
        yg = yg + dyg
        if res < 1.0e-6:
            break
    return [xg, yg, xpix, ypix]


def GetPixelsTri(xn, yn, xpix, ypix):
    """Finds the pixels that belong to a triangle element and"""
    """inverse the mapping to know their corresponding position in """
    """the parent element."""
    wg = IsPointInElem2d(xn, yn, xpix, ypix)
    ind = np.where(wg)
    xpix = xpix[ind]
    ypix = ypix[ind]
    xg = 0 * xpix
    yg = 0 * ypix
    res = 1
    for k in range(7):
        N = np.array([1 - xg - yg, xg, yg]).T
        N_r = np.array([-np.ones(xg.shape), np.ones(xg.shape), np.zeros(xg.shape)]).T
        N_s = np.array([-np.ones(xg.shape), np.zeros(xg.shape), np.ones(xg.shape)]).T
        dxdr = np.dot(N_r, xn)
        dydr = np.dot(N_r, yn)
        dxds = np.dot(N_s, xn)
        dyds = np.dot(N_s, yn)
        detJ = dxdr * dyds - dydr * dxds
        invJ = np.array([dyds / detJ, -dxds / detJ, -dydr / detJ, dxdr / detJ]).T
        xp = np.dot(N, xn)
        yp = np.dot(N, yn)
        dxg = invJ[:, 0] * (xpix - xp) + invJ[:, 1] * (ypix - yp)
        dyg = invJ[:, 2] * (xpix - xp) + invJ[:, 3] * (ypix - yp)
        res = np.dot(dxg, dxg) + np.dot(dyg, dyg)
        xg = xg + dxg
        yg = yg + dyg
        if res < 1.0e-6:
            break
    return [xg, yg, xpix, ypix]

@njit(cache=True)
def SubQuaIso(nx, ny):
    """Subdivide a Quadrilateral to build the quadrature rule"""
    px = 1.0 / nx
    xi = np.linspace(px - 1, 1 - px, int(nx))
    py = 1.0 / ny
    yi = np.linspace(py - 1, 1 - py, int(ny))
    xg, yg = meshgrid(xi, yi)
    wg = 4.0 / (nx * ny)
    return xg.ravel(), yg.ravel(), wg

@njit(cache=True)
def meshgrid(a, b):
    A = a.repeat(len(b)).reshape((-1, len(b))).T
    B = b.repeat(len(a)).reshape((-1, len(a)))
    return A, B

@njit(cache=True)
def SubTriIso(nx, ny):
    """Subdivide a Triangle to build the quadrature rule (possibly heterogeneous subdivision)"""
    # M1M2 is divided in nx and M1M3 in ny, the meshing being heterogeneous, we
    # end up with trapezes on the side of hypothenuse, the remainder are rectangles
    px = 1 / nx
    py = 1 / ny
    if nx > ny:
        xg = np.zeros(int(np.sum(np.floor(ny * (1 - np.arange(1, nx + 1) / nx))) + nx))
        yg = xg.copy()
        wg = xg.copy()
        j = 1
        for i in range(1, nx + 1):
            niy = int(ny * (1 - i / nx)) # nb of full rectangles in the vertical dir
            v = np.array([[(i - 1) * px, niy * py],
                          [(i - 1) * px, 1 - (i - 1) * px],
                          [i * px, niy * py],
                          [i * px, 1 - i * px]])
            neww = px * (v[3, 1] - v[0, 1]) + px * (v[1, 1] - v[3, 1]) / 2
            newx = ((v[3, 1] - v[0, 1]) * (v[2, 0] + v[0, 0]) / 2
                    + (v[1, 1] - v[3, 1]) / 2 * (v[0, 0] + px / 3)) * px / neww
            newy = ((v[3, 1] - v[0, 1]) * (v[0, 1] + v[3, 1]) / 2
                    + (v[1, 1] - v[3, 1]) / 2 * (v[3, 1] + (v[1, 1] - v[3, 1]) / 3)) * px / neww
            xg[(j - 1) : j + niy] = np.append((px / 2 + (i - 1) * px) * np.ones(niy), newx)
            yg[(j - 1) : j + niy] = np.append(py / 2 + np.arange(niy) * py, newy)
            wg[(j - 1) : j + niy] = np.append(px * py * np.ones(niy), neww)
            j = j + niy + 1
    else:
        xg = np.zeros(int(np.sum(np.floor(nx * (1 - np.arange(1, ny + 1) / ny))) + ny))
        yg = xg.copy()
        wg = xg.copy()
        j = 1
        for i in range(1, ny + 1):
            nix = int(nx * (1 - i / ny))  # number of full rectangles in the horizontal dir
            v = np.array([[nix * px, (i - 1) * py],
                          [nix * px, i * py],
                          [1 - (i - 1) * py, (i - 1) * py],
                          [1 - i * py, i * py]])
            neww = py * (v[3, 0] - v[0, 0]) + py * (v[2, 0] - v[3, 0]) / 2
            newx = ((v[3, 0] - v[0, 0]) * (v[3, 0] + v[0, 0]) / 2
                    + (v[2, 0] - v[3, 0]) / 2 * (v[3, 0] + (v[2, 0] - v[3, 0]) / 3)) * py / neww
            newy = ((v[3, 0] - v[0, 0]) * (v[1, 1] + v[0, 1]) / 2
                    + (v[2, 0] - v[3, 0]) / 2 * (v[0, 1] + py / 3)) * py / neww
            xg[(j - 1) : j + nix] = np.append(px / 2 + np.arange(nix) * px, newx)
            yg[(j - 1) : j + nix] = np.append((py / 2 + (i - 1) * py) * np.ones(nix), newy)
            wg[(j - 1) : j + nix] = np.append(px * py * np.ones(nix), neww)
            j = j + nix + 1
    return xg, yg, wg


@njit(cache=True)
def SubTriIso2(nx, ny=None):
    """Subdivide a Triangle to build the quadrature rule (homogeneous subdivision, faster)"""
    # M1M2 and M1M3 are divided into (nx+ny)/2, the meshing being homogeneous, we
    # end up with triangles on the side of hypothenuse, the remainder are rectangles
    if ny is None:
        n = nx
    else:
        n = (nx + ny) // 2
    pxy = 1 / n
    xg = np.zeros(n * (n + 1) // 2)
    yg = np.zeros(n * (n + 1) // 2)
    wg = np.zeros(n * (n + 1) // 2)
    xi = np.arange(n - 1) / n + 0.5 * pxy
    [qx, qy] = meshgrid(xi, xi)
    qx = qx.ravel()
    qy = qy.ravel()
    (rep,) = np.where(qy - (1 - qx) < -1e-5)
    xg[: n * (n - 1) // 2] = qx[rep]
    yg[: n * (n - 1) // 2] = qy[rep]
    wg[: n * (n - 1) // 2] = pxy ** 2
    yi = np.arange(n) / n + 2 / 3 * pxy
    xg[n * (n - 1) // 2 :] = 1 - yi
    yg[n * (n - 1) // 2 :] = yi - pxy * 1 / 3
    wg[n * (n - 1) // 2 :] = pxy ** 2 / 2
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # plt.plot([0,1,0,0],[0,0,1,0],'k-')
    # plt.axis('equal')
    # ticks = np.linspace(0,1,nxy+1)
    # ax.set_xticks(ticks)
    # ax.set_yticks(ticks)
    # ax.grid(which='both')
    # plt.plot(qx,qy,'ko')
    # plt.plot(wx,wy,'ro')
    return xg, yg, wg

def SubTriGmsh(nx,ny):
    import gmsh as gmsh
    gmsh.initialize()
    gmsh.model.add("P")
    gmsh.model.geo.addPoint(nx[0], ny[0], 0, 1, 1)
    gmsh.model.geo.addPoint(nx[1], ny[1], 0, 1, 2)
    gmsh.model.geo.addPoint(nx[2], ny[2], 0, 1, 3)
    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 3, 2)
    gmsh.model.geo.addLine(3, 1, 3)
    gmsh.model.geo.addCurveLoop([1,2,3],1)
    gmsh.model.geo.addPlaneSurface([1],1)
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.setRecombine(2,1)
    gmsh.model.mesh.generate(2)
    nums, nodes, e = gmsh.model.mesh.getNodes()
    nodes = nodes.reshape((len(nums), 3))
    nodes = nodes[:,:-1]
    c = np.empty((0,2))
    a = []
    if 2 in gmsh.model.mesh.getElementTypes():
        nums, els = gmsh.model.mesh.getElementsByType(2)
        nnd = len(els) // len(nums)
        elems = els.reshape((len(nums), nnd)).astype(int) - 1
        a = np.append(a, 0.5 * abs((nodes[elems[:,1],0] - nodes[elems[:,0],0]) * 
                    (nodes[elems[:,2],1] - nodes[elems[:,0],1]) - 
                    (nodes[elems[:,2],0] - nodes[elems[:,0],0]) * 
                    (nodes[elems[:,1],1] - nodes[elems[:,0],1])))
        c = np.vstack((c, (nodes[elems[:,0]] + nodes[elems[:,1]] + nodes[elems[:,2]]) / 3))
    if 3 in gmsh.model.mesh.getElementTypes():
        nums, els = gmsh.model.mesh.getElementsByType(3)
        nnd = len(els) // len(nums)
        elems = els.reshape((len(nums), nnd)).astype(int) - 1
        a = np.append(a, 0.5 * ( abs((nodes[elems[:,0],0] - nodes[elems[:,2],0]) * 
                    (nodes[elems[:,1],1] - nodes[elems[:,3],1])) + 
                    abs((nodes[elems[:,1],0] - nodes[elems[:,3],0]) * 
                    (nodes[elems[:,0],1] - nodes[elems[:,2],1]))))
        c = np.vstack((c, (nodes[elems[:,0]] + nodes[elems[:,1]] + nodes[elems[:,2]] + nodes[elems[:,3]]) / 4))
    # m=px.Gmsh2Mesh(gmsh)
    # m.Plot()
    gmsh.finalize()
    return c[:,0], c[:,1], a

def StructuredMeshQ4(box, dx):
    """Build a structured linear Q4 mesh from two points coordinates (box)
    box = np.array([[xmin, ymin],
                    [xmax, ymax]])   in mesh unit
    dx = [dx, dy]: average element size (can be scalar)  in mesh unit"""
    dbox = box[1] - box[0]
    NE = (dbox / dx).astype(np.int64)
    X, Y = meshgrid(np.linspace(box[0, 0], box[1, 0], NE[0] + 1),
                    np.linspace(box[0, 1], box[1, 1], NE[1] + 1))
    n = np.vstack((X.T.ravel(), Y.T.ravel())).T
    nel = np.prod(NE)
    e = np.zeros((nel, 4), dtype=int)
    for ix in range(NE[0]):
        for iy in range(NE[1]):
            p1 = ix * (NE[1] + 1) + iy
            p4 = ix * (NE[1] + 1) + iy + 1
            p2 = ix * (NE[1] + 1) + iy + NE[1] + 1
            p3 = ix * (NE[1] + 1) + iy + NE[1] + 2
            e[ix * NE[1] + iy, :] = np.array([p1, p2, p3, p4])
    el = {3: e}
    m = Mesh(el, n)
    return m

def StructuredMeshQ9(box, dx):
    """Build a structured quadratic Q9 mesh from two points coordinates (box)
    box = np.array([[xmin, ymin],
                    [xmax, ymax]])    in mesh unit
    dx = [dx, dy]: average element size (can be scalar)  in mesh unit"""
    dbox = box[1] - box[0]
    NE = (dbox / dx).astype(np.int64)
    X, Y = np.meshgrid(np.linspace(box[0, 0], box[1, 0], 2 * NE[0] + 1),
                       np.linspace(box[0, 1], box[1, 1], 2 * NE[1] + 1))
    n = np.vstack((X.T.ravel(), Y.T.ravel())).T
    nel = np.prod(NE)
    e = np.zeros((nel, 9), dtype=int)
    for ix in range(NE[0]):
        for iy in range(NE[1]):
            p1 = 2 * ix * (2 * NE[1] + 1) + 2 * iy
            p8 = p1 + 1
            p4 = p1 + 2
            p5 = 2 * ix * (2 * NE[1] + 1) + 2 * iy + 2 * NE[1] + 1
            p9 = p5 + 1
            p7 = p5 + 2
            p2 = 2 * ix * (2 * NE[1] + 1) + 2 * iy + 2 * (2 * NE[1] + 1)
            p6 = p2 + 1
            p3 = p2 + 2
            e[ix * NE[1] + iy, :] = np.array([p1, p2, p3, p4, p5, p6, p7, p8, p9])
    el = {10: e}
    m = Mesh(el, n)
    return m

def StructuredMeshQ8(box, dx):
    """Build a structured quadratic Q8 mesh from two points coordinates (box)
    box = np.array([[xmin, ymin],
                    [xmax, ymax]])   in mesh unit
    dx = [dx, dy]: average element size (can be scalar)  in mesh unit"""
    dbox = box[1] - box[0]
    NE = (dbox / dx).astype(np.int64)
    X, Y = np.meshgrid(np.linspace(box[0, 0], box[1, 0], 2 * NE[0] + 1),
                       np.linspace(box[0, 1], box[1, 1], 2 * NE[1] + 1))
    n = np.vstack((X.T.ravel(), Y.T.ravel())).T
    nel = np.prod(NE)
    e = np.zeros((nel, 8), dtype=int)
    for ix in range(NE[0]):
        for iy in range(NE[1]):
            p1 = 2 * ix * (2 * NE[1] + 1) + 2 * iy
            p8 = p1 + 1
            p4 = p1 + 2
            p5 = 2 * ix * (2 * NE[1] + 1) + 2 * iy + 2 * NE[1] + 1
            p7 = p5 + 2
            p2 = 2 * ix * (2 * NE[1] + 1) + 2 * iy + 2 * (2 * NE[1] + 1)
            p6 = p2 + 1
            p3 = p2 + 2
            n[p5 + 1,:] = n[0,:] * np.nan
            e[ix * NE[1] + iy] = np.array([p1, p2, p3, p4, p5, p6, p7, p8])
    el = {16: e}
    m = Mesh(el, n)
    return m

def StructuredMeshT3(box, dx):
    """Build a structured linear T3 mesh from two points coordinates (box)
    box = np.array([[xmin, ymin],
                    [xmax, ymax]])    in mesh unit
    dx = [dx, dy]: average element size (can be scalar) in mesh unit"""
    dbox = box[1] - box[0]
    NE = (dbox / dx).astype(np.int64)
    X, Y = meshgrid(np.linspace(box[0, 0], box[1, 0], NE[0] + 1),
                    np.linspace(box[0, 1], box[1, 1], NE[1] + 1))
    n = np.vstack((X.T.ravel(), Y.T.ravel())).T
    nel = np.prod(NE) * 2
    e = np.zeros((nel, 3), dtype=int)
    for ix in range(NE[0]):
        for iy in range(NE[1]):
            p1 = ix * (NE[1] + 1) + iy
            p4 = ix * (NE[1] + 1) + iy + 1
            p2 = ix * (NE[1] + 1) + iy + NE[1] + 1
            p3 = ix * (NE[1] + 1) + iy + NE[1] + 2
            e[2 * (ix * NE[1] + iy), :] = np.array([p1, p2, p4])
            e[2 * (ix * NE[1] + iy) + 1, :] = np.array([p2, p3, p4])
    el = {2: e}
    m = Mesh(el, n)
    return m

def StructuredMeshT6(box, dx):
    """Build a structured quadratic T6 mesh from two points coordinates (box)
    box = np.array([[xmin, ymin],
                    [xmax, ymax]])    in mesh unit
    dx = [dx, dy]: average element size (can be scalar) in mesh unit"""
    dbox = box[1] - box[0]
    NE = (dbox / dx).astype(np.int64)
    X, Y = meshgrid(np.linspace(box[0, 0], box[1, 0], 2 * NE[0] + 1),
                       np.linspace(box[0, 1], box[1, 1], 2 * NE[1] + 1))
    n = np.vstack((X.T.ravel(), Y.T.ravel())).T
    nel = np.prod(NE) * 2
    e = np.zeros((nel, 6), dtype=int)
    for ix in range(NE[0]):
        for iy in range(NE[1]):
            p1 = 2 * ix * (2 * NE[1] + 1) + 2 * iy
            p8 = p1 + 1
            p4 = p1 + 2
            p5 = 2 * ix * (2 * NE[1] + 1) + 2 * iy + 2 * NE[1] + 1
            p9 = p5 + 1
            p7 = p5 + 2
            p2 = 2 * ix * (2 * NE[1] + 1) + 2 * iy + 2 * (2 * NE[1] + 1)
            p6 = p2 + 1
            p3 = p2 + 2
            e[2 * (ix * NE[1] + iy), :] = np.array([p1, p2, p4, p5, p9, p8])
            e[2 * (ix * NE[1] + iy) + 1, :] = np.array([p2, p3, p4, p6, p7, p9])
    el = {9: e}
    m = Mesh(el, n)
    return m


def MeshFromROI(roi, dx, f, typel=3):
    """Build a structured FE mesh and a pyxel.camera object from a region
    of interest (ROI) selected in an image f

    Parameters
    ----------
    roi : numpy.array
        The Region of Interest made using  f.SelectROI(), f being a pyxel.Image
    dx : numpy or python array
        dx  = [dx, dy]: average element size (can be scalar) in pixels
    f : pyxel.Image
        The image on which is defined the region of interest
    typel : int
        type of element: {3: 'qua4',2: 'tri3',9: 'tri6',16: 'qua8',10: 'qua9'}

    Returns
    -------
    m : pyxel.Mesh
        The finite element mesh
    cam : pyxel.Camera
        The corresponding camera

    Example:
    -------
    f.SelectROI()  -> select the region with rectangle selector
                      and copy - paste the roi in the python terminal
    m, cam = px.MeshFromROI(roi, [20, 20], f)
    """

    droi = roi[1] - roi[0]
    xmin = np.min(roi[:, 0])
    ymin = f.pix.shape[0] - np.max(roi[:, 1])
    roi = np.array([[0, 0], droi])
    if typel == 3:
        m = StructuredMeshQ4(roi, dx)
    elif typel == 2:
        m = StructuredMeshT3(roi, dx)
    elif typel == 10:
        m = StructuredMeshQ9(roi, dx)
    elif typel == 16:
        m = StructuredMeshQ8(roi, dx)
    elif typel == 9:
        m = StructuredMeshT6(roi, dx)
    else:
        print('Unknown Element Type... !')
    p = np.array([1.0, xmin, ymin - f.pix.shape[0], 0.0])
    cam = Camera(p)
    return m, cam

def ShapeFunctions(eltype):
    """For any type of 2D elements, gives the quadrature rule and
    the shape functions and their derivative"""
    xg = 0
    yg = 0
    wg = 0
    if eltype == 2:
        """
        #############
            tri3
        #############
        """
        def N(x, y):
            return np.concatenate(
                (1 - x - y, x, y)).reshape((3,len(x))).T
        
        def dN_xi(x, y):
            return np.concatenate(
                (-1.0 + 0 * x, 1.0 + 0 * x, 0.0 * x)).reshape((3,len(x))).T

        def dN_eta(x, y):
            return np.concatenate(
                (-1.0 + 0 * x, 0.0 * x, 1.0 + 0 * x)).reshape((3,len(x))).T 

        xg = np.array([1. / 6, 2. / 3, 1. / 6])
        yg = np.array([1. / 6, 1. / 6, 2. / 3])
        wg = 1. / 6 * np.ones(3)
        # xg = np.array([1.0 / 3])
        # yg = np.array([1.0 / 3])
        # wg = np.array([0.5])
    elif eltype == 3:
        """
        #############
            qua4
        #############
        """
        def N(x, y):
            return 0.25 * np.concatenate(((1 - x) * (1 - y),
                                 (1 + x) * (1 - y),
                                 (1 + x) * (1 + y),
                                 (1 - x) * (1 + y))).reshape((4,len(x))).T 

        def dN_xi(x, y):
            return 0.25 * np.concatenate(
                (y - 1, 1 - y, 1 + y, -1 - y)).reshape((4,len(x))).T 

        def dN_eta(x, y):
            return 0.25 * np.concatenate(
                (x - 1, -1 - x, 1 + x, 1 - x)).reshape((4,len(x))).T 

        xg = np.sqrt(3) / 3 * np.array([-1, 1, -1, 1])
        yg = np.sqrt(3) / 3 * np.array([-1, -1, 1, 1])
        wg = np.ones(4)
    elif eltype == 9:
        """
        #############
            tri6
        #############
        """
        def N(x, y):
            return np.concatenate(
                ((1 - x - y) * (2 * (1 - x - y) - 1),
                                   x * (2 * x - 1),
                                   y * (2 * y - 1),
                     4 * (1 - x - y) * x,
                               4 * x * y,
                               4 * y * (1 - x - y))).reshape((6,len(x))).T 

        def dN_xi(x, y):
            return np.concatenate(
                (4 * x + 4 * y - 3, 4 * x - 1, x * 0, 4 * (1 - 2 * x - y), 4 * y, -4 * y)
                ).reshape((6,len(x))).T 

        def dN_eta(x, y):
            return np.concatenate(
                (4 * x + 4 * y - 3, x * 0, 4 * y - 1, -4 * x, 4 * x, 4 * (1 - x - 2 * y))
                ).reshape((6,len(x))).T 
        
        # quadrature using 3 gp
        # xg = np.array([1. / 6, 2. / 3, 1. / 6])
        # yg = np.array([1. / 6, 1. / 6, 2. / 3])
        # wg = 1. / 6 * np.ones(3)
        # quadrature using 6 gp
        a = 0.445948490915965
        b = 0.091576213509771
        xg = np.array([a, 1 - 2 * a, a, b, 1 - 2 * b, b])
        yg = np.array([a, a, 1 - 2 * a, b, b, 1 - 2 * b])
        a = 0.111690794839005
        b = 0.054975871827661
        wg = np.array([a, a, a, b, b, b])
    elif eltype == 10:
        """
        #############
            qua9
        #############
        """

        def N(x, y):
            return np.concatenate(
               ((x * (x - 1) * 0.5) * (y * (y - 1) * 0.5),
                (x * (x + 1) * 0.5) * (y * (y - 1) * 0.5),
                (x * (x + 1) * 0.5) * (y * (y + 1) * 0.5),
                (x * (x - 1) * 0.5) * (y * (y + 1) * 0.5),
                (1 - x ** 2) * (y * (y - 1) * 0.5),
                (x * (x + 1) * 0.5) * (1 - y ** 2),
                (1 - x ** 2) * (y * (y + 1) * 0.5),
                (x * (x - 1) * 0.5) * (1 - y ** 2),
                (1 - x ** 2) * (1 - y ** 2))
               ).reshape((9,len(x))).T

        def dN_xi(x, y):
            return np.concatenate(
               ((x - 0.5) * (y * (y - 1) * 0.5),
                (x + 0.5) * (y * (y - 1) * 0.5),
                (x + 0.5) * (y * (y + 1) * 0.5),
                (x - 0.5) * (y * (y + 1) * 0.5),
                (-2 * x) * (y * (y - 1) * 0.5),
                (x + 0.5) * (1 - y ** 2),
                (-2 * x) * (y * (y + 1) * 0.5),
                (x - 0.5) * (1 - y ** 2),
                (-2 * x) * (1 - y ** 2))
               ).reshape((9,len(x))).T 

        def dN_eta(x, y):
            return np.concatenate(
               ((x * (x - 1) * 0.5) * (y - 0.5),
                (x * (x + 1) * 0.5) * (y - 0.5),
                (x * (x + 1) * 0.5) * (y + 0.5),
                (x * (x - 1) * 0.5) * (y + 0.5),
                (1 - x ** 2) * (y - 0.5),
                (x * (x + 1) * 0.5) * (-2 * y),
                (1 - x ** 2) * (y + 0.5),
                (x * (x - 1) * 0.5) * (-2 * y),
                (1 - x ** 2) * (-2 * y))
            ).reshape((9,len(x))).T 

        a = 0.774596669241483
        xg = a * np.array([-1, 1, -1, 1, 0, 1, 0, -1, 0])
        yg = a * np.array([-1, -1, 1, 1, -1, 0, 1, 0, 0])
        wg = np.array([25, 25, 25, 25, 40, 40, 40, 40, 64]) / 81
    elif eltype == 16:
        """
        #############
            qua8
        #############
        """

        def N(x, y):
            return np.concatenate(
               (-0.25 * (1 - x) * (1 - y) * (1 + x + y),
                -0.25 * (1 + x) * (1 - y) * (1 - x + y),
                -0.25 * (1 + x) * (1 + y) * (1 - x - y),
                -0.25 * (1 - x) * (1 + y) * (1 + x - y),
                 0.50 * (1 - x) * (1 + x) * (1 - y),
                 0.50 * (1 + x) * (1 + y) * (1 - y),
                 0.50 * (1 - x) * (1 + x) * (1 + y),
                 0.50 * (1 - x) * (1 + y) * (1 - y))
            ).reshape((8,len(x))).T 

        def dN_xi(x, y):
            return np.concatenate(
               (-0.25 * (-1 + y) * (2 * x + y),
                0.25 * (-1 + y) * (y - 2 * x),
                0.25 * (1 + y) * (2 * x + y),
                -0.25 * (1 + y) * (y - 2 * x),
                -x * (1 - y),
                -0.5 * (1 + y) * (-1 + y),
                -x * (1 + y),
                -0.5 * (1 + y) * (1 - y))
            ).reshape((8,len(x))).T 

        def dN_eta(x, y):
            return np.concatenate(
               (-0.25 * (-1 + x) * (x + 2 * y),
                0.25 * (1 + x) * (2 * y - x),
                0.25 * (1 + x) * (x + 2 * y),
                -0.25 * (-1 + x) * (2 * y - x),
                0.50 * (1 + x) * (-1 + x),
                -y * (1 + x),
                -0.50 * (1 + x) * (-1 + x),
                y * (-1 + x))
            ).reshape((8,len(x))).T 

        # quadrature using 4 gp
        # xg = np.sqrt(3) / 3 * np.array([-1, 1, -1, 1])
        # yg = np.sqrt(3) / 3 * np.array([-1, -1, 1, 1])
        # wg = np.ones(4)
        # quadrature using 9 gp
        a = 0.774596669241483
        xg = a * np.array([-1, 1, -1, 1, 0, 1, 0, -1, 0])
        yg = a * np.array([-1, -1, 1, 1, -1, 0, 1, 0, 0])
        wg = np.array([25, 25, 25, 25, 40, 40, 40, 40, 64]) / 81
    return xg, yg, wg, N, dN_xi, dN_eta

#%%
class Mesh:
    def __init__(self, e, n, dim=2):
        """Contructor from elems and node arrays"""
        self.e = e
        self.n = n
        self.conn = []
        self.ndof = []
        self.npg = []
        self.pgx = []
        self.pgy = []
        self.phix = None
        self.phiy = None
        self.wdetJ = []
        self.dim = dim

    def Copy(self):
        m = Mesh(self.e.copy(), self.n.copy())
        m.conn = self.conn.copy()
        m.ndof = self.ndof
        m.dim = self.dim
        m.npg = self.npg
        m.pgx = self.pgx.copy()
        m.pgy = self.pgy.copy()
        m.phix = self.phix
        m.phiy = self.phiy
        m.wdetJ = self.wdetJ.copy()
        return m

    def Connectivity(self):
        print("Connectivity.")
        """ Computes connectivity """
        used_nodes = np.zeros(0, dtype=int)
        for je in self.e.keys():
            used_nodes = np.unique(np.append(used_nodes, self.e[je].ravel()))
        nn = len(used_nodes)
        self.conn = -np.ones(self.n.shape[0], dtype=int)
        self.conn[used_nodes] = np.arange(nn)
        if self.dim == 2:
            self.conn = np.c_[self.conn, self.conn + nn * (self.conn >= 0)]
        else:
            self.conn = np.c_[
                self.conn,
                self.conn + nn * (self.conn >= 0),
                self.conn + 2 * nn * (self.conn >= 0),
            ]
        self.ndof = nn * self.dim

    def DICIntegration(self, cam, G=False, EB=False, tri_same=False):
        """Compute FE-DIC quadrature rule along with FE shape functions operators
    
        Parameters
        ----------
        cam : pyxel.Camera
            Calibrated Camera model.
        G : bool
            Set TRUE to get also shape function gradients (default FALSE).
        EB : bool
            Set TRUE to get also the elementary brightness
            and contrast correction operators (default FALSE).   
        tri_same : bool
            Set TRUE to subdivide triangles with the same number of integration
            points in each direction (default FALSE).   
        """
        self.wdetJ = np.array([])
        col = np.array([], dtype=int)
        row = np.array([], dtype=int)
        val = np.array([])
        if G: # shape function gradient
            valx = np.array([])
            valy = np.array([])
        if EB: # Elementary Brightness and Contrast Correction
            cole = np.array([], dtype=int)
            rowe = np.array([], dtype=int)
            vale = np.array([], dtype=int)
        un, vn = cam.P(self.n[:, 0], self.n[:, 1])
        ne = 0
        for et in self.e.keys():
            ne += len(self.e[et])
            repdof = self.conn[self.e[et], 0]
            xn = self.n[self.e[et], 0]
            yn = self.n[self.e[et], 1]
            u = un[self.e[et]]
            v = vn[self.e[et]]
            _, _, _, N, Ndx, Ndy = ShapeFunctions(et)
            nfun = N(np.zeros(1), np.zeros(1)).shape[1]
            if et == 3 or et == 10 or et == 16:  # qua4 or qua9 or qua8
                dist = np.floor(
                    np.sqrt((u[:, :2] - u[:, 1:3]) ** 2 + (v[:, :2] - v[:, 1:3]) ** 2)
                ).astype(int)
                a, b = np.where(dist < 1)
                if len(a):  # at least one integration point in each element
                    dist[a, b] = 1
                npg = np.sum(np.prod(dist, axis=1))
                wdetJj = np.zeros(npg)
                rowj = np.zeros(npg * nfun, dtype=int)
                colj = np.zeros(npg * nfun, dtype=int)
                valj = np.zeros(npg * nfun)
                if G: # shape function gradient
                    valxj = np.zeros(npg * nfun)
                    valyj = np.zeros(npg * nfun)
                if EB: # Elementary Brightness and Contrast Correction
                    rowej = np.zeros(npg, dtype=int)
                    colej = np.zeros(npg, dtype=int)
                    valej = np.zeros(npg, dtype=int)
                npg = 0
                for je in range(len(self.e[et])):
                    xg, yg, wg = SubQuaIso(dist[je, 0], dist[je, 1])
                    phi = N(xg, yg)
                    dN_xi = Ndx(xg, yg)
                    dN_eta = Ndy(xg, yg)
                    repg = npg + np.arange(len(xg))
                    dxdr = dN_xi @ xn[je, :]
                    dydr = dN_xi @ yn[je, :]
                    dxds = dN_eta @ xn[je, :]
                    dyds = dN_eta @ yn[je, :]
                    detJ = dxdr * dyds - dydr * dxds
                    wdetJj[repg] = wg * abs(detJ)
                    [repcol, reprow] = np.meshgrid(repdof[je, :], repg + len(self.wdetJ))
                    rangephi = nfun * npg + np.arange(np.prod(phi.shape))
                    rowj[rangephi] = reprow.ravel()
                    colj[rangephi] = repcol.ravel()
                    valj[rangephi] = phi.ravel()
                    if G:
                        dphidx = (dyds / detJ)[:, np.newaxis] * dN_xi + (-dydr / detJ)[
                            :, np.newaxis
                        ] * dN_eta
                        dphidy = (-dxds / detJ)[:, np.newaxis] * dN_xi + (dxdr / detJ)[
                            :, np.newaxis
                        ] * dN_eta
                        valxj[rangephi] = dphidx.ravel()
                        valyj[rangephi] = dphidy.ravel()
                    if EB:
                        rangeone = npg + np.arange(len(repg))
                        rowej[rangeone] = repg
                        colej[rangeone] = je
                        valej[rangeone] = 1
                    npg += len(xg)
            elif et == 2 or et == 9:  # tri3 or tri6
                if et == 2:
                    n0 = np.array([[0, 1], [0, 0], [1, 0]])
                    n2 = np.array([[1, 0], [0, 1], [0, 0]])
                elif et == 9:
                    n0 = np.array([[0, 1], [0, 0], [1, 0], [0, 0.5], [0.5, 0], [0.5, 0.5]])
                    n2 = np.array([[1, 0], [0, 1], [0, 0], [0.5, 0.5], [0, 0.5], [0.5, 0]])
                uu = np.diff(np.c_[u, u[:, 0]])
                vv = np.diff(np.c_[v, v[:, 0]])
                nn = np.floor(np.sqrt(uu ** 2 + vv ** 2) / 1.1).astype(int)
                b1, b2 = np.where(nn < 1)
                if len(b1):  # at least one integration point in each element
                    nn[b1,b2] = 1
                a = np.argmax(nn, axis=1)  # a is the largest triangle side
                if tri_same:
                    nn = (np.sum(nn, axis=1) - np.amax(nn, axis=1)) // 2  # take the average of nx and ny for subtriso2
                    npg = np.sum(nn * (nn + 1) // 2) # exact number of integration points
                else:
                    nx = nn[np.arange(len(nn)), np.array([2, 0, 1])[a]]
                    ny = nn[np.arange(len(nn)), np.array([1, 2, 0])[a]]
                    npg = np.sum(((nx+1) * (ny+1)) // 2) # overestimate the number of integration points                    
                wdetJj = np.zeros(npg)
                rowj = np.zeros(npg * nfun, dtype=int)
                colj = np.zeros(npg * nfun, dtype=int)
                valj = np.zeros(npg * nfun)
                if G: # shape function gradient
                    valxj = np.zeros(npg * nfun)
                    valyj = np.zeros(npg * nfun)
                if EB: # Elementary Brightness and Contrast Correction
                    rowej = np.zeros(npg, dtype=int)
                    colej = np.zeros(npg, dtype=int)
                    valej = np.zeros(npg, dtype=int)
                npg = 0
                for je in range(len(self.e[et])):
                    if tri_same:
                        xg, yg, wg = SubTriIso2(nn[je])
                    else:
                        xg, yg, wg = SubTriIso(nx[je], ny[je])
                    if a[je] == 0:
                        pp = N(xg, yg) @ n0
                        xg = pp[:, 0]
                        yg = pp[:, 1]
                    elif a[je] == 2:
                        pp = N(xg, yg) @ n2
                        xg = pp[:, 0]
                        yg = pp[:, 1]
                    phi = N(xg, yg)
                    dN_xi = Ndx(xg, yg)
                    dN_eta = Ndy(xg, yg)
                    repg = npg + np.arange(len(xg))
                    dxdr = dN_xi @ xn[je, :]
                    dydr = dN_xi @ yn[je, :]
                    dxds = dN_eta @ xn[je, :]
                    dyds = dN_eta @ yn[je, :]
                    detJ = dxdr * dyds - dydr * dxds
                    wdetJj[repg] = wg * abs(detJ)
                    [repcol, reprow] = meshgrid(repdof[je, :], repg + len(self.wdetJ))
                    rangephi = nfun * npg + np.arange(np.prod(phi.shape))
                    rowj[rangephi] = reprow.ravel()
                    colj[rangephi] = repcol.ravel()
                    valj[rangephi] = phi.ravel()
                    if G:
                        dphidx = (dyds / detJ)[:, np.newaxis] * dN_xi + (-dydr / detJ)[
                            :, np.newaxis
                        ] * dN_eta
                        dphidy = (-dxds / detJ)[:, np.newaxis] * dN_xi + (dxdr / detJ)[
                            :, np.newaxis
                        ] * dN_eta
                        valxj[rangephi] = dphidx.ravel()
                        valyj[rangephi] = dphidy.ravel()
                    if EB:
                        rangeone = npg + np.arange(len(repg))
                        rowej[rangeone] = repg
                        colej[rangeone] = je
                        valej[rangeone] = 1
                    npg += len(xg)
            else:
                print("Oops!  That is not a valid element type...")
            col = np.append(col, colj[:npg * nfun])
            row = np.append(row, rowj[:npg * nfun])
            val = np.append(val, valj[:npg * nfun])
            if G:
                valx = np.append(valx, valxj[:npg * nfun])
                valy = np.append(valy, valyj[:npg * nfun])
            if EB:
                cole = np.append(cole, colej[:npg])
                rowe = np.append(rowe, rowej[:npg])
                vale = np.append(vale, valej[:npg])
            self.wdetJ = np.append(self.wdetJ, wdetJj[:npg])
        self.npg = len(self.wdetJ)
        self.phix = sp.sparse.csc_matrix(
            (val, (row, col)), shape=(self.npg, self.ndof))
        self.phiy = sp.sparse.csc_matrix(
            (val, (row, col + self.ndof // 2)), shape=(self.npg, self.ndof))
        if G:
            self.dphixdx = sp.sparse.csc_matrix(
                (valx, (row, col)), shape=(self.npg, self.ndof))
            self.dphixdy = sp.sparse.csc_matrix(
                (valy, (row, col)), shape=(self.npg, self.ndof))
            self.dphiydx = sp.sparse.csc_matrix(
                (valx, (row, col + self.ndof // 2)), shape=(self.npg, self.ndof))
            self.dphiydy = sp.sparse.csc_matrix(
                (valy, (row, col + self.ndof // 2)), shape=(self.npg, self.ndof))
        if EB:
            self.Me = sp.sparse.csc_matrix(
                (vale, (rowe, cole)), shape=(self.npg, ne))
        qx = np.zeros(self.ndof)
        (rep,) = np.where(self.conn[:, 0] >= 0)
        qx[self.conn[rep, :]] = self.n[rep, :]
        self.pgx = self.phix.dot(qx)
        self.pgy = self.phiy.dot(qx)

    def DICIntegrationPixel(self, cam):
        """ Builds a pixel integration scheme with integration points at the
        center of the image pixels 

        Parameters
        ----------
        cam : pyxel.Camera
            Calibrated Camera model.
        """
        nzv = 0  # nb of non zero values in phix
        repg = 0  # nb of integration points
        ne = 0  # element count
        un, vn = cam.P(self.n[:, 0], self.n[:, 1])
        nelem = 0
        for et in self.e.keys():
            nelem += len(self.e[et])
        elem = np.empty(nelem, dtype=object)
        for et in self.e.keys():
            repdof = self.conn[self.e[et], 0]
            u = un[self.e[et]]
            v = vn[self.e[et]]
            _, _, _, N, _, _ = ShapeFunctions(et)
            if et == 3:  # qua4
                for je in range(len(self.e[et])):
                    elem[ne] = Elem()
                    elem[ne].repx = repdof[je]
                    rx = np.arange(
                        np.floor(min(u[je])), np.ceil(max(u[je])) + 1
                    ).astype("int")
                    ry = np.arange(
                        np.floor(min(v[je])), np.ceil(max(v[je])) + 1
                    ).astype("int")
                    [ypix, xpix] = meshgrid(ry, rx)
                    [xg, yg, elem[ne].pgx, elem[ne].pgy] = GetPixelsQua(
                        u[je], v[je], xpix.ravel(), ypix.ravel()
                    )
                    elem[ne].phi = N(xg, yg)
                    elem[ne].repg = repg + np.arange(xg.shape[0])
                    repg += xg.shape[0]
                    nzv += np.prod(elem[ne].phi.shape)
                    ne += 1
            elif et == 2:  # tri3
                for je in range(len(self.e[et])):
                    elem[ne] = Elem()
                    elem[ne].repx = repdof[je]
                    rx = np.arange(
                        np.floor(min(u[je])), np.ceil(max(u[je])) + 1
                    ).astype("int")
                    ry = np.arange(
                        np.floor(min(v[je])), np.ceil(max(v[je])) + 1
                    ).astype("int")
                    [ypix, xpix] = meshgrid(ry, rx)
                    [xg, yg, elem[ne].pgx, elem[ne].pgy] = GetPixelsTri(
                        u[je], v[je], xpix.ravel(), ypix.ravel()
                    )
                    elem[ne].phi = N(xg, yg)
                    elem[ne].repg = repg + np.arange(xg.shape[0])
                    repg += xg.shape[0]
                    nzv += np.prod(elem[ne].phi.shape)
                    ne += 1
            else:
                print("Oops!  That is not a valid element type...")
        self.npg = repg
        self.pgx = np.zeros(self.npg)
        self.pgy = np.zeros(self.npg)
        for je in range(len(elem)):
            self.pgx[elem[je].repg] = elem[je].pgx
            self.pgy[elem[je].repg] = elem[je].pgy
        """ Inverse Mapping """
        pgX = np.zeros(self.pgx.shape[0])
        pgY = np.zeros(self.pgx.shape[0])
        for ii in range(10):
            pgx, pgy = cam.P(pgX, pgY)
            resx = self.pgx - pgx
            resy = self.pgy - pgy
            dPxdX, dPxdY, dPydX, dPydY = cam.dPdX(pgX, pgY)
            detJ = dPxdX * dPydY - dPxdY * dPydX
            dX = dPydY / detJ * resx - dPxdY / detJ * resy
            dY = -dPydX / detJ * resx + dPxdX / detJ * resy
            pgX += dX
            pgY += dY
            res = np.linalg.norm(dX) + np.linalg.norm(dY)
            if res < 1e-4:
                break
        self.pgx = pgX
        self.pgy = pgY

        """ Builds the FE interpolation """
        self.wdetJ = np.ones(self.npg) / cam.f**2 # f**2 = area of a pixel
        row = np.zeros(nzv)
        col = np.zeros(nzv)
        val = np.zeros(nzv)
        nzv = 0
        for je in range(len(elem)):
            [repj, repi] = meshgrid(elem[je].repx, elem[je].repg)
            rangephi = nzv + np.arange(np.prod(elem[je].phi.shape))
            row[rangephi] = repi.ravel()
            col[rangephi] = repj.ravel()
            val[rangephi] = elem[je].phi.ravel()
            nzv += np.prod(elem[je].phi.shape)
        self.phix = sp.sparse.csc_matrix(
            (val, (row, col)), shape=(self.npg, self.ndof))
        self.phiy = sp.sparse.csc_matrix(
            (val, (row, col + self.ndof // 2)), shape=(self.npg, self.ndof))

    def GaussIntegElem(self, e, et):
        # parent element
        xg, yg, wg, N, Ndx, Ndy = ShapeFunctions(et)
        phi = N(xg, yg)
        dN_xi = Ndx(xg, yg)
        dN_eta = Ndy(xg, yg)
        # elements
        ne = len(e)  # nb of elements
        nfun = phi.shape[1]  # nb of shape fun per element
        npg = len(xg)  # nb of gauss point per element
        nzv = nfun * npg * ne  # nb of non zero values in dphixdx
        wdetJ = np.zeros(npg * ne)
        row = np.zeros(nzv, dtype=int)
        col = np.zeros(nzv, dtype=int)
        val = np.zeros(nzv)
        valx = np.zeros(nzv)
        valy = np.zeros(nzv)
        repdof = self.conn[e, 0]
        xn = self.n[e, 0]
        yn = self.n[e, 1]
        for i in range(len(xg)):
            dxdr = xn @ dN_xi[i, :]
            dydr = yn @ dN_xi[i, :]
            dxds = xn @ dN_eta[i, :]
            dyds = yn @ dN_eta[i, :]
            detJ = dxdr * dyds - dxds * dydr
            wdetJ[np.arange(ne) + i * ne] = abs(detJ) * wg[i]
            dphidx = (dyds / detJ)[:, np.newaxis] * dN_xi[i, :] + (-dydr / detJ)[
                :, np.newaxis] * dN_eta[i, :]
            dphidy = (-dxds / detJ)[:, np.newaxis] * dN_xi[i, :] + (dxdr / detJ)[
                :, np.newaxis] * dN_eta[i, :]
            repnzv = np.arange(ne * nfun) + i * ne * nfun
            col[repnzv] = repdof.ravel()
            row[repnzv] = np.tile(np.arange(ne) + i * ne, [nfun, 1]).T.ravel()
            val[repnzv] = np.tile(phi[i, :], [ne, 1]).ravel()
            valx[repnzv] = dphidx.ravel()
            valy[repnzv] = dphidy.ravel()
        return col, row, val, valx, valy, wdetJ

    def GaussIntegration(self):
        """Builds a Gauss integration scheme"""
        print('Gauss Integration.')
        self.wdetJ = np.array([])
        col = np.array([])
        row = np.array([])
        val = np.array([])
        valx = np.array([])
        valy = np.array([])
        for je in self.e.keys():
            colj, rowj, valj, valxj, valyj, wdetJj = self.GaussIntegElem(self.e[je], je)
            col = np.append(col, colj)
            row = np.append(row, rowj)
            val = np.append(val, valj)
            valx = np.append(valx, valxj)
            valy = np.append(valy, valyj)
            self.wdetJ = np.append(self.wdetJ, wdetJj)
        self.npg = len(self.wdetJ)
        self.phix = sp.sparse.csc_matrix(
            (val, (row, col)), shape=(self.npg, self.ndof))
        self.phiy = sp.sparse.csc_matrix(
            (val, (row, col + self.ndof // 2)), shape=(self.npg, self.ndof))
        self.dphixdx = sp.sparse.csc_matrix(
            (valx, (row, col)), shape=(self.npg, self.ndof))
        self.dphixdy = sp.sparse.csc_matrix(
            (valy, (row, col)), shape=(self.npg, self.ndof))
        self.dphiydx = sp.sparse.csc_matrix(
            (valx, (row, col + self.ndof // 2)), shape=(self.npg, self.ndof))
        self.dphiydy = sp.sparse.csc_matrix(
            (valy, (row, col + self.ndof // 2)), shape=(self.npg, self.ndof))
        rep, = np.where(self.conn[:, 0] >= 0)
        qx = np.zeros(self.ndof)
        qx[self.conn[rep, :]] = self.n[rep, :]
        self.pgx = self.phix.dot(qx)
        self.pgy = self.phiy.dot(qx)

    def Stiffness(self, hooke):
        """Assembles Stiffness Operator"""
        if not hasattr(self, "dphixdx"):
            m = self.Copy()
            m.GaussIntegration()
            wdetJ = sp.sparse.diags(m.wdetJ)
            Bxy = m.dphixdy + m.dphiydx
            K = (hooke[0, 0] * m.dphixdx.T @ wdetJ @ m.dphixdx
               + hooke[1, 1] * m.dphiydy.T @ wdetJ @ m.dphiydy
               + hooke[2, 2] * Bxy.T @ wdetJ @ Bxy
               + hooke[0, 1] * m.dphixdx.T @ wdetJ @ m.dphiydy
               + hooke[0, 2] * m.dphixdx.T @ wdetJ @ Bxy
               + hooke[1, 2] * m.dphiydy.T @ wdetJ @ Bxy
               + hooke[1, 0] * m.dphiydy.T @ wdetJ @ m.dphixdx
               + hooke[2, 0] * Bxy.T @ wdetJ @ m.dphixdx
               + hooke[2, 1] * Bxy.T @ wdetJ @ m.dphiydy)
        else:
            wdetJ = sp.sparse.diags(self.wdetJ)
            Bxy = self.dphixdy + self.dphiydx
            K = (hooke[0, 0] * self.dphixdx.T @ wdetJ @ self.dphixdx
               + hooke[1, 1] * self.dphiydy.T @ wdetJ @ self.dphiydy
               + hooke[2, 2] * Bxy.T @ wdetJ @ Bxy
               + hooke[0, 1] * self.dphixdx.T @ wdetJ @ self.dphiydy
               + hooke[0, 2] * self.dphixdx.T @ wdetJ @ Bxy
               + hooke[1, 2] * self.dphiydy.T @ wdetJ @ Bxy
               + hooke[1, 0] * self.dphiydy.T @ wdetJ @ self.dphixdx
               + hooke[2, 0] * Bxy.T @ wdetJ @ self.dphixdx
               + hooke[2, 1] * Bxy.T @ wdetJ @ self.dphiydy)
        return K
    
    def Tikhonov(self):
        """Assembles Tikhonov (Laplacian) Operator"""
        if not hasattr(self, "dphixdx"):
            m = self.Copy()
            m.GaussIntegration()
            wdetJ = sp.sparse.diags(m.wdetJ)
            L = (m.dphixdx.T @ wdetJ @ m.dphixdx
                + m.dphiydy.T @ wdetJ @ m.dphiydy
                + m.dphixdy.T @ wdetJ @ m.dphixdy
                + m.dphiydx.T @ wdetJ @ m.dphiydx)
        else:
            wdetJ = sp.sparse.diags(self.wdetJ)
            L = (self.dphixdx.T @ wdetJ @ self.dphixdx
                + self.dphiydy.T @ wdetJ @ self.dphiydy
                + self.dphixdy.T @ wdetJ @ self.dphixdy
                + self.dphiydx.T @ wdetJ @ self.dphiydx)
        return L

    def TikoSprings(self, liste, l=None, dim=2):
        """
        Builds a Tikhonov like operator from bar elements.
        liste is a list of bar elements and the dofs concerned.
          liste = [node1, node2, dofu=1, dofv=1(, dofw=0)]
          liste=np.array([[0, 1, 1(, 1)],
                          [1, 2, 0(, 1)]])"""
        nzv = np.sum(liste[:, -dim:], dtype=int) * 4
        row = np.zeros(nzv)
        col = np.zeros(nzv)
        val = np.zeros(nzv)
        nzv = 0
        for ei in liste:
            dofn = self.conn[ei[:2]]
            xn = self.n[ei[:2]]
            if l is not None:
                d = l
            else:
                d = np.linalg.norm(np.diff(xn, axis=0))
            if ei[2] == 1:  # dof u
                row[nzv + np.arange(4)] = dofn[[0, 0, 1, 1], 0]
                col[nzv + np.arange(4)] = dofn[[0, 1, 0, 1], 0]
                val[nzv + np.arange(4)] = np.array([1, -1, -1, 1]) / d
                nzv += 4
            if ei[3] == 1:  # dof v
                row[nzv + np.arange(4)] = dofn[[0, 0, 1, 1], 1]
                col[nzv + np.arange(4)] = dofn[[0, 1, 0, 1], 1]
                val[nzv + np.arange(4)] = np.array([1, -1, -1, 1]) / d
                nzv += 4
            if dim == 3:
                if ei[4] == 1:  # dof w
                    row[nzv + np.arange(4)] = dofn[[0, 0, 1, 1], 2]
                    col[nzv + np.arange(4)] = dofn[[0, 1, 0, 1], 2]
                    val[nzv + np.arange(4)] = np.array([1, -1, -1, 1]) / d
                    nzv += 4
        return sp.sparse.csc_matrix((val, (row, col)), shape=(self.ndof, self.ndof))

    def Mass(self, rho):
        """Assembles Mass Matrix"""
        if not hasattr(self, "dphixdx"):
            m = self.Copy()
            m.GaussIntegration()
            wdetJ = sp.sparse.diags(m.wdetJ)
            M = rho * m.phix.T @ wdetJ @ m.phix \
              + rho * m.phiy.T @ wdetJ @ m.phiy
        else:
            wdetJ = sp.sparse.diags(self.wdetJ)
            M = rho * self.phix.T @ wdetJ @ self.phix \
              + rho * self.phiy.T @ wdetJ @ self.phiy
        return M

    def VTKMesh(self, filename="mesh"):
        """
        Writes a VTK mesh file for vizualisation using Paraview.
        Usage:
            m.VTKMesh('FileName')
        """
        nnode = self.n.shape[0]
        if self.n.shape[1] == 2:
            new_node = np.append(self.n, np.zeros((nnode, 1)), axis=1).ravel()
        else:
            new_node = self.n.ravel()
        new_conn = np.array([], dtype="int")
        new_offs = np.array([], dtype="int")
        new_type = np.array([], dtype="int")  # paraview type
        new_typp = np.array([], dtype="int")  # pyxel (gmsh) type
        new_num = np.array([], dtype="int")
        nelem = 0
        coffs = 0
        for je in self.e.keys():
            net = len(self.e[je])  # nb of elements of type je
            if je == 3:  # quad4
                nne = 4  # nb of node of this elem
                typel = 9  # type of elem for paraview
            elif je == 2:  # tri3
                nne = 3
                typel = 5
            elif je == 9:  # tri6
                nne = 6
                typel = 22
            elif je == 16:  # quad8
                nne = 8
                typel = 23
            elif je == 10:  # quad9
                nne = 8
                typel = 23  # qua9 also for Paraview!
            elif je == 5:  # hex8
                nne = 8
                typel = 12
            else:
                print("Oops!  That is not a valid element type...")
            new_type = np.append(new_type, typel * np.ones(net, dtype=int))
            new_typp = np.append(new_typp, je * np.ones(net, dtype=int))
            new_conn = np.append(new_conn, self.e[je][:, :nne])
            new_offs = np.append(new_offs, np.arange(1, net + 1) * nne + coffs)
            new_num = np.append(new_num, np.arange(net))
            coffs += nne * net
            nelem += net
        vtkfile = vtk.VTUWriter(nnode, nelem, new_node, new_conn, new_offs, new_type)
        vtkfile.addCellData("num", 1, new_num)
        vtkfile.addCellData("type", 1, new_typp)
        # Write the VTU file in the VTK dir
        dir0, filename = os.path.split(filename)
        if not os.path.isdir(os.path.join("vtk", dir0)):
            os.makedirs(os.path.join("vtk", dir0))
        vtkfile.write(os.path.join("vtk", dir0, filename))

    def VTKSolSeries(self, filename, UU):
        """
        Writes a set of VTK files for vizualisation using Paraview.
        Writes also the PVD file that collects the VTK files.
        Usage:
            m.VTKSolSeries('FileName', UU)
        UU is a (ndof x nstep) Numpy array containing the displacement dofs
         where ndof is the numer of dofs
         ans   nstep the number of time steps
        """
        for ig in range(UU.shape[1]):
            fname = filename + "_0_" + str(ig)
            self.VTKSol(fname, UU[:, ig])
        self.PVDFile(filename, "vtu", 1, UU.shape[1])

    def PVDFile(self, fileName, ext, npart, nstep):
        dir0, fileName = os.path.split(fileName)
        if not os.path.isdir(os.path.join("vtk", dir0)):
            os.makedirs(os.path.join("vtk", dir0))
        vtk.PVDFile(os.path.join("vtk", dir0, fileName), ext, npart, nstep)

    def VTKSol(self, filename, U, E=[], S=[], T=[]):
        """
        Writes a VTK Result file for vizualisation using Paraview.
        Usage:
            m.VTKSol('FileName', U)
        U is a (ndof x 1) Numpy array containing the displacement dofs
         (ndof is the numer of dofs)
        """
        nnode = self.n.shape[0]
        new_node = np.append(self.n, np.zeros((nnode, 1)), axis=1).ravel()
        # unused nodes displacement
        conn2 = self.conn.copy()
        (unused,) = np.where(conn2[:, 0] == -1)
        conn2[unused, 0] = 0
        new_u = np.append(U[self.conn], np.zeros((nnode, 1)), axis=1).ravel()
        new_conn = np.array([], dtype="int")
        new_offs = np.array([], dtype="int")
        new_type = np.array([], dtype="int")
        new_num = np.array([], dtype="int")
        nelem = 0
        coffs = 0
        for je in self.e.keys():
            net = len(self.e[je])  # nb of elements of type je
            if je == 3:  # quad4
                nne = 4  # nb of node of this elem
                typel = 9  # type of elem for paraview
            elif je == 2:  # tri3
                nne = 3
                typel = 5
            elif je == 9:  # tri6
                nne = 6
                typel = 22
            elif je == 16:  # quad8
                nne = 8
                typel = 23
            elif je == 10:  # quad9
                nne = 8
                typel = 23  # quad9 also for paraview!
            elif je == 5:  # hex8
                nne = 8
                typel = 12
            else:
                print("Oops!  That is not a valid element type...")
            new_type = np.append(new_type, typel * np.ones(net, dtype=int))
            new_conn = np.append(new_conn, self.e[je][:, :nne])
            new_offs = np.append(new_offs, np.arange(1, net + 1) * nne + coffs)
            new_num = np.append(new_num, np.arange(net))
            coffs += nne * net
            nelem += net
        vtkfile = vtk.VTUWriter(nnode, nelem, new_node, new_conn, new_offs, new_type)
        vtkfile.addPointData("displ", 3, new_u)
        vtkfile.addCellData("num", 1, new_num)
        if len(T) > 0:
            vtkfile.addPointData("temp", 1, T[self.conn[:, 0]])
        # Strain
        if len(E) == 0:
            Ex, Ey, Exy = self.StrainAtNodes(U)
            E = np.c_[Ex, Ey, Exy]
            C = (Ex + Ey) / 2
            R = np.sqrt((Ex - C) ** 2 + Exy ** 2)
            EP = np.sort(np.c_[C + R, C - R], axis=1)
        new_e = np.c_[
            E[self.conn[:, 0], 0], E[self.conn[:, 0], 1], E[self.conn[:, 0], 2]
        ].ravel()
        vtkfile.addPointData("strain", 3, new_e)
        new_ep = np.c_[EP[self.conn[:, 0], 0], EP[self.conn[:, 0], 1]].ravel()
        vtkfile.addPointData("pcpal_strain", 2, new_ep)
        # Stress
        if len(S) > 0:
            new_s = np.c_[
                S[self.conn[:, 0], 0], S[self.conn[:, 0], 1], S[self.conn[:, 0], 2]
            ].ravel()
            vtkfile.addPointData("stress", 3, new_s)
        # Write the VTU file in the VTK dir
        dir0, filename = os.path.split(filename)
        if not os.path.isdir(os.path.join("vtk", dir0)):
            os.makedirs(os.path.join("vtk", dir0))
        vtkfile.write(os.path.join("vtk", dir0, filename))

    def StrainAtGP(self, U):
        if not hasattr(self, "dphixdx"):
            m = self.Copy()
            m.GaussIntegration()
            epsx = m.dphixdx @ U
            epsy = m.dphiydy @ U
            epsxy = 0.5 * m.dphixdy @ U + 0.5 * m.dphiydx @ U
        else:
            epsx = self.dphixdx @ U
            epsy = self.dphiydy @ U
            epsxy = 0.5 * self.dphixdy @ U + 0.5 * self.dphiydx @ U
        return epsx, epsy, epsxy

    def StrainAtNodes(self, U):
        nnodes = self.ndof // 2
        m = self.Copy()
        m.GaussIntegration()
        exxgp = m.dphixdx @ U
        eyygp = m.dphiydy @ U
        exygp = 0.5 * m.dphixdy @ U + 0.5 * m.dphiydx @ U
        EpsXX = np.zeros(nnodes)
        EpsYY = np.zeros(nnodes)
        EpsXY = np.zeros(nnodes)
        for jn in range(len(self.n)):
            if self.conn[jn, 0] >= 0:
                sig = 0  # max over all element types in the neighborhood
                for je in self.e.keys():
                    eljn, _ = np.where(self.e[je] == jn)
                    if len(eljn) != 0:
                        xm = np.mean(self.n[self.e[je][eljn, :], 0], axis=1)
                        ym = np.mean(self.n[self.e[je][eljn, :], 1], axis=1)
                        sig = max(sig, np.max(np.sqrt((xm - self.n[jn, 0]) ** 2
                                                      + (ym - self.n[jn, 1]) ** 2)) / 3)
                D = np.sqrt((m.pgx - self.n[jn, 0]) ** 2 + (m.pgy - self.n[jn, 1]) ** 2)
                gauss = np.exp(-(D ** 2) / (2 * sig ** 2))
                if np.sum(gauss) < 1e-15:
                    print(jn)
                gauss /= np.sum(gauss)
                EpsXX[self.conn[jn, 0]] = gauss @ exxgp
                EpsYY[self.conn[jn, 0]] = gauss @ eyygp
                EpsXY[self.conn[jn, 0]] = gauss @ exygp
        return EpsXX, EpsYY, EpsXY

    def StrainAtNodesOld(self, UU):
        # LS projection... not working so good!
        m = self.Copy()
        m.GaussIntegration()
        wdetJ = sp.sparse.diags(m.wdetJ)
        phi = m.phix[:, : m.ndof // 2]
        if not hasattr(self, "Bx"):
            self.Bx = splalg.splu(phi.T @ wdetJ @ phi)
        epsx = self.Bx.solve(phi.T @ wdetJ @ m.dphixdx @ UU)
        epsy = self.Bx.solve(phi.T @ wdetJ @ m.dphiydy @ UU)
        epsxy = self.Bx.solve(phi.T @ wdetJ @ (m.dphixdy @ UU + m.dphiydx @ UU)) * 0.5
        return epsx, epsy, epsxy

    def Elem2Node(self, edata):
        # LS projection... not working so good!
        wdetJ = sp.sparse.diags(self.wdetJ)
        phi = self.phix[:, : self.ndof // 2]
        M = splalg.splu(phi.T.dot(wdetJ.dot(phi)).T)
        ndata = M.solve(phi.T.dot(wdetJ.dot(edata)))
        return ndata

    def VTKIntegrationPoints(self, cam, f, g, U, filename="IntPts", iscale=2):
        """
        Writes a VTK Result file for vizualisation using Paraview.
        It builds a points cloud corresponding to the integration points
        where it is possible to visualize the displacement, strain,
        but also the graylevel values and the residual map.
        Usage:
            m.VTKIntegrationPoints(cam, f, g, U)
        - U is a (ndof x 1) Numpy array containing the displacement dofs
           (ndof is the numer of dofs)
        - f, g the reference and deformed state images respectively
        - cam a pyxel camera object
        - filename (OPTIONAL)
        - iscale (OPTIONAL) to generate less points (faster). Default iscale=0
            can be iscale=1 or iscale=2
        """
        cam2 = cam.SubSampleCopy(iscale)
        m2 = self.Copy()
        m2.DICIntegrationWithGrad(cam2)
        nnode = m2.pgx.shape[0]
        nelem = nnode
        new_node = np.array([m2.pgx, m2.pgy, 0 * m2.pgx]).T.ravel()
        new_conn = np.arange(nelem)
        new_offs = np.arange(nelem) + 1
        new_type = 2 * np.ones(nelem).astype("int")
        vtkfile = vtk.VTUWriter(nnode, nelem, new_node, new_conn, new_offs, new_type)
        """ Reference image """
        u, v = cam.P(m2.pgx, m2.pgy)
        if hasattr(f, "tck") == 0:
            f.BuildInterp()
        imref = f.Interp(u, v)
        vtkfile.addCellData("f", 1, imref)
        """ Deformed image """
        if hasattr(g, "tck") == 0:
            g.BuildInterp()
        imdef = g.Interp(u, v)
        vtkfile.addCellData("g", 1, imdef)
        """ Residual Map """
        pgu = m2.phix.dot(U)
        pgv = m2.phiy.dot(U)
        pgxu = m2.pgx + pgu
        pgyv = m2.pgy + pgv
        u, v = cam.P(pgxu, pgyv)
        imdefu = g.Interp(u, v)
        vtkfile.addCellData("res", 1, imdefu - imref)
        """ Advected Deformed image """
        imdef = g.Interp(u, v)
        vtkfile.addCellData("gu", 1, imdefu)
        """ Displacement field """
        new_u = np.array([pgu, pgv, 0 * pgu]).T.ravel()
        vtkfile.addPointData("disp", 3, new_u)
        """ Strain field """
        epsxx, epsyy, epsxy = m2.StrainAtGP(U)
        new_eps = np.array([epsxx, epsyy, epsxy]).T.ravel()
        vtkfile.addCellData("epsilon", 3, new_eps)

        # Write the VTU file in the VTK dir
        dir0, filename = os.path.split(filename)
        if not os.path.isdir(os.path.join("vtk", dir0)):
            os.makedirs(os.path.join("vtk", dir0))
        vtkfile.write(os.path.join("vtk", dir0, filename))

    def VTKNodes(self, cam, f, g, U, filename="IntPts"):
        """The same as VTKIntegrationPoints
        but on nodes instead of quadrature points"""
        nnode = self.n.shape[0]
        nelem = nnode
        new_node = np.array([self.n[:, 0], self.n[:, 1], 0 * self.n[:, 0]]).T.ravel()
        new_conn = np.arange(nelem)
        new_offs = np.arange(nelem) + 1
        new_type = 2 * np.ones(nelem).astype("int")
        vtkfile = vtk.VTUWriter(nnode, nelem, new_node, new_conn, new_offs, new_type)
        """ Reference image """
        u, v = cam.P(self.n[:, 0], self.n[:, 1])
        if hasattr(f, "tck") == 0:
            f.BuildInterp()
        imref = f.Interp(u, v)
        vtkfile.addCellData("f", 1, imref)
        """ Deformed image """
        if hasattr(g, "tck") == 0:
            g.BuildInterp()
        imdef = g.Interp(u, v)
        vtkfile.addCellData("g", 1, imdef)
        """ Residual Map """
        pgu = U[self.conn[:, 0]]
        pgv = U[self.conn[:, 1]]
        pgxu = self.n[:, 0] + pgu
        pgyv = self.n[:, 1] + pgv
        u, v = cam.P(pgxu, pgyv)
        imdefu = g.Interp(u, v)
        vtkfile.addCellData("gu", 1, (imdefu - imref) / f.Dynamic() * 100)
        """ Displacement field """
        new_u = np.array([pgu, pgv, 0 * pgu]).T.ravel()
        vtkfile.addPointData("disp", 3, new_u)

        # Write the VTU file in the VTK dir
        # Write the VTU file in the VTK dir
        dir0, filename = os.path.split(filename)
        if not os.path.isdir(os.path.join("vtk", dir0)):
            os.makedirs(os.path.join("vtk", dir0))
        vtkfile.write(os.path.join("vtk", dir0, filename))
        
    def Morphing(self, U):
        """
        Morphs a mesh with a displacement field defined by a DOF vector U.
        Warning, the nodes coordinates are modified.
        This step usually require to rebuild the Integration operators
        """
        self.n += U[self.conn]
        self.pgx += self.phix @ U
        self.pgy += self.phiy @ U

    def Plot(self, U=None, coef=1, n=None, **kwargs):
        """
        Plots the (possibly warped) mesh using Matplotlib Library.
        Usage:
            m.Plot()      > plots the mesh
            m.Plot(U)     > plots the mesh warped by the displacement U
            m.Plot(U, 30) > ... with a displacement amplification factor = 30

            Supports other Matplotlib arguments:
            m.Plot(U, edgecolor='r', facecolor='b', alpha=0.2)
        """
        edgecolor = kwargs.pop("edgecolor", "k")
        facecolor = kwargs.pop("facecolor", "none")
        alpha = kwargs.pop("alpha", 0.8)
        # plt.figure()
        ax = plt.gca()
        """ Plot deformed or undeformes Mesh """
        if n is None:
            n = self.n.copy()
        if U is not None:
            n += coef * U[self.conn]
        # plt.plot(n[:,0],n[:,1],'.',color=edgecolor,alpha=0.5)
        qua = np.zeros((0, 4), dtype="int64")
        tri = np.zeros((0, 3), dtype="int64")
        bar = np.zeros((0, 2), dtype="int64")
        plotnodes = False
        for ie in self.e.keys():
            if ie == 3 or ie == 16 or ie == 10:  # quadrangles
                qua = np.vstack((qua, self.e[ie][:, :4]))
                if ie == 16 or ie == 10:
                    plotnodes = True
            elif ie == 2 or ie == 9:  # triangles
                tri = np.vstack((tri, self.e[ie][:, :3]))
                if ie == 9:
                    plotnodes = True
            elif ie == 1:  # bars
                bar = np.vstack((bar, self.e[ie]))
        if len(qua) > 0:
            pc = cols.PolyCollection(
                n[qua], facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, **kwargs
            )
            ax.add_collection(pc)
        if len(tri) > 0:
            pc = cols.PolyCollection(
                n[tri], facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, **kwargs
            )
            ax.add_collection(pc)
        if len(bar) > 0:
            pc = cols.PolyCollection(
                n[bar], facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, **kwargs
            )
            ax.add_collection(pc)
        ax.autoscale()
        if plotnodes:
            plt.plot(
                n[:, 0],
                n[:, 1],
                linestyle="None",
                marker="o",
                color=edgecolor,
                alpha=alpha,
            )
        plt.axis("equal")
        plt.show()

    def PlotResidualMap(self, res, cam, npts=1e4):
        """
        Plots the residual map using Matplotlib Library.
        Usage:
            m.PlotResidualMap(res, cam)
            where res is a numpy.array containing the residual at integration points

            m.PlotResidualMap(res, cam, npts=1e3)
            to limit the number of integration points visualization (faster)

            m.PlotResidualMap(res, cam, npts='all')
            to visualize all integration points (less fast)
        """
        if npts == "all":
            rep = np.arange(self.npg)
        else:
            rep = np.unique((np.random.sample(int(npts)) * (len(self.pgx) - 1)).astype("int"))
        u, v = cam.P(self.pgx[rep], self.pgy[rep])
        stdr = np.std(res)
        plt.figure()
        plt.scatter(self.pgx[rep], self.pgy[rep], c=res[rep], cmap="RdBu", s=1)
        plt.axis("equal")
        plt.clim(-3 * stdr, 3 * stdr)
        plt.colorbar()
        plt.show()

    def PlotResidualMapRecompute(self, f, g, cam, U, npts=1e4):
        """
        Plots the residual map using Matplotlib Library.
        Usage:
            m.PlotResidualMap(f, g, cam, U)
            where f and g the pyxel.images, cam the pyxel.camera, U dof vector

            m.PlotResidualMap(f, g, cam, U, npts=1e3)
            to limit the number of integration points visualization (faster)

            m.PlotResidualMap(f, g, cam, U, npts='all')
            to visualize all integration points (less fast)
        """
        if npts == "all":
            rep = np.arange(self.npg)
        else:
            rep = np.unique((np.random.sample(int(npts)) * (len(self.pgx) - 1)).astype("int"))
        u, v = cam.P(self.pgx[rep], self.pgy[rep])
        if hasattr(f, "tck") == 0:
            f.BuildInterp()
        imref = f.Interp(u, v)
        pgxu = self.pgx + self.phix.dot(U)
        pgyv = self.pgy + self.phiy.dot(U)
        u, v = cam.P(pgxu[rep], pgyv[rep])
        if hasattr(g, "tck") == 0:
            g.BuildInterp()
        res = g.Interp(u, v)
        res -= np.mean(res)
        res = imref - np.mean(imref) - np.std(imref) / np.std(res) * res
        stdr = np.std(res)
        plt.figure()
        plt.scatter(self.pgx[rep], self.pgy[rep], c=res, cmap="RdBu", s=1)
        plt.axis("equal")
        plt.clim(-3 * stdr, 3 * stdr)
        plt.colorbar()
        plt.show()

    def PlotContourDispl(self, U=None, n=None, s=1.0, **kwargs):
        """
        Plots the 2 components of a displacement field using Matplotlib Library.
        Usage:
            m.PlotContourDispl(U)
            where U is the dof vector

            m.PlotContourDispl(U, s=30)   >  deformation magnification = 30
        """
        if n is None:
            n = self.n.copy()
        n += U[self.conn] * s  # s: amplification scale factor
        """ Plot mesh and field contour """
        triangles = np.zeros((0, 3), dtype=int)
        for ie in self.e.keys():
            if ie == 3 or ie == 16 or ie == 10:  # quadrangles
                triangles = np.vstack(
                    (triangles, self.e[ie][:, [0, 1, 3]], self.e[ie][:, [1, 2, 3]])
                )
            elif ie == 2 or ie == 9:  # triangles
                triangles = np.vstack((triangles, self.e[ie]))
        plt.figure()
        plt.tricontourf(n[:, 0], n[:, 1], triangles, U[self.conn[:, 0]], 20)
        self.Plot(n=n, alpha=0.1)
        plt.axis("off")
        plt.title("Ux")
        plt.colorbar()
        plt.figure()
        plt.tricontourf(n[:, 0], n[:, 1], triangles, U[self.conn[:, 1]], 20)
        self.Plot(n=n, alpha=0.1)
        plt.axis("equal")
        plt.title("Uy")
        plt.axis("off")
        plt.colorbar()
        plt.show()

    def PlotContourStrain(self, U, n=None, s=1.0, **kwargs):
        """
        Plots the 3 components of a strain field using Matplotlib Library.
        Usage:
            m.PlotContourStrain(U)
            where U is the dof vector

            m.PlotContourDispl(U, s=30)   >  deformation magnification = 30
        """
        if n is None:
            n = self.n.copy()
        n += U[self.conn] * s  # s: amplification scale factor
        triangles = np.zeros((0, 3), dtype=int)
        for ie in self.e.keys():
            if ie == 3 or ie == 16 or ie == 10:  # quadrangles
                triangles = np.vstack(
                    (triangles, self.e[ie][:, [0, 1, 3]], self.e[ie][:, [1, 2, 3]])
                )
            elif ie == 2 or ie == 9:  # triangles
                triangles = np.vstack((triangles, self.e[ie]))
        EX, EY, EXY = self.StrainAtNodes(U)
        """ Plot mesh and field contour """
        plt.figure()
        plt.tricontourf(n[:, 0], n[:, 1], triangles, EX[self.conn[:, 0]], 20)
        self.Plot(n=n, alpha=0.1)
        plt.axis("off")
        plt.axis("equal")
        plt.title("EPS_X")
        plt.colorbar()
        plt.figure()
        plt.tricontourf(n[:, 0], n[:, 1], triangles, EY[self.conn[:, 0]], 20)
        self.Plot(n=n, alpha=0.1)
        plt.axis("equal")
        plt.title("EPS_Y")
        plt.axis("off")
        plt.colorbar()
        plt.figure()
        plt.tricontourf(n[:, 0], n[:, 1], triangles, EXY[self.conn[:, 0]], 20)
        self.Plot(n=n, alpha=0.1)
        plt.axis("equal")
        plt.title("EPS_XY")
        plt.axis("off")
        plt.colorbar()
        plt.show()

    def PlotNodeLabels(self, **kwargs):
        """
        Plots the mesh with the node labels (may be slow for large mesh size).
        """
        self.Plot(**kwargs)
        color = kwargs.get("edgecolor", "k")
        plt.plot(self.n[:, 0], self.n[:, 1], ".", color=color)
        for i in range(len(self.n[:, 1])):
            plt.text(self.n[i, 0], self.n[i, 1], str(i), color=color)

    def PlotElemLabels(self, **kwargs):
        """
        Plots the mesh with the elems labels (may be slow for large mesh size).
        """
        self.Plot(**kwargs)
        color = kwargs.get("edgecolor", "k")
        for je in self.e.keys():
            for ie in range(len(self.e[je])):
                ce = np.mean(self.n[self.e[je][ie, :], :], axis=0)
                plt.text(
                    ce[0],
                    ce[1],
                    str(ie),
                    horizontalalignment="center",
                    verticalalignment="center",
                    color=color,
                )

    def VTKIntegrationPointsTh(self, cam, f, U, filename="IntPtsT"):
        nnode = self.pgx.shape[0]
        nelem = nnode
        new_node = np.array([self.pgx, self.pgy, 0 * self.pgx]).T.ravel()
        new_conn = np.arange(nelem)
        new_offs = np.arange(nelem) + 1
        new_type = 2 * np.ones(nelem).astype("int")
        vtkfile = vtk.VTUWriter(nnode, nelem, new_node, new_conn, new_offs, new_type)
        """ Reference image """
        u, v = cam.P(self.pgx, self.pgy)
        if hasattr(f, "tck") == 0:
            f.BuildInterp()
        imref = f.Interp(u, v)
        vtkfile.addCellData("Th_init", 1, imref)
        """ ReMaped thermal field """
        pgu = self.phix.dot(U)
        pgv = self.phiy.dot(U)
        pgxu = self.pgx + pgu
        pgyv = self.pgy + pgv
        u, v = cam.P(pgxu, pgyv)
        imdefu = f.Interp(u, v)
        vtkfile.addCellData("Th_advected", 1, imdefu)
        """ Displacement field """
        new_u = np.array([pgu, pgv, 0 * pgu]).T.ravel()
        vtkfile.addPointData("disp", 3, new_u)
        """ Strain field """
        epsxx, epsyy, epsxy = self.StrainAtGP(U)
        new_eps = np.array([epsxx, epsyy, epsxy]).T.ravel()
        vtkfile.addCellData("epsilon", 3, new_eps)

        # Write the VTU file in the VTK dir
        dir0, filename = os.path.split(filename)
        if not os.path.isdir(os.path.join("vtk", dir0)):
            os.makedirs(os.path.join("vtk", dir0))
        vtkfile.write(os.path.join("vtk", dir0, filename))

    def FindDOFinBox(self, box):
        """
        Returns the dof of all the nodes lying within a rectangle defined
        by the coordinates of two diagonal points (in the mesh coordinate sys).
        Used to apply BC for instance.
        box = np.array([[xmin, ymin],
                        [xmax, ymax]])    in mesh unit
        """
        dofs = np.zeros((0, 2), dtype="int")
        for jn in range(len(self.n)):
            if isInBox(box, self.n[jn, 0], self.n[jn, 1]):
                dofs = np.vstack((dofs, self.conn[jn]))
        return dofs

    def KeepEdgeElems(self):
        """
        Removes every but edge elements.
        """
        newe = dict()
        if 1 in self.e:
            newe[1] = self.e[1]
        self.e = newe

    def KeepSurfElems(self):
        """
        Removes every but surface (or 2D) elements.
        """
        newe = dict()
        for je in self.e.keys():
            if je in [2, 3, 9, 16, 10]:
                newe[je] = self.e[je]
        self.e = newe

    def KeepVolElems(self):
        """
        Removes every but volume (or 3D) elements.
        """
        newe = dict()
        for je in self.e.keys():
            if je in [4, 5, 10, 11, 12]:
                newe[je] = self.e[je]
        self.e = newe

    def RemoveElemsOutsideRoi(self, cam, roi):
        """
        Removes all the elements whose center lie in the Region of Interest of
        an image f.
        Usage :
            m.RemoveElemsOutsideRoi(cam, roi)

        where  roi = f.SelectROI()
        """
        for je in self.e.keys():
            xc = np.mean(self.n[self.e[je], 0], axis=1)
            yc = np.mean(self.n[self.e[je], 1], axis=1)
            u, v = cam.P(xc, yc)
            inside = isInBox(roi, v, u)
            self.e[je] = self.e[je][inside, :]

    def BuildBoundaryMesh(self):
        """
        Builds edge elements corresponding to the edges of Mesh m.
        """
        edges = {}
        for je in self.e.keys():
            n1 = self.e[je].ravel()
            n2 = np.c_[self.e[je][:, 1:], self.e[je][:, 0]].ravel()
            a = np.sort(np.c_[n1, n2], axis=1)
            for i in range(len(a)):
                tedge = tuple(a[i, :])
                if tedge in edges.keys():
                    edges[tedge] += 1
                else:
                    edges[tedge] = 1
        (rep,) = np.where(np.array(list(edges.values())) == 1)
        edges = np.array(list(edges.keys()))[rep, :]
        elems = {1: edges}
        edgem = Mesh(elems, self.n)
        return edgem

    def SelectPoints(self, n=-1, title=None):
        """
        Selection of points coordinates by hand in a mesh.
        """
        plt.figure()
        self.Plot()
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        if title is None:
            if n < 0:
                plt.title("Select several points... and press enter")
            else:
                plt.title("Select " + str(n) + " points... and press enter")
        else:
            plt.title(title)
        pts1 = np.array(plt.ginput(n, timeout=0))
        plt.close()
        return pts1

    def SelectNodes(self, n=-1):
        """
        Selection of nodes by hand in a mesh.
        """
        plt.figure()
        self.Plot()
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.title("Select " + str(n) + " points... and press enter")
        pts1 = np.array(plt.ginput(n, timeout=0))
        plt.close()
        dx = np.kron(np.ones(pts1.shape[0]), self.n[:, [0]]) - np.kron(
            np.ones((self.n.shape[0], 1)), pts1[:, 0]
        )
        dy = np.kron(np.ones(pts1.shape[0]), self.n[:, [1]]) - np.kron(
            np.ones((self.n.shape[0], 1)), pts1[:, 1]
        )
        nset = np.argmin(np.sqrt(dx ** 2 + dy ** 2), axis=0)
        self.Plot()
        plt.plot(self.n[nset, 0], self.n[nset, 1], "ro")
        return nset

    def SelectNodesBox(self):
        """
        Selection of all the nodes of a mesh lying in a box defined by two
        points clics.
        """
        plt.figure()
        self.Plot()
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.title("Select 2 points... and press enter")
        pts1 = np.array(plt.ginput(2, timeout=0))
        plt.close()
        inside = (
            (self.n[:, 0] > pts1[0, 0])
            * (self.n[:, 0] < pts1[1, 0])
            * (self.n[:, 1] > pts1[0, 1])
            * (self.n[:, 1] < pts1[1, 1])
        )
        (nset,) = np.where(inside)
        self.Plot()
        plt.plot(self.n[nset, 0], self.n[nset, 1], "ro")
        return nset

    def SelectLine(self):
        """
        Selection of the nodes along a line defined by 2 nodes.
        """
        plt.figure()
        self.Plot()
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.title("Select 2 points of a line... and press enter")
        pts1 = np.array(plt.ginput(2, timeout=0))
        plt.close()
        n1 = np.argmin(np.linalg.norm(self.n - pts1[0, :], axis=1))
        n2 = np.argmin(np.linalg.norm(self.n - pts1[1, :], axis=1))
        v = np.diff(self.n[[n1, n2]], axis=0)[0]
        nv = np.linalg.norm(v)
        v = v / nv
        n = np.array([v[1], -v[0]])
        c = n.dot(self.n[n1, :])
        (rep,) = np.where(abs(self.n.dot(n) - c) < 1e-8)
        c1 = v.dot(self.n[n1, :])
        c2 = v.dot(self.n[n2, :])
        nrep = self.n[rep, :]
        (rep2,) = np.where(((nrep.dot(v) - c1) * (nrep.dot(v) - c2)) < nv * 1e-2)
        nset = rep[rep2]
        self.Plot()
        plt.plot(self.n[nset, 0], self.n[nset, 1], "ro")
        return nset

    def SelectCircle(self):
        """
        Selection of the nodes around a circle defined by 3 nodes.
        """
        plt.figure()
        self.Plot()
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.title("Select 3 points on a circle... and press enter")
        pts1 = np.array(plt.ginput(3, timeout=0))
        plt.close()
        n1 = np.argmin(np.linalg.norm(self.n - pts1[0, :], axis=1))
        n2 = np.argmin(np.linalg.norm(self.n - pts1[1, :], axis=1))
        n3 = np.argmin(np.linalg.norm(self.n - pts1[2, :], axis=1))
        pts1 = self.n[[n1, n2, n3], :]
        meanu = np.mean(pts1, axis=0)
        pts = pts1 - meanu
        pts2 = pts ** 2
        A = pts.T.dot(pts)
        b = 0.5 * np.sum(pts.T.dot(pts2), axis=1)
        cpos = np.linalg.solve(A, b)
        R = np.sqrt(np.linalg.norm(cpos) ** 2 + np.sum(pts2) / pts.shape[0])
        cpos += meanu
        (nset,) = np.where(
            np.sqrt(
                abs(
                    (self.n[:, 0] - cpos[0]) ** 2
                    + (self.n[:, 1] - cpos[1]) ** 2
                    - R ** 2
                )
            )
            < (R * 1e-2)
        )
        # self.Plot()
        # plt.plot(self.n[nset,0],self.n[nset,1],'ro')
        return nset  # ,R
    
#%%
class Image:
    def Load(self):
        """Load image data"""
        if self.fname.split(".")[-1] == "npy":
            self.pix = np.load(self.fname)
        else:
            self.pix = np.asarray(image.open(self.fname)).astype(float)
            # self.pix = image.imread(self.fname).astype(float)
        if len(self.pix.shape) == 3:
            self.ToGray()
        return self

    def Load_cv2(self):
        """Load image data using OpenCV"""
        import cv2 as cv
        self.pix = cv.imread(self.fname).astype(float)
        if len(self.pix.shape) == 3:
            self.ToGray()
        return self

    def Copy(self):
        """Image Copy"""
        newimg = Image("Copy")
        newimg.pix = self.pix.copy()
        return newimg

    def Save(self, fname):
        """Image Save"""
        PILimg = image.fromarray(self.pix.astype("uint8"))
        PILimg.save(fname)
        # image.imsave(fname,self.pix.astype('uint8'),vmin=0,vmax=255,format='tif')

    def __init__(self, fname):
        """Contructor"""
        self.fname = fname

    def BuildInterp(self):
        """build bivariate Spline interp"""
        x = np.arange(0, self.pix.shape[0])
        y = np.arange(0, self.pix.shape[1])
        self.tck = spi.RectBivariateSpline(x, y, self.pix)

    def Interp(self, x, y):
        """evaluate interpolator at non-integer pixel position x, y"""
        return self.tck.ev(x, y)

    def InterpGrad(self, x, y):
        """evaluate gradient of the interpolator at non-integer pixel position x, y"""
        return self.tck.ev(x, y, 1, 0), self.tck.ev(x, y, 0, 1)

    def InterpHess(self, x, y):
        """evaluate Hessian of the interpolator at non-integer pixel position x, y"""
        return self.tck.ev(x, y, 2, 0), self.tck.ev(x, y, 0, 2), self.tck.ev(x, y, 1, 1)

    def Plot(self):
        """Plot Image"""
        plt.imshow(self.pix, cmap="gray", interpolation="none", origin="upper")
        # plt.axis('off')
        # plt.colorbar()

    def Dynamic(self):
        """Compute image dynamic"""
        g = self.pix.ravel()
        return max(g) - min(g)

    def GaussianFilter(self, sigma=0.7):
        """Performs a Gaussian filter on image data. 

        Parameters
        ----------
        sigma : float
            variance of the Gauss filter."""
        from scipy.ndimage import gaussian_filter

        self.pix = gaussian_filter(self.pix, sigma)

    def PlotHistogram(self):
        """Plot Histogram of graylevels"""
        plt.hist(self.pix.ravel(), bins=125, range=(0.0, 255), fc="k", ec="k")
        plt.show()

    def SubSample(self, n):
        """Image copy with subsampling for multiscale initialization"""
        scale = 2 ** n
        sizeim1 = np.array([self.pix.shape[0] // scale, self.pix.shape[1] // scale])
        nn = scale * sizeim1
        im0 = np.mean(
            self.pix[0 : nn[0], 0 : nn[1]].T.reshape(np.prod(nn) // scale, scale),
            axis=1,
        )
        nn[0] = nn[0] // scale
        im0 = np.mean(
            im0.reshape(nn[1], nn[0]).T.reshape(np.prod(nn) // scale, scale), axis=1
        )
        nn[1] = nn[1] // scale
        self.pix = im0.reshape(nn)

    def ToGray(self, type="lum"):
        """Convert RVG to Grayscale :

        Parameters
        ----------
        type : string
            lig : lightness
            lum : luminosity (DEFAULT)
            avg : average"""
        if type == "lum":
            self.pix = (
                0.21 * self.pix[:, :, 0]
                + 0.72 * self.pix[:, :, 1]
                + 0.07 * self.pix[:, :, 2]
            )
        elif type == "lig":
            self.pix = 0.5 * np.maximum(
                np.maximum(self.pix[:, :, 0], self.pix[:, :, 1]), self.pix[:, :, 2]
            ) + 0.5 * np.minimum(
                np.minimum(self.pix[:, :, 0], self.pix[:, :, 1]), self.pix[:, :, 2]
            )
        else:
            self.pix = np.mean(self.pix, axis=2)

    def SelectPoints(self, n=-1, title=None):
        """Select a point in the image. 
        
        Parameters
        ----------
        n : int
            number of expected points
        title : string (OPTIONNAL)
            modify the title of the figure when clic is required.
            
        """
        plt.figure()
        self.Plot()
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        if title is None:
            if n < 0:
                plt.title("Select some points... and press enter")
            else:
                plt.title("Select " + str(n) + " points... and press enter")
        else:
            plt.title(title)
        pts1 = np.array(plt.ginput(n, timeout=0))
        plt.close()
        return pts1

    def SelectROI(self, m=None, cam=None):
        """Select a Region of Interest within the image. 
        
        Parameters
        ----------
        m : pyxel.Mesh object (OPTIONNAL)
        cam : pyxel.Camera object (OPTIONNAL)
            To superimpose the mesh in the image
        
        The result of the ROI is displayed in the python command. 
        It can be copy-pasted.
        """
        from matplotlib.widgets import RectangleSelector

        fig, ax = plt.subplots()
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        if m is not None:
            PlotMeshImage(self, m, cam, newfig=False)
        else:
            self.Plot()

        def line_select_callback(eclick, erelease):
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            print(
                "roi = np.array([[%4d, %4d], [%4d, %4d]])"
                % (int(x1), int(y1), int(x2), int(y2))
            )

        rs = RectangleSelector(
            ax,
            line_select_callback,
            drawtype="box",
            useblit=False,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=True,
        )
        plt.show()
        return rs

#%%
class Camera:
    def __init__(self, p):
        self.set_p(p)

    def set_p(self, p):
        self.f = p[0]
        self.tx = p[1]
        self.ty = p[2]
        self.rz = p[3]

    def get_p(self):
        """Returns the vector of parameters."""
        return np.array([self.f, self.tx, self.ty, self.rz])

    def SubSampleCopy(self, nscale):
        """Camera model copy with subsampling for multiscale initialization"""
        p = self.get_p()
        p[0] /= 2 ** nscale
        return Camera(p)

    def P(self, X, Y):
        """Camera model projection. Maps a point of the mesh to a point
        in the image plane"""
        u = -self.f * (-np.sin(self.rz) * X + np.cos(self.rz) * Y + self.ty)
        v = self.f * (np.cos(self.rz) * X + np.sin(self.rz) * Y + self.tx)
        return u, v

    def Pinv(self, u, v):
        """Inverse of the Camera model. Maps a point in the image to a point
        in the mesh coordinate sys"""
        X = -np.sin(self.rz) * (-u / self.f - self.ty) + np.cos(self.rz) * (
            v / self.f - self.tx
        )
        Y = np.cos(self.rz) * (-u / self.f - self.ty) + np.sin(self.rz) * (
            v / self.f - self.tx
        )
        return X, Y

    def dPdX(self, X, Y):
        """Camera model derivative wrt physical position """
        dudx = self.f * np.sin(self.rz) * np.ones(X.shape[0])
        dudy = -self.f * np.cos(self.rz) * np.ones(X.shape[0])
        dvdx = self.f * np.cos(self.rz) * np.ones(X.shape[0])
        dvdy = self.f * np.sin(self.rz) * np.ones(X.shape[0])
        return dudx, dudy, dvdx, dvdy

    def dPdp(self, X, Y):
        """Camera model first order derivative wrt camera parameters"""
        dudf = -1 * (-np.sin(self.rz) * X + np.cos(self.rz) * Y + self.ty)
        dudtx = 0 * X
        dudty = 0 * X - self.f
        dudrz = self.f * (np.cos(self.rz) * X + np.sin(self.rz) * Y)
        dvdf = np.cos(self.rz) * X + np.sin(self.rz) * Y + self.tx
        dvdtx = 0 * X + self.f
        dvdty = 0 * X
        dvdrz = self.f * (-np.sin(self.rz) * X + np.cos(self.rz) * Y)
        return np.c_[dudf, dudtx, dudty, dudrz], np.c_[dvdf, dvdtx, dvdty, dvdrz]

    def d2Pdp2(self, X, Y):
        """Camera model second order derivative wrt camera parameters"""
        d2udf2 = 0 * X
        d2udtx2 = 0 * X
        d2udty2 = 0 * X
        d2udrz2 = self.f * (-np.sin(self.rz) * X + np.cos(self.rz) * Y)
        d2udftx = 0 * X
        d2udfty = 0 * X - 1
        d2udfrz = np.cos(self.rz) * X + np.sin(self.rz) * Y
        d2udtxty = 0 * X
        d2udtxrz = 0 * X
        d2udtyrz = 0 * X
        d2vdf2 = 0 * X
        d2vdtx2 = 0 * X
        d2vdty2 = 0 * X
        d2vdrz2 = -self.f * (np.cos(self.rz) * X + np.sin(self.rz) * Y)
        d2vdftx = 0 * X + 1
        d2vdfty = 0 * X
        d2vdfrz = -np.sin(self.rz) * X + np.cos(self.rz) * Y
        d2vdtxty = 0 * X
        d2vdtxrz = 0 * X
        d2vdtyrz = 0 * X
        d2udp2 = np.c_[d2udf2, d2udtx2, d2udty2, d2udrz2, d2udftx, 
                       d2udfty, d2udfrz, d2udtxty, d2udtxrz, d2udtyrz]
        d2vdp2 = np.c_[d2vdf2, d2vdtx2, d2vdty2, d2vdrz2, d2vdftx,
                       d2vdfty, d2vdfrz, d2vdtxty, d2vdtxrz, d2vdtyrz]
        return d2udp2, d2vdp2

    def ImageFiles(self, fname, imnums):
        self.fname = fname
        self.imnums = imnums

#%% LevelSet Calibration tools

def LSfromLine(f, pts1):
    """Compute level set of a line from a point cloud"""
    b = pts1.T.dot(np.ones_like(pts1[:, 0]))
    A = pts1.T.dot(pts1)
    res = np.linalg.solve(A, b)
    ui = np.arange(0, f.pix.shape[0])
    vi = np.arange(0, f.pix.shape[1])
    [Yi, Xi] = np.meshgrid(vi, ui)
    lvlset = (Xi * res[1] + Yi * res[0] - 1) / np.linalg.norm(res)
    lvl = Image("lvl")
    lvl.pix = abs(lvlset)
    return lvl

def LSfromPoint(f, pts1):
    """Compute level set from one single point"""
    ui = np.arange(f.pix.shape[0]) - pts1[1]
    vi = np.arange(f.pix.shape[1]) - pts1[0]
    [Yi, Xi] = np.meshgrid(vi, ui)
    lvl = Image("lvl")
    lvl.pix = np.sqrt(Xi ** 2 + Yi ** 2)
    return lvl

def LSfromCircle(f, pts1):
    """Compute level set of a circle from a point cloud"""
    meanu = np.mean(pts1, axis=0)
    pts = pts1 - meanu
    pts2 = pts ** 2
    A = pts.T.dot(pts)
    b = 0.5 * np.sum(pts.T.dot(pts2), axis=1)
    cpos = np.linalg.solve(A, b)
    R = np.sqrt(np.linalg.norm(cpos) ** 2 + np.sum(pts2) / pts.shape[0])
    cpos += meanu
    ui = np.arange(0, f.pix.shape[0])
    vi = np.arange(0, f.pix.shape[1])
    [Yi, Xi] = np.meshgrid(vi, ui)
    lvlset = abs(np.sqrt((Xi - cpos[1]) ** 2 + (Yi - cpos[0]) ** 2) - R)
    lvl = Image("lvl")
    lvl.pix = abs(lvlset)
    return lvl

class LSCalibrator:
    """Calibration of a front parallel setting 2D-DIC"""
    def __init__(self, f, m):
        self.f = f
        self.m = m
        self.ptsi = dict()
        self.ptsm = dict()
        self.feat = dict()
        self.nfeat = 0
        self.lvl = dict()
        self.cam = None

    def Init3Pts(self, ptsm=None, ptsM=None):
        """Initialization of the calibration using 3 points.

        Parameters
        ----------
        ptsm : Numpy array
            points coordinates in the images (DEFAULT = defined by clic)
        ptsM : Numpy array
            points coordinates in the mesh (DEFAULT = defined by clic)
            
        """
        if ptsm is None:
            print(" ************************************************* ")
            print(" *  SELECT 3 characteristic points in the image  * ")
            print(" ************************************************* ")
            ptsm = self.f.SelectPoints(3)[:, [1, 0]]
        if ptsM is None:
            print(" ************************************************* ")
            print(" * SELECT the 3 corresponding points on the mesh * ")
            print(" ************************************************* ")
            ptsM = self.m.SelectPoints(3)
        cm = np.mean(ptsm, axis=0)
        cM = np.mean(ptsM, axis=0)
        dm = np.linalg.norm(ptsm - cm, axis=1)
        dM = np.linalg.norm(ptsM - cM, axis=1)
        scale = np.mean(dm / dM)
        dmax = np.argmax(dm)
        vm = ptsm[dmax] - cm
        vM = ptsM[dmax] - cM
        vm /= np.linalg.norm(vm)
        vM /= np.linalg.norm(vM)
        angl = np.arccos(vM @ vm)
        p = np.array([scale, 0, 0, np.pi / 2 - angl])
        self.cam = Camera(p)
        for i in range(40):
            up, vp = self.cam.P(ptsM[:, 0], ptsM[:, 1])
            dPudp, dPvdp = self.cam.dPdp(ptsM[:, 0], ptsM[:, 1])
            A = np.vstack((dPudp, dPvdp))
            M = A.T @ A
            b = A.T @ (ptsm.T.ravel() - np.append(up, vp))
            dp = np.linalg.solve(M, b)
            p += 0.8 * dp
            self.cam.set_p(p)
            err = np.linalg.norm(dp) / np.linalg.norm(p)
            print(
                "Iter # %2d | disc=%2.2f %% | dU/U=%1.2e"
                % (i + 1, np.linalg.norm(ptsm.T.ravel() - np.append(up, vp))
                    / np.linalg.norm(ptsm.T.ravel()) * 100, err))
            if err < 1e-5:
                break

    def Plot(self):
        """Plot the level sets of each feature"""
        for i in self.feat.keys():
            plt.figure()
            self.f.Plot()
            plt.contour(self.lvl[i].pix, np.array([0.4]), colors=["y"])
            plt.figure()
            plt.contourf(self.lvl[i].pix, 16, origin="image")
            plt.colorbar()
            plt.contour(self.lvl[i].pix, np.array([0.4]), colors=["y"], origin="image")
            plt.axis("image")

    def NewCircle(self):
        print(" ******************************* ")
        print(" *        SELECT Circle        * ")
        self.ptsi[self.nfeat] = self.f.SelectPoints(
            -1, title="Select n points of a circle... and press enter"
        )  # [:,[1,0]]
        self.ptsm[self.nfeat] = self.m.SelectCircle()
        self.feat[self.nfeat] = "circle"
        self.nfeat += 1
        print(" ******************************* ")

    def NewLine(self):
        print(" ******************************* ")
        print(" *        SELECT Line          *")
        self.ptsi[self.nfeat] = self.f.SelectPoints(
            -1, title="Select n points of a straight line... and press enter"
        )  # [:,[1,0]]
        self.ptsm[self.nfeat] = self.m.SelectLine()
        self.feat[self.nfeat] = "line"
        self.nfeat += 1
        print(" ******************************* ")

    def NewPoint(self):
        print(" ******************************* ")
        print(" *        SELECT Point         * ")
        self.ptsi[self.nfeat] = self.f.SelectPoints(1)  # [:,[1,0]]
        self.ptsm[self.nfeat] = self.m.SelectNodes(1)
        self.feat[self.nfeat] = "point"
        self.nfeat += 1
        print(" ******************************* ")

    def DisableFeature(self, i):
        """Disable one of the features. Used to redefine an inappropriate mesh selection.

        Parameters
        ----------
        i : int
            the feature number
            
        """
        if i in self.feat.keys():
            del self.lvl[i]
            del self.ptsi[i]
            del self.ptsm[i]
            del self.feat[i]

    def FineTuning(self, im=None):
        """Redefine and refine the points selected in the images.

        Parameters
        ----------
        im : int (OPTIONNAL)
            the feature number is only one feature has to be redefined. Default = all
            
        """

        # Arg: f pyxel image or Array of pyxel images
        if im is None:
            rg = self.ptsi.keys()
        else:
            rg = np.array([im])
        for i in rg:  # loop on features
            for j in range(len(self.ptsi[i][:, 1])):  # loop on points
                x = int(self.ptsi[i][j, 0])
                y = int(self.ptsi[i][j, 1])
                umin = max(0, x - 50)
                vmin = max(0, y - 50)
                umax = min(self.f.pix.shape[1] - 1, x + 50)
                vmax = min(self.f.pix.shape[0] - 1, y + 50)
                fsub = self.f.pix[vmin:vmax, umin:umax]
                plt.imshow(fsub, cmap="gray", interpolation="none")
                plt.plot(x - umin, y - vmin, "y+")
                figManager = plt.get_current_fig_manager()
                figManager.window.showMaximized()
                self.ptsi[i][j, :] = np.array(plt.ginput(1))[0] + np.array([umin, vmin])
                plt.close()

    def Calibration(self):
        """Performs the calibration provided that sufficient features have been 
        selected using NewPoint(), NewLine() or NewCircle().
            
        Returns
        -------
        pyxel Camera object
            The calibrated camera model    
        """
        # Compute Levelsets
        for i in self.feat.keys():
            if "circle" in self.feat[i]:
                self.lvl[i] = LSfromCircle(self.f, self.ptsi[i])
            elif "line" in self.feat[i]:
                self.lvl[i] = LSfromLine(self.f, self.ptsi[i])
            elif "point" in self.feat[i]:
                self.lvl[i] = LSfromPoint(self.f, self.ptsi[i])
        # Calibration
        xp = dict()
        yp = dict()
        for i in self.feat.keys():
            self.lvl[i].BuildInterp()
            xp[i] = self.m.n[self.ptsm[i], 0]
            yp[i] = self.m.n[self.ptsm[i], 1]
        if self.cam is None:
            if len(self.feat) > 2:
                ptsm = np.empty((0, 2))
                ptsM = np.empty((0, 2))
                for i in self.feat.keys():
                    ptsm = np.vstack((ptsm, np.mean(self.ptsi[i], axis=0)))
                    ptsM = np.vstack((ptsM, np.mean(self.m.n[self.ptsm[i]], axis=0)))
                self.Init3Pts(ptsm, ptsM)
            else:
                self.Init3Pts()
        p = self.cam.get_p()
        C = np.diag(p)
        if p[-1] == 0:
            C[-1, -1] = 1
        for i in range(40):
            M = np.zeros((len(p), len(p)))
            b = np.zeros(len(p))
            for j in self.feat.keys():
                up, vp = self.cam.P(xp[j], yp[j])
                lp = self.lvl[j].Interp(up, vp)
                dPudp, dPvdp = self.cam.dPdp(xp[j], yp[j])
                ldxr, ldyr = self.lvl[j].InterpGrad(up, vp)
                dPdl = np.diag(ldxr) @ dPudp + np.diag(ldyr).dot(dPvdp)
                M += C.T.dot(dPdl.T.dot(dPdl.dot(C)))
                b += C.T.dot(dPdl.T.dot(lp))
            dp = C.dot(np.linalg.solve(M, -b))
            p += 0.8 * dp
            self.cam.set_p(p)
            err = np.linalg.norm(dp) / np.linalg.norm(p)
            print("Iter # %2d | disc=%2.2f %% | dU/U=%1.2e"
                % (i + 1, np.mean(lp) / max(self.f.pix.shape) * 100, err))
            if err < 1e-5:
                break
        print("cam = px.Camera(np.array([%f, %f, %f, %f]))" % (p[0], p[1], p[2], p[3]))
        return self.cam

#%%
class DICEngine:
    def __init__(self):
        self.f = []
        self.wphiJdf = []
        self.dyn = []
        self.mean0 = []
        self.std0 = []

    def ComputeLHS(self, f, m, cam):
        """Compute the FE-DIC Left hand side operator with the modified GN
    
        Parameters
        ----------
        f : pyxel.Image
            Reference State Image
        m : pyxel.Mesh
            The FE mesh
        cam : pyxel.Camera
            Calibrated Camera model.
            
        Returns
        -------
        scipy sparse
            The DIC Hessian (in the modified GN sense)
    
        """
        if hasattr(f, "tck") == 0:
            f.BuildInterp()
        pgu, pgv = cam.P(m.pgx, m.pgy)
        self.f = f.Interp(pgu, pgv)
        fdxr, fdyr = f.InterpGrad(pgu, pgv)
        Jxx, Jxy, Jyx, Jyy = cam.dPdX(m.pgx, m.pgy)
        phiJdf = (
            sp.sparse.diags(fdxr * Jxx + fdyr * Jyx) @ m.phix
            + sp.sparse.diags(fdxr * Jxy + fdyr * Jyy) @ m.phiy
        )
        self.wphiJdf = sp.sparse.diags(m.wdetJ) @ phiJdf
        self.dyn = np.max(self.f) - np.min(self.f)
        self.mean0 = np.mean(self.f)
        self.std0 = np.std(self.f)
        self.f -= self.mean0
        return phiJdf.T @ self.wphiJdf

    def ComputeLHS_EB(self, f, m, cam):
        """Compute the FE-DIC Left hand side operator with the modified GN
        and with elementary correction of brigthness and contrast
    
        Parameters
        ----------
        f : pyxel.Image
            Reference State Image
        m : pyxel.Mesh
            The FE mesh
        cam : pyxel.Camera
            Calibrated Camera model.
            
        Returns
        -------
        scipy sparse
            The DIC Hessian (in the modified GN sense)
    
        """
        if hasattr(f, "tck") == 0:
            f.BuildInterp()
        pgu, pgv = cam.P(m.pgx, m.pgy)
        self.f = f.Interp(pgu, pgv)
        fdxr, fdyr = f.InterpGrad(pgu, pgv)
        Jxx, Jxy, Jyx, Jyy = cam.dPdX(m.pgx, m.pgy)
        phiJdf = (
            sp.sparse.diags(fdxr * Jxx + fdyr * Jyx) @ m.phix
            + sp.sparse.diags(fdxr * Jxy + fdyr * Jyy) @ m.phiy
        )
        self.wphiJdf = sp.sparse.diags(m.wdetJ) @ phiJdf
        self.dyn = np.max(self.f) - np.min(self.f)
        ff = sp.sparse.diags(self.f) @ m.Me
        mean0 = np.asarray(np.mean(ff, axis=0))[0]
        self.std0 = np.asarray(np.sqrt(np.mean(ff.power(2), axis=0) - mean0 ** 2))[0]
        self.f -= m.Me @ mean0.T
        return phiJdf.T @ self.wphiJdf

    def ComputeLHS2(self, f, g, m, cam, U):
        """Compute the FE-DIC right hand side operator with the true Gauss Newton
    
        Parameters
        ----------
        f : pyxel.Image
            Reference State Image
        g : pyxel.Image
            Deformed State Image
        m : pyxel.Mesh
            The FE mesh
        cam : pyxel.Camera
            Calibrated Camera model.
        U : Numpy Array (OPTIONAL)
            The running displacement dof vector.
            
        Returns
        -------
        scipy sparse
            The DIC Hessian (in the GN sense)
    
        """
        if hasattr(f, "tck") == 0:
            f.BuildInterp()
        if hasattr(g, "tck") == 0:
            g.BuildInterp()
        pgu, pgv = cam.P(m.pgx, m.pgy)
        self.f = f.Interp(pgu, pgv)
        pgu, pgv = cam.P(m.pgx + m.phix.dot(U), m.pgy + m.phiy.dot(U))
        fdxr, fdyr = g.InterpGrad(pgu, pgv)
        Jxx, Jxy, Jyx, Jyy = cam.dPdX(m.pgx, m.pgy)
        phiJdf = sp.sparse.diags(fdxr * Jxx + fdyr * Jyx).dot(m.phix) + sp.sparse.diags(
            fdxr * Jxy + fdyr * Jyy
        ).dot(m.phiy)
        self.wphiJdf = sp.sparse.diags(m.wdetJ).dot(phiJdf)
        self.dyn = np.max(self.f) - np.min(self.f)
        self.mean0 = np.mean(self.f)
        self.std0 = np.std(self.f)
        self.f -= self.mean0
        return phiJdf.T.dot(self.wphiJdf)

    def ComputeRHS(self, g, m, cam, U=[]):
        """Compute the FE-DIC right hand side operator with the modified GN
    
        Parameters
        ----------
        g : pyxel.Image
            Deformed State Image
        m : pyxel.Mesh
            The FE mesh
        cam : pyxel.Camera
            Calibrated Camera model.
        U : Numpy Array (OPTIONAL)
            The running displacement dof vector.
            
        Returns
        -------
        Numpy array
            DIC right hand side vector
        Numpy array
            The residual vector.
    
        """
        if hasattr(g, "tck") == 0:
            g.BuildInterp()
        if len(U) != m.ndof:
            U = np.zeros(m.ndof)
        u, v = cam.P(m.pgx + m.phix @ U, m.pgy + m.phiy @ U)
        res = g.Interp(u, v)
        res -= np.mean(res)
        std1 = np.std(res)
        res = self.f - self.std0 / std1 * res
        B = self.wphiJdf.T @ res
        return B, res

    def ComputeRHS2(self, g, m, cam, U=[]):
        """Compute the FE-DIC right hand side operator with the true Gauss-Newton
    
        Parameters
        ----------
        g : pyxel.Image
            Deformed State Image
        m : pyxel.Mesh
            The FE mesh
        cam : pyxel.Camera
            Calibrated Camera model.
        U : Numpy Array (OPTIONAL)
            The running displacement dof vector.
            
        Returns
        -------
        Numpy array
            DIC right hand side vector
        Numpy array
            The residual vector.
    
        """
        if hasattr(g, "tck") == 0:
            g.BuildInterp()
        if len(U) != m.ndof:
            U = np.zeros(m.ndof)
        u, v = cam.P(m.pgx + m.phix @ U, m.pgy + m.phiy @ U)
        res = g.Interp(u, v)
        res -= np.mean(res)
        std1 = np.std(res)
        res = self.f - self.std0 / std1 * res
        fdxr, fdyr = g.InterpGrad(u, v)
        Jxx, Jxy, Jyx, Jyy = cam.dPdX(m.pgx, m.pgy)
        wphiJdf = (
            sp.sparse.diags(m.wdetJ * (fdxr * Jxx + fdyr * Jyx)) @ m.phix
            + sp.sparse.diags(m.wdetJ * (fdxr * Jxy + fdyr * Jyy)) @ m.phiy
        )
        B = wphiJdf.T @ res
        return B, res

    def ComputeRES(self, g, m, cam, U=[]):
        """Compute the FE-DIC residual
    
        Parameters
        ----------
        g : pyxel.Image
            Deformed State Image
        m : pyxel.Mesh
            The FE mesh
        cam : pyxel.Camera
            Calibrated Camera model.
        U : Numpy Array (OPTIONAL)
            The displacement dof vector.

        Returns
        -------
        Numpy array
            the residual vector.
    
        """
        if hasattr(g, "tck") == 0:
            g.BuildInterp()
        if len(U) != m.ndof:
            U = np.zeros(m.ndof)
        pgxu = m.pgx + m.phix.dot(U)
        pgyv = m.pgy + m.phiy.dot(U)
        u, v = cam.P(pgxu, pgyv)
        res = g.Interp(u, v)
        res -= np.mean(res)
        std1 = np.std(res)
        res = self.f - self.std0 / std1 * res
        return res

    def ComputeRHS_EB(self, g, m, cam, U=[]):
        """Compute the FE-DIC right hand side operator with the modified GN
        and with elementary correction of brigthness and contrast
    
        Parameters
        ----------
        g : pyxel.Image
            Deformed State Image
        m : pyxel.Mesh
            The FE mesh
        cam : pyxel.Camera
            Calibrated Camera model.
        U : Numpy Array (OPTIONAL)
            The running displacement dof vector.
            
        Returns
        -------
        Numpy array
            DIC right hand side vector
        Numpy array
            The residual vector.
        """
        if hasattr(g, "tck") == 0:
            g.BuildInterp()
        if len(U) != m.ndof:
            U = np.zeros(m.ndof)
        pgxu = m.pgx + m.phix.dot(U)
        pgyv = m.pgy + m.phiy.dot(U)
        u, v = cam.P(pgxu, pgyv)
        res = g.Interp(u, v)
        ff = sp.sparse.diags(res).dot(m.Me)
        mean0 = np.asarray(np.mean(ff, axis=0))[0]
        std0 = np.asarray(np.sqrt(np.mean(ff.power(2), axis=0) - mean0 ** 2))[0]
        res -= m.Me @ mean0
        res = self.f - sp.sparse.diags(m.Me @ (self.std0 / std0)) @ res
        B = self.wphiJdf.T @ res
        return B, res

#%% 
def PlotMeshImage(f, m, cam, U=None, newfig=True):
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
    newfig : Bool
        (DEFAULT = True)

    """
    n = m.n.copy()
    if U is not None:
        n += U[m.conn]
    if newfig:
        plt.figure()
    f.Plot()
    u, v = cam.P(n[:, 0], n[:, 1])
    m.Plot(n=np.c_[v, u], edgecolor="y", alpha=0.6)
    # plt.xlim([0,f.pix.shape[1]])
    # plt.ylim([f.pix.shape[0],0])
    plt.axis("on")


def Correlate(f, g, m, cam, dic=None, H=None, U0=None, l0=None, Basis=None, 
              L=None, eps=None, disp=True):
    """Perform FE-Digital Image Correlation.

    Parameters
    ----------
    f : pyxel.Image
        Reference Image
    g : pyxel.Image
        Deformed State Image
    m : pyxel.Mesh
        The FE mesh
    cam : pyxel.Camera
        Calibrated Camera model.
    dic : pyxel.DICEngine (OPTIONAL)
        An existing DIC engine where ComputeLHS is pre-computed.
        Allow to perform multi time step correlation faster. 
    H : scipy sparse (OPTIONAL)
        DIC Hessian operator (avoid recomputing when constant)
    U0 : Numpy Array (OPTIONAL)
        Initial guess for the displacement dof vector.
    l0 : float (OPTIONAL)
        regularization length in physical (mesh) unit
    Basis : Numpy array (OPTIONAL)
        Reduced basis for use in iDIC for instance
    L : scipy sparse (OPTIONAL)
        Regularization operator, for instance computed with L = pyxel.Tikhonov()
    eps : float (OPTIONAL)
        stopping criterion for dU/U
    disp : Bool (DEFAULT=True)
        Display error and residual magnitude at each iteration.
        
    Returns
    -------
    Numpy array
        Displacement DOF vector
    Numpy array
        Residual vector

    """
    if dic is None:
        dic = DICEngine()
    if len(m.conn) == 0:
        m.Connectivity()
    if U0 is None:
        U = np.zeros(m.ndof)
    else:
        U = U0.copy()
    if m.phix is None:
        m.DICIntegration(cam)
    if H is None:
        H = dic.ComputeLHS(f, m, cam)
    if eps is None:
        eps = 1e-3
    if Basis is not None:
        # Reduced Basis
        H_LU = splalg.splu(Basis.T @ H @ Basis)
    else:
        if l0 is not None:
            # Tikhonov regularisation
            if L is None:
                L = m.Tikhonov()
            used_nodes = m.conn[:, 0] > 0
            V = np.zeros(m.ndof)
            V[m.conn[used_nodes, 0]] = np.cos(m.n[used_nodes, 1] / l0 * 2 * np.pi)
            H0 = V.dot(H.dot(V))
            L0 = V.dot(L.dot(V))
            l = H0 / L0
            H_LU = splalg.splu(H + l * L)
        else:
            if disp:
                print("no reg")
            H_LU = splalg.splu(H)
    for ik in range(0, 100):
        [b, res] = dic.ComputeRHS(g, m, cam, U)
        if Basis is not None:
            da = H_LU.solve(Basis.T @ b)
            dU = Basis @ da
        elif l0 is not None:
            dU = H_LU.solve(b - l * L.dot(U))
            err = np.max(abs(dU))
        else:
            dU = H_LU.solve(b)
        U += dU
        err = np.linalg.norm(dU) / np.linalg.norm(U)
        if disp:
            print("Iter # %2d | std(res)=%2.2f gl | dU/U=%1.2e" % (ik + 1, np.std(res), err))
        if err < eps:
            break
    return U, res


def MultiscaleInit(imf, img, m, cam, scales=[3, 2, 1], l0=None, U0=None,
                   Basis=None, eps=None, disp=True):
    """Perform Multigrid initialization for FE-Digital Image Correlation.

    Parameters
    ----------
    f : pyxel.Image
        Reference Image
    g : pyxel.Image
        Deformed State Image
    m : pyxel.Mesh
        The FE mesh
    cam : pyxel.Camera
        Calibrated Camera model.
    scales : python list (DEFAULT=[3,2,1])
        An ordered list of scales for the multigrid initialization.
        Each time image is subsampled by 2**scale.
        Scale 0 correspond to initial image
    l0 : float (OPTIONAL)
        regularization length in physical (mesh) unit
    U0 : Numpy Array (OPTIONAL)
        Initial guess for the displacement dof vector.
    Basis : Numpy array (OPTIONAL)
        Reduced basis for use in iDIC for instance
    L : scipy sparse (OPTIONAL)
        Regularization operator, for instance computed with L = pyxel.Tikhonov()
    eps : float (OPTIONAL)
        stopping criterion for dU/U
    disp : Bool (DEFAULT=True)
        Display error and residual magnitude at each iteration.
        
    Returns
    -------
    Numpy array
        Displacement DOF vector

    """
    if len(m.conn) == 0:
        m.Connectivity()
    if l0 is None:
        l0 = 0.0
        for je in m.e.keys():
            n1 = m.n[m.e[je][:, 0]]
            n2 = m.n[m.e[je][:, 1]]
            l0 = max(l0, 4 * min(np.linalg.norm(n1 - n2, axis=1)))
            print('Auto reg. length l0 = %2.3e' % l0)
    if U0 is None:
        U = np.zeros(m.ndof)
    else:
        U = U0.copy()
    L = m.Tikhonov()
    for js in range(len(scales)):
        iscale = scales[js]
        if disp:
            print("SCALE %2d" % (iscale))
        f = imf.Copy()
        f.SubSample(iscale)
        g = img.Copy()
        g.SubSample(iscale)
        cam2 = cam.SubSampleCopy(iscale)
        m2 = m.Copy()
        m2.DICIntegration(cam2, tri_same=True)
        U, r = Correlate(f, g, m2, cam2, l0=l0 * 2 ** iscale, 
                         Basis=Basis, L=L, U0=U, eps=eps, disp=disp)
    return U


def CorrelateTimeIncr(m, f, imagefile, imnums, cam, scales):
    """Performs FE-DIC for a time image series.

    Parameters
    ----------
    m : pyxel.Mesh
        The FE mesh
    f : pyxel.Image
        Reference Image
    imagefile : string
        a generic filename for the deformed state images.
        example: imagefile = os.path.join('data', 'dic_composite', 'zoom-0%03d_1.tif')
        such that imagefile % 30 is the filename 'data/dic_composite/zoom-0030_1.tif'
    imnums : Numpy Array
        The array containing the deformed state image numbers
    cam : pyxel.Camera
        Calibrated Camera model.
    scales : python list (DEFAULT=[3,2,1])
        An ordered list of scales for the multigrid initialization.
        Each time image is subsampled by 2**scale.
        Scale 0 correspond to initial image
        
    Returns
    -------
    Numpy array
        An Array containing the displacement DOF vector, one column for one timestep.

    """
    UU = np.zeros((m.ndof, len(imnums)))
    if len(m.pgx) == 0:
        m.DICIntegration(cam)
    dic = DICEngine()
    H = dic.ComputeLHS(f, m, cam)
    im = 1
    print(" ==== IMAGE %3d === " % imnums[im])
    imdef = imagefile % imnums[im]
    g = Image(imdef).Load()
    UU[:, im] = MultiscaleInit(f, g, m, cam, scales=scales)
    UU[:, im], r = Correlate(f, g, m, cam, dic=dic, H=H, U0=UU[:, im])
    for im in range(2, len(imnums)):
        print(" ==== IMAGE %3d === " % imnums[im])
        imdef = imagefile % imnums[im]
        g = Image(imdef).Load()
        if True:
            UU[:, im] = MultiscaleInit(
                f, g, m, cam, scales=scales, U0=UU[:, im - 1], eps=1e-4
            )
            UU[:, im], r = Correlate(f, g, m, cam, dic=dic, H=H, U0=UU[:, im], eps=1e-4)
        else:
            V = UU[:, [im - 1]]
            UU[:, im] = MultiscaleInit(
                f, g, m, cam, scales=scales, Basis=V, U0=UU[:, im - 1], eps=1e-4
            )
            UU[:, im], r = Correlate(
                f, g, m, cam, dic=dic, H=H, Basis=V, U0=UU[:, im], eps=1e-4
            )
            UU[:, im], r = Correlate(f, g, m, cam, dic=dic, H=H, U0=UU[:, im], eps=1e-4)
        if not os.path.isdir('tmp'):
            os.makedirs('tmp')
        np.save(os.path.join('tmp', 'multiscale_init_tmp'), UU)
    return UU

def Hooke(p,typc='isotropic'):
    """Compute 2D Hooke tensor from elastic constants

    Parameters
    ----------
    p : Numpy Array
        p = [E, nu] for isotropic material
        p = [E1, E2, nu12, G12] for orthotropic material
    typc : string
        'isotropic' (DEFAULT)
        'orthotropic'  

    Returns
    -------
    Numpy array
        2D Hooke tensor.

    """
    if typc == 'isotropic':
        E = p[0]
        v = p[1]
        return E / (1 - v**2) * np.array([[1, v, 0], [v, 1, 0], [0, 0, (1 - v) / 2]])
    elif typc == 'orthotropic':
        El = p[0]
        Et = p[1]
        vtl = p[2]
        Glt = p[3]
        vlt = vtl * El / Et
        alp = 1 / (1 - vlt * vtl)
        return np.array([[alp * El, alp * vtl * El, 0],
                        [alp * vlt * Et, alp * Et, 0],
                        [0, 0, 2 * Glt]])
    else:
        print('Unknown elastic constitutive regime')

#%%
def Gmsh2Mesh(gmsh, dim=2):
    """
    Bulding pyxel mesh from gmsh python object

    Parameters
    ----------
        gmsh : python gmsh object

    EXAMPLE:
    ----------
        import gmsh
        gmsh.initialize()
        gmsh.model.add("P")
        lc = 0.02
        gmsh.model.geo.addPoint(0, 0.0, 0, 4 * lc, 1)
        gmsh.model.geo.addPoint(1, 0.0, 0, lc, 2)
        gmsh.model.geo.addPoint(1, 0.5, 0, lc, 3)
        gmsh.model.geo.addPoint(0, 0.5, 0, 4 * lc, 4)
        gmsh.model.geo.addLine(1, 2, 1)
        gmsh.model.geo.addLine(2, 3, 2)
        gmsh.model.geo.addLine(3, 4, 3)
        gmsh.model.geo.addLine(4, 1, 4)
        gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
        gmsh.model.geo.addPlaneSurface([1], 1)
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(2)
        m = px.gmsh2pyxel(gmsh)
        m.Plot()
        
    """
    # Get direct full node list
    nums, nodes, e = gmsh.model.mesh.getNodes()
    nodes = nodes.reshape((len(nums), 3))
    elems = dict()
    # Get the Element by type
    for et in gmsh.model.mesh.getElementTypes():
        nums, els = gmsh.model.mesh.getElementsByType(et)
        nnd = len(els) // len(nums)
        elems[et] = els.reshape((len(nums), nnd)).astype(int) - 1
    del elems[15]  # remove points
    del elems[1]  # remove segments
    m = Mesh(elems, nodes[:, :dim], dim)
    return m

def ReadMeshGMSH(fn, dim=2):
    """
    Beta GMSH parser and converter to pyxel.Mesh object.
    
    Parameters
    ----------
        fn : string (filename)
        dim : int (DEFAULT dim=2)
    """
    mshfid = open(fn, "r")
    line = mshfid.readline()
    while line.find("$Nodes") < 0:
        line = mshfid.readline()
        pass
    line = mshfid.readline()
    nnodes = int(line)
    nodes = np.zeros((nnodes, 3))
    for jn in range(nnodes):
        sl = mshfid.readline().split()
        nodes[jn] = np.double(sl[1:])
    while line.find("$Elements") < 0:
        line = mshfid.readline()
        pass
    line = mshfid.readline()
    nelems = int(line)
    elems = dict()
    ne = 1
    line = np.int32(mshfid.readline().split())
    while ne < nelems:
        et = line[1]
        if et == 3:  # qua4
            nn = 4
            rep = np.arange(5, 9)
        elif et == 2:  # tri3
            nn = 3
            rep = np.arange(5, 8)
        elif et == 15:  # point
            nn = 1
            rep = [4]
        elif et == 1:  # segment
            nn = 2
            rep = np.arange(5, 7)
        elif et == 9:  # tri6
            nn = 6
            rep = np.arange(5, 11)
        elif et == 16:  # qua8
            nn = 8
            rep = np.arange(5, 13)
        elif et == 10:  # qua9
            nn = 8
            rep = np.arange(5, 14)
        elems[et] = np.empty((0, nn), dtype=int)
        while line[1] == et:
            elems[et] = np.vstack((elems[et], line[rep] - 1))
            try:
                line = np.int32(mshfid.readline().split())
            except:
                break
            ne += 1
    if dim == 2:
        nodes = np.delete(nodes, 2, 1)
    del elems[15]  # remove points
    del elems[1]  # remove segments
    m = Mesh(elems, nodes, dim)
    return m

def ReadMeshINP(fn):
    """
    Beta ABAQUS INP parser with 2D mesh and converter to pyxel.Mesh object.
    
    Parameters
    ----------
        fn: string (filename)
    
    2D ONLY : any type of S4, E4, S3, E3, S6, E6, S8, E8
    """

    mshfid = open(fn, "r")
    line = mshfid.readline()
    while line.find("*Node") < 0:
        line = mshfid.readline()
        pass
    nodes = np.zeros((0, 2))
    line = mshfid.readline()
    while line.find("*Element") < 0:
        nodes = np.vstack((nodes, np.double(line.split(",")[1:])))
        line = mshfid.readline()
    elems = dict()
    while "*Element" in line:  # Loop on different element types
        print(line[:-1])
        if "S4" in line or "E4" in line:
            et = 3
            nn = 4
            rep = [1, 2, 3, 4]
        elif "S3" in line or "E3" in line:
            et = 2
            nn = 3
            rep = [1, 2, 3]
        elif "S6" in line or "E6" in line:
            et = 9
            nn = 6
            rep = [1, 2, 3, 4, 5, 6]
        elif "S8" in line or "E8" in line:
            et = 16
            nn = 8
            rep = [1, 2, 3, 4, 5, 6, 7, 8]
        elems[et] = np.empty((0, nn), dtype=int)
        while True:
            line = mshfid.readline().split(",")
            try:
                line = np.int32(line)
            except:
                break
            elems[et] = np.vstack((elems[et], line[rep] - 1))
    m = Mesh(elems, nodes)
    return m

def ReadMeshINPwithElset(fn):
    """
    Beta ABAQUS INP parser with 2D mesh and converter to pyxel.Mesh object.
    Exports also the element sets.

    Parameters
    ----------
        fn : string (filename)
    
    2D ONLY : any type of S4, E4, S3, E3, S6, E6, S8, E8
    """
    mshfid = open(fn, "r")
    line = mshfid.readline()
    while line.find("*Node") < 0:
        line = mshfid.readline()
        pass
    nodes = np.zeros((0, 2))
    line = mshfid.readline()
    while line.find("*Element") < 0:
        nodes = np.vstack((nodes, np.double(line.split(",")[1:])))
        line = mshfid.readline()
    # nnodes = nodes.shape[0]
    elems = dict()
    while "*Element" in line:  # Loop on different element types
        print(line[:-1])
        if "S4" in line or "E4" in line:
            et = 3
            nn = 4
            rep = [1, 2, 3, 4]
        elif "S3" in line or "E3" in line:
            et = 2
            nn = 3
            rep = [1, 2, 3]
        elif "S6" in line or "E6" in line:
            et = 9
            nn = 6
            rep = [1, 2, 3, 4, 5, 6]
        elif "S8" in line or "E8" in line:
            et = 16
            nn = 8
            rep = [1, 2, 3, 4, 5, 6, 7, 8]
        else:
            print("Unknown Element!")
            print(line)
        elems[et] = np.empty((0, nn), dtype=int)
        while True:
            line = mshfid.readline()
            try:
                line = np.int32(line.split(","))
            except:
                break
            elems[et] = np.vstack((elems[et], line[rep] - 1))
    elset = dict()
    nelset = 0
    while line.find("*") >= 0:
        if line.find("*Elset") >= 0:
            print(line[:-1])
            if line.find("generate") >= 0:
                line = mshfid.readline()
                gen = np.int32(line.split(","))
                elset[nelset] = np.arange(gen[0] - 1, gen[1], gen[2])
                line = mshfid.readline()
            else:
                line = mshfid.readline()
                lineconcat = ""
                while line.find("*") < 0:
                    lineconcat += "," + line
                    line = mshfid.readline()
                if lineconcat[-2] == ",":
                    lineconcat = lineconcat[:-2]
                elset[nelset] = np.int32(lineconcat[1:].split(",")) - 1
            nelset += 1
        elif line.find("*End Part") >= 0:
            break
        else:
            line = mshfid.readline()
            while line.find("*") < 0:
                line = mshfid.readline()
    m = Mesh(elems, nodes)
    return m, elset

def ReadMeshINP3D(fn):
    """BETA"""
    lines = open(fn, "r").readlines()
    k = 0
    while lines[k] != "*Node\r\n":
        k += 1
    k += 1
    nodes = np.zeros((0, 3))
    while lines[k][0:8] != "*Element":
        nodes = np.vstack((nodes, np.fromstring(lines[k], sep=",")[1:]))
        k += 1
    # here lines[k] == '*Element, type=C3D8R\r\n'
    k += 1
    elems = np.zeros((0, 9), dtype="int")
    while lines[k][0:1] != "*":
        elems = np.vstack((elems, np.fromstring(lines[k], sep=",", dtype="int") - 1))
        k += 1
    elems[:, 0] = 5
    m = Mesh(elems, nodes)
    return m
