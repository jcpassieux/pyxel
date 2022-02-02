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
import matplotlib.pyplot as plt
import matplotlib.collections as cols
import matplotlib.animation as animation
from numba import njit
from .utils import meshgrid, isInBox
from .vtktools import VTUWriter, PVDFile
from .camera import Camera

def ElTypes():
    """
    Returns a dictionnary of GMSH element types which some of them are used in the library.
    """
    return {1: "2-node line.",
        2: "3-node triangle.",
        3: "4-node quadrangle.",
        4: "4-node tetrahedron.",
        5: "8-node hexahedron.",
        6: "6-node prism.",
        7: "5-node pyramid.",
        8: "3-node second order line (2 nodes associated with the vertices and 1 with the edge).",
        9: "6-node second order triangle (3 nodes associated with the vertices and 3 with the edges).",
        10: "9-node second order quadrangle (4 nodes associated with the vertices, 4 with the edges and 1 with the face).",
        11: "10-node second order tetrahedron (4 nodes associated with the vertices and 6 with the edges).",
        12: "27-node second order hexahedron (8 nodes associated with the vertices, 12 with the edges, 6 with the faces and 1 with the volume).",
        13: "18-node second order prism (6 nodes associated with the vertices, 9 with the edges and 3 with the quadrangular faces).",
        14: "14-node second order pyramid (5 nodes associated with the vertices, 8 with the edges and 1 with the quadrangular face).",
        15: "1-node point.",
        16: "8-node second order quadrangle (4 nodes associated with the vertices and 4 with the edges).",
        17: "20-node second order hexahedron (8 nodes associated with the vertices and 12 with the edges).",
        18: "15-node second order prism (6 nodes associated with the vertices and 9 with the edges).",
        19: "13-node second order pyramid (5 nodes associated with the vertices and 8 with the edges).",
        20: "9-node third order incomplete triangle (3 nodes associated with the vertices, 6 with the edges)",
        21: "10-node third order triangle (3 nodes associated with the vertices, 6 with the edges, 1 with the face)",
        22: "12-node fourth order incomplete triangle (3 nodes associated with the vertices, 9 with the edges)",
        23: "15-node fourth order triangle (3 nodes associated with the vertices, 9 with the edges, 3 with the face)",
        24: "15-node fifth order incomplete triangle (3 nodes associated with the vertices, 12 with the edges)",
        25: "21-node fifth order complete triangle (3 nodes associated with the vertices, 12 with the edges, 6 with the face)",
        26: "4-node third order edge (2 nodes associated with the vertices, 2 internal to the edge)",
        27: "5-node fourth order edge (2 nodes associated with the vertices, 3 internal to the edge)",
        28: "6-node fifth order edge (2 nodes associated with the vertices, 4 internal to the edge)",
        29: "20-node third order tetrahedron (4 nodes associated with the vertices, 12 with the edges, 4 with the faces)",
        30: "35-node fourth order tetrahedron (4 nodes associated with the vertices, 18 with the edges, 12 with the faces, 1 in the volume)",
        31: "56-node fifth order tetrahedron (4 nodes associated with the vertices, 24 with the edges, 24 with the faces, 4 in the volume)",
        92: "64-node third order hexahedron (8 nodes associated with the vertices, 24 with the edges, 24 with the faces, 8 in the volume)",
        93: "125-node fourth order hexahedron (8 nodes associated with the vertices, 36 with the edges, 54 with the faces, 27 in the volume)",
        }

#%%
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

def InverseFEMapping(xn, yn, xpix, ypix, eltype):
    """ Inverse the Finite Element mapping in order to map the coordinates """
    """ of any physical points (xpix, ypix) to their corresponding position in """
    """ the parent element (xg, yg)."""
    _,_,_,N,dN_xi,dN_eta = ShapeFunctions(eltype)
    xg = 0 * xpix
    yg = 0 * ypix
    res = 1
    for k in range(7):
        phi = N(xg, yg)
        N_r = dN_xi(xg, yg)
        N_s = dN_eta(xg, yg)
        dxdr = np.dot(N_r, xn)
        dydr = np.dot(N_r, yn)
        dxds = np.dot(N_s, xn)
        dyds = np.dot(N_s, yn)
        detJ = dxdr * dyds - dydr * dxds
        invJ = np.array([dyds / detJ, -dxds / detJ, -dydr / detJ, dxdr / detJ]).T
        xp = np.dot(phi, xn)
        yp = np.dot(phi, yn)
        dxg = invJ[:, 0] * (xpix - xp) + invJ[:, 1] * (ypix - yp)
        dyg = invJ[:, 2] * (xpix - xp) + invJ[:, 3] * (ypix - yp)
        res = np.dot(dxg, dxg) + np.dot(dyg, dyg)
        xg = xg + dxg
        yg = yg + dyg
        if res < 1.0e-6:
            break
    return xg, yg

def GetPixelsElem(xn, yn, xpix, ypix, eltype):
    """Finds the pixels that belong to any 2D element and"""
    """inverse the mapping to know their corresponding position in """
    """the parent element."""
    wg = IsPointInElem2d(xn, yn, xpix, ypix)
    ind = np.where(wg)
    xg, yg = InverseFEMapping(xn, yn, xpix[ind], ypix[ind], eltype)
    return xg, yg, xpix[ind], ypix[ind]

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

def SubTriGmsh(n):
    import gmsh as gmsh
    gmsh.initialize()
    gmsh.model.add("P")
    gmsh.model.geo.addPoint(0, 0, 0, 1/n, 1)
    gmsh.model.geo.addPoint(1, 0, 0, 1/n, 2)
    gmsh.model.geo.addPoint(0, 1, 0, 1/n, 3)
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
    NE = np.max(np.c_[NE,np.ones(2,dtype=int)],axis=1)
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
    NE = np.max(np.c_[NE,np.ones(2,dtype=int)],axis=1)
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
    NE = np.max(np.c_[NE,np.ones(2,dtype=int)],axis=1)
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
    NE = np.max(np.c_[NE,np.ones(2,dtype=int)],axis=1)
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

def StructuredMesh(box, dx, typel=3):
    """Build a structured mesh from two points coordinates (box)
    
    parameters
    ----------
    box: numpy.array
        box = np.array([[xmin, ymin],
                    [xmax, ymax]])    in mesh unit
    dx: numpy or python array
        dx = [dx, dy]: average element size (can be scalar) in mesh unit
    typel: int
        Element types: (see pyxel.ElTypes)
        2: first order triangles (T3)
        3: first order quadrangles (Q4)
        9: second order triangles (T6)
        10: 9-node second order quadrangles (Q9)
        16: 8-node second order quadrangles (Q8)
    """
    if typel == 2:
        return StructuredMeshT3(box, dx)
    elif typel == 9:
        return StructuredMeshT6(box, dx)
    elif typel == 3:
        return StructuredMeshQ4(box, dx)
    elif typel == 16:
        return StructuredMeshQ8(box, dx)
    elif typel == 10:
        return StructuredMeshQ9(box, dx)
    
#%%
def ShapeFunctions(eltype):
    """For any type of 2D elements, gives the quadrature rule and
    the shape functions and their derivative"""
    xg = 0
    yg = 0
    wg = 0
    if eltype == 1:
        """
        #############
            seg2
        #############
        """
        def N(x):
            return np.concatenate(
                (1 - x, x)).reshape((2,len(x))).T
        
        def dN_xi(x):
            return np.concatenate(
                (-1.0 + 0 * x, 1.0 + 0 * x)).reshape((2,len(x))).T

        def dN_eta(x):
            return False

        xg = np.array([0.])
        wg = np.array([2.])
    elif eltype == 8:
        """
        #############
            seg3
        #############
        """
        def N(x):
            return np.concatenate(
                ((x**2 - x) * 0.5, 1 - x**2, (x**2 + x) * 0.5)).reshape((3,len(x))).T
        
        def dN_xi(x):
            return np.concatenate(
                (x - 0.5, -2 * x, x + 1)).reshape((3,len(x))).T

        def dN_eta(x):
            return False
        
        xg = np.sqrt(3) / 3 * np.array([-1, 1])
        wg = np.array([1., 1.])
    elif eltype == 2:
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
                [xg, yg, elem[ne].pgx, elem[ne].pgy] = GetPixelsElem(
                    u[je], v[je], xpix.ravel(), ypix.ravel(), et
                )
                elem[ne].phi = N(xg, yg)
                elem[ne].repg = repg + np.arange(xg.shape[0])
                repg += xg.shape[0]
                nzv += np.prod(elem[ne].phi.shape)
                ne += 1
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

    def FastDICIntegElem(self, e, et, n=10, G=False):
        # parent element
        _, _, _, N, Ndx, Ndy = ShapeFunctions(et)
        if et in [2, 9]: # Triangles
            if et == 2:
                n = max(n, 1) # minimum 1 integration point for first order
            else:
                n = max(n, 2) # minimum 2 integration points for second order
            xg, yg, wg = SubTriIso2(n)
            # xi = np.linspace(0, 1, n+3)[1:-1]
            # xg, yg = np.meshgrid(1-xi, xi)
            # rep=xg==np.triu(xg, 1)
            # repi,repj=np.where(rep)
            # xg = xg[repi,repj]
            # yg = yg[repi,repj]
            # wg = 0.5 / len(xg) * np.ones(len(xg))

            # plt.plot(xg,yg,'k.')
            # plt.plot([0,1,0,0],[0,0,1,0],'k-')
            # plt.axis('equal')
        elif et in [3, 10, 16]: # Quadrangles
            n = max(n, 2) # minimum 2 integration points
            # xi = np.linspace(-1, 1, n+2)[1:-1]
            xi = np.linspace(-1, 1, n+1)[:-1] + 1/n
            xg, yg = np.meshgrid(xi, xi)
            xg = xg.ravel()
            yg = yg.ravel()
            # plt.plot(xg,yg,'k.')
            # plt.plot([-1,1,1,-1,-1],[-1,-1,1,1,-1],'k-')
            wg = 4 / len(xg) * np.ones(len(xg))
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
        if G:
            valx = np.zeros(nzv)
            valy = np.zeros(nzv)
        else :
            valx = []
            valy = []
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
            repnzv = np.arange(ne * nfun) + i * ne * nfun
            col[repnzv] = repdof.ravel()
            row[repnzv] = np.tile(np.arange(ne) + i * ne, [nfun, 1]).T.ravel()
            val[repnzv] = np.tile(phi[i, :], [ne, 1]).ravel()
            if G:
                dphidx = (dyds / detJ)[:, np.newaxis] * dN_xi[i, :] + (-dydr / detJ)[
                    :, np.newaxis] * dN_eta[i, :]
                dphidy = (-dxds / detJ)[:, np.newaxis] * dN_xi[i, :] + (dxdr / detJ)[
                    :, np.newaxis] * dN_eta[i, :]
                valx[repnzv] = dphidx.ravel()
                valy[repnzv] = dphidy.ravel()
        return col, row, val, valx, valy, wdetJ

    def GetApproxElementSize(self, cam=None, method='max'):
        """ Estimate average/min/max element size
        input
        -----
        cam : pyxel.Camera (OPTIONAL)
            To get the size in pixels
        method: string
            'max': estimation of the maximum element size
            'min': estimation of the minimum element size
            'mean': estimation of the mean element size
        """
        if cam is None:
            u = self.n[:, 0]
            v = self.n[:, 1]            
        else:
            u, v = cam.P(self.n[:,0], self.n[:,1])
        aes = []
        for et in self.e.keys():
            um = u[self.e[et]]-np.mean(u[self.e[et]], axis=1)[:,np.newaxis]
            vm = v[self.e[et]]-np.mean(v[self.e[et]], axis=1)[:,np.newaxis]
            if method == 'max':
                aes = np.append(aes, np.max(np.sqrt(um**2 + vm**2), axis=1))
            elif method == 'min':
                aes = np.append(aes, np.min(np.sqrt(um**2 + vm**2), axis=1))
            elif method == 'mean':
                aes = np.append(aes, np.sqrt(um**2 + vm**2))
        if method == 'max':
            return np.mean(aes) + np.std(aes) * 0.5
        elif method == 'min':
            return np.mean(aes) - np.std(aes) * 0.5
        elif method == 'mean':
            return np.mean(aes)

    def DICIntegrationFast(self, n=10, G=False):
        """Builds a homogeneous (and fast) integration scheme for DIC"""
        if hasattr(n, 'rz'):
            # if n is a camera and n is autocomputed
            n = self.GetApproxElementSize(n)
        if type(n) is not int:
            n = int(n)
        self.wdetJ = np.array([])
        col = np.array([])
        row = np.array([])
        val = np.array([])
        if G: # compute also the shape function gradients
            valx = np.array([])
            valy = np.array([])
        npg = 0
        for je in self.e.keys():
            colj, rowj, valj, valxj, valyj, wdetJj = self.FastDICIntegElem(self.e[je], je, n, G=G)
            col = np.append(col, colj)
            row = np.append(row, rowj + npg)
            val = np.append(val, valj)
            if G:
                valx = np.append(valx, valxj)
                valy = np.append(valy, valyj)
            self.wdetJ = np.append(self.wdetJ, wdetJj)
            npg += len(wdetJj)
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
        rep, = np.where(self.conn[:, 0] >= 0)
        qx = np.zeros(self.ndof)
        qx[self.conn[rep, :]] = self.n[rep, :]
        self.pgx = self.phix.dot(qx)
        self.pgy = self.phiy.dot(qx)

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
        npg = 0
        for je in self.e.keys():
            colj, rowj, valj, valxj, valyj, wdetJj = self.GaussIntegElem(self.e[je], je)
            col = np.append(col, colj)
            row = np.append(row, rowj + npg)
            val = np.append(val, valj)
            valx = np.append(valx, valxj)
            valy = np.append(valy, valyj)
            self.wdetJ = np.append(self.wdetJ, wdetJj)
            npg += len(wdetJj)
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
            print('Gauss Integ.')
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
        vtkfile = VTUWriter(nnode, nelem, new_node, new_conn, new_offs, new_type)
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
        PVDFile(os.path.join("vtk", dir0, fileName), ext, npart, nstep)

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
        vtkfile = VTUWriter(nnode, nelem, new_node, new_conn, new_offs, new_type)
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

    # def StrainAtNodes(self, U):
    #     nnodes = self.ndof // 2
    #     m = self.Copy()
    #     m.GaussIntegration()
    #     exxgp = m.dphixdx @ U
    #     eyygp = m.dphiydy @ U
    #     exygp = 0.5 * m.dphixdy @ U + 0.5 * m.dphiydx @ U
    #     EpsXX = np.zeros(nnodes)
    #     EpsYY = np.zeros(nnodes)
    #     EpsXY = np.zeros(nnodes)
    #     for jn in range(len(self.n)):
    #         if self.conn[jn, 0] >= 0:
    #             sig = 0  # max over all element types in the neighborhood
    #             for je in self.e.keys():
    #                 eljn, _ = np.where(self.e[je] == jn)
    #                 if len(eljn) != 0:
    #                     xm = np.mean(self.n[self.e[je][eljn, :], 0], axis=1)
    #                     ym = np.mean(self.n[self.e[je][eljn, :], 1], axis=1)
    #                     sig = max(sig, np.max(np.sqrt((xm - self.n[jn, 0]) ** 2
    #                                                   + (ym - self.n[jn, 1]) ** 2)) / 3)
    #             D = np.sqrt((m.pgx - self.n[jn, 0]) ** 2 + (m.pgy - self.n[jn, 1]) ** 2)
    #             gauss = np.exp(-(D ** 2) / (2 * sig ** 2))
    #             if np.sum(gauss) < 1e-15:
    #                 print(jn)
    #             gauss /= np.sum(gauss)
    #             EpsXX[self.conn[jn, 0]] = gauss @ exxgp
    #             EpsYY[self.conn[jn, 0]] = gauss @ eyygp
    #             EpsXY[self.conn[jn, 0]] = gauss @ exygp
    #     return EpsXX, EpsYY, EpsXY

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
        phi = m.phix[:,:nnodes]
        w = np.array(np.sum(phi, axis=0))[0]
        phi = phi @ sp.sparse.diags(1/w)
        EpsXX = phi.T @ exxgp
        EpsYY = phi.T @ eyygp
        EpsXY = phi.T @ exygp
        return EpsXX, EpsYY, EpsXY


    # def StrainAtNodesOld(self, UU):
    #     # LS projection... not working so good!
    #     m = self.Copy()
    #     m.GaussIntegration()
    #     wdetJ = sp.sparse.diags(m.wdetJ)
    #     phi = m.phix[:, : m.ndof // 2]
    #     if not hasattr(self, "Bx"):
    #         self.Bx = splalg.splu(phi.T @ wdetJ @ phi)
    #     epsx = self.Bx.solve(phi.T @ wdetJ @ m.dphixdx @ UU)
    #     epsy = self.Bx.solve(phi.T @ wdetJ @ m.dphiydy @ UU)
    #     epsxy = self.Bx.solve(phi.T @ wdetJ @ (m.dphixdy @ UU + m.dphiydx @ UU)) * 0.5
    #     return epsx, epsy, epsxy

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
        vtkfile = VTUWriter(nnode, nelem, new_node, new_conn, new_offs, new_type)
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
        vtkfile = VTUWriter(nnode, nelem, new_node, new_conn, new_offs, new_type)
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

    def PreparePlot(self, U=None, coef=1., n=None, **kwargs):
        """
        Prepare the matplotlib collections for a plot
        """
        edgecolor = kwargs.pop("edgecolor", "k")
        facecolor = kwargs.pop("facecolor", "none")
        alpha = kwargs.pop("alpha", 0.8)
        """ Plot deformed or undeformes Mesh """
        if n is None:
            n = self.n.copy()
        if U is not None:
            n += coef * U[self.conn]
        qua = np.zeros((0, 4), dtype="int64")
        tri = np.zeros((0, 3), dtype="int64")
        bar = np.zeros((0, 2), dtype="int64")
        pn = np.zeros(0, dtype=int) # nodes to plot for quad elems
        for ie in self.e.keys():
            if ie in [3, 16, 10]:  # quadrangles
                qua = np.vstack((qua, self.e[ie][:, :4]))
                if ie in [16, 10]:
                    pn = np.append(pn, self.e[ie].ravel())
            elif ie in [2, 9]:  # triangles
                tri = np.vstack((tri, self.e[ie][:, :3]))
                if ie == 9:
                    pn = np.append(pn, self.e[ie].ravel())
            elif ie in [1, 8]:  # lin and quad bars
                bar = np.vstack((bar, self.e[ie][:, :2]))
                if ie == 8:
                    pn = np.append(pn, self.e[ie].ravel())

        ### Join the 2 lists of vertices
        nn = n[qua].tolist() + n[tri].tolist() + n[bar].tolist()
        ### Create the collection
        pn = np.unique(pn)
        n = n[pn,:]
        pc = cols.PolyCollection(nn, facecolor=facecolor, edgecolor=edgecolor, 
                                 alpha=alpha, **kwargs)
        ### Return the matplotlib collection and the list of vertices
        return pc, nn, n

    def Plot(self, U=None, coef=1, n=None, plotnodes=True, **kwargs):
        """
        Plots the (possibly warped) mesh using Matplotlib Library.

        Inputs: 
        -------
            -U: displacement fields for a deformed mesh plot
            -coef: amplification coefficient
            -n: nodes coordinates

        Usage:
        ------
            m.Plot()      > plots the mesh
            m.Plot(U)     > plots the mesh warped by the displacement U
            m.Plot(U, 30) > ... with a displacement amplification factor = 30

            Supports other Matplotlib arguments:
            m.Plot(U, edgecolor='r', facecolor='b', alpha=0.2)
        """
        ax = plt.gca()
        pc, nn, n = self.PreparePlot(U, coef, n, **kwargs)
        ax.add_collection(pc)
        ax.autoscale()
        edgecolor = kwargs.pop("edgecolor", "k")
        alpha = kwargs.pop("alpha", 0.8)
        if plotnodes:
            plt.plot(
                n[:, 0],
                n[:, 1],
                linestyle="None",
                marker="o",
                color=edgecolor,
                alpha=alpha,
            )
        plt.axis('equal')
        plt.show()

    def AnimatedPlot(self, U, coef=1, n=None, timeAnim=5, color=('k','b','r','g','c')):
        """
        Animated plot with funcAnimation 
        Inputs:
            -U: displacement field, stored in column for each time step
            -coef: amplification coefficient
            -n: nodes coordinates
            -timeAnim: time of the animation
        """
        if not(isinstance(U,list)):
            U = [U]
        ntimes = U[0].shape[1]
        fig = plt.figure()
        ax = plt.gca()
        pc = dict()
        nn = dict()
        for jj, u in enumerate(U):
            pc[jj], nn[jj], _ = self.PreparePlot(u[:,0], coef, n, edgecolor=color[jj])
            ax.add_collection(pc[jj])
            ax.autoscale()
            plt.axis('equal')
        
        def updateMesh(ii):
            """
            Function to update the matplotlib collections
            """
            for jj, u in enumerate(U):
                titi, nn, _ = self.PreparePlot(u[:,ii], coef, n)
                pc[jj].set_paths(nn)
            return pc.values()
        
        line_ani = animation.FuncAnimation(fig, updateMesh, range(ntimes),
                                           blit=True, 
                                           interval=timeAnim/ntimes*1000)
        return line_ani

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
        vtkfile = VTUWriter(nnode, nelem, new_node, new_conn, new_offs, new_type)
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
        Builds edge elements corresponding to the edges of 2d Mesh m.
        """
        edgel = {} #lin
        edgeq = {} #qua
        for je in self.e.keys():
            if je in [9, 16, 10]: # quadratic
                if je in [16, 10]: # qua8 et qua9
                    n1 = self.e[je][:,:4].ravel()
                    n2 = np.c_[self.e[je][:, 1:4], self.e[je][:, 0]].ravel()
                    n3 = self.e[je][:,4:8].ravel()
                else: # tri6
                    n1 = self.e[je][:,:3].ravel()
                    n2 = np.c_[self.e[je][:, 1:3], self.e[je][:, 0]].ravel()
                    n3 = self.e[je][:,3:].ravel()
                a = np.sort(np.c_[n1, n2, n3], axis=1)
                for i in range(len(a)):
                    tedge = tuple(a[i, :])
                    if tedge in edgeq.keys():
                        edgeq[tedge] += 1
                    else:
                        edgeq[tedge] = 1
            else: #linear
                n1 = self.e[je].ravel()
                n2 = np.c_[self.e[je][:, 1:], self.e[je][:, 0]].ravel()
                a = np.sort(np.c_[n1, n2], axis=1)
                for i in range(len(a)):
                    tedge = tuple(a[i, :])
                    if tedge in edgel.keys():
                        edgel[tedge] += 1
                    else:
                        edgel[tedge] = 1
        # linear edges
        elems = dict()
        if len(edgel):
            (rep,) = np.where(np.array(list(edgel.values())) == 1)
            edgel = np.array(list(edgel.keys()))[rep, :]
            elems[1] = edgel
        # quadratic edges
        if len(edgeq):
            (rep,) = np.where(np.array(list(edgeq.values())) == 1)
            edgeq = np.array(list(edgeq.keys()))[rep, :]
            elems[8] = edgeq
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

    def RBM(self):
        """
        INFINITESIMAL RIGID BODY MODES
        
        Returns
        -------
        tx : 1D NUMPY ARRAY
            Give the dof vector corresponding to a unitary rigid body 
            translation in direction x.
        ty : 1D NUMPY ARRAY
            Give the dof vector corresponding to a unitary rigid body 
            translation in direction y.
        rz : 1D NUMPY ARRAY
            Give the dof vector corresponding to a infinitesimal unitary rigid
            body rotation around direction z.

        """
        tx = np.zeros(self.ndof)
        tx[self.conn[:,0]]=1
        ty = np.zeros(self.ndof)
        ty[self.conn[:,1]]=1
        v = self.n-np.mean(self.n,axis=0)
        v = np.c_[-v[:,1],v[:,0]] / np.max(np.linalg.norm(v,axis=1))
        rz = np.zeros(self.ndof)
        rz[self.conn]=v
        return tx, ty, rz