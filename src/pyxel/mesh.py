# -*- coding: utf-8 -*-
"""
Finite Element Digital Image Correlation method

@author: JC Passieux, INSA Toulouse, 2023

pyxel

PYthon library for eXperimental mechanics using Finite ELements

"""

import os
import numpy as np
import scipy as sp
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import art3d
import matplotlib.pyplot as plt
import matplotlib.collections as cols
import matplotlib.animation as animation
# from numba import njit # uncomment for just in time compilation
from .utils import meshgrid, isInBox
from .vtktools import PVDFile
from .material import *
import meshio

from numpy.polynomial.legendre import leggauss 

# %%
def ElTypes():
    """
    Returns a dictionnary of GMSH element types which some of them are used
    in the library.
    """
    return {
      1: "2-node line.",
      2: "3-node triangle.",
      3: "4-node quadrangle.",
      4: "4-node tetrahedron.",
      5: "8-node hexahedron.",
      6: "6-node prism.",
      7: "5-node pyramid.",
      8: "3-node second order line (2 nodes associated with the vertices and"
      + " 1 with the edge).",
      9: "6-node second order triangle (3 nodes associated with the vertices"
      + " and 3 with the edges).",
      10: "9-node second order quadrangle (4 nodes associated with the"
      + " vertices, 4 with the edges and 1 with the face).",
      11: "10-node second order tetrahedron (4 nodes associated with the"
      + " vertices and 6 with the edges).",
      12: "27-node second order hexahedron (8 nodes associated with the verti"
      + "ces, 12 with the edges, 6 with the faces and 1 with the volume).",
      13: "18-node second order prism (6 nodes associated with the vertices,"
      + " 9 with the edges and 3 with the quadrangular faces).",
      14: "14-node second order pyramid (5 nodes associated with the vertices,"
      + " 8 with the edges and 1 with the quadrangular face).",
      15: "1-node point.",
      16: "8-node second order quadrangle (4 nodes associated with the vertice"
      + "s and 4 with the edges).",
      17: "20-node second order hexahedron (8 nodes associated with the vertic"
      + "es and 12 with the edges).",
      18: "15-node second order prism (6 nodes associated with the vertices an"
      + "d 9 with the edges).",
      19: "13-node second order pyramid (5 nodes associated with the vertices"
      + " and 8 with the edges).",
      20: "9-node third order incomplete triangle (3 nodes associated with the"
      + " vertices, 6 with the edges)",
      21: "10-node third order triangle (3 nodes associated with the vertices,"
      + " 6 with the edges, 1 with the face)",
      22: "12-node fourth order incomplete triangle (3 nodes associated with"
      + " the vertices, 9 with the edges)",
      23: "15-node fourth order triangle (3 nodes associated with the vertices"
      + ", 9 with the edges, 3 with the face)",
      24: "15-node fifth order incomplete triangle (3 nodes associated with th"
      + "e vertices, 12 with the edges)",
      25: "21-node fifth order complete triangle (3 nodes associated with the"
      + " vertices, 12 with the edges, 6 with the face)",
      26: "4-node third order edge (2 nodes associated with the vertices, 2"
      + " internal to the edge)",
      27: "5-node fourth order edge (2 nodes associated with the vertices, 3"
      + " internal to the edge)",
      28: "6-node fifth order edge (2 nodes associated with the vertices, 4"
      + " internal to the edge)",
      29: "20-node third order tetrahedron (4 nodes associated with the vertic"
      + "es, 12 with the edges, 4 with the faces)",
      30: "35-node fourth order tetrahedron (4 nodes associated with the verti"
      + "ces, 18 with the edges, 12 with the faces, 1 in the volume)",
      31: "56-node fifth order tetrahedron (4 nodes associated with the vertic"
      + "es, 24 with the edges, 24 with the faces, 4 in the volume)",
      92: "64-node third order hexahedron (8 nodes associated with the vertice"
      + "s, 24 with the edges, 24 with the faces, 8 in the volume)",
      93: "125-node fourth order hexahedron (8 nodes associated with the verti"
      + "ces, 36 with the edges, 54 with the faces, 27 in the volume)",
      }


eltype_n2s = {1: "line",
              2: "triangle",
              3: "quad",
              4: "tetra",
              5: "hexahedron",
              6: "wedge",
              7: "pyramid",
              8: "line3",
              9: "triangle6",
              10: "quad9",
              11: "tetra10",
              12: "hexahedron27",
              14: "pyramid14",
              15: "vertex",
              16: "quad8",
              17: "hexahedron20",
              18: "wedge15",
              19: "pyramid13"}

eltype_s2n = {}
for jn in eltype_n2s.keys():
    eltype_s2n[eltype_n2s[jn]] = jn


def ReadMesh(fn, dim=2):
    mesh = meshio.read(fn)
    if mesh.points.shape[1] > dim:       # too much node coordinate
        # Remove coordinate with minimal std.
        rmdim = np.argmin(np.std(mesh.points, axis=0))
        n = np.delete(mesh.points, rmdim, 1)
    elif mesh.points.shape[1] < dim:     # not enough node coordinates
        n = np.hstack((mesh.points, np.zeros((len(mesh.points), 1))))
    else:
        n = mesh.points
    e = dict()
    for et in mesh.cells_dict.keys():
        e[eltype_s2n[et]] = mesh.cells_dict[et]
    m = Mesh(e, n, dim)
    m.point_data = mesh.point_data
    m.cell_data = mesh.cell_data
    return m


# %%
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
    """ Inverse the Finite Element mapping in order to map the coordinates
    of any physical points (xpix, ypix) to their corresponding position in
    the parent element (xg, yg)."""
    _, _, _, N, dN_xi, dN_eta = ShapeFunctions(eltype)
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
        invJ = np.array([dyds / detJ, -dxds / detJ,
                         -dydr / detJ, dxdr / detJ]).T
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


def SubQuaIso(nx, ny):
    """Subdivide a Quadrilateral to build the quadrature rule"""
    px = 1.0 / nx
    xi = np.linspace(px - 1, 1 - px, int(nx))
    py = 1.0 / ny
    yi = np.linspace(py - 1, 1 - py, int(ny))
    xg, yg = meshgrid(xi, yi)
    wg = 4.0 / (nx * ny)
    return xg.ravel(), yg.ravel(), wg


def SubTriIso(nx, ny):
    """ Subdivide a Triangle to build the quadrature rule (possibly
    heterogeneous subdivision)
    M1M2 is divided in nx and M1M3 in ny, the meshing being heterogeneous, we
    end up with trapezes on the side of hypothenuse, the remainder are
    rectangles """
    px = 1 / nx
    py = 1 / ny
    if nx > ny:
        xg = np.zeros(int(np.sum(np.floor(ny * (1 - np.arange(1, nx + 1)
                                                / nx))) + nx))
        yg = xg.copy()
        wg = xg.copy()
        j = 1
        for i in range(1, nx + 1):
            niy = int(ny * (1 - i / nx))  # nb of full rect in vertical dir
            v = np.array([[(i - 1) * px, niy * py],
                          [(i - 1) * px, 1 - (i - 1) * px],
                          [i * px, niy * py],
                          [i * px, 1 - i * px]])
            neww = px * (v[3, 1] - v[0, 1]) + px * (v[1, 1] - v[3, 1]) / 2
            newx = ((v[3, 1] - v[0, 1]) * (v[2, 0] + v[0, 0]) / 2
                    + (v[1, 1] - v[3, 1]) / 2 * (v[0, 0] + px / 3)) * px / neww
            newy = ((v[3, 1] - v[0, 1]) * (v[0, 1] + v[3, 1]) / 2
                    + (v[1, 1] - v[3, 1]) / 2
                    * (v[3, 1] + (v[1, 1] - v[3, 1]) / 3)) * px / neww
            xg[(j - 1):j + niy] = np.append((px / 2 + (i - 1) * px)
                                            * np.ones(niy), newx)
            yg[(j - 1):j + niy] = np.append(py / 2 + np.arange(niy)
                                            * py, newy)
            wg[(j - 1):j + niy] = np.append(px * py * np.ones(niy), neww)
            j = j + niy + 1
    else:
        xg = np.zeros(int(np.sum(np.floor(nx * (1 - np.arange(1, ny + 1)
                                                / ny))) + ny))
        yg = xg.copy()
        wg = xg.copy()
        j = 1
        for i in range(1, ny + 1):
            nix = int(nx * (1 - i / ny))  # number of full rect in horizontal
            v = np.array([[nix * px, (i - 1) * py],
                          [nix * px, i * py],
                          [1 - (i - 1) * py, (i - 1) * py],
                          [1 - i * py, i * py]])
            neww = py * (v[3, 0] - v[0, 0]) + py * (v[2, 0] - v[3, 0]) / 2
            newx = ((v[3, 0] - v[0, 0]) * (v[3, 0] + v[0, 0]) / 2
                    + (v[2, 0] - v[3, 0]) / 2
                    * (v[3, 0] + (v[2, 0] - v[3, 0]) / 3)) * py / neww
            newy = ((v[3, 0] - v[0, 0]) * (v[1, 1] + v[0, 1]) / 2
                    + (v[2, 0] - v[3, 0]) / 2 * (v[0, 1] + py / 3)) * py / neww
            xg[(j - 1):j + nix] = np.append(px / 2 + np.arange(nix) * px, newx)
            yg[(j - 1):j + nix] = np.append((py / 2 + (i - 1) * py)
                                            * np.ones(nix), newy)
            wg[(j - 1):j + nix] = np.append(px * py * np.ones(nix), neww)
            j = j + nix + 1
    return xg, yg, wg


def SubTriIso2(nx, ny=None):
    """Subdivide a Triangle to build the quadrature rule (homogeneous
    subdivision, faster)
    M1M2 and M1M3 are divided into (nx+ny)/2, the meshing being homogeneous, we
    end up with triangles on the side of hypothenuse, the remainder
    are rectangles """
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
    xg[n * (n - 1) // 2:] = 1 - yi
    yg[n * (n - 1) // 2:] = yi - pxy * 1 / 3
    wg[n * (n - 1) // 2:] = pxy ** 2 / 2
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

def SubTetIso(n):
    """Subdivide a Tetraedron to build the quadrature rule
    (homogeneous subdivision)"""
    # Method 1:
    # n = max(n, 2)
    # dx = 1 / n
    # x = dx/2 + np.arange(n)*dx
    # y = dx/2 + np.arange(n)*dx
    # z = dx/2 + np.arange(n)*dx
    # X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    # iTetra = (Z <= 1 - X - Y)
    # xg = X[iTetra]
    # yg = Y[iTetra]
    # zg = Z[iTetra]
    # wg = np.ones(xg.size)/(6*xg.size)

    # Method 2: Using a FE mesh
    if n == 1:
        xg = np.array([1/4])
        yg = np.array([1/4])
        zg = np.array([1/4])
        wg = np.array([1/6])
    else:
        import gmsh as gmsh
        gmsh.initialize()
        gmsh.model.add("")
        gmsh.model.geo.addPoint(0, 0, 0, 1/(n-1), 1)
        gmsh.model.geo.addPoint(1, 0, 0, 1/(n-1), 2)
        gmsh.model.geo.addPoint(0, 1, 0, 1/(n-1), 3)
        gmsh.model.geo.addPoint(0, 0, 1, 1/(n-1), 4)
        gmsh.model.geo.addLine(1, 2, tag=5)
        gmsh.model.geo.addLine(2, 3, tag=6)
        gmsh.model.geo.addLine(3, 1, tag=7)
        gmsh.model.geo.addLine(1, 4, tag=8)
        gmsh.model.geo.addLine(2, 4, tag=9)
        gmsh.model.geo.addLine(3, 4, tag=10)
        gmsh.model.geo.addCurveLoop([5, 6, 7], 1)
        gmsh.model.geo.addCurveLoop([5, 9, -8], 2)
        gmsh.model.geo.addCurveLoop([6, 10, -9], 3)
        gmsh.model.geo.addCurveLoop([7, 8, -10], 4)
        gmsh.model.geo.addPlaneSurface([1], 1)
        gmsh.model.geo.addPlaneSurface([2], 2)
        gmsh.model.geo.addPlaneSurface([3], 3)
        gmsh.model.geo.addPlaneSurface([4], 4)
        gmsh.model.geo.addSurfaceLoop([1, 2, 3, 4], 12)
        gmsh.model.geo.addVolume([12, ], 13)
        gmsh.model.geo.synchronize()
        # gmsh.option.setNumber('Mesh.RecombineAll', 1)
        # gmsh.option.setNumber('Mesh.RecombinationAlgorithm', 1)
        # gmsh.option.setNumber('Mesh.Recombine3DLevel', 2)
        gmsh.option.setNumber('General.Verbosity', 1)
        gmsh.model.mesh.generate(3)
        # gmsh.fltk.run()
        # nums, nodes, e = gmsh.model.mesh.getNodes()
        # nodes = nodes.reshape((len(nums), 3))
        # gmsh.model.mesh.getElementTypes()
        # nums, els = gmsh.model.mesh.getElementsByType(4)
        # els = np.reshape(els.astype(int) - 1, (len(nums), 4))
        # mtet = px.Mesh({4: els}, nodes, 3)
        # mtet.Plot()
        # x = nodes[els, 0]
        # y = nodes[els, 0]
        # z = nodes[els, 0]
        # x = np.mean(nodes[els, 0], axis=1)
        # y = np.mean(nodes[els, 0], axis=1)
        # z = np.mean(nodes[els, 0], axis=1)
        X = gmsh.model.mesh.getBarycenters(4, 13, False, True)
        X = X.reshape(len(X)//3, 3)
        xg = X[:, 0]
        yg = X[:, 1]
        zg = X[:, 2]
        wg = np.ones(len(xg)) / (6*len(xg))
        # gmsh.finalize()
    print('Number of integration points in elems %3d' % len(xg))
    return xg, yg, zg, wg


def SubHexIso(n):
    """Subdivide a Hexaedron to build the quadrature rule
    (homogeneous subdivision)"""
    dx = 2 / n                # (subdiving the domain [-1,1])
    x = - 1 + dx/2 + np.arange(n)*dx
    y = - 1 + dx/2 + np.arange(n)*dx
    z = - 1 + dx/2 + np.arange(n)*dx
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    xg = X.ravel()
    yg = Y.ravel()
    zg = Z.ravel()
    wg = np.ones(xg.size) * 8 / xg.size
    return xg, yg, zg, wg


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
    gmsh.model.geo.addCurveLoop([1, 2, 3], 1)
    gmsh.model.geo.addPlaneSurface([1], 1)
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.setRecombine(2, 1)
    gmsh.model.mesh.generate(2)
    nums, nodes, e = gmsh.model.mesh.getNodes()
    nodes = nodes.reshape((len(nums), 3))
    nodes = nodes[:, :-1]
    c = np.empty((0, 2))
    a = []
    if 2 in gmsh.model.mesh.getElementTypes():
        nums, els = gmsh.model.mesh.getElementsByType(2)
        nnd = len(els) // len(nums)
        elems = els.reshape((len(nums), nnd)).astype(int) - 1
        a = np.append(a, 0.5 * abs(
            (nodes[elems[:, 1], 0] - nodes[elems[:, 0], 0])
            * (nodes[elems[:, 2], 1] - nodes[elems[:, 0], 1])
            - (nodes[elems[:, 2], 0] - nodes[elems[:, 0], 0])
            * (nodes[elems[:, 1], 1] - nodes[elems[:, 0], 1])))
        c = np.vstack((c, (nodes[elems[:, 0]] + nodes[elems[:, 1]]
                           + nodes[elems[:, 2]]) / 3))
    if 3 in gmsh.model.mesh.getElementTypes():
        nums, els = gmsh.model.mesh.getElementsByType(3)
        nnd = len(els) // len(nums)
        elems = els.reshape((len(nums), nnd)).astype(int) - 1
        a = np.append(a, 0.5 * (
            abs((nodes[elems[:, 0], 0] - nodes[elems[:, 2], 0])
                * (nodes[elems[:, 1], 1] - nodes[elems[:, 3], 1]))
            + abs((nodes[elems[:, 1], 0] - nodes[elems[:, 3], 0])
                  * (nodes[elems[:, 0], 1] - nodes[elems[:, 2], 1]))))
        c = np.vstack((c, (nodes[elems[:, 0]] + nodes[elems[:, 1]]
                           + nodes[elems[:, 2]] + nodes[elems[:, 3]]) / 4))
    # m=px.Gmsh2Mesh(gmsh)
    # m.Plot()
    gmsh.finalize()
    return c[:, 0], c[:, 1], a


# %%
def AddChildElem(child_list, new_child, sorted_child_list):
    sorted_new_child = tuple(np.sort(new_child))
    if sorted_new_child in child_list.keys():
        child_list[sorted_new_child] += 1
    else:
        child_list[sorted_new_child] = 1
        sorted_child_list[sorted_new_child] = new_child
    return child_list, sorted_child_list

# %%
def ShapeFunctions(eltype):
    """For any type of 2D elements, gives the quadrature rule and
    the shape functions and their derivative"""
    if eltype == 1:
        """
        #############
            seg2
        #############
        """
        def N(x):
            return np.concatenate(
                (0.5 * (1 - x), 0.5*(x + 1))).reshape((2, len(x))).T

        def dN_xi(x):
            return np.concatenate(
                (-0.5 + 0 * x, 0.5 + 0 * x)).reshape((2, len(x))).T

        # def dN_eta(x):
        #     return False
        xg, wg = leggauss(1)
        return xg, wg, N, dN_xi
    elif eltype == 8:
        """
        #############
            seg3
        #############
        """
        def N(x):
            return np.concatenate(
                ((x**2 - x) * 0.5, 1 - x**2, (x**2 + x) * 0.5)
                ).reshape((3, len(x))).T

        def dN_xi(x):
            return np.concatenate(
                (x - 0.5, -2 * x, x + 1)).reshape((3, len(x))).T

        xg, wg = leggauss(2)
        return xg, wg, N, dN_xi
    elif eltype == 2:
        """
        #############
            tri3
        #############
        """
        def N(x, y):
            return np.concatenate(
                (1 - x - y, x, y)).reshape((3, len(x))).T

        def dN_xi(x, y):
            return np.concatenate(
                (-1.0 + 0 * x, 1.0 + 0 * x, 0.0 * x)).reshape((3, len(x))).T

        def dN_eta(x, y):
            return np.concatenate(
                (-1.0 + 0 * x, 0.0 * x, 1.0 + 0 * x)).reshape((3, len(x))).T

        # xg = np.array([1. / 6, 2. / 3, 1. / 6])
        # yg = np.array([1. / 6, 1. / 6, 2. / 3])
        # wg = 1. / 6 * np.ones(3)
        xg = np.array([1. / 3])
        yg = np.array([1. / 3])
        wg = np.array([0.5])
        return xg, yg, wg, N, dN_xi, dN_eta
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
                                          (1 - x) * (1 + y))
                                         ).reshape((4, len(x))).T

        def dN_xi(x, y):
            return 0.25 * np.concatenate(
                (y - 1, 1 - y, 1 + y, -1 - y)).reshape((4, len(x))).T

        def dN_eta(x, y):
            return 0.25 * np.concatenate(
                (x - 1, -1 - x, 1 + x, 1 - x)).reshape((4, len(x))).T

        
        # deg = 1  # reduced integration 1 gp
        deg = 2  # full integration 4 gp
        xg, wg = leggauss(deg)
        xg, yg = np.meshgrid(xg, xg)
        xg = xg.ravel()
        yg = yg.ravel()
        wg = np.kron(wg, wg)
        return xg, yg, wg, N, dN_xi, dN_eta
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
                 4 * y * (1 - x - y))).reshape((6, len(x))).T

        def dN_xi(x, y):
            return np.concatenate(
                (4 * x + 4 * y - 3, 4 * x - 1, x * 0,
                 4 * (1 - 2 * x - y), 4 * y, -4 * y)
                ).reshape((6, len(x))).T

        def dN_eta(x, y):
            return np.concatenate(
                (4 * x + 4 * y - 3, x * 0, 4 * y - 1,
                 -4 * x, 4 * x, 4 * (1 - x - 2 * y))
                ).reshape((6, len(x))).T

        # quadrature using 3 gp
        xg = np.array([1. / 6, 2. / 3, 1. / 6])
        yg = np.array([1. / 6, 1. / 6, 2. / 3])
        wg = 1. / 6 * np.ones(3)
        # quadrature using 6 gp
        # a = 0.445948490915965
        # b = 0.091576213509771
        # xg = np.array([a, 1 - 2 * a, a, b, 1 - 2 * b, b])
        # yg = np.array([a, a, 1 - 2 * a, b, b, 1 - 2 * b])
        # a = 0.111690794839005
        # b = 0.054975871827661
        # wg = np.array([a, a, a, b, b, b])
        return xg, yg, wg, N, dN_xi, dN_eta
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
               ).reshape((9, len(x))).T

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
               ).reshape((9, len(x))).T

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
            ).reshape((9, len(x))).T

        deg = 3  # 9 gp
        xg, wg = leggauss(deg)
        xg, yg = np.meshgrid(xg, xg)
        xg = xg.ravel()
        yg = yg.ravel()
        wg = np.kron(wg, wg)
        return xg, yg, wg, N, dN_xi, dN_eta
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
            ).reshape((8, len(x))).T

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
            ).reshape((8, len(x))).T

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
            ).reshape((8, len(x))).T

        # deg = 2  # 4 p
        deg = 3  # 9 p
        xg, wg = leggauss(deg)
        xg, yg = np.meshgrid(xg, xg)
        xg = xg.ravel()
        yg = yg.ravel()
        wg = np.kron(wg, wg)
        return xg, yg, wg, N, dN_xi, dN_eta
    elif eltype == 5:
        """
        #############
            Hex8
        #############
        """
        def N(x, y, z):
            return 0.125 * np.concatenate(((1-x) * (1-y) * (1-z),
                                           (1+x) * (1-y) * (1-z),
                                           (1+x) * (1+y) * (1-z),
                                           (1-x) * (1+y) * (1-z),
                                           (1-x) * (1-y) * (1+z),
                                           (1+x) * (1-y) * (1+z),
                                           (1+x) * (1+y) * (1+z),
                                           (1-x) * (1+y) * (1+z))
                                          ).reshape((8, len(x))).T

        def dN_xi(x, y, z):
            return 0.125 * np.concatenate((-(1-y)*(1-z),  (1-y)*(1-z),
                                           (1+y)*(1-z), -(1+y)*(1-z),
                                           -(1-y)*(1+z),  (1-y)*(1+z),
                                           (1+y)*(1+z), -(1+y)*(1+z))
                                          ).reshape((8, len(x))).T

        def dN_eta(x, y, z):
            return 0.125 * np.concatenate((-(1-x)*(1-z), -(1+x)*(1-z),
                                           (1+x)*(1-z),  (1-x)*(1-z),
                                           -(1-x)*(1+z), -(1+x)*(1+z),
                                           (1+x)*(1+z),  (1-x)*(1+z))
                                          ).reshape((8, len(x))).T

        def dN_zeta(x, y, z):
            return 0.125 * np.concatenate((-(1-x)*(1-y), -(1+x)*(1-y),
                                          -(1+x)*(1+y), -(1-x)*(1+y),
                                           (1-x)*(1-y),  (1+x)*(1-y),
                                           (1+x)*(1+y),  (1-x)*(1+y))
                                          ).reshape((8, len(x))).T
        deg = 2
        xg, wg = leggauss(deg)
        xg, yg, zg = np.meshgrid(xg, xg, xg)
        xg = xg.ravel()
        yg = yg.ravel()
        zg = zg.ravel()
        wg = np.kron(np.kron(wg, wg), wg)  # WORKS WITH DEG 2, TO BE CHECKED!
        return xg, yg, zg, wg, N, dN_xi, dN_eta, dN_zeta
    elif eltype == 4:
        """
        #############
            Tet4
        #############
        """
        def N(x, y, z):
            return np.concatenate((1-x-y-z, x, y, z)).reshape((4, len(x))).T

        def dN_xi(x, y, z):
            zer = np.zeros(len(x))
            one = np.ones(len(x))
            return np.concatenate((-one, one, zer, zer)).reshape((4, len(x))).T

        def dN_eta(x, y, z):
            zer = np.zeros(len(x))
            one = np.ones(len(x))
            return np.concatenate((-one, zer, one, zer)).reshape((4, len(x))).T

        def dN_zeta(x, y, z):
            zer = np.zeros(len(x))
            one = np.ones(len(x))
            return np.concatenate((-one, zer, zer, one)).reshape((4, len(x))).T
        xg = 0.25 * np.array([1])
        yg = 0.25 * np.array([1])
        zg = 0.25 * np.array([1])
        wg = np.array([0.1666666666666666])
        return xg, yg, zg, wg, N, dN_xi, dN_eta, dN_zeta
    elif eltype == 11:
        """
        #############
            Tet10
        #############
        """
        def N(x, y, z):
            return np.concatenate(((1-x-y-z)*(1-2*x-2*y-2*z),
                                   x*(2*x-1),
                                   y*(2*y-1),
                                   z*(2*z-1),
                                   4*z*(1-x-y-z),
                                   4*x*y,
                                   4*y*z,
                                   4*y*(1-x-y-z),
                                   4*x*z,
                                   4*x*(1-x-y-z),
                                   )).reshape((10, len(x))).T

        def dN_xi(x, y, z):
            zer = np.zeros(len(x))
            return np.concatenate((-3+4*x+4*y+4*z,
                                   4*x-1,
                                   zer,
                                   zer,
                                   -4*z,
                                   4*y,
                                   zer,
                                   -4*y,
                                   4*z,
                                   4*(1-2*x-y-z),
                                   )).reshape((10, len(x))).T

        def dN_eta(x, y, z):
            zer = np.zeros(len(x))
            return np.concatenate((-3+4*x+4*y+4*z,
                                   zer,
                                   4*y-1,
                                   zer,
                                   -4*z,
                                   4*x,
                                   4*z,
                                   4*(1-x-2*y-z),
                                   zer,
                                   -4*x,
                                   )).reshape((10, len(x))).T

        def dN_zeta(x, y, z):
            zer = np.zeros(len(x))
            return np.concatenate((-3+4*x+4*y+4*z,
                                   zer,
                                   zer,
                                   4*z-1,
                                   4*(1-x-y-2*z),
                                   zer,
                                   4*y,
                                   -4*y,
                                   4*x,
                                   -4*x,
                                   )).reshape((10, len(x))).T
        # 4 Gauss points
        c = 0.585410196624968
        b = 0.138196601125010
        xg = np.array([b, b, b, c])
        yg = np.array([b, b, c, b])
        zg = np.array([b, c, b, b])
        wg = np.array([1, 1, 1, 1]) * 0.25

        # 15 Gauss points
        # a = 0.25
        # b1 = (7+np.sqrt(15))/34
        # b2 = (7-np.sqrt(15))/34
        # c1 = (13-3*np.sqrt(15))/34
        # c2 = (13+3*np.sqrt(15))/34
        # d = (5-np.sqrt(15))/20
        # e = (5+np.sqrt(15))/20
        # w1 = 8/405
        # w2 = (2665-14*np.sqrt(15))/226800
        # w3 = (2665+14*np.sqrt(15))/226800
        # w4 = 5/567
        # xg = np.array([a,b1,b1,b1,c1,b2,b2,b2,c2,d,d,e,d,e,e])
        # yg = np.array([a,b1,b1,c1,b1,b2,b2,c2,b2,d,e,d,e,d,e])
        # zg = np.array([a,b1,c1,b1,b1,b2,c2,b2,b2,e,d,d,e,e,d])
        # wg = np.array([w1,w2,w2,w2,w2,w3,w3,w3,w3,w4,w4,w4,w4,w4,w4])
        return xg, yg, zg, wg, N, dN_xi, dN_eta, dN_zeta
    elif eltype == 17:
        """
        #############
            Hex20
        #############
        """
        # import sympy as sp
        # x, y, z = sp.symbols('x, y, z')
        # f = [(1-x)*(1-y)*(1-z)*(-2-x-y-z),
        #                 (1+x)*(1-y)*(1-z)*(-2+x-y-z),
        #                 (1+x)*(1+y)*(1-z)*(-2+x+y-z),
        #                 (1-x)*(1+y)*(1-z)*(-2-x+y-z),
        #                 (1-x)*(1-y)*(1+z)*(-2-x-y+z),
        #                 (1+x)*(1-y)*(1+z)*(-2+x-y+z),
        #                 (1+x)*(1+y)*(1+z)*(-2+x+y+z),
        #                 (1-x)*(1+y)*(1+z)*(-2-x+y+z),
        #                 2*(1-x**2)*(1-y)*(1-z),
        #                 2*(1+x)*(1-y**2)*(1-z),
        #                 2*(1-x**2)*(1+y)*(1-z),
        #                 2*(1-x)*(1-y**2)*(1-z),
        #                 2*(1-x**2)*(1-y)*(1+z),
        #                 2*(1+x)*(1-y**2)*(1+z),
        #                 2*(1-x**2)*(1+y)*(1+z),
        #                 2*(1-x)*(1-y**2)*(1+z),
        #                 2*(1-x)*(1-y)*(1-z**2),
        #                 2*(1+x)*(1-y)*(1-z**2),
        #                 2*(1+x)*(1+y)*(1-z**2),
        #                 2*(1-x)*(1+y)*(1-z**2)]
        # for fi in f:
        #     print(sp.pycode(sp.diff(fi, x)))
        # for fi in f:
        #     print(sp.pycode(sp.diff(fi, y)))
        # for fi in f:
        #     print(sp.pycode(sp.diff(fi, z)))
        def N(x, y, z):
            return 0.125 * np.concatenate(((1-x)*(1-y)*(1-z)*(-2-x-y-z),
                                           (1+x)*(1-y)*(1-z)*(-2+x-y-z),
                                           (1+x)*(1+y)*(1-z)*(-2+x+y-z),
                                           (1-x)*(1+y)*(1-z)*(-2-x+y-z),
                                           (1-x)*(1-y)*(1+z)*(-2-x-y+z),
                                           (1+x)*(1-y)*(1+z)*(-2+x-y+z),
                                           (1+x)*(1+y)*(1+z)*(-2+x+y+z),
                                           (1-x)*(1+y)*(1+z)*(-2-x+y+z),
                                           2*(1-x**2)*(1-y)*(1-z),
                                           2*(1+x)*(1-y**2)*(1-z),
                                           2*(1-x**2)*(1+y)*(1-z),
                                           2*(1-x)*(1-y**2)*(1-z),
                                           2*(1-x**2)*(1-y)*(1+z),
                                           2*(1+x)*(1-y**2)*(1+z),
                                           2*(1-x**2)*(1+y)*(1+z),
                                           2*(1-x)*(1-y**2)*(1+z),
                                           2*(1-x)*(1-y)*(1-z**2),
                                           2*(1+x)*(1-y)*(1-z**2),
                                           2*(1+x)*(1+y)*(1-z**2),
                                           2*(1-x)*(1+y)*(1-z**2))
                                          ).reshape((20, len(x))).T

        def dN_xi(x, y, z):
            return 0.125 * np.concatenate((
                -(1 - x)*(1 - y)*(1 - z) - (1 - y)*(1 - z)*(-x - y - z - 2),
                (1 - y)*(1 - z)*(x + 1) + (1 - y)*(1 - z)*(x - y - z - 2),
                (1 - z)*(x + 1)*(y + 1) + (1 - z)*(y + 1)*(x + y - z - 2),
                -(1 - x)*(1 - z)*(y + 1) - (1 - z)*(y + 1)*(-x + y - z - 2),
                -(1 - x)*(1 - y)*(z + 1) - (1 - y)*(z + 1)*(-x - y + z - 2),
                (1 - y)*(x + 1)*(z + 1) + (1 - y)*(z + 1)*(x - y + z - 2),
                (x + 1)*(y + 1)*(z + 1) + (y + 1)*(z + 1)*(x + y + z - 2),
                -(1 - x)*(y + 1)*(z + 1) - (y + 1)*(z + 1)*(-x + y + z - 2),
                -4*x*(1 - y)*(1 - z),
                2*(1 - y**2)*(1 - z),
                -4*x*(1 - z)*(y + 1),
                -2*(1 - y**2)*(1 - z),
                -4*x*(1 - y)*(z + 1),
                2*(1 - y**2)*(z + 1),
                -4*x*(y + 1)*(z + 1),
                -2*(1 - y**2)*(z + 1),
                -2*(1 - y)*(1 - z**2),
                2*(1 - y)*(1 - z**2),
                2*(1 - z**2)*(y + 1),
                -2*(1 - z**2)*(y + 1)
                )).reshape((20, len(x))).T

        def dN_eta(x, y, z):
            return 0.125 * np.concatenate((
                -(1 - x)*(1 - y)*(1 - z) - (1 - x)*(1 - z)*(-x - y - z - 2),
                -(1 - y)*(1 - z)*(x + 1) - (1 - z)*(x + 1)*(x - y - z - 2),
                (1 - z)*(x + 1)*(y + 1) + (1 - z)*(x + 1)*(x + y - z - 2),
                (1 - x)*(1 - z)*(y + 1) + (1 - x)*(1 - z)*(-x + y - z - 2),
                -(1 - x)*(1 - y)*(z + 1) - (1 - x)*(z + 1)*(-x - y + z - 2),
                -(1 - y)*(x + 1)*(z + 1) - (x + 1)*(z + 1)*(x - y + z - 2),
                (x + 1)*(y + 1)*(z + 1) + (x + 1)*(z + 1)*(x + y + z - 2),
                (1 - x)*(y + 1)*(z + 1) + (1 - x)*(z + 1)*(-x + y + z - 2),
                -(1 - z)*(2 - 2*x**2),
                -2*y*(1 - z)*(2*x + 2),
                (1 - z)*(2 - 2*x**2),
                -2*y*(1 - z)*(2 - 2*x),
                -(2 - 2*x**2)*(z + 1),
                -2*y*(2*x + 2)*(z + 1),
                (2 - 2*x**2)*(z + 1),
                -2*y*(2 - 2*x)*(z + 1),
                -(1 - z**2)*(2 - 2*x),
                -(1 - z**2)*(2*x + 2),
                (1 - z**2)*(2*x + 2),
                (1 - z**2)*(2 - 2*x)
                )).reshape((20, len(x))).T

        def dN_zeta(x, y, z):
            return 0.125 * np.concatenate((
                -(1 - x)*(1 - y)*(1 - z) - (1 - x)*(1 - y)*(-x - y - z - 2),
                -(1 - y)*(1 - z)*(x + 1) - (1 - y)*(x + 1)*(x - y - z - 2),
                -(1 - z)*(x + 1)*(y + 1) - (x + 1)*(y + 1)*(x + y - z - 2),
                -(1 - x)*(1 - z)*(y + 1) - (1 - x)*(y + 1)*(-x + y - z - 2),
                (1 - x)*(1 - y)*(z + 1) + (1 - x)*(1 - y)*(-x - y + z - 2),
                (1 - y)*(x + 1)*(z + 1) + (1 - y)*(x + 1)*(x - y + z - 2),
                (x + 1)*(y + 1)*(z + 1) + (x + 1)*(y + 1)*(x + y + z - 2),
                (1 - x)*(y + 1)*(z + 1) + (1 - x)*(y + 1)*(-x + y + z - 2),
                -(1 - y)*(2 - 2*x**2),
                -(1 - y**2)*(2*x + 2),
                -(2 - 2*x**2)*(y + 1),
                -(1 - y**2)*(2 - 2*x),
                (1 - y)*(2 - 2*x**2),
                (1 - y**2)*(2*x + 2),
                (2 - 2*x**2)*(y + 1),
                (1 - y**2)*(2 - 2*x),
                -2*z*(1 - y)*(2 - 2*x),
                -2*z*(1 - y)*(2*x + 2),
                -2*z*(2*x + 2)*(y + 1),
                -2*z*(2 - 2*x)*(y + 1),
                )).reshape((20, len(x))).T
        # 8 gauss points
        xg = np.sqrt(3) / 3 * np.array([-1, 1, -1, 1, -1, 1, -1, 1])
        yg = np.sqrt(3) / 3 * np.array([-1, -1, 1, 1, -1, -1, 1, 1])
        zg = np.sqrt(3) / 3 * np.array([-1, -1, -1, -1, 1, 1, 1, 1])
        wg = np.ones(8)

        # 27 gauss points
        # xi = np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)])
        # xg, yg, zg = np.meshgrid(xi, xi, xi)
        # wi = np.array([0.5555555555555556,
        #                0.8888888888888888,
        #                0.5555555555555556])
        # wx, wy, wz = np.meshgrid(wi, wi, wi)
        # wg = wy*wx*wz
        return xg.ravel(), yg.ravel(), zg.ravel(), wg.ravel(), N, dN_xi, dN_eta, dN_zeta


# %%
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

    def Connectivity(self, order='C'):
        """
        Associate DOFs to each node

        Parameters
        ----------
        order : STRING, optional
            DESCRIPTION. The default is 'C'.
            'C' ordered by component.
            'N' ordered by node.

        """
        print("Connectivity.")
        used_nodes = np.zeros(0, dtype=int)
        for je in self.e.keys():
            used_nodes = np.unique(np.append(used_nodes, self.e[je].ravel()))
        nn = len(used_nodes)
        self.ndof = nn * self.dim
        if order == 'C':
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
        elif order == 'N':
            self.conn = -np.ones((self.n.shape[0], self.dim), dtype=int)
            self.conn[used_nodes] = np.arange(self.ndof).reshape(nn, self.dim)

    def DOF2Nodes(self, Udof, fillzero=False):
        """
        Switch from DOF vector to a table similar to the node coordinate table
        PYXEL.MESH.N

        Parameters
        ----------
        Udof : NUMPY.ARRAY
            displacement DOF vector:
            Udof=[u1, u2, ..., uN, v1, v2, ... vN]
        fillzero : TYPE, optional
            if in 2D, fill add a full zero column to get a 3D displacement
            (useful for VTK for instance)

        Returns
        -------
        Unodes : NUMPY.ARRAY
            Table of the same size as PYXEL.MESH.N
            Unodes = [[ui, vi, [0]],...]

        """
        conn = self.conn.copy()
        not_used, = np.where(self.conn[:, 0] < 0)
        conn[not_used, 0] = np.max(conn[:, 0])
        conn[not_used, 1] = np.max(conn[:, 1])
        if self.dim == 2:
            Unodes = Udof[conn]
            if fillzero:
                Unodes = np.hstack((Unodes, np.zeros((len(self.n), 1))))
        else:
            conn[not_used, 2] = np.max(conn[:, 2])
            Unodes = Udof[conn]
        return Unodes

    def Nodes2DOF(self, Unodes):
        """
        Switch from a table similar to the node coordinate table PYXEL.MESH.N
        to a DOF vector.

        Parameters
        ----------
        Unodes : NUMPY.ARRAY
            Table of the same size as PYXEL.MESH.N
            Unodes = [[ui, vi, [0]],...]

        Returns
        -------
        Udof : NUMPY.ARRAY
            displacement DOF vector:
            Udof=[u1, u2, ..., uN, v1, v2, ... vN]
        """
        used_nodes = np.where(self.conn[:, 0] > -1)
        Udof = np.zeros(self.ndof)
        Udof[self.conn[used_nodes]] = Unodes[used_nodes, :self.dim]
        return Udof

    def DICIntegration(self, cam, G=False, EB=False, tri_same=False):
        """Compute FE-DIC quadrature rule along with FE shape functions
        operators

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
        if G:   # shape function gradient
            valx = np.array([])
            valy = np.array([])
        if EB:   # Elementary Brightness and Contrast Correction
            cole = np.array([], dtype=int)
            rowe = np.array([], dtype=int)
            vale = np.array([], dtype=int)
        un, vn = cam.P(self.n[:, 0], self.n[:, 1])
        ne = 0
        for et in self.e.keys():
            ne += len(self.e[et])
            # repdof = self.conn[self.e[et], 0]
            repdof = self.e[et]
            xn = self.n[self.e[et], 0]
            yn = self.n[self.e[et], 1]
            u = un[self.e[et]]
            v = vn[self.e[et]]
            _, _, _, N, Ndx, Ndy = ShapeFunctions(et)
            nfun = N(np.zeros(1), np.zeros(1)).shape[1]
            if et in (3, 10, 16):  # qua4 or qua9 or qua8
                dist = np.floor(
                    np.sqrt((u[:, :2] - u[:, 1:3]) ** 2
                            + (v[:, :2] - v[:, 1:3]) ** 2)
                ).astype(int)
                a, b = np.where(dist < 1)
                if len(a):  # at least one integration point in each element
                    dist[a, b] = 1
                npg = np.sum(np.prod(dist, axis=1))
                wdetJj = np.zeros(npg)
                rowj = np.zeros(npg * nfun, dtype=int)
                colj = np.zeros(npg * nfun, dtype=int)
                valj = np.zeros(npg * nfun)
                if G:   # shape function gradient
                    valxj = np.zeros(npg * nfun)
                    valyj = np.zeros(npg * nfun)
                if EB:   # Elementary Brightness and Contrast Correction
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
                    [repcol, reprow] = np.meshgrid(repdof[je, :],
                                                   repg + len(self.wdetJ)
                                                   )
                    rangephi = nfun * npg + np.arange(np.prod(phi.shape))
                    rowj[rangephi] = reprow.ravel()
                    colj[rangephi] = repcol.ravel()
                    valj[rangephi] = phi.ravel()
                    if G:
                        dphidx = (dyds / detJ)[:, np.newaxis] * dN_xi\
                            + (-dydr / detJ)[:, np.newaxis] * dN_eta
                        dphidy = (-dxds / detJ)[:, np.newaxis] * dN_xi\
                            + (dxdr / detJ)[:, np.newaxis] * dN_eta
                        valxj[rangephi] = dphidx.ravel()
                        valyj[rangephi] = dphidy.ravel()
                    if EB:
                        rangeone = npg + np.arange(len(repg))
                        rowej[rangeone] = repg
                        colej[rangeone] = je
                        valej[rangeone] = 1
                    npg += len(xg)
            elif et in (2, 9):   # tri3 or tri6
                if et == 2:
                    n0 = np.array([[0, 1], [0, 0], [1, 0]])
                    n2 = np.array([[1, 0], [0, 1], [0, 0]])
                elif et == 9:
                    n0 = np.array([[0, 1], [0, 0], [1, 0],
                                   [0, 0.5], [0.5, 0], [0.5, 0.5]])
                    n2 = np.array([[1, 0], [0, 1], [0, 0],
                                   [0.5, 0.5], [0, 0.5], [0.5, 0]])
                uu = np.diff(np.c_[u, u[:, 0]])
                vv = np.diff(np.c_[v, v[:, 0]])
                nn = np.floor(np.sqrt(uu ** 2 + vv ** 2) / 1.1).astype(int)
                b1, b2 = np.where(nn < 1)
                if len(b1):  # at least one integration point in each element
                    nn[b1, b2] = 1
                a = np.argmax(nn, axis=1)  # a is the largest triangle side
                if tri_same:
                    # takes the average of nx and ny for subtriso2
                    nn = (np.sum(nn, axis=1) - np.amax(nn, axis=1)) // 2
                    # exact number of integration points
                    npg = np.sum(nn * (nn + 1) // 2)
                else:
                    nx = nn[np.arange(len(nn)), np.array([2, 0, 1])[a]]
                    ny = nn[np.arange(len(nn)), np.array([1, 2, 0])[a]]
                    # overestimate the number of integration points
                    npg = np.sum(((nx+1) * (ny+1)) // 2)
                wdetJj = np.zeros(npg)
                rowj = np.zeros(npg * nfun, dtype=int)
                colj = np.zeros(npg * nfun, dtype=int)
                valj = np.zeros(npg * nfun)
                if G:   # shape function gradient
                    valxj = np.zeros(npg * nfun)
                    valyj = np.zeros(npg * nfun)
                if EB:   # Elementary Brightness and Contrast Correction
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
                    [repcol, reprow] = meshgrid(repdof[je, :],
                                                repg + len(self.wdetJ))
                    rangephi = nfun * npg + np.arange(np.prod(phi.shape))
                    rowj[rangephi] = reprow.ravel()
                    colj[rangephi] = repcol.ravel()
                    valj[rangephi] = phi.ravel()
                    if G:
                        dphidx = (dyds / detJ)[:, np.newaxis] * dN_xi\
                            + (-dydr / detJ)[:, np.newaxis] * dN_eta
                        dphidy = (-dxds / detJ)[:, np.newaxis] * dN_xi\
                            + (dxdr / detJ)[:, np.newaxis] * dN_eta
                        valxj[rangephi] = dphidx.ravel()
                        valyj[rangephi] = dphidy.ravel()
                    if EB:
                        rangeone = npg + np.arange(len(repg))
                        rowej[rangeone] = repg
                        colej[rangeone] = je
                        valej[rangeone] = 1
                    npg += len(xg)
            else:
                raise Exception("Oops! %d is not a valid element type..." % et)
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
        self.phix = sp.sparse.csr_matrix(
            (val, (row, self.conn[col, 0])), shape=(self.npg, self.ndof))
        self.phiy = sp.sparse.csr_matrix(
            (val, (row, self.conn[col, 1])), shape=(self.npg, self.ndof))
        if G:
            self.dphixdx = sp.sparse.csr_matrix(
                (valx, (row, self.conn[col, 0])), shape=(self.npg, self.ndof))
            self.dphixdy = sp.sparse.csr_matrix(
                (valy, (row, self.conn[col, 0])), shape=(self.npg, self.ndof))
            self.dphiydx = sp.sparse.csr_matrix(
                (valx, (row, self.conn[col, 1])), shape=(self.npg, self.ndof))
            self.dphiydy = sp.sparse.csr_matrix(
                (valy, (row, self.conn[col, 1])), shape=(self.npg, self.ndof))
        if EB:
            self.Me = sp.sparse.csr_matrix(
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
            # repdof = self.conn[self.e[et], 0]
            u = un[self.e[et]]
            v = vn[self.e[et]]
            _, _, _, N, _, _ = ShapeFunctions(et)
            for je in range(len(self.e[et])):
                elem[ne] = Elem()
                elem[ne].repx = self.e[et][je]   # repdof[je]
                rx = np.arange(
                    np.floor(min(u[je])), np.ceil(max(u[je])) + 1
                ).astype("int")
                ry = np.arange(
                    np.floor(min(v[je])), np.ceil(max(v[je])) + 1
                ).astype("int")
                [ypix, xpix] = np.meshgrid(ry, rx)
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
        self.wdetJ = np.ones(self.npg) / cam.f**2   # f**2 = area of a pixel
        row = np.zeros(nzv, dtype=int)
        col = np.zeros(nzv, dtype=int)
        val = np.zeros(nzv)
        nzv = 0
        for je in range(len(elem)):
            [repj, repi] = np.meshgrid(elem[je].repx, elem[je].repg)
            rangephi = nzv + np.arange(np.prod(elem[je].phi.shape))
            row[rangephi] = repi.ravel()
            col[rangephi] = repj.ravel()
            val[rangephi] = elem[je].phi.ravel()
            nzv += np.prod(elem[je].phi.shape)
        self.phix = sp.sparse.csr_matrix(
            (val, (row, self.conn[col, 0])), shape=(self.npg, self.ndof))
        self.phiy = sp.sparse.csr_matrix(
            (val, (row, self.conn[col, 1])), shape=(self.npg, self.ndof))

    def __FastDICIntegElem(self, e, et, n=10, G=False):
        # parent element
        _, _, _, N, Ndx, Ndy = ShapeFunctions(et)
        if et in [2, 9]:   # Triangles
            if et == 2:
                n = max(n, 1)   # minimum 1 integration point for first order
            else:
                n = max(n, 2)   # minimum 2 integration points for second order
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
        elif et in (3, 10, 16):   # Quadrangles
            n = max(n, 2)   # minimum 2 integration points
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
        else:
            valx = []
            valy = []
        # repdof = self.conn[e, 0]
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
            col[repnzv] = e.ravel()   # repdof.ravel()
            row[repnzv] = np.tile(np.arange(ne) + i * ne, [nfun, 1]).T.ravel()
            val[repnzv] = np.tile(phi[i, :], [ne, 1]).ravel()
            if G:
                dphidx = (dyds / detJ)[:, np.newaxis] * dN_xi[i, :]\
                    + (-dydr / detJ)[:, np.newaxis] * dN_eta[i, :]
                dphidy = (-dxds / detJ)[:, np.newaxis] * dN_xi[i, :]\
                    + (dxdr / detJ)[:, np.newaxis] * dN_eta[i, :]
                valx[repnzv] = dphidx.ravel()
                valy[repnzv] = dphidy.ravel()
        return col, row, val, valx, valy, wdetJ

    def GetApproxElementSize(self, cam=None, method='min'):
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
        if self.dim == 3:
            aes = []
            if cam is None:
                u = self.n[:, 0]
                v = self.n[:, 1]
                w = self.n[:, 2]
            else:
                u, v, w = cam.P(self.n[:, 0], self.n[:, 1], self.n[:, 2])
            for et in self.e.keys():
                eet = self.e[et]
                x1 = u[eet[:, 0]]
                y1 = v[eet[:, 0]]
                z1 = w[eet[:, 0]]
                x2 = u[eet[:, 1]]
                y2 = v[eet[:, 1]]
                z2 = w[eet[:, 1]]
                x3 = u[eet[:, 2]]
                y3 = v[eet[:, 2]]
                z3 = w[eet[:, 2]]
                x4 = u[eet[:, 3]]
                y4 = v[eet[:, 3]]
                z4 = w[eet[:, 3]]
                l1 = np.sqrt((x1-x4)**2 + (y1-y4)**2 + (z1-z4)**2)
                l2 = np.sqrt((x1-x3)**2 + (y1-y3)**2 + (z1-z3)**2)
                l3 = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
                aes = np.append(aes, np.hstack((l1, l2, l3)))
            if method == 'max':
                return np.mean(aes) + np.std(aes) * 0.5
            elif method == 'min':
                return np.mean(aes) - np.std(aes) * 0.5
            elif method == 'mean':
                return np.mean(aes)
        else:   # 2D
            if cam is None:
                u = self.n[:, 0]
                v = self.n[:, 1]
            else:
                u, v = cam.P(self.n[:, 0], self.n[:, 1])
            aes = []
            for et in self.e.keys():
                um = u[self.e[et]]\
                    - np.mean(u[self.e[et]], axis=1)[:, np.newaxis]
                vm = v[self.e[et]]\
                    - np.mean(v[self.e[et]], axis=1)[:, np.newaxis]
                if method == 'max':
                    aes = np.append(aes,
                                    np.max(np.sqrt(um**2 + vm**2), axis=1)
                                    )
                elif method == 'min':
                    aes = np.append(aes,
                                    np.min(np.sqrt(um**2 + vm**2), axis=1)
                                    )
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
        col = np.array([], dtype=int)
        row = np.array([], dtype=int)
        val = np.array([])
        if G:   # compute also the shape function gradients
            valx = np.array([])
            valy = np.array([])
        npg = 0
        for je in self.e.keys():
            colj, rowj, valj, valxj, valyj, wdetJj = self.__FastDICIntegElem(
                self.e[je], je, n, G=G)
            col = np.append(col, colj)
            row = np.append(row, rowj + npg)
            val = np.append(val, valj)
            if G:
                valx = np.append(valx, valxj)
                valy = np.append(valy, valyj)
            self.wdetJ = np.append(self.wdetJ, wdetJj)
            npg += len(wdetJj)
        self.npg = len(self.wdetJ)
        self.phix = sp.sparse.csr_matrix(
            (val, (row, self.conn[col, 0])), shape=(self.npg, self.ndof))
        self.phiy = sp.sparse.csr_matrix(
            (val, (row, self.conn[col, 1])), shape=(self.npg, self.ndof))
        if G:
            self.dphixdx = sp.sparse.csr_matrix(
                (valx, (row, self.conn[col, 0])), shape=(self.npg, self.ndof))
            self.dphixdy = sp.sparse.csr_matrix(
                (valy, (row, self.conn[col, 0])), shape=(self.npg, self.ndof))
            self.dphiydx = sp.sparse.csr_matrix(
                (valx, (row, self.conn[col, 1])), shape=(self.npg, self.ndof))
            self.dphiydy = sp.sparse.csr_matrix(
                (valy, (row, self.conn[col, 1])), shape=(self.npg, self.ndof))
        rep, = np.where(self.conn[:, 0] >= 0)
        qx = np.zeros(self.ndof)
        qx[self.conn[rep, :]] = self.n[rep, :]
        self.pgx = self.phix.dot(qx)
        self.pgy = self.phiy.dot(qx)

    def __DVCIntegElem(self, e, et, n=10, G=False):
        # parent element
        _, _, _, _, N, Ndx, Ndy, Ndz = ShapeFunctions(et)
        if et in [4, ]:   # Tet4
            n = max(n, 1)   # minimum 1 integration point for first order
            xg, yg, zg, wg = SubTetIso(n)
        elif et in [5, ]:   # Hex8
            n = max(n, 2)   # minimum 2 integration points
            xg, yg, zg, wg = SubHexIso(n)
        phi = N(xg, yg, zg)
        dN_xi = Ndx(xg, yg, zg)
        dN_eta = Ndy(xg, yg, zg)
        dN_zeta = Ndz(xg, yg, zg)
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
            valz = np.zeros(nzv)
        else:
            valx = []
            valy = []
            valz = []
        # repdof = self.conn[e, 0]
        xn = self.n[e, 0]
        yn = self.n[e, 1]
        zn = self.n[e, 2]
        for i in range(len(xg)):
            dxdr = xn @ dN_xi[i, :]
            dydr = yn @ dN_xi[i, :]
            dzdr = zn @ dN_xi[i, :]
            dxds = xn @ dN_eta[i, :]
            dyds = yn @ dN_eta[i, :]
            dzds = zn @ dN_eta[i, :]
            dxdt = xn @ dN_zeta[i, :]
            dydt = yn @ dN_zeta[i, :]
            dzdt = zn @ dN_zeta[i, :]
            detJ = dxdr * dyds * dzdt + dxds * dydt * dzdr\
                + dydr * dzds * dxdt - dzdr * dyds * dxdt\
                - dxdr * dzds * dydt - dydr * dxds * dzdt
            wdetJ[np.arange(ne) + i * ne] = abs(detJ) * wg[i]
            repnzv = np.arange(ne * nfun) + i * ne * nfun
            col[repnzv] = e.ravel()
            # col[repnzv] = repdof.ravel()
            row[repnzv] = np.tile(np.arange(ne) + i * ne, [nfun, 1]).T.ravel()
            val[repnzv] = np.tile(phi[i, :], [ne, 1]).ravel()
            if G:
                dphidx = (
                    (dyds*dzdt - dzds*dydt)/detJ)[:, np.newaxis]*dN_xi[i, :]\
                - ((dydr*dzdt - dzdr*dydt)/detJ)[:, np.newaxis]*dN_eta[i, :]\
                + ((dydr*dzds - dzdr*dyds)/detJ)[:, np.newaxis]*dN_zeta[i, :]
                dphidy = - (
                    (dxds*dzdt - dzds*dxdt)/detJ)[:, np.newaxis]*dN_xi[i, :]\
                + ((dxdr*dzdt - dzdr*dxdt)/detJ)[:, np.newaxis]*dN_eta[i, :]\
                - ((dxdr*dzds - dzdr*dxds)/detJ)[:, np.newaxis]*dN_zeta[i, :]
                dphidz = (
                    (dxds*dydt - dyds*dxdt)/detJ)[:, np.newaxis]*dN_xi[i, :]\
                - ((dxdr*dydt - dydr*dxdt)/detJ)[:, np.newaxis]*dN_eta[i, :]\
                + ((dxdr*dyds - dydr*dxds)/detJ)[:, np.newaxis]*dN_zeta[i, :]
                valx[repnzv] = dphidx.ravel()
                valy[repnzv] = dphidy.ravel()
                valz[repnzv] = dphidz.ravel()
        return col, row, val, valx, valy, valz, wdetJ

    def DVCIntegration(self, n=None, G=False):
        """Builds a homogeneous (and fast) integration scheme for DVC"""
        if hasattr(n, 'rz') or n is None:
            # if n is a camera then n is autocomputed
            n = self.GetApproxElementSize(n)
        if type(n) is not int:
            n = int(n)
        print('Nb quadrature in each direction = %d' % n)
        self.wdetJ = np.array([])
        col = np.array([], dtype=int)
        row = np.array([], dtype=int)
        val = np.array([])
        if G:   # compute also the shape function gradients
            valx = np.array([])
            valy = np.array([])
            valz = np.array([])
        npg = 0
        for je in self.e.keys():
            colj, rowj, valj, valxj, valyj, valzj, wdetJj = self.__DVCIntegElem(
                self.e[je], je, n, G=G
                )
            col = np.append(col, colj)
            row = np.append(row, rowj + npg)
            val = np.append(val, valj)
            if G:
                valx = np.append(valx, valxj)
                valy = np.append(valy, valyj)
                valz = np.append(valz, valzj)
            self.wdetJ = np.append(self.wdetJ, wdetJj)
            npg += len(wdetJj)
        self.npg = len(self.wdetJ)
        self.phix = sp.sparse.csr_matrix(
            (val, (row, self.conn[col, 0])), shape=(self.npg, self.ndof))
        self.phiy = sp.sparse.csr_matrix(
            (val, (row, self.conn[col, 1])), shape=(self.npg, self.ndof))
        self.phiz = sp.sparse.csr_matrix(
            (val, (row, self.conn[col, 2])), shape=(self.npg, self.ndof))
        if G:
            self.dphixdx = sp.sparse.csr_matrix(
                (valx, (row, self.conn[col, 0])), shape=(self.npg, self.ndof))
            self.dphixdy = sp.sparse.csr_matrix(
                (valy, (row, self.conn[col, 0])), shape=(self.npg, self.ndof))
            self.dphixdz = sp.sparse.csr_matrix(
                (valz, (row, self.conn[col, 0])), shape=(self.npg, self.ndof))
            self.dphiydx = sp.sparse.csr_matrix(
                (valx, (row, self.conn[col, 1])), shape=(self.npg, self.ndof))
            self.dphiydy = sp.sparse.csr_matrix(
                (valy, (row, self.conn[col, 1])), shape=(self.npg, self.ndof))
            self.dphiydz = sp.sparse.csr_matrix(
                (valz, (row, self.conn[col, 1])), shape=(self.npg, self.ndof))
            self.dphizdx = sp.sparse.csr_matrix(
                (valx, (row, self.conn[col, 2])), shape=(self.npg, self.ndof))
            self.dphizdy = sp.sparse.csr_matrix(
                (valy, (row, self.conn[col, 2])), shape=(self.npg, self.ndof))
            self.dphizdz = sp.sparse.csr_matrix(
                (valz, (row, self.conn[col, 2])), shape=(self.npg, self.ndof))
        rep, = np.where(self.conn[:, 0] >= 0)
        qx = np.zeros(self.ndof)
        qx[self.conn[rep, :]] = self.n[rep, :]
        self.pgx = self.phix.dot(qx)
        self.pgy = self.phiy.dot(qx)
        self.pgz = self.phiz.dot(qx)

    def __GaussIntegElem(self, e, et):
        # parent element
        if et in (1, 8):  # bar element
            xg, yg, wg, N, Ndx, Ndy = ShapeFunctions(et)
            phi = N(xg)
            dN_xi = Ndx(xg)
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
            # repdof = self.conn[e, 0]
            xn = self.n[e, 0]
            yn = self.n[e, 1]
            v = np.hstack((np.diff(xn), np.diff(yn)))
            L = np.linalg.norm(v, axis=1)
            c = v[:, 0]/L
            s = v[:, 1]/L
            for i in range(len(xg)):
                dxdr = xn @ dN_xi[i, :]
                dydr = yn @ dN_xi[i, :]
                detJ = np.sqrt(dxdr**2 + dydr**2)
                wdetJ[np.arange(ne) + i * ne] = detJ * wg[i]
                dphidx = (c/detJ)[np.newaxis].T * dN_xi[i, :]
                dphidy = (s/detJ)[np.newaxis].T * dN_xi[i, :]
                repnzv = np.arange(ne * nfun) + i * ne * nfun
                col[repnzv] = e.ravel()   # repdof.ravel()
                row[repnzv] = np.tile(np.arange(ne)+i*ne, [nfun, 1]).T.ravel()
                val[repnzv] = np.tile(phi[i, :], [ne, 1]).ravel()
                valx[repnzv] = dphidx.ravel()
                valy[repnzv] = dphidy.ravel()
            return col, row, val, valx, valy, wdetJ
        elif et in (2, 3, 9, 10, 16):   # 2D elements
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
            # repdof = self.conn[e, 0]
            xn = self.n[e, 0]
            yn = self.n[e, 1]
            for i in range(len(xg)):
                dxdr = xn @ dN_xi[i, :]
                dydr = yn @ dN_xi[i, :]
                dxds = xn @ dN_eta[i, :]
                dyds = yn @ dN_eta[i, :]
                detJ = dxdr * dyds - dxds * dydr
                wdetJ[np.arange(ne) + i * ne] = abs(detJ) * wg[i]
                dphidx = (dyds / detJ)[:, np.newaxis] * dN_xi[i, :]\
                         + (-dydr / detJ)[:, np.newaxis] * dN_eta[i, :]
                dphidy = (-dxds / detJ)[:, np.newaxis] * dN_xi[i, :]\
                         + (dxdr / detJ)[:, np.newaxis] * dN_eta[i, :]
                repnzv = np.arange(ne * nfun) + i * ne * nfun
                col[repnzv] = e.ravel()  # repdof.ravel()
                row[repnzv] = np.tile(np.arange(ne)+i*ne, [nfun, 1]).T.ravel()
                val[repnzv] = np.tile(phi[i, :], [ne, 1]).ravel()
                valx[repnzv] = dphidx.ravel()
                valy[repnzv] = dphidy.ravel()
            return col, row, val, valx, valy, wdetJ
        else:   # 3D elements
            xg, yg, zg, wg, N, Ndx, Ndy, Ndz = ShapeFunctions(et)
            phi = N(xg, yg, zg)
            dN_xi = Ndx(xg, yg, zg)
            dN_eta = Ndy(xg, yg, zg)
            dN_zeta = Ndz(xg, yg, zg)
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
            valz = np.zeros(nzv)
            # repdof = self.conn[e, 0]
            xn = self.n[e, 0]
            yn = self.n[e, 1]
            zn = self.n[e, 2]
            for i in range(len(xg)):
                dxdr = xn @ dN_xi[i, :]
                dydr = yn @ dN_xi[i, :]
                dzdr = zn @ dN_xi[i, :]
                dxds = xn @ dN_eta[i, :]
                dyds = yn @ dN_eta[i, :]
                dzds = zn @ dN_eta[i, :]
                dxdt = xn @ dN_zeta[i, :]
                dydt = yn @ dN_zeta[i, :]
                dzdt = zn @ dN_zeta[i, :]
                detJ = dxdr * dyds * dzdt + dxds * dydt * dzdr\
                    + dydr * dzds * dxdt - dzdr * dyds * dxdt\
                    - dxdr * dzds * dydt - dydr * dxds * dzdt
                dphidx = (
                    (dyds*dzdt - dzds*dydt)/detJ)[:, np.newaxis]*dN_xi[i, :]\
                - ((dydr*dzdt - dzdr*dydt)/detJ)[:, np.newaxis]*dN_eta[i, :]\
                + ((dydr*dzds - dzdr*dyds)/detJ)[:, np.newaxis]*dN_zeta[i, :]
                dphidy = -(
                    (dxds*dzdt - dzds*dxdt)/detJ)[:, np.newaxis]*dN_xi[i, :]\
                + ((dxdr*dzdt - dzdr*dxdt)/detJ)[:, np.newaxis]*dN_eta[i, :]\
                - ((dxdr*dzds - dzdr*dxds)/detJ)[:, np.newaxis]*dN_zeta[i, :]
                dphidz = (
                    (dxds*dydt - dyds*dxdt)/detJ)[:, np.newaxis]*dN_xi[i, :]\
                - ((dxdr*dydt - dydr*dxdt)/detJ)[:, np.newaxis]*dN_eta[i, :]\
                + ((dxdr*dyds - dydr*dxds)/detJ)[:, np.newaxis]*dN_zeta[i, :]
                wdetJ[np.arange(ne) + i * ne] = abs(detJ) * wg[i]
                repnzv = np.arange(ne * nfun) + i * ne * nfun
                col[repnzv] = e.ravel()   # repdof.ravel()
                row[repnzv] = np.tile(np.arange(ne)+i*ne, [nfun, 1]).T.ravel()
                val[repnzv] = np.tile(phi[i, :], [ne, 1]).ravel()
                valx[repnzv] = dphidx.ravel()
                valy[repnzv] = dphidy.ravel()
                valz[repnzv] = dphidz.ravel()
            return col, row, val, valx, valy, valz, wdetJ

    def GaussIntegration(self):
        """Builds a Gauss integration scheme"""
        print('Gauss Integration.')
        if self.dim == 3:
            self.wdetJ = np.array([])
            col = np.array([], dtype=int)
            row = np.array([], dtype=int)
            val = np.array([])
            valx = np.array([])
            valy = np.array([])
            valz = np.array([])
            npg = 0
            for je in self.e.keys():
                colj, rowj, valj, valxj, valyj, valzj, wdetJj = self.__GaussIntegElem(
                    self.e[je], je)
                col = np.append(col, colj)
                row = np.append(row, rowj + npg)
                val = np.append(val, valj)
                valx = np.append(valx, valxj)
                valy = np.append(valy, valyj)
                valz = np.append(valz, valzj)
                self.wdetJ = np.append(self.wdetJ, wdetJj)
                npg += len(wdetJj)
            self.npg = len(self.wdetJ)
            colx = self.conn[col, 0]
            coly = self.conn[col, 1]
            colz = self.conn[col, 2]
            # shape funs
            self.phix = sp.sparse.csr_matrix(
                (val, (row, colx)), shape=(self.npg, self.ndof))
            self.phiy = sp.sparse.csr_matrix(
                (val, (row, coly)), shape=(self.npg, self.ndof))
            self.phiz = sp.sparse.csr_matrix(
                (val, (row, colz)), shape=(self.npg, self.ndof))
            # phix
            self.dphixdx = sp.sparse.csr_matrix(
                (valx, (row, colx)), shape=(self.npg, self.ndof))
            self.dphixdy = sp.sparse.csr_matrix(
                (valy, (row, colx)), shape=(self.npg, self.ndof))
            self.dphixdz = sp.sparse.csr_matrix(
                (valz, (row, colx)), shape=(self.npg, self.ndof))
            # phiy
            self.dphiydx = sp.sparse.csr_matrix(
                (valx, (row, coly)), shape=(self.npg, self.ndof))
            self.dphiydy = sp.sparse.csr_matrix(
                (valy, (row, coly)), shape=(self.npg, self.ndof))
            self.dphiydz = sp.sparse.csr_matrix(
                (valz, (row, coly)), shape=(self.npg, self.ndof))
            # phiz
            self.dphizdx = sp.sparse.csr_matrix(
                (valx, (row, colz)), shape=(self.npg, self.ndof))
            self.dphizdy = sp.sparse.csr_matrix(
                (valy, (row, colz)), shape=(self.npg, self.ndof))
            self.dphizdz = sp.sparse.csr_matrix(
                (valz, (row, colz)), shape=(self.npg, self.ndof))
            # gp coordinates
            rep, = np.where(self.conn[:, 0] >= 0)
            qx = np.zeros(self.ndof)
            qx[self.conn[rep, :]] = self.n[rep, :]
            self.pgx = self.phix.dot(qx)
            self.pgy = self.phiy.dot(qx)
            self.pgz = self.phiz.dot(qx)
        else:   # dim 2
            self.wdetJ = np.array([])
            col = np.array([], dtype=int)
            row = np.array([], dtype=int)
            val = np.array([])
            valx = np.array([])
            valy = np.array([])
            npg = 0
            for je in self.e.keys():
                colj, rowj, valj, valxj, valyj, wdetJj = self.__GaussIntegElem(
                    self.e[je], je)
                col = np.append(col, colj)
                row = np.append(row, rowj + npg)
                val = np.append(val, valj)
                valx = np.append(valx, valxj)
                valy = np.append(valy, valyj)
                self.wdetJ = np.append(self.wdetJ, wdetJj)
                npg += len(wdetJj)
            self.npg = len(self.wdetJ)
            colx = self.conn[col, 0]
            coly = self.conn[col, 1]
            self.phix = sp.sparse.csr_matrix(
                (val, (row, colx)), shape=(self.npg, self.ndof))
            self.phiy = sp.sparse.csr_matrix(
                (val, (row, coly)), shape=(self.npg, self.ndof))
            self.dphixdx = sp.sparse.csr_matrix(
                (valx, (row, colx)), shape=(self.npg, self.ndof))
            self.dphixdy = sp.sparse.csr_matrix(
                (valy, (row, colx)), shape=(self.npg, self.ndof))
            self.dphiydx = sp.sparse.csr_matrix(
                (valx, (row, coly)), shape=(self.npg, self.ndof))
            self.dphiydy = sp.sparse.csr_matrix(
                (valy, (row, coly)), shape=(self.npg, self.ndof))
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
        else:
            m = self
        wdetJ = sp.sparse.diags(m.wdetJ)
        if self.dim == 3:
            Bxy = m.dphixdy + m.dphiydx
            Bxz = m.dphixdz + m.dphizdx
            Byz = m.dphiydz + m.dphizdy
            K = (
                 hooke[0, 0] * m.dphixdx.T @ wdetJ @ m.dphixdx
                 + hooke[1, 1] * m.dphiydy.T @ wdetJ @ m.dphiydy
                 + hooke[2, 2] * m.dphizdz.T @ wdetJ @ m.dphizdz
                 + hooke[3, 3] * Bxy.T @ wdetJ @ Bxy
                 + hooke[4, 4] * Bxz.T @ wdetJ @ Bxz
                 + hooke[5, 5] * Byz.T @ wdetJ @ Byz
                 + hooke[0, 1] * m.dphixdx.T @ wdetJ @ m.dphiydy
                 + hooke[0, 2] * m.dphixdx.T @ wdetJ @ m.dphizdz
                 + hooke[1, 0] * m.dphiydy.T @ wdetJ @ m.dphixdx
                 + hooke[1, 2] * m.dphiydy.T @ wdetJ @ m.dphizdz
                 + hooke[2, 0] * m.dphizdz.T @ wdetJ @ m.dphixdx
                 + hooke[2, 1] * m.dphizdz.T @ wdetJ @ m.dphiydy
               )
        else:
            Bxy = m.dphixdy + m.dphiydx
            K = (
                 hooke[0, 0] * m.dphixdx.T @ wdetJ @ m.dphixdx
                 + hooke[1, 1] * m.dphiydy.T @ wdetJ @ m.dphiydy
                 + hooke[2, 2] * Bxy.T @ wdetJ @ Bxy
                 + hooke[0, 1] * m.dphixdx.T @ wdetJ @ m.dphiydy
                 # + hooke[0, 2] * m.dphixdx.T @ wdetJ @ Bxy
                 # + hooke[1, 2] * m.dphiydy.T @ wdetJ @ Bxy
                 + hooke[1, 0] * m.dphiydy.T @ wdetJ @ m.dphixdx
                 # + hooke[2, 0] * Bxy.T @ wdetJ @ m.dphixdx
                 # + hooke[2, 1] * Bxy.T @ wdetJ @ m.dphiydy
               )
        return K

    def StiffnessAxi(self, hooke):
        """Assembles Stiffness Operator"""
        if not hasattr(self, "dphixdx"):
            m = self.Copy()
            print('Gauss Integ.')
            m.GaussIntegration()
        else:
            m = self
        wdetJr = sp.sparse.diags(m.wdetJ*m.pgx)   # r dr dz !
        Bxy = m.dphixdy + m.dphiydx
        Nr = sp.sparse.diags(1/m.pgx) @ m.phix
        K = (
             hooke[0, 0] * m.dphixdx.T @ wdetJr @ m.dphixdx
             + hooke[1, 1] * m.dphiydy.T @ wdetJr @ m.dphiydy
             + hooke[2, 2] * Nr.T @ wdetJr @ Nr
             + hooke[3, 3] * Bxy.T @ wdetJr @ Bxy
             + hooke[0, 1] * m.dphixdx.T @ wdetJr @ m.dphiydy
             + hooke[0, 2] * m.dphixdx.T @ wdetJr @ Nr
             + hooke[1, 0] * m.dphiydy.T @ wdetJr @ m.dphixdx
             + hooke[1, 2] * m.dphiydy.T @ wdetJr @ Nr
             + hooke[2, 0] * Nr.T @ wdetJr @ m.dphixdx
             + hooke[2, 1] * Nr.T @ wdetJr @ m.dphiydy
           )
        return K

    def Laplacian(self):
        """Assembles Tikhonov (Laplacian) Operator"""
        if not hasattr(self, "dphixdx"):
            m = self.Copy()
            m.GaussIntegration()
        else:
            m = self
        wdetJ = sp.sparse.diags(m.wdetJ)
        if self.dim == 3:
            L = m.dphixdx.T @ wdetJ @ m.dphixdx + \
                m.dphixdy.T @ wdetJ @ m.dphixdy + \
                m.dphixdz.T @ wdetJ @ m.dphixdz + \
                m.dphiydx.T @ wdetJ @ m.dphiydx + \
                m.dphiydy.T @ wdetJ @ m.dphiydy + \
                m.dphiydz.T @ wdetJ @ m.dphiydz + \
                m.dphizdx.T @ wdetJ @ m.dphizdx + \
                m.dphizdy.T @ wdetJ @ m.dphizdy + \
                m.dphizdz.T @ wdetJ @ m.dphizdz
        else:
            L = m.dphixdx.T @ wdetJ @ m.dphixdx + \
                m.dphiydy.T @ wdetJ @ m.dphiydy + \
                m.dphixdy.T @ wdetJ @ m.dphixdy + \
                m.dphiydx.T @ wdetJ @ m.dphiydx
        return L

    def TikoSprings(self, liste, l0=None, dim=2):
        """
        Builds a Laplacian like operator from bar elements.
        liste is a list of bar elements and the dofs concerned.
          liste = [node1, node2, dofu=1, dofv=1(, dofw=0)]
          liste=np.array([[0, 1, 1(, 1)],
                          [1, 2, 0(, 1)]])"""
        nzv = np.sum(liste[:, -dim:], dtype=int) * 4
        row = np.zeros(nzv, dtype=int)
        col = np.zeros(nzv, dtype=int)
        val = np.zeros(nzv)
        nzv = 0
        for ei in liste:
            dofn = self.conn[ei[:2]]
            xn = self.n[ei[:2]]
            if l0 is not None:
                d = l0
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
        return sp.sparse.csr_matrix((val, (row, col)),
                                    shape=(self.ndof, self.ndof))

    def Mass(self, rho):
        """Assembles Mass Matrix"""
        if not hasattr(self, "phix"):
            m = self.Copy()
            m.GaussIntegration()
            wdetJ = sp.sparse.diags(m.wdetJ)
            M = rho * m.phix.T @ wdetJ @ m.phix\
                + rho * m.phiy.T @ wdetJ @ m.phiy
        else:
            wdetJ = sp.sparse.diags(self.wdetJ)
            M = rho * self.phix.T @ wdetJ @ self.phix\
                + rho * self.phiy.T @ wdetJ @ self.phiy
        return M

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
        PVDFile(os.path.join('vtk', filename), "vtu", 1, UU.shape[1])

    def Write(self, filename, point_data={}, cell_data={}):
        """
        Export a meshfile using meshio.
        mesh format depends on the chosen extension.
        Abaqus (.inp), ANSYS msh (.msh), AVS-UCD (.avs), CGNS (.cgns),
        DOLFIN XML (.xml), Exodus (.e, .exo), FLAC3D (.f3grid), H5M (.h5m),
        Kratos/MDPA (.mdpa), Medit (.mesh, .meshb), MED/Salome (.med),
        Nastran (bulk data, .bdf, .fem, .nas), Netgen (.vol, .vol.gz),
        Neuroglancer precomputed format, Gmsh (format versions 2.2, 4.0, and
        4.1, .msh), OBJ (.obj), OFF (.off), PERMAS (.post, .post.gz, .dato,
        .dato.gz), PLY (.ply), STL (.stl), Tecplot .dat, TetGen .node/.ele,
        SVG (2D output only) (.svg), SU2 (.su2), UGRID (.ugrid), VTK (.vtk),
        VTU (.vtu), WKT (TIN) (.wkt), XDMF (.xdmf, .xmf).

        Parameters
        ----------
        filename : STRING
            name of the mesh file, including extension
        U : NUMPY.ARRAY, optional
            Displacement DOF vector.

        """
        cells = dict()
        for et in self.e.keys():
            cells[eltype_n2s[et]] = self.e[et].astype('int32')
            
        points = self.n
        if self.dim == 2:
            points = np.hstack((points, np.zeros((len(self.n), 1))))
        mesh = meshio.Mesh(points, cells)
        mesh.cell_data = cell_data
        mesh.point_data = point_data
        mesh.write(filename)
        print("Meshfile " + filename + " written.")

    def VTKSol(self, filename, U):
        """
        Writes a VTK Result file for vizualisation using Paraview.
        Usage:
            m.VTKSol('FileName', U)

        Parameters
        ----------
        filename : STRING
            mesh file name, without no extension. Example: 'result'
        U : NUMPY.ARRAY
            a (ndof x 1) Numpy array containing the displacement dofs
            (ndof is the numer of dofs)

        Returns
        -------
        None.

        """
        cells = dict()
        for et in self.e.keys():
            cells[eltype_n2s[et]] = self.e[et]
        points = self.n
        if self.dim == 2:
            points = np.hstack((points, np.zeros((len(self.n), 1))))
        mesh = meshio.Mesh(points, cells)
        mesh.cell_data = {}
        new_u = self.DOF2Nodes(U, fillzero=True)
        if self.dim == 2:
            ES, EN = self.StrainAtNodes(U)
            Ex = ES[:, 0]
            Ey = ES[:, 1]
            Exy = EN[:, 0]
            new_e = np.c_[Ex, Ey, Exy]
            C = (Ex + Ey) / 2
            R = np.sqrt((Ex - C) ** 2 + Exy ** 2)
            new_ep = np.sort(np.c_[C + R, C - R], axis=1)
            mesh.point_data = {'U': new_u, 'strain': new_e,
                               'pcp_strain': new_ep}
        else:
            EN, ES = self.StrainAtNodes(U)
            new_e = np.c_[EN, ES]
            mesh.point_data = {'U': new_u, 'strain': new_e}
        # write
        dir0, filename = os.path.split(filename)
        if not os.path.isdir(os.path.join("vtk", dir0)):
            os.makedirs(os.path.join("vtk", dir0))
        mesh.write(os.path.join("vtk", dir0, filename) + '.vtu')
        print("Meshfile " + os.path.join("vtk", dir0, filename)
              + '.vtu' + " written.")

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

    def GP2DOFs(self, gp_field):
        # Chaml2Chpoint
        if not hasattr(self, "dphixdx"):
            m = self.Copy()
            m.GaussIntegration()
        else:
            m = self
        phi = m.phix[:, m.conn[:,0]]
        wx = np.sum(phi, axis=0).A[0] + 1e-12
        if type(gp_field) is list:
            nd_field = []
            for i in range(len(gp_field)):
                nd_field += [sp.sparse.diags(1/wx) @ phi.T @ gp_field[i]]
        else:
            nd_field = sp.sparse.diags(1/wx) @ phi.T @ gp_field
        return nd_field

    def StrainAtNodes(self, U):
        if not hasattr(self, "dphixdx"):
            m = self.Copy()
            m.GaussIntegration()
        else:
            m = self
        if self.dim == 2:
            exxgp = m.dphixdx @ U
            eyygp = m.dphiydy @ U
            exygp = 0.5 * m.dphixdy @ U + 0.5 * m.dphiydx @ U
            wx = np.sum(m.phix, axis=0).A[0] + 1e-12
            wy = np.sum(m.phiy, axis=0).A[0] + 1e-12
            EpsN = sp.sparse.diags(1/wx) @ m.phix.T @ exxgp\
            + sp.sparse.diags(1/wy) @ m.phiy.T @ eyygp
            EpsS = sp.sparse.diags(1/wx) @ m.phix.T @ exygp
            EpsN = self.DOF2Nodes(EpsN)
            EpsS = self.DOF2Nodes(EpsS)
            return EpsN, EpsS
        else:   # dim 3
            exxgp = m.dphixdx @ U
            eyygp = m.dphiydy @ U
            ezzgp = m.dphizdz @ U
            exygp = 0.5 * m.dphixdy @ U + 0.5 * m.dphiydx @ U
            exzgp = 0.5 * m.dphixdz @ U + 0.5 * m.dphizdx @ U
            eyzgp = 0.5 * m.dphiydz @ U + 0.5 * m.dphizdy @ U
            wx = np.sum(m.phix, axis=0).A[0] + 1e-12
            wy = np.sum(m.phiy, axis=0).A[0] + 1e-12
            wz = np.sum(m.phiz, axis=0).A[0] + 1e-12
            EpsN = sp.sparse.diags(1/wx) @ m.phix.T @ exxgp\
            + sp.sparse.diags(1/wy) @ m.phiy.T @ eyygp\
            + sp.sparse.diags(1/wz) @ m.phiz.T @ ezzgp
            EpsS = sp.sparse.diags(1/wx) @ m.phix.T @ exygp\
            + sp.sparse.diags(1/wy) @ m.phiy.T @ exzgp\
            + sp.sparse.diags(1/wz) @ m.phiz.T @ eyzgp
            EpsN = self.DOF2Nodes(EpsN)
            EpsS = self.DOF2Nodes(EpsS)
            return EpsN, EpsS

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
        m2.DICIntegrationFast(cam2)
        nnode = m2.pgx.shape[0]
        cells = {'vertex': np.arange(nnode)[np.newaxis].T}
        points = np.array([m2.pgx, m2.pgy, 0 * m2.pgx]).T
        mesh = meshio.Mesh(points, cells)
        mesh.cell_data = {}
        mesh.point_data = {}
        """ Displacement field """
        pgu = m2.phix.dot(U)
        pgv = m2.phiy.dot(U)
        mesh.point_data['U'] = np.array([pgu, pgv]).T
        """ Reference image """
        u, v = cam.P(m2.pgx, m2.pgy)
        if hasattr(f, "tck") == 0:
            f.BuildInterp()
        mesh.point_data['f'] = f.Interp(u, v)
        """ Deformed image """
        if hasattr(g, "tck") == 0:
            g.BuildInterp()
        mesh.point_data['g'] = g.Interp(u, v)
        """ Residual Map """
        pgu = m2.phix.dot(U)
        pgv = m2.phiy.dot(U)
        pgxu = m2.pgx + pgu
        pgyv = m2.pgy + pgv
        u, v = cam.P(pgxu, pgyv)
        mesh.point_data['res'] = mesh.point_data['f'] - g.Interp(u, v)
        # Write the VTU file in the VTK dir
        dir0, filename = os.path.split(filename)
        if not os.path.isdir(os.path.join("vtk", dir0)):
            os.makedirs(os.path.join("vtk", dir0))
        mesh.write(os.path.join("vtk", dir0, filename)+'.vtu')
        print("Meshfile " + os.path.join("vtk", dir0, filename)
              + '.vtu' + " written.")

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
        facecolor = kwargs.pop("facecolor", "w")
        alpha = kwargs.pop("alpha", 0.8)
        """ Plot deformed or undeformes Mesh """
        if n is None:
            n = self.n.copy()
        if U is not None:
            n += coef * U[self.conn[:, :self.dim]]
        qua = np.zeros((0, 4), dtype="int64")
        tri = np.zeros((0, 3), dtype="int64")
        bar = np.zeros((0, 2), dtype="int64")
        pn = np.zeros(0, dtype=int)   # nodes to plot for quad elems
        for ie in self.e.keys():
            if ie in [3, 16, 10]:   # quadrangles
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
        # Join the 2 lists of vertices
        nn = tuple(n[qua]) + tuple(n[tri]) + tuple(n[bar])
        # Create the collection
        pn = np.unique(pn)
        n = n[pn, :]
        if self.dim == 3:   # 3D collection
            pc = art3d.Poly3DCollection(nn, facecolors=facecolor,
                                        edgecolor=edgecolor,
                                        alpha=alpha)
        else:  # 2D collection
            pc = cols.PolyCollection(nn, facecolor=facecolor,
                                     edgecolor=edgecolor,
                                     alpha=alpha, **kwargs)

        # Return the matplotlib collection and the list of vertices
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
        # if Volumetric mesh > build surface mesh and plot it
        alpha = kwargs.pop("alpha", 1)
        edgecolor = kwargs.pop("edgecolor", "k")
        if list(self.e.keys())[0] in [5, 4, 11, 17]:
            mb = self.BuildBoundaryMesh()
            mb.conn = self.conn
            facecolor = kwargs.pop("facecolor", "w")
            mb.Plot(U, coef, n, plotnodes, alpha=alpha, edgecolor=edgecolor,
                    facecolor=facecolor, **kwargs)
        else:   # otherwise
            fig = plt.figure()
            if self.dim == 2:
                facecolor = kwargs.pop("facecolor", "None")
                ax = plt.gca()
            else:
                facecolor = kwargs.pop("facecolor", "w")  # "w"
                ax = fig.add_subplot(111, projection="3d")
                # ax = Axes3D(plt.figure())
                # ax = plt.figure().add_subplot(projection='3d')
            pc, nn, n = self.PreparePlot(U, coef, n, alpha=alpha,
                                         edgecolor=edgecolor,
                                         facecolor=facecolor, **kwargs)

            if self.dim == 2:
                ax.add_collection(pc)
            else:
                ax.add_collection3d(pc)
            ax.autoscale()
            if self.dim == 2:
                plt.axis('equal')
                if plotnodes:
                    plt.plot(
                        n[:, 0],
                        n[:, 1],
                        linestyle="None",
                        marker="o",
                        color=edgecolor,
                        alpha=alpha,
                    )
            else:
                n = np.vstack(nn)
                X = n[:, 0]
                Y = n[:, 1]
                Z = n[:, 2]
                max_range = np.array([X.max()-X.min(),
                                      Y.max()-Y.min(),
                                      Z.max()-Z.min()]).max() / 2.0
                mid_x = (X.max()+X.min()) * 0.5
                mid_y = (Y.max()+Y.min()) * 0.5
                mid_z = (Z.max()+Z.min()) * 0.5
                ax.set_xlim(mid_x - max_range, mid_x + max_range)
                ax.set_ylim(mid_y - max_range, mid_y + max_range)
                ax.set_zlim(mid_z - max_range, mid_z + max_range)
                ax.set_xlim(mid_x - max_range, mid_x + max_range)
                ax.set_ylim(mid_y - max_range, mid_y + max_range)
                # if plotnodes:
                #     ax.plot(
                #         n[:, 0],
                #         n[:, 1],
                #         mnew.n[:, 2],
                #         linestyle="None",
                #         marker="o",
                #         color=edgecolor,
                #         alpha=alpha,
                #     )
            plt.show()

    def AnimatedPlot(self, U, coef=1, n=None, timeAnim=5,
                     color=('k', 'b', 'r', 'g', 'c')):
        """
        Animated plot with funcAnimation
        Inputs:
            -U: displacement field, stored in column for each time step
            -coef: amplification coefficient
            -n: nodes coordinates
            -timeAnim: time of the animation
        """
        if not(isinstance(U, list)):
            U = [U]
        ntimes = U[0].shape[1]
        fig = plt.figure()
        if self.dim == 2:
            ax = plt.gca()
        else:
            ax = fig.add_subplot(111, projection="3d")
        pc = dict()
        nn = dict()
        for jj, u in enumerate(U):
            pc[jj], nn[jj], _ = self.PreparePlot(u[:, 0], coef,
                                                 n, edgecolor=color[jj])
            if self.dim == 2:
                ax.add_collection(pc[jj])
                ax.autoscale()
                plt.axis('equal')
            else:
                ax.add_collection3d(pc[jj])
                ax.autoscale()
                nn = np.vstack(nn[jj])
                X = nn[:, 0]
                Y = nn[:, 1]
                Z = nn[:, 2]
                max_range = np.array([X.max()-X.min(),
                                      Y.max()-Y.min(),
                                      Z.max()-Z.min()]).max() / 2.0
                mid_x = (X.max()+X.min()) * 0.5
                mid_y = (Y.max()+Y.min()) * 0.5
                mid_z = (Z.max()+Z.min()) * 0.5
                ax.set_xlim(mid_x - max_range, mid_x + max_range)
                ax.set_ylim(mid_y - max_range, mid_y + max_range)
                ax.set_zlim(mid_z - max_range, mid_z + max_range)
                ax.set_xlim(mid_x - max_range, mid_x + max_range)
                ax.set_ylim(mid_y - max_range, mid_y + max_range)

        def updateMesh(ii):
            """
            Function to update the matplotlib collections
            """
            for jj, u in enumerate(U):
                titi, nn, _ = self.PreparePlot(u[:, ii], coef, n)
                # pc[jj].set_paths(nn)
                pc[jj].set_verts(nn)
            return pc.values()

        line_ani = animation.FuncAnimation(fig, updateMesh, range(ntimes),
                                           blit=False,
                                           interval=timeAnim/ntimes*1000)
        return line_ani

    def PlotResidualMap(self, res, cam, npts=1e4):
        """
        Plots the residual map using Matplotlib Library.
        Usage:
            m.PlotResidualMap(res, cam)
            where res is a numpy.array containing the residual at integration
            points

            m.PlotResidualMap(res, cam, npts=1e3)
            to limit the number of integration points visualization (faster)

            m.PlotResidualMap(res, cam, npts='all')
            to visualize all integration points (less fast)
        """
        if npts == "all":
            rep = np.arange(self.npg)
        else:
            rep = np.unique((np.random.sample(int(npts))
                             * (len(self.pgx) - 1)).astype("int"))
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
            rep = np.unique((np.random.sample(int(npts))
                             * (len(self.pgx) - 1)).astype("int"))
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

    def PlotContourDispl(self, U=None, n=None, s=1.0, stype='comp',
                         newfig=True, plotmesh=True, cmap='RdBu', **kwargs):
        """
        Plots the displacement field using Matplotlib Library.

        Parameters
        ----------
        U : 1D NUMPY.ARRAY
            displacement dof vector
        n : NUMPY.ARRAY, optional
            Coordinate of the nodes. The default is None, which corresponds
            to using self.n instead.
        s : FLOAT, optional
            Deformation scale factor. The default is 1.0.
        stype : STRING, optional
            'comp' > plots the 3 components of the strain field
            'mag' > plots the 'VonMises' equivalent strain
             The default is 'comp'.
        newfig : BOOL
            if TRUE plot in a new figure (default)
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        plot_y = True
        if self.ndof % len(U):
            raise Exception('Problem: number of dofs in U ='
                            + ' %d and number of dof in the mesh = %d'
                            % (len(U), self.ndof))
        else:
            V = np.zeros(self.ndof)
            V[:len(U)] = U
            if self.ndof != len(U):
                plot_y = False
        if n is None:
            n = self.n.copy()
            n += V[self.conn] * s  # s: amplification scale factor
        """ Plot mesh and field contour """
        triangles = np.zeros((0, 3), dtype=int)
        for ie in self.e.keys():
            if ie == 3 or ie == 16 or ie == 10:  # quadrangles
                triangles = np.vstack(
                    (triangles, self.e[ie][:, [0, 1, 3]],
                     self.e[ie][:, [1, 2, 3]])
                )
            elif ie == 2 or ie == 9:  # triangles
                triangles = np.vstack((triangles, self.e[ie]))
        alpha = kwargs.pop("alpha", 1)
        if stype == 'mag':
            Vmag = np.sqrt(V[self.conn[:, 0]]**2 + V[self.conn[:, 1]]**2)
            if newfig:
                plt.figure()
            plt.tricontourf(n[:, 0], n[:, 1], triangles, Vmag, 20,
                            alpha=alpha, cmap=cmap)
            plt.colorbar()
            if plotmesh:
                self.Plot(n=n, alpha=0.1)
            plt.axis('equal')
            plt.axis("off")
            plt.title("Magnitude")
        else:
            plt.figure()
            plt.tricontourf(n[:, 0], n[:, 1], triangles, V[self.conn[:, 0]],
                            20, alpha=alpha, cmap=cmap)
            if plotmesh:
                self.Plot(n=n, alpha=0.1)
            plt.axis('equal')
            plt.axis("off")
            plt.title("x")
            plt.colorbar()
            if plot_y:
                plt.figure()
                plt.tricontourf(n[:, 0], n[:, 1], triangles, V[self.conn[:, 1]],
                                20, alpha=alpha, cmap=cmap)
                plt.colorbar()
                if plotmesh:
                    self.Plot(n=n, alpha=0.1)
                plt.axis("equal")
                plt.title("y")
                plt.axis("off")
                plt.show()

    def PlotContourTensorField(self, U, Fn, Fs, n=None, s=1.0, stype='comp',
                          newfig=True, cmap='rainbow', field_name='Field', **kwargs):
        """
        Plots the STRESS/STRAIN field using Matplotlib Library.

        Parameters
        ----------
        U : 1D NUMPY.ARRAY
            displacement dof vector
        Fn : The normal fields ex: [Ex, Ey] should be nodal values
        Fs : The tangential fields ex: [Exy, 0] should be nodal values
        n : NUMPY.ARRAY, optional
            Coordinate of the nodes. The default is None, which corresponds
            to using self.n instead.
        s : FLOAT, optional
            Deformation scale factor. The default is 1.0.
        stype : STRING, optional
            'comp' > plots the 3 components of the field
            'mag' > plots the 'VonMises' equivalent field
            'pcp'> plots the 2 principal fields
            'maxpcp'> plots the maximal principal fields
            The default is 'comp'.
        newfigure : BOOL
            if TRUE plot in a new figure (default)
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if n is None:
            n = self.n.copy()
            n += U[self.conn] * s  # s: amplification scale factor
        triangles = np.zeros((0, 3), dtype=int)
        for ie in self.e.keys():
            if ie == 3 or ie == 16 or ie == 10:  # quadrangles
                triangles = np.vstack(
                    (triangles, self.e[ie][:, [0, 1, 3]],
                     self.e[ie][:, [1, 2, 3]])
                )
            elif ie == 2 or ie == 9:  # triangles
                triangles = np.vstack((triangles, self.e[ie]))
        EX = Fn[:, 0]
        EY = Fn[:, 1]
        EXY = Fs[:, 0]
        alpha = kwargs.pop("alpha", 1)
        if stype == 'pcp':
            E1 = 0.5*EX + 0.5*EY\
            - 0.5*np.sqrt(EX**2 - 2*EX*EY + EY**2 + 4*EXY**2)
            E2 = 0.5*EX + 0.5*EY\
            + 0.5*np.sqrt(EX**2 - 2*EX*EY + EY**2 + 4*EXY**2)
            plt.figure()
            plt.tricontourf(n[:, 0], n[:, 1], triangles, E1[self.conn[:, 0]],
                            20, alpha=alpha)
            self.Plot(n=n, alpha=0.1)
            plt.axis("off")
            plt.axis("equal")
            plt.title(field_name+"_1")
            plt.colorbar()
            plt.figure()
            plt.tricontourf(n[:, 0], n[:, 1], triangles, E2[self.conn[:, 0]],
                            20, alpha=alpha)
            self.Plot(n=n, alpha=0.1)
            plt.axis("off")
            plt.axis("equal")
            plt.title(field_name+"_2")
            plt.colorbar()
            plt.show()
        elif stype == 'maxpcp':
            E1 = 0.5*EX + 0.5*EY\
            - 0.5*np.sqrt(EX**2 - 2*EX*EY + EY**2 + 4*EXY**2)
            E2 = 0.5*EX + 0.5*EY\
            + 0.5*np.sqrt(EX**2 - 2*EX*EY + EY**2 + 4*EXY**2)
            rep, = np.where(abs(E1) < abs(E2))
            E1[rep] = E2[rep]
            if newfig:
                plt.figure()
            plt.tricontourf(n[:, 0], n[:, 1], triangles, E1, 20, alpha=alpha, cmap=cmap)
            self.Plot(n=n, alpha=0.1)
            plt.axis("off")
            plt.axis("equal")
            plt.title(field_name+"_max")
            plt.colorbar()
        elif stype == 'mag':
            EVM = np.sqrt(EX**2 + EY**2 + EX * EY + 3 * EXY**2)
            if newfig:
                plt.figure()
            plt.tricontourf(n[:, 0], n[:, 1], triangles, EVM, 20, alpha=alpha, cmap=cmap)
            self.Plot(n=n, alpha=0.1)
            plt.axis("off")
            plt.axis("equal")
            plt.title(field_name+"_VM")
            plt.colorbar()
        else:
            """ Plot mesh and field contour """
            plt.figure()
            plt.tricontourf(n[:, 0], n[:, 1], triangles, EX, 20, alpha=alpha, cmap=cmap)
            self.Plot(n=n, alpha=0.1)
            plt.axis("off")
            plt.axis("equal")
            plt.title(field_name+"_X")
            plt.colorbar()
            plt.figure()
            plt.tricontourf(n[:, 0], n[:, 1], triangles, EY, 20, alpha=alpha, cmap=cmap)
            self.Plot(n=n, alpha=0.1)
            plt.axis("equal")
            plt.title(field_name+"_Y")
            plt.axis("off")
            plt.colorbar()
            plt.figure()
            plt.tricontourf(n[:, 0], n[:, 1], triangles, EXY, 20, alpha=alpha, cmap=cmap)
            self.Plot(n=n, alpha=0.1)
            plt.axis("equal")
            plt.title(field_name+"_XY")
            plt.axis("off")
            plt.colorbar()
            plt.show()

    def PlotContourStrain(self, U, n=None, s=1.0, stype='comp',
                          newfig=True, cmap='rainbow', **kwargs):
        """
        Plots the strain field using Matplotlib Library.

        Parameters
        ----------
        U : 1D NUMPY.ARRAY
            displacement dof vector
        n : NUMPY.ARRAY, optional
            Coordinate of the nodes. The default is None, which corresponds
            to using self.n instead.
        s : FLOAT, optional
            Deformation scale factor. The default is 1.0.
        stype : STRING, optional
            'comp' > plots the 3 components of the strain field
            'mag' > plots the 'VonMises' equivalent strain
            'pcp'> plots the 2 principal strain fields
            'maxpcp'> plots the maximal principal strain fields
            The default is 'comp'.
        newfigure : BOOL
            if TRUE plot in a new figure (default)
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        EN, ES = self.StrainAtNodes(U)
        self.PlotContourTensorField(U, EN, ES, n=n, s=s, stype=stype,
                          newfig=newfig, cmap=cmap, field_name='EPS', **kwargs)


    def PlotContourStress(self, U, hooke, n=None, s=1.0, stype='comp',
                          newfig=True, cmap='rainbow', **kwargs):
        """
        Plots the stress field using Matplotlib Library.

        Parameters
        ----------
        U : 1D NUMPY.ARRAY
            displacement dof vector
        hooke : Hooke operator
        n : NUMPY.ARRAY, optional
            Coordinate of the nodes. The default is None, which corresponds
            to using self.n instead.
        s : FLOAT, optional
            Deformation scale factor. The default is 1.0.
        stype : STRING, optional
            'comp' > plots the 3 components of the stress field
            'mag' > plots the 'VonMises' equivalent stress
            'pcp'> plots the 2 principal stress fields
            'maxpcp'> plots the maximal principal stress fields
            The default is 'comp'.
        newfigure : BOOL
            if TRUE plot in a new figure (default)
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        EN, ES = self.StrainAtNodes(U)
        SN, SS = Strain2Stress(hooke, EN, ES)
        self.PlotContourTensorField(U, SN, SS, n=n, 
                        s=s, stype=stype, newfig=newfig, cmap=cmap, 
                        field_name='SIG', **kwargs)

    def PlotNodeLabels(self, d=[0, 0], **kwargs):
        """
        Plots the mesh with the node labels (may be slow for large mesh size).
        """
        if type(d) is float:
            d = [0.707*d, 0.707*d]
        self.Plot(**kwargs)
        color = kwargs.get("edgecolor", "k")
        plt.plot(self.n[:, 0], self.n[:, 1], ".", color=color)
        for i in range(len(self.n[:, 1])):
            plt.text(self.n[i, 0] + d[0], self.n[i, 1] + d[1], str(i),
                     color=color)

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

    # def VTKIntegrationPointsTh(self, cam, f, U, filename="IntPtsT"):
    #     nnode = self.pgx.shape[0]
    #     nelem = nnode
    #     new_node = np.array([self.pgx, self.pgy, 0 * self.pgx]).T.ravel()
    #     new_conn = np.arange(nelem)
    #     new_offs = np.arange(nelem) + 1
    #     new_type = 2 * np.ones(nelem).astype("int")
    #     vtkfile = VTUWriter(nnode, nelem, new_node, new_conn,
    #                        new_offs, new_type)
    #     """ Reference image """
    #     u, v = cam.P(self.pgx, self.pgy)
    #     if hasattr(f, "tck") == 0:
    #         f.BuildInterp()
    #     imref = f.Interp(u, v)
    #     vtkfile.addCellData("Th_init", 1, imref)
    #     """ ReMaped thermal field """
    #     pgu = self.phix.dot(U)
    #     pgv = self.phiy.dot(U)
    #     pgxu = self.pgx + pgu
    #     pgyv = self.pgy + pgv
    #     u, v = cam.P(pgxu, pgyv)
    #     imdefu = f.Interp(u, v)
    #     vtkfile.addCellData("Th_advected", 1, imdefu)
    #     """ Displacement field """
    #     new_u = np.array([pgu, pgv, 0 * pgu]).T.ravel()
    #     vtkfile.addPointData("disp", 3, new_u)
    #     """ Strain field """
    #     epsxx, epsyy, epsxy = self.StrainAtGP(U)
    #     new_eps = np.array([epsxx, epsyy, epsxy]).T.ravel()
    #     vtkfile.addCellData("epsilon", 3, new_eps)

    #     # Write the VTU file in the VTK dir
    #     dir0, filename = os.path.split(filename)
    #     if not os.path.isdir(os.path.join("vtk", dir0)):
    #         os.makedirs(os.path.join("vtk", dir0))
    #     vtkfile.write(os.path.join("vtk", dir0, filename))

    def FindDOFinBox(self, box):
        """
        Returns the dof of all the nodes lying within a rectangle defined
        by the coordinates of two diagonal points (in the mesh coordinate sys).
        Used to apply BC for instance.
        box = np.array([[xmin, ymin],
                        [xmax, ymax]])    in mesh unit
        """
        if self.dim == 2:
            rep, = np.where(isInBox(box,
                                    self.n[:, 0],
                                    self.n[:, 1]))
        else:
            rep, = np.where(isInBox(box,
                                    self.n[:, 0],
                                    self.n[:, 1],
                                    self.n[:, 2]))
        reprep, = np.where(self.conn[rep, 0] > -1)
        dofs = self.conn[rep[reprep]]
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
            if je in [4, 5, 11, 12, 14, 17]:
                newe[je] = self.e[je]
        self.e = newe

    def RemoveElemsOutsideRoi(self, roi, cam=None):
        """
        Removes all the elements whose center lie in the Region of Interest of
        an image f.
        Usage :
            m.RemoveElemsOutsideRoi(cam, roi)

        where  roi = f.SelectROI()
        """
        if self.dim == 3:
            if cam is None:
                u, v, w = self.n[:, 0], self.n[:, 1], self.n[:, 2]
            else:
                u, v, w = cam.P(self.n[:, 0], self.n[:, 1], self.n[:, 2])
            for je in self.e.keys():
                umoy = np.mean(u[self.e[je]], axis=1)
                vmoy = np.mean(v[self.e[je]], axis=1)
                wmoy = np.mean(w[self.e[je]], axis=1)
                inside = isInBox(roi, umoy, vmoy, wmoy)
        else:
            if cam is None:
                u, v = self.n[:, 0], self.n[:, 1]
            else:
                v, u = cam.P(self.n[:, 0], self.n[:, 1])
            for je in self.e.keys():
                umoy = np.mean(u[self.e[je]], axis=1)
                vmoy = np.mean(v[self.e[je]], axis=1)
                inside = isInBox(roi, umoy, vmoy)
        self.e[je] = self.e[je][inside, :]

    def RemoveDoubleNodes(self):
        """
        Removes the double nodes thus changes connectivity
        Warning: both self.e and self.n are modified!

        Usage :
            m.RemoveDoubleNodes()

        """
        nnew = np.unique(self.n, axis=0)
        table = -1 * np.ones(len(self.n), dtype='int')
        eps = 1e-5 * self.GetApproxElementSize()
        for jn in range(len(self.n)):
            rep, = np.where(np.linalg.norm(nnew-self.n[jn, :][np.newaxis],
                                           axis=1) < eps)
            table[jn] = rep
        for k in self.e.keys():
            self.e[k] = table[self.e[k]]
        self.n = nnew

    def RemoveUnusedNodes(self):
        """
        Removes all the nodes that are not connected to an element and
        renumbers the element table. Both self.e and self.n are changed

        Usage :
            m.RemoveUnusedNodes()

        Returns
        -------
        None.

        """
        used_nodes = np.zeros(0, dtype=int)
        for ie in self.e.keys():
            used_nodes = np.hstack((used_nodes, self.e[ie].ravel()))
        used_nodes = np.unique(used_nodes)
        table = np.zeros(len(self.n), dtype=int)
        table[used_nodes] = np.arange(len(used_nodes))
        self.n = self.n[used_nodes, :]
        for ie in self.e.keys():
            self.e[ie] = table[self.e[ie]]

    def BuildBoundaryMesh(self):
        """
        Builds edge elements corresponding to the edges of 2d Mesh m
        and Tet in 3d.
        """
        edgel = {}  # edge lin
        edgeq = {}  # edge qua
        tril = {}  # tri face lin
        tril_sort = {}
        qual = {}  # qua face lin
        qual_sort = {}
        for je in self.e.keys():
            if je in [9, 16, 10]:  # quadratic 2d
                if je in [16, 10]:  # qua8 et qua9
                    n1 = self.e[je][:, :4].ravel()
                    n2 = np.c_[self.e[je][:, 1:4], self.e[je][:, 0]].ravel()
                    n3 = self.e[je][:, 4:8].ravel()
                else:  # tri6
                    n1 = self.e[je][:, :3].ravel()
                    n2 = np.c_[self.e[je][:, 1:3], self.e[je][:, 0]].ravel()
                    n3 = self.e[je][:, 3:].ravel()
                a = np.sort(np.c_[n1, n2, n3], axis=1)
                for i in range(len(a)):
                    tedge = tuple(a[i, :])
                    if tedge in edgeq.keys():
                        edgeq[tedge] += 1
                    else:
                        edgeq[tedge] = 1
            elif je in [2, 3]:  # linear 2d
                n1 = self.e[je].ravel()
                n2 = np.c_[self.e[je][:, 1:], self.e[je][:, 0]].ravel()
                a = np.sort(np.c_[n1, n2], axis=1)
                for i in range(len(a)):
                    tedge = tuple(a[i, :])
                    if tedge in edgel.keys():
                        edgel[tedge] += 1
                    else:
                        edgel[tedge] = 1
            elif je in [4, 11]:  # tet4
                for ie in range(len(self.e[je])):
                    ei = self.e[je][ie, :4]
                    tril, tril_sort = AddChildElem(tril, ei[[0, 1, 2]],
                                                   tril_sort)
                    tril, tril_sort = AddChildElem(tril, ei[[0, 1, 3]],
                                                   tril_sort)
                    tril, tril_sort = AddChildElem(tril, ei[[0, 2, 3]],
                                                   tril_sort)
                    tril, tril_sort = AddChildElem(tril, ei[[1, 2, 3]],
                                                   tril_sort)
            elif je in [5, 17]:  # hex8
                for ie in range(len(self.e[je])):
                    ei = self.e[je][ie][:8]
                    qual, qual_sort = AddChildElem(qual, ei[[0, 1, 2, 3]],
                                                   qual_sort)
                    qual, qual_sort = AddChildElem(qual, ei[[4, 5, 6, 7]],
                                                   qual_sort)
                    qual, qual_sort = AddChildElem(qual, ei[[0, 1, 5, 4]],
                                                   qual_sort)
                    qual, qual_sort = AddChildElem(qual, ei[[1, 2, 6, 5]],
                                                   qual_sort)
                    qual, qual_sort = AddChildElem(qual, ei[[2, 3, 7, 6]],
                                                   qual_sort)
                    qual, qual_sort = AddChildElem(qual, ei[[0, 3, 7, 4]],
                                                   qual_sort)
        elems = dict()
        # linear edges
        if len(edgel):
            (rep,) = np.where(np.array(list(edgel.values())) == 1)
            edgel = np.array(list(edgel.keys()))[rep, :]
            elems[1] = edgel
        # quadratic edges
        if len(edgeq):
            (rep,) = np.where(np.array(list(edgeq.values())) == 1)
            edgeq = np.array(list(edgeq.keys()))[rep, :]
            elems[8] = edgeq
        # tri faces
        if len(tril):
            (rep,) = np.where(np.array(list(tril.values())) == 1)
            tril = np.array(list(tril_sort.values()))[rep, :]
            elems[2] = tril
        # qua faces
        if len(qual):
            (rep,) = np.where(np.array(list(qual.values())) == 1)
            qual = np.array(list(qual_sort.values()))[rep, :]
            elems[3] = qual
        edgem = Mesh(elems, self.n, self.dim)
        return edgem

    def SelectPoints(self, n=-1, title=None):
        """
        Selection of points coordinates by hand in a mesh.
        """
        plt.figure()
        self.Plot()
        figManager = plt.get_current_fig_manager()
        if hasattr(figManager.window, 'showMaximized'):
            figManager.window.showMaximized()
        else:
            if hasattr(figManager.window, 'maximize'):
                figManager.resize(figManager.window.maximize())
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
        if hasattr(figManager.window, 'showMaximized'):
            figManager.window.showMaximized()
        else:
            if hasattr(figManager.window, 'maximize'):
                figManager.resize(figManager.window.maximize())
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

    def SelectNodesBox(self, box=None):
        """
        Selection of all the nodes of a mesh lying in a box defined by two
        points clics.
        """
        if box is None:
            plt.figure()
            self.Plot()
            figManager = plt.get_current_fig_manager()
            if hasattr(figManager.window, 'showMaximized'):
                figManager.window.showMaximized()
            else:
                if hasattr(figManager.window, 'maximize'):
                    figManager.resize(figManager.window.maximize())
            plt.title("Select 2 points... and press enter")
            pts1 = np.array(plt.ginput(2, timeout=0))
            plt.close()
            inside = (
                (self.n[:, 0] > pts1[0, 0])
                * (self.n[:, 0] < pts1[1, 0])
                * (self.n[:, 1] > pts1[0, 1])
                * (self.n[:, 1] < pts1[1, 1])
            )
        else:
            if self.dim == 3:
                inside = isInBox(box, self.n[:, 0], self.n[:, 1], self.n[:, 2])
            else:
                inside = isInBox(box, self.n[:, 0], self.n[:, 1])
        (nset,) = np.where(inside)
        self.Plot()
        if self.dim == 3:
            ax = plt.gca()
            ax.plot(self.n[nset, 0], self.n[nset, 1], self.n[nset, 2], "ro")
        else:
            plt.plot(self.n[nset, 0], self.n[nset, 1], "ro")
        return nset

    def SelectLine(self, eps=1e-8):
        """
        Selection of the nodes along a line defined by 2 nodes.
        """
        plt.figure()
        self.Plot()
        figManager = plt.get_current_fig_manager()
        if hasattr(figManager.window, 'showMaximized'):
            figManager.window.showMaximized()
        else:
            if hasattr(figManager.window, 'maximize'):
                figManager.resize(figManager.window.maximize())
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
        (rep,) = np.where(abs(self.n.dot(n) - c) < eps)
        c1 = v.dot(self.n[n1, :])
        c2 = v.dot(self.n[n2, :])
        nrep = self.n[rep, :]
        (rep2,) = np.where(((nrep.dot(v) - c1)
                            * (nrep.dot(v) - c2)) < nv * 1e-2)
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
        if hasattr(figManager.window, 'showMaximized'):
            figManager.window.showMaximized()
        else:
            if hasattr(figManager.window, 'maximize'):
                figManager.resize(figManager.window.maximize())
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

    def PlaneWave(self, T):
        used_nodes = self.conn[:, 0] > -1
        V = np.zeros(self.ndof)
        V[self.conn[used_nodes, 0]] = np.cos(self.n[used_nodes, 1] / T * 2 * np.pi)
        return V

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
        if self.dim == 3:
            tx = np.zeros(self.ndof)
            tx[self.conn[:, 0]] = 1
            ty = np.zeros(self.ndof)
            ty[self.conn[:, 1]] = 1
            tz = np.zeros(self.ndof)
            tz[self.conn[:, 2]] = 1
            v = self.n - np.mean(self.n, axis=0)
            amp = np.max(np.linalg.norm(v, axis=1))
            rx = np.zeros(self.ndof)
            rx[self.conn] = np.c_[0*v[:, 0], v[:, 2], -v[:, 1]] / amp
            ry = np.zeros(self.ndof)
            ry[self.conn] = np.c_[-v[:, 2], 0*v[:, 1], v[:, 0]] / amp
            rz = np.zeros(self.ndof)
            rz[self.conn] = np.c_[v[:, 1], -v[:, 0], 0*v[:, 2]] / amp
            return tx, ty, tz, rx, ry, rz
        else:
            tx = np.zeros(self.ndof)
            tx[self.conn[:, 0]] = 1
            ty = np.zeros(self.ndof)
            ty[self.conn[:, 1]] = 1
            v = self.n-np.mean(self.n, axis=0)
            v = np.c_[-v[:, 1], v[:, 0]] / np.max(np.linalg.norm(v, axis=1))
            rz = np.zeros(self.ndof)
            rz[self.conn] = v
            return tx, ty, rz

    def MedianFilter(self, U):
        """
        Compute the median filter of a displacement field. Replace the
        nodal values by the median value of the first neighbors.

        Parameters
        ----------
        U : NUMPY.ARRAY
            DOF vector of the input displacement field

        Returns
        -------
        Um : NUMPY.ARRAY
            DOF vector of the filtered displacement field

        """
        usnd, = np.where(self.conn[:, 0] > -1)
        Um = U.copy()
        for j in range(len(usnd)):
            jn = usnd[j]
            vjn = np.zeros(0, dtype=int)
            for et in self.e.keys():
                e, p = np.where(self.e[et] == jn)
                vjn = np.append(vjn, self.e[et][e, :].ravel())
            vjn = np.unique(vjn)
            Um[self.conn[jn, 0]] = np.median(U[self.conn[vjn, 0]])
            Um[self.conn[jn, 1]] = np.median(U[self.conn[vjn, 1]])
        return Um


    def Extrude(self, lz, nz, order=1):
        import gmsh
        gmsh.initialize()
        gmsh.model.add("lug")
        m = self.Copy()
        m.RemoveUnusedNodes()
        m.KeepSurfElems()
        for i in range(len(m.n)):
            gmsh.model.geo.addPoint(m.n[i, 0], m.n[i, 1], 0, tag=i+1)
        ied = 1
        iel = 1
        recombine = True
        for ik in m.e.keys():
            if ik in {2, 9}:  # triangles
                recombine = False
                m.e[ik] += 1
                for i in range(len(m.e[ik])):
                    gmsh.model.geo.addLine(m.e[ik][i, 0], m.e[ik][i, 1], tag=ied)
                    gmsh.model.geo.mesh.setTransfiniteCurve(ied, 1)
                    ied += 1
                    gmsh.model.geo.addLine(m.e[ik][i, 1], m.e[ik][i, 2], tag=ied)
                    gmsh.model.geo.mesh.setTransfiniteCurve(ied, 1)
                    ied += 1
                    gmsh.model.geo.addLine(m.e[ik][i, 2], m.e[ik][i, 0], tag=ied)
                    gmsh.model.geo.mesh.setTransfiniteCurve(ied, 1)
                    ied += 1
                    gmsh.model.geo.addCurveLoop([ied-3, ied-2, ied-1], iel)
                    gmsh.model.geo.addPlaneSurface([iel], iel)
                    gmsh.model.geo.mesh.setTransfiniteSurface(
                        iel, "Left", m.e[ik][i, :3].tolist())
                    iel += 1
            elif ik in {3, 10, 16}:  # quadranles
                m.e[ik] += 1
                recombine = True
                for i in range(len(m.e[ik])):
                    gmsh.model.geo.addLine(m.e[ik][i, 0], m.e[ik][i, 1], tag=ied)
                    gmsh.model.geo.mesh.setTransfiniteCurve(ied, 1)
                    ied += 1
                    gmsh.model.geo.addLine(m.e[ik][i, 1], m.e[ik][i, 2], tag=ied)
                    gmsh.model.geo.mesh.setTransfiniteCurve(ied, 1)
                    ied += 1
                    gmsh.model.geo.addLine(m.e[ik][i, 2], m.e[ik][i, 3], tag=ied)
                    gmsh.model.geo.mesh.setTransfiniteCurve(ied, 1)
                    ied += 1
                    gmsh.model.geo.addLine(m.e[ik][i, 3], m.e[ik][i, 0], tag=ied)
                    gmsh.model.geo.mesh.setTransfiniteCurve(ied, 1)
                    ied += 1
                    gmsh.model.geo.addCurveLoop([ied-4, ied-3, ied-2, ied-1], iel)
                    gmsh.model.geo.addPlaneSurface([iel], iel)
                    gmsh.model.geo.mesh.setTransfiniteSurface(
                        iel, "Left", m.e[ik][i, :4].tolist())
                    iel += 1
        # gmsh.model.geo.addSurfaceLoop([*range(1, iel)], iel)
        if recombine:
            gmsh.option.setNumber('Mesh.RecombineAll', 1)
            gmsh.option.setNumber('Mesh.RecombinationAlgorithm', 1)
            gmsh.option.setNumber('Mesh.Recombine3DLevel', 2)
        listpairs = [(2, i) for i in range(1, iel)]
        gmsh.model.geo.extrude(listpairs, 0, 0, lz, [nz, ], recombine=recombine)
        gmsh.option.setNumber('Mesh.ElementOrder', order)
        gmsh.option.setNumber('General.Verbosity', 1)
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(3)
        gmsh.write("lug.msh")
        # if '-nopopup' not in sys.argv:
        #     gmsh.fltk.run()
        gmsh.finalize()
        m = ReadMesh("lug.msh", 3)
        m.KeepVolElems()
        # m.Plot()
        return m
