# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 08:16:13 2022

@author: passieux
"""
import numpy as np
import matplotlib.pyplot as plt
import gmsh
from .mesh import Mesh
from .camera import Camera
from skimage import measure

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
    if 15 in elems:
        del elems[15]  # remove points
    if 1 in elems:
        del elems[1]  # remove segments
    if 8 in elems:
        del elems[8]  # remove quadratic segments
    m = Mesh(elems, nodes[:, :dim], dim)
    return m


#%% Plot a polygon
def PlotLS(ls, coul):
    for i in range(len(ls)):
        plt.plot(ls[i][:,1], ls[i][:,0], coul)
    plt.axis('equal')

def RemoveSegments(ls):
    PlotLS(ls, 'k-')
    rep = []
    for i in range(len(ls)):
        if len(ls[i]) < 4:
            rep += [i]
            PlotLS([ls[i]], 'r-')
    return np.delete(ls, rep)

#%% finds the polygons included in other polygons
def lsio(lsi):
    # determine if the line loop truns clockwise or counterclockwise
    c = np.mean(lsi, axis=0)
    sc = 0
    for i in range(len(lsi)-1):
        xn = lsi[[i,i+1], :]
        v = np.diff(xn, axis=0)[0]
        v /= np.linalg.norm(v)
        w = lsi[i,:] - c
        w /= np.linalg.norm(w)
        sc = np.diff(w[::-1]*v)[0]
    return int(sc>0)

def RayCasting(lsi, p):
    # Ray Casting Algorithm (finds if a point p is in a polygon lsi)
    #      cast a ray from p vertical in the top direction and count the
    #      number of intersected segments.
    nc = 0
    # compute eps with first edge length
    eps = 1e-5 * np.linalg.norm(np.diff(lsi[[0,1],:], axis=0))
    for i in range(len(lsi)-1):
        xn = lsi[[i,i+1], :]
        w = np.mean(xn, axis=0) - p
        # comparing the x coordinates of the 3 points.
        if w[1] > 0 and abs(xn[0,0] - p[0]) + abs(xn[1,0] - p[0]) - abs(xn[1,0] - xn[0,0]) < eps:
            # if xp coorespond to the x the first node, dont consider this edge
            if abs(xn[0,0] - p[0]) > eps:
                nc += 1
    return nc % 2

#%% bulk meshing from polygon

def MeshFromLS(ls, lc):
    lsc = np.zeros(len(ls), dtype=int)
    for i in range(len(ls)):
        lsc[i] = lsio(ls[i])    
    white, = np.where(lsc==0)
    black, = np.where(lsc)
    connect = -np.ones(len(ls))
    for i in range(len(white)):
        for j in range(len(black)):
            if RayCasting(ls[black[j]], ls[white[i]][0,:]):
                connect[white[i]] = black[j]
                break
    gmsh.initialize()
    gmsh.model.add("P")    
    nn = 1
    nl = 1
    curvedloop = dict()
    for j in range(len(ls)):
        for i in range(len(ls[j])-1):
            gmsh.model.geo.addPoint(ls[j][i,0], ls[j][i,1], 0, lc, nn+i)
        for i in range(len(ls[j])-2):
            gmsh.model.geo.addLine(nn+i, nn+i+1, nl+i)
        gmsh.model.geo.addLine(nn+i+1, nn, nl+i+1)
        curvedloop[j] = nl + np.arange(len(ls[j]) - 1)
        nn += len(ls[j]) - 1
        nl += len(ls[j]) - 1    
    ncl=1
    for i in range(len(ls)):
        if lsc[i]:
            rep, = np.where(connect==i)
            cvl = curvedloop[i]
            for j in rep:
                cvl = np.append(cvl ,-curvedloop[j])
            gmsh.model.geo.addCurveLoop(cvl, ncl)
            gmsh.model.geo.addPlaneSurface([ncl], ncl)
            ncl += 1    
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)
    m = Gmsh2Mesh(gmsh)
    gmsh.finalize()    
    return m

def MeshFromImage(f, h, appls=1):
    """
    Builds a mesh from a graylevel image.

    Parameters
    ----------
    f : NUMPY ARRAY (bool pixel map)
        1: inside, 0: outside the domain to mesh
    h : FLOAT
        approximate finite elements size in pixels
    appls : BOOL
        If 0, does not approximate the list of segments 
        Set appls=4 to allow to move each point of the polygon up to
        4 pixels to regularize its shape. Default 1.

    Returns
    -------
    m : PYXEL.MESH
        mesh of the domain where f equals to 0

    """
    pix = ((1-f)*255).astype('uint8')
    ls = measure.find_contours(pix, 127 ,fully_connected='low') 
    ls = [measure.approximate_polygon(i, appls) for i in ls]
    m = MeshFromLS(ls, h)
    p = np.array([1, 0., 0., np.pi/2])
    cam = Camera(p)
    return m, cam
