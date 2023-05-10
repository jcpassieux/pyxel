# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 08:16:13 2022

@author: passieux
"""
import numpy as np
import matplotlib.pyplot as plt
import gmsh
from .mesh import ReadMesh
from .camera import Camera
from .image import Image
from .mesh import Mesh
from skimage import measure

#%%

def StructuredMeshQ4(box, dx):
    """Build a structured linear Q4 mesh from two points coordinates (box)
    box = np.array([[xmin, ymin],
                    [xmax, ymax]])   in mesh unit
    dx = [dx, dy]: average element size (can be scalar)  in mesh unit"""
    dbox = box[1] - box[0]
    NE = (dbox / dx).astype(np.int64)
    NE = np.max(np.c_[NE,np.ones(2,dtype=int)],axis=1)
    X, Y = np.meshgrid(np.linspace(box[0, 0], box[1, 0], NE[0] + 1),
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
    X, Y = np.meshgrid(np.linspace(box[0, 0], box[1, 0], NE[0] + 1),
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
    X, Y = np.meshgrid(np.linspace(box[0, 0], box[1, 0], 2 * NE[0] + 1),
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


def StructuredMeshHex8(box, lc):
    """
    Meshes a box with hexahedral elements 

    Parameters
    ----------
    box = np.array([[xmin, ymin, zmin],
                    [xmax, ymax, zmax]])    in mesh unit
    lc : mesh length.

    Returns
    -------
    m : Structured C8 mesh.

    """
    
    # Create a structured hexahedral mesh 
    lx = box[1, 0] - box[0, 0]
    ly = box[1, 1] - box[0, 1]
    lz = box[1, 2] - box[0, 2]

    Nx = int( np.round( lx / lc ) ) 
    Ny = int( np.round( ly / lc ) ) 
    Nz = int( np.round( lz / lc ) )  
     
    x = np.linspace( 0, lx, Nx+1 )
    y = np.linspace( 0, ly, Ny+1 )
    z = np.linspace( 0, lz, Nz+1 )
    
    nbf_xi = len(x); nbf_eta = len(y); nbf_zeta = len(z) 
 
    Xtot = np.kron(np.ones(nbf_eta*nbf_zeta), x)
    Ytot = np.kron(np.ones(nbf_zeta), np.kron(y,np.ones(nbf_xi)))
    Ztot = np.kron(z,np.ones(nbf_eta*nbf_xi)) 
    
    nodes  = np.c_[Xtot,Ytot,Ztot]
    
    # Creates element connectivity 
    p = 1 ; q = 1 ; r = 1 
    nxi, neta, nzeta = Nx, Ny, Nz 
    bf_xi_index   = (np.kron( np.ones(nxi), np.arange(p+1) ) + np.kron(np.arange(nxi), np.ones(p+1) )).reshape((nxi,p+1))
    bf_eta_index  = (np.kron( np.ones(neta), np.arange(q+1) ) + np.kron(np.arange(neta), np.ones(q+1) )).reshape((neta,q+1))
    bf_zeta_index = (np.kron( np.ones(nzeta), np.arange(r+1) ) + np.kron(np.arange(nzeta), np.ones(r+1) )).reshape((nzeta,r+1))
    
    noelem = np.kron(  np.ones_like(bf_zeta_index) ,  np.kron(np.ones_like(bf_eta_index)  ,bf_xi_index) )  + \
          (nxi+p)*np.kron(  np.ones_like(bf_zeta_index), np.kron( bf_eta_index, np.ones_like(bf_xi_index) ) )  + \
          (nxi+p)*(neta+q)*np.kron(  bf_zeta_index, np.kron( np.ones_like(bf_eta_index),np.ones_like(bf_xi_index) ) )  
    
    els = noelem.astype('int32') 
    els = els[:,[2,0,1,3,6,4,5,7]] # In order to get the correct ordering of Meshio
    # els = els[:,[]] # In order to get tge correct ordering of Gmsh 
    e = {5: els}
    m = Mesh(e , nodes, 3)
    
    m.n[:,0] += box[0, 0]
    m.n[:,1] += box[0, 1]
    m.n[:,2] += box[0, 2]

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
        4: 4-node first order tetrahedron (Tet4)
        5: 8-node first order hexahedron (Hex8)        
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
    # elif typel == 4:
    #     return StructuredMeshTet4(box, dx)
    elif typel == 5:
        return StructuredMeshHex8(box, dx)

#%%

def TetraMeshCylinder(x0, y0, z0, R, h, lc):
    """
    Meshes a cylinder geometry with tetrahedral elements 

    Parameters
    ----------
    x0,y0,z0 : Center of the base of cylinder  
    R : Radius
    h : height
    lc : mesh length.

    Returns
    -------
    m : mesh object

    """
    print('cylinder')
    gmsh.initialize()
    gmsh.clear()
    gmsh.model.add("cylinder")
    gmsh.model.geo.addPoint( x0, y0, z0, lc, 1)  
    gmsh.model.geo.addPoint( x0-R, y0, z0, lc, 2) 
    gmsh.model.geo.addPoint(x0+R, y0, z0, lc, 3) 
    gmsh.model.geo.addPoint( x0, y0+R, z0, lc, 4) 
    gmsh.model.geo.addPoint( x0, y0-R, z0, lc, 5)
    gmsh.model.geo.addCircleArc(2, 1, 4, 1)   
    gmsh.model.geo.addCircleArc(4, 1, 3, 2) 
    gmsh.model.geo.addCircleArc(3, 1, 5, 3)  
    gmsh.model.geo.addCircleArc(5, 1, 2, 4)   
    gmsh.model.geo.addCurveLoop([1,2,3,4],1)  
    gmsh.model.geo.addPlaneSurface([1],1)    
    gmsh.model.geo.extrude([(2,1)], 0, 0, h)       
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)
    
    nums, nodes, e = gmsh.model.mesh.getNodes()
    nodes = nodes.reshape((len(nums), 3))
    
    nums, els = gmsh.model.mesh.getElementsByType(4)
    nnd = len(els)//len(nums)
    els = els.reshape((len(nums),nnd)) - 1
    e = {4: els.astype(int)}
    m= Mesh(e, nodes, 3)
    # m.CleanMesh()
    return m 
 
def TetraMeshBox(box, dx):
    """
    Meshes a box with tetrahedral elements 

    Parameters
    ----------
    box = np.array([[xmin, ymin, zmin],
                    [xmax, ymax, zmax]])    in mesh unit
    dx : average element size (scalar) in mesh unit

    Returns
    -------
    m : mesh object 

    """
    gmsh.initialize()
    gmsh.clear()
    gmsh.model.add("box")
    xmin, ymin, zmin = box[0, :]
    xmax, ymax, zmax = box[1, :]
    gmsh.model.geo.addPoint( xmin, ymin , zmin, dx, 1)
    gmsh.model.geo.addPoint( xmax, ymin , zmin, dx, 2)
    gmsh.model.geo.addPoint( xmax, ymax , zmin, dx, 3)
    gmsh.model.geo.addPoint( xmin, ymax , zmin, dx, 4)    
    gmsh.model.geo.addLine(1,2,1)
    gmsh.model.geo.addLine(2,3,2)
    gmsh.model.geo.addLine(3,4,3)
    gmsh.model.geo.addLine(4,1,4)    
    gmsh.model.geo.addCurveLoop([1,2,3,4], 1)
    gmsh.model.geo.addPlaneSurface([1],1)    
    gmsh.model.geo.extrude([(2, 1)], 0, 0, zmax-zmin)
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)
    nums, nodes, e = gmsh.model.mesh.getNodes()
    nodes = nodes.reshape((len(nums), 3))
    nums, els = gmsh.model.mesh.getElementsByType(4)
    nnd = len(els) // len(nums)
    els = els.reshape((len(nums), nnd)) - 1
    e = {4: els.astype(int)}
    m  = Mesh(e , nodes, 3)
    return m 

#%% 

# def Gmsh2Mesh(gmsh, dim=2):
#     """
#     Bulding pyxel mesh from gmsh python object

#     Parameters
#     ----------
#         gmsh : python gmsh object

#     EXAMPLE:
#     ----------
#         import gmsh
#         gmsh.initialize()
#         gmsh.model.add("P")
#         lc = 0.02
#         gmsh.model.geo.addPoint(0, 0.0, 0, 4 * lc, 1)
#         gmsh.model.geo.addPoint(1, 0.0, 0, lc, 2)
#         gmsh.model.geo.addPoint(1, 0.5, 0, lc, 3)
#         gmsh.model.geo.addPoint(0, 0.5, 0, 4 * lc, 4)
#         gmsh.model.geo.addLine(1, 2, 1)
#         gmsh.model.geo.addLine(2, 3, 2)
#         gmsh.model.geo.addLine(3, 4, 3)
#         gmsh.model.geo.addLine(4, 1, 4)
#         gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
#         gmsh.model.geo.addPlaneSurface([1], 1)
#         gmsh.model.geo.synchronize()
#         gmsh.model.mesh.generate(2)
#         m = px.Gmsh2Mesh(gmsh)
#         m.Plot()
        
#     """
#     # Get direct full node list
#     nums, nodes, e = gmsh.model.mesh.getNodes()
#     nodes = nodes.reshape((len(nums), 3))
#     elems = dict()
#     # Get the Element by type
#     for et in gmsh.model.mesh.getElementTypes():
#         nums, els = gmsh.model.mesh.getElementsByType(et)
#         nnd = len(els) // len(nums)
#         elems[et] = els.reshape((len(nums), nnd)).astype(int) - 1
#     if 15 in elems:
#         del elems[15]  # remove points
#     if 1 in elems:
#         del elems[1]  # remove segments
#     if 8 in elems:
#         del elems[8]  # remove quadratic segments
#     m = Mesh(elems, nodes[:, :dim], dim)
#     return m


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
    # ls list of segments
    # lc approximate element size
    lsc = np.zeros(len(ls), dtype=int)
    for i in range(len(ls)):
        lsc[i] = lsio(ls[i])
    if np.max(lsc) == 0:
        raise Exception('in pyxel::MeshFromLS, Only counterclockwise list of segments, use flipud')
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
    gmsh.write('tmp.msh')
    gmsh.finalize()
    m = ReadMesh('tmp.msh')
    print(m.e.keys())
    if 15 in m.e:
        del m.e[15]  # remove vertex
    if 1 in m.e:
        del m.e[1]  # remove segments
    if 8 in m.e:
        del m.e[8]  # remove quadratic segments
    return m

def MeshFromImage(fpix, h, appls=None):
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
        4 pixels to regularize its shape. Default None (=no approx.)

    Returns
    -------
    m : PYXEL.MESH
        mesh of the domain where f equals to 0

    """
    pix = ((1-fpix)*255).astype('uint8')
    ls = measure.find_contours(pix, 127 ,fully_connected='low') 
    #PlotLS(ls,'r.-')
    if appls is not None:
        ls = [measure.approximate_polygon(i, appls) for i in ls]
        # PlotLS(ls,'k.-')
    m = MeshFromLS(ls, h)
    # p = np.array([1, 0., 0., np.pi/2])
    # a = Image('')
    # a.pix = f
    m.n = np.c_[m.n[:,1], -m.n[:,0]]
    p = np.array([1, 0., 0., 0])
    cam = Camera(p)
    return m, cam

xi = np.arange(-50,51)
X, Y = np.meshgrid(xi, xi)
Z = np.sqrt(X**2+Y**2)
fpix = Z<20

plt.imshow(fpix)
