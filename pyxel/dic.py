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
from .image import Image
from .camera import Camera
from .mesh import StructuredMesh
#import matplotlib.pyplot as plt

#import pdb
#pdb.set_trace()
    
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
    m = StructuredMesh(roi, dx, typel=typel)
    m.n[:,1] *= -1
    p = np.array([1., 0., 0., 0.])
    cam = Camera(p)
    return m, cam


def Correlate(f, g, m, cam, dic=None, H=None, U0=None, l0=None, Basis=None, 
              L=None, eps=None, maxiter=30, disp=True):
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
        print('reduced basis')
        H_LU = splalg.splu(Basis.T @ H @ Basis)
        print(Basis.T @ H @ Basis)
    else:
        if l0 is not None:
            # Tikhonov regularisation
            if L is None:
                L = m.Tikhonov()
            used_nodes = m.conn[:, 0] > 0
            T = 10 * m.GetApproxElementSize()
            V = np.zeros(m.ndof)
            V[m.conn[used_nodes, 0]] = np.cos(m.n[used_nodes, 1] / T * 2 * np.pi)
            H0 = V.dot(H.dot(V))
            L0 = V.dot(L.dot(V))
            l = (l0/T)**2 * H0 / L0
            H_LU = splalg.splu(H + l * L)
            print(l)
        else:
            if disp:
                print("no reg")
            H_LU = splalg.splu(H)
    stdr_old = 100
    for ik in range(0, maxiter):
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
        stdr = np.std(res)
        if disp:
            print("Iter # %2d | std(res)=%2.2f gl | dU/U=%1.2e" % (ik + 1, stdr, err))
        if err < eps or abs(stdr - stdr_old) < 1e-3:
            break
        stdr_old = stdr
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
        - set l0 to None to automatically compute l0
        - set l0 to 0 to descativate regularization in the multiscale process
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
    # estimate average element size in pixels
    aes = int(m.GetApproxElementSize(cam))
    if l0 is None:
        # l0 = 0.0
        # for et in m.e.keys():
        #     n1 = m.n[m.e[et][:, 0]]
        #     n2 = m.n[m.e[et][:, 1]]
        #     l0 = max(l0, 4 * min(np.linalg.norm(n1 - n2, axis=1)))
        l0 = 30/cam.get_p()[0]
        print('Auto reg. length l0 = %2.3e' % l0)
    if l0 == 0:
        l0 = None

    print('Average Element Size in px: %3d' % aes)
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
        m2.DICIntegrationFast(aes // (2**iscale))
        # m2.DICIntegration(cam2)
        # plt.figure()
        # g.Plot()
        # u, v = cam2.P(m2.n[:, 0], m2.n[:, 1])
        # m.Plot(n=np.c_[v, u], edgecolor="y", alpha=0.6)
        # plt.figure()
        # m2.Plot()
        # plt.plot(m2.pgx,m2.pgy,'k.')
        # plt.axis('equal')

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


def DISFlowInit(imf, img, m=None, cam=None, meth='MEDIUM'):
    """
    Compute initial guess using OpenCV DISFlow routine

    Parameters
    ----------
    imf : PYXEL.IMAGE
        Reference image
    img : PYXEL.IMAGE
        Deformed state image
    m : PYXEL.MESH
        finite element mesh
        if None > return the result of DISFlow
    cam : PYXEL.CAMERA
        Camera model
        if None > return the result of DISFlow
    meth : STRING, optional
        'MEDIUM': medium option of DISFlow
        'FAST': fast option of DISFlow
        'ULTRAFAST': ultrafast option of DISFlow
        otherwise: manual settings for DISFlow
        DESCRIPTION. The default is 'MEDIUM'.

    Returns
    -------
    u : NUMPY.ARRAY
        initial guess DOF vector if m and cam are given
        returns pixmaps U, V is cam is None

    """
    import cv2
    if meth == 'MEDIUM':
        flow=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    elif meth == 'FAST':
        flow=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    elif meth == 'ULTRAFAST':
        flow=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
    else:
        # MANUAL
        flow = cv2.DISOpticalFlow_create()
        flow.setVariationalRefinementAlpha(20.0)		# Laplacian of displacment
        flow.setVariationalRefinementGamma(10.0)		# Gradient of image consistency
        flow.setVariationalRefinementDelta(5.0) 	    # Optical flow
        flow.setVariationalRefinementIterations(5)	    # Number of iterations
        flow.setFinestScale(0)
        flow.setPatchSize(13)
        flow.setPatchStride(7)
    UV = flow.calc(imf.pix.astype('uint8'), img.pix.astype('uint8'), None)
    U = UV[::,::,0]
    V = UV[::,::,1]
    if m is None:
        return U, V
    else: 
        u, v = cam.P(m.n[:,0],m.n[:,1])
        fp = imf.Copy()
        fp.pix = V
        fp.BuildInterp()
        du = fp.Interp(u, v)
        fp.pix = U
        fp.BuildInterp()
        dv = fp.Interp(u, v)
        
        Xdx, Ydy = cam.PinvNL(u+du, v+dv)
        Ux = Xdx - m.n[:,0]
        Uy = Ydy - m.n[:,1]
        
        if len(m.conn) == 0 :
            m.Connectivity()
        u = np.zeros(m.ndof)
        u[m.conn[:,0]] = Ux
        u[m.conn[:,1]] = Uy
        return u