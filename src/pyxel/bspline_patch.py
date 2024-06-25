# -*- coding: utf-8 -*-
"""
Finite Element Digital Image Correlation method

@author: A. Rouawne, INSA Toulouse, 2023

pyxel

PYthon library for eXperimental mechanics using Finite ELements

"""
import os
import numpy as np
import scipy as sp
from .bspline_routines import bspdegelev, bspkntins, global_basisfuns, Get2dBasisFunctionsAtPts, global_basisfunsWd
import scipy.sparse as sps
import matplotlib.pyplot as plt
from .camera import Camera
from .mesher import StructuredMeshQ4


class BSplinePatch(object):
    def __init__(self, ctrlPts, degree, knotVect):
        """
        Nurbs surface from R^2 (xi,eta)--->R^2 (x,y)
        ctrlPts = [X,Y]
        """
        self.dim = 2
        self.ctrlPts = ctrlPts
        self.n = self.CrtlPts2N()
        self.degree = np.array(degree)
        self.knotVect = knotVect

        self.ien = 0              # NURBS Connectivity p-uplet (IENu,IENv)
        # Connectivity: Control Point number (column) of each element (line)
        self.noelem = 0
        self.tripleien = 0        # Correspondance between 2D elements and 1D elements
        self.iperM = 0            # Sparse matrix for coincident control points

        # Dictionary containing the value det(Jacobian)*Gauss_Weights*ElementMeasure/4
        self.wdetJmes = 0
        # Dictionary containing the measure of the elements in the ordre of listeltot
        self.mes = 0

        # Dictionary of basis functions evaluated at gauss points
        self.phi = np.empty(0)
        # Dictionary containing the derivative of Basis function in x direction
        self.dphidx = np.empty(0)
        # Dictionary containing the derivative of Basis function in y direction
        self.dphidy = np.empty(0)

        """ Attributes when using vectorization  """
        """ In this case, the implicit connectivity of the structured B-spline parametric space is used """
        self.npg = 0
        self.phix = np.empty(0)      # Matrix (N,0)
        self.phiy = np.empty(0)      # Matrix (0,N)
        self.dphixdx = np.empty(0)  # Matrix (dNdx,0)
        self.dphixdy = np.empty(0)  # Matrix (dNdy,0)
        self.dphiydx = np.empty(0)  # Matrix (0,dNdx)
        self.dphiydy = np.empty(0)  # Matrix (0,dNdy)
        self.wdetJ = np.empty(0)        # Integration weights diagonal matrix
        self.pgx = np.empty(0)
        self.pgy = np.empty(0)

        self.phiMatrix = 0
        self.n_elems = 0
        self.pix = 0
        self.piy = 0
        self.piz = 0
        self.integrationCellsCoord = 0
        """ fixed parameters for integration """
        self.nbg_xi = 0
        self.nbg_eta = 0
        self.Gauss_xi = 0
        self.Gauss_eta = 0
        self.wgRef = 0
        self.refGaussTriangle = 0

    def Copy(self):
        m = BSplinePatch(self.ctrlPts.copy(), self.degree.copy(),
                         self.knotVect.copy())
        m.conn = self.conn.copy()
        m.ndof = self.ndof
        m.dim = self.dim
        m.npg = self.npg
        m.pgx = self.pgx.copy()
        m.pgy = self.pgy.copy()
        m.phix = self.phix.copy()
        m.phiy = self.phiy.copy()
        m.wdetJ = self.wdetJ.copy()
        return m

    def IsRational(self):
        return (self.ctrlPts[3] != 1).any()

    def Get_nbf_1d(self):
        """ Get the number of basis functions per parametric direction """
        return self.ctrlPts.shape[1:]

    def Get_nbf(self):
        """ Total number of basis functions """
        return np.product(self.Get_nbf_1d())

    def Get_nbf_elem_1d(self):
        return self.degree + 1

    def Get_nbf_elem(self):
        return np.product(self.degree+1)

    def Get_listeltot(self):
        """ Indices of elements """
        return np.arange(self.ien[0].shape[1]*self.ien[1].shape[1])

    def Get_P(self):
        """  Returns the total"""
        P = np.c_[self.ctrlPts[0].ravel(order='F'),
                  self.ctrlPts[1].ravel(order='F')]
        return P

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

    def Connectivity(self):
        nn = len(self.n)
        self.ndof = nn * self.dim
        self.conn = np.c_[np.arange(nn), np.arange(nn) + nn]

    def CrtlPts2N(self, ctrlPts=None):
        if ctrlPts is None:
            ctrlPts = self.ctrlPts.copy()
        n = np.c_[ctrlPts[0].ravel(order='F'),
                  ctrlPts[1].ravel(order='F')]
        return n

    def N2CrtlPts(self, n=None):
        # n should be in the right order (xi, eta) meshgrid
        if n is None:
            n = self.n.copy()
        nbf = self.Get_nbf_1d()
        ctrlPts = np.array([n[:, 0].reshape(nbf, order='F'),
                            n[:, 1].reshape(nbf, order='F')])
        return ctrlPts

    def BS2FE(self, U, n=[30, 30]):
        xi = np.linspace(self.knotVect[0][self.degree[0]],
                         self.knotVect[0][-self.degree[0]], n[0])
        eta = np.linspace(self.knotVect[1][self.degree[1]],
                          self.knotVect[1][-self.degree[1]], n[1])
        phi, _, _ = self.ShapeFunctionsAtGridPoints(xi, eta)
        x = phi.dot(self.n[:, 0])
        y = phi.dot(self.n[:, 1])
        roi = np.c_[np.ones(2), n].T-1
        mfe = StructuredMeshQ4(roi, 1)
        mfe.n = np.c_[x, y]
        # mfe.Plot()
        mfe.Connectivity()
        V = np.zeros(mfe.ndof)
        V[mfe.conn[:, 0]] = phi.dot(U[self.conn[:, 0]])
        V[mfe.conn[:, 1]] = phi.dot(U[self.conn[:, 1]])
        # mfe.Plot(V*30)
        return mfe, V, phi

    def VTKSol(self, filename, U=None, n=[30, 30]):
        # Surface
        if U is None:
            U = np.zeros(self.ndof)
        m, V, phi = self.BS2FE(U, n)
        m.VTKSol(filename, V)
        # Control mesh
        nbf = self.Get_nbf_1d()
        roi = np.c_[np.ones(2, dtype=int), nbf].T-1
        mfe = StructuredMeshQ4(roi, 1)
        mfe.n = np.c_[self.ctrlPts[0].ravel(),
                      self.ctrlPts[1].ravel()]
        mfe.Connectivity()
        V = U.copy()
        V[self.conn[:, 0]] = U[self.conn[:, 0]].reshape(nbf, order='F').ravel()
        V[self.conn[:, 1]] = U[self.conn[:, 1]].reshape(nbf, order='F').ravel()
        mfe.VTKSol(filename+'_cp', V)

    def Plot(self, U=None, n=None, neval=[30, 30], **kwargs):
        """ Physical elements = Image of the parametric elements on Python """
        alpha = kwargs.pop("alpha", 1)
        edgecolor = kwargs.pop("edgecolor", "k")
        nbf = self.Get_nbf()
        if n is None:
            n = self.Get_P()  # control points
        if U is None:
            U = np.zeros(2*nbf)
        Pxm = n[:, 0] + U[:nbf]
        Pym = n[:, 1] + U[nbf:]

        xi = np.linspace(
            self.knotVect[0][self.degree[0]], self.knotVect[0][-self.degree[0]], neval[0])
        eta = np.linspace(
            self.knotVect[1][self.degree[1]], self.knotVect[1][-self.degree[1]], neval[1])
        # Iso parameters for the elemnts
        xiu = np.unique(self.knotVect[0])
        etau = np.unique(self.knotVect[1])

        # Basis functions
        phi_xi1 = global_basisfunsWd(self.degree[0], self.knotVect[0], xiu)
        phi_eta1 = global_basisfunsWd(self.degree[1], self.knotVect[1], eta)
        phi_xi2 = global_basisfunsWd(self.degree[0], self.knotVect[0], xi)
        phi_eta2 = global_basisfunsWd(self.degree[1], self.knotVect[1], etau)

        phi1 = sps.kron(phi_eta1,  phi_xi1,  'csc')
        phi2 = sps.kron(phi_eta2,  phi_xi2,  'csc')

        xe1 = phi1.dot(Pxm)
        ye1 = phi1.dot(Pym)
        xe2 = phi2.dot(Pxm)
        ye2 = phi2.dot(Pym)

        xe1 = xe1.reshape((xiu.size, neval[1]), order='F')
        ye1 = ye1.reshape((xiu.size, neval[1]), order='F')
        xe2 = xe2.reshape((neval[0], etau.size), order='F')
        ye2 = ye2.reshape((neval[0], etau.size), order='F')

        for i in range(xiu.size):
            # loop on xi
            # Getting one eta iso-curve
            plt.plot(xe1[i, :], ye1[i, :], color=edgecolor,
                     alpha=alpha, **kwargs)

        for i in range(etau.size):
            # loop on eta
            # Getting one xi iso-curve
            plt.plot(xe2[:, i], ye2[:, i], color=edgecolor,
                     alpha=alpha, **kwargs)
        plt.plot(Pxm, Pym, color=edgecolor,
                 alpha=alpha, marker='o', linestyle='')
        plt.axis('equal')

    def DegreeElevation(self, new_degree):
        # m.DegreeElevation(np.array([3, 3]))
        new_degree = np.array(new_degree)
        t = new_degree - self.degree
        # to homogeneous coordinates
        nbf_xi, nbf_eta = self.Get_nbf_1d()
        # Degree elevation along the eta direction
        if t[1] != 0:
            coefs = self.ctrlPts.reshape((2*nbf_xi, nbf_eta), order="F")
            self.ctrlPts, self.knotVect[1] = bspdegelev(
                self.degree[1], coefs, self.knotVect[1], t[1])
            nbf_eta = self.ctrlPts.shape[1]
            self.ctrlPts = self.ctrlPts.reshape(
                (2, nbf_xi, nbf_eta), order="F")
        # Degree elevation along the xi direction
        if t[0] != 0:
            coefs = np.transpose(self.ctrlPts, (0, 2, 1))
            coefs = coefs.reshape((2*nbf_eta, nbf_xi), order="F")
            self.ctrlPts, self.knotVect[0] = bspdegelev(
                self.degree[0], coefs, self.knotVect[0], t[0])
            nbf_xi = self.ctrlPts.shape[1]
            self.ctrlPts = self.ctrlPts.reshape(
                (2, nbf_eta, nbf_xi), order="F")
            self.ctrlPts = np.transpose(self.ctrlPts, (0, 2, 1))

        self.degree = new_degree
        self.n = self.CrtlPts2N()

    def KnotInsertion(self, knots):
        # to homogeneous coordinates
        # Example: m.KnotInsertion([np.array([0.5]), np.array([0.5])])
        nbf_xi, nbf_eta = self.Get_nbf_1d()
        nxi = np.size(knots[0])
        neta = np.size(knots[1])
        # Degree elevate along the eta direction
        if neta != 0:
            coefs = self.ctrlPts.reshape((2*nbf_xi, nbf_eta), order="F")
            self.ctrlPts, self.knotVect[1] = bspkntins(
                self.degree[1], coefs, self.knotVect[1], knots[1])
            nbf_eta = self.ctrlPts.shape[1]
            self.ctrlPts = self.ctrlPts.reshape(
                (2, nbf_xi, nbf_eta), order="F")
        # Degree elevate along the xi direction
        if nxi != 0:
            coefs = np.transpose(self.ctrlPts, (0, 2, 1))
            coefs = coefs.reshape((2*nbf_eta, nbf_xi), order="F")
            self.ctrlPts, self.knotVect[0] = bspkntins(
                self.degree[0], coefs, self.knotVect[0], knots[0])
            nbf_xi = self.ctrlPts.shape[1]
            self.ctrlPts = self.ctrlPts.reshape(
                (2, nbf_eta, nbf_xi), order="F")
            self.ctrlPts = np.transpose(self.ctrlPts, (0, 2, 1))
        self.n = self.CrtlPts2N()

    def Stiffness(self, hooke):
        """ 
        Stiffness Matrix 
        """
        wg = sps.diags(self.wdetJ)
        Bxy = self.dphixdy+self.dphiydx
        K = hooke[0, 0]*self.dphixdx.T.dot(wg.dot(self.dphixdx)) +   \
            hooke[1, 1]*self.dphiydy.T.dot(wg.dot(self.dphiydy)) +   \
            hooke[2, 2]*Bxy.T.dot(wg.dot(Bxy)) + \
            hooke[0, 1]*self.dphixdx.T.dot(wg.dot(self.dphiydy)) +   \
            hooke[0, 2]*self.dphixdx.T.dot(wg.dot(Bxy)) +  \
            hooke[1, 2]*self.dphiydy.T.dot(wg.dot(Bxy)) +  \
            hooke[1, 0]*self.dphiydy.T.dot(wg.dot(self.dphixdx)) +   \
            hooke[2, 0]*Bxy.T.dot(wg.dot(self.dphixdx)) +  \
            hooke[2, 1]*Bxy.T.dot(wg.dot(self.dphiydy))
        return K

    def Laplacian(self):
        wg = sps.diags(self.wdetJ)
        return self.dphixdx.T.dot(wg.dot(self.dphixdx)) + self.dphixdy.T.dot(wg.dot(self.dphixdy)) +\
            self.dphiydx.T.dot(wg.dot(self.dphiydx)) + \
            self.dphiydy.T.dot(wg.dot(self.dphiydy))

    def DoubleLaplacian(self):
        wg = sps.diags(self.wdetJ)
        return 2*self.dphixdxx.T.dot(wg.dot(self.dphixdyy)) +\
            2*self.dphiydxx.T.dot(wg.dot(self.dphiydyy)) +\
            self.dphixdxx.T.dot(wg.dot(self.dphixdxx)) +\
            self.dphixdyy.T.dot(wg.dot(self.dphixdyy)) +\
            self.dphiydxx.T.dot(wg.dot(self.dphiydxx)) +\
            self.dphiydyy.T.dot(wg.dot(self.dphiydyy))

    def GaussIntegration(self, npg=None, P=None):
        """ Gauss integration: build of the global differential operators """
        if npg is None:
            nbg_xi = self.degree[0]+1
            nbg_eta = self.degree[1]+1
        else:
            nbg_xi = npg[0]
            nbg_eta = npg[1]

        Gauss_xi = GaussLegendre(nbg_xi)
        Gauss_eta = GaussLegendre(nbg_eta)
        nbf = self.Get_nbf()

        e_xi = np.unique(self.knotVect[0])
        ne_xi = e_xi.shape[0]-1
        e_eta = np.unique(self.knotVect[1])
        ne_eta = e_eta.shape[0]-1
        xi_min = np.kron(e_xi[:-1], np.ones(nbg_xi))
        eta_min = np.kron(e_eta[:-1], np.ones(nbg_eta))
        xi_g = np.kron(np.ones(ne_xi), Gauss_xi[0])
        eta_g = np.kron(np.ones(ne_eta), Gauss_eta[0])

        """ Measures of elements """
        mes_xi = e_xi[1:] - e_xi[:-1]
        mes_eta = e_eta[1:] - e_eta[:-1]

        mes_xi = np.kron(mes_xi, np.ones(nbg_xi))
        mes_eta = np.kron(mes_eta, np.ones(nbg_eta))

        """ Going from the reference element to the parametric space  """
        xi = xi_min + 0.5*(xi_g+1) * \
            mes_xi     # Aranged gauss points in  xi direction
        # Aranged gauss points in  eta direction
        eta = eta_min + 0.5*(eta_g+1)*mes_eta

        phi_xi, dphi_xi = global_basisfuns(
            self.degree[0], self.knotVect[0], xi)
        phi_eta, dphi_eta = global_basisfuns(
            self.degree[1], self.knotVect[1], eta)

        phi = sps.kron(phi_eta,  phi_xi,  'csc')
        dphidxi = sps.kron(phi_eta,  dphi_xi,  'csc')
        dphideta = sps.kron(dphi_eta,  phi_xi,  'csc')
        self.npg = phi.shape[0]

        wg_xi = np.kron(np.ones(ne_xi), Gauss_xi[1])
        wg_eta = np.kron(np.ones(ne_eta), Gauss_eta[1])

        mes_xi = np.kron(np.ones(eta.shape[0]), mes_xi)
        mes_eta = np.kron(mes_eta, np.ones(xi.shape[0]))

        if P is None:
            P = self.Get_P()

        """ Jacobian elements"""
        dxdxi = dphidxi.dot(P[:, 0])
        dxdeta = dphideta.dot(P[:, 0])
        dydxi = dphidxi.dot(P[:, 1])
        dydeta = dphideta.dot(P[:, 1])
        detJ = dxdxi*dydeta - dydxi*dxdeta
        """ Spatial derivatives """
        dphidx = sps.diags(dydeta/detJ).dot(dphidxi) + \
            sps.diags(-dydxi/detJ).dot(dphideta)
        dphidy = sps.diags(-dxdeta/detJ).dot(dphidxi) + \
            sps.diags(dxdxi/detJ).dot(dphideta)
        """ Integration weights + measures + Jacobian of the transformation """
        self.wdetJ = np.kron(wg_eta, wg_xi)*np.abs(detJ)*mes_xi*mes_eta/4
        zero = sps.csr_matrix((self.npg, nbf))
        self.phi = phi
        self.phix = sps.hstack((phi, zero),  'csc')
        self.phiy = sps.hstack((zero, phi),  'csc')
        self.dphixdx = sps.hstack((dphidx, zero),  'csc')
        self.dphixdy = sps.hstack((dphidy, zero),  'csc')
        self.dphiydx = sps.hstack((zero, dphidx),  'csc')
        self.dphiydy = sps.hstack((zero, dphidy),  'csc')

    def GetApproxElementSize(self, cam=None):
        if cam is None:
            # in physical unit
            u, v = self.n[:, 0], self.n[:, 1]
            m2 = self.Copy()
            m2.GaussIntegration(npg=[1, 1], P=np.c_[u, v])
            n = np.max(np.sqrt(m2.wdetJ))
        else:
            # in pyxel unit (int)
            u, v = cam.P(self.n[:, 0], self.n[:, 1])
            m2 = self.Copy()
            m2.GaussIntegration(npg=[1, 1], P=np.c_[u, v])
            n = int(np.floor(np.max(np.sqrt(m2.wdetJ))))
        return n

    def DICIntegrationFast(self, n=10):
        self.DICIntegration(n)

    def DICIntegration(self, n=10):
        """ DIC integration: build of the global differential operators """
        if 'Camera' in str(type(n)):
            # if n is a camera then n is autocomputed
            n = self.GetApproxElementSize(n)
        if type(n) == int:
            n = np.array([n, n], dtype=int)
        n = np.maximum(self.degree + 1, n)
        nbg_xi = n[0]
        nbg_eta = n[1]

        Rect_xi = np.linspace(-1, 1, nbg_xi)
        Weight_xi = 2/n[0] * np.ones(nbg_xi)
        Rect_eta = np.linspace(-1, 1, nbg_eta)
        Weight_eta = 2/n[1] * np.ones(nbg_eta)

        nbf = self.Get_nbf()

        e_xi = np.unique(self.knotVect[0])
        ne_xi = e_xi.shape[0]-1
        e_eta = np.unique(self.knotVect[1])
        ne_eta = e_eta.shape[0]-1
        xi_min = np.kron(e_xi[:-1], np.ones(nbg_xi))
        eta_min = np.kron(e_eta[:-1], np.ones(nbg_eta))
        xi_g = np.kron(np.ones(ne_xi), Rect_xi)
        eta_g = np.kron(np.ones(ne_eta), Rect_eta)

        """ Measures of elements """
        mes_xi = e_xi[1:] - e_xi[:-1]
        mes_eta = e_eta[1:] - e_eta[:-1]

        mes_xi = np.kron(mes_xi, np.ones(nbg_xi))
        mes_eta = np.kron(mes_eta, np.ones(nbg_eta))

        """ Going from the reference element to the parametric space  """
        xi = xi_min + 0.5*(xi_g+1) * \
            mes_xi     # Aranged gauss points in  xi direction
        # Aranged gauss points in  eta direction
        eta = eta_min + 0.5*(eta_g+1)*mes_eta

        phi_xi, dphi_xi = global_basisfuns(
            self.degree[0], self.knotVect[0], xi)
        phi_eta, dphi_eta = global_basisfuns(
            self.degree[1], self.knotVect[1], eta)

        phi = sps.kron(phi_eta,  phi_xi,  'csc')
        dphidxi = sps.kron(phi_eta,  dphi_xi,  'csc')
        dphideta = sps.kron(dphi_eta,  phi_xi,  'csc')
        self.npg = phi.shape[0]

        wg_xi = np.kron(np.ones(ne_xi), Weight_xi)
        wg_eta = np.kron(np.ones(ne_eta), Weight_eta)

        mes_xi = np.kron(np.ones(eta.shape[0]), mes_xi)
        mes_eta = np.kron(mes_eta, np.ones(xi.shape[0]))

        P = self.Get_P()

        """ Jacobian elements"""
        dxdxi = dphidxi.dot(P[:, 0])
        dxdeta = dphideta.dot(P[:, 0])
        dydxi = dphidxi.dot(P[:, 1])
        dydeta = dphideta.dot(P[:, 1])
        detJ = dxdxi*dydeta - dydxi*dxdeta
        """ Spatial derivatives """
        dphidx = sps.diags(dydeta/detJ).dot(dphidxi) + \
            sps.diags(-dydxi/detJ).dot(dphideta)
        dphidy = sps.diags(-dxdeta/detJ).dot(dphidxi) + \
            sps.diags(dxdxi/detJ).dot(dphideta)
        """ Integration weights + measures + Jacobian of the transformation """
        self.wdetJ = np.kron(wg_eta, wg_xi)*np.abs(detJ)*mes_xi*mes_eta/4

        zero = sps.csr_matrix((self.npg, nbf))
        self.phi = phi
        self.phix = sps.hstack((phi, zero),  'csc')
        self.phiy = sps.hstack((zero, phi),  'csc')
        self.dphixdx = sps.hstack((dphidx, zero),  'csc')
        self.dphixdy = sps.hstack((dphidy, zero),  'csc')
        self.dphiydx = sps.hstack((zero, dphidx),  'csc')
        self.dphiydy = sps.hstack((zero, dphidy),  'csc')

        self.pgx = self.phi @ P[:, 0]
        self.pgy = self.phi @ P[:, 1]

    def SetBasisFunctionsAtIntegrationPoints(self):
        phi, dphidx, dphidy = Get2dBasisFunctionsAtPts(
            self.pix, self.piy, self.knotVect[0], self.knotVect[1], self.degree[0], self.degree[1])
        nbf = self.Get_nbf()
        self.wg = sps.diags(self.wg)
        zero = sps.csr_matrix((self.npg, nbf))
        self.phiMatrix = phi
        self.phix = sps.hstack((phi, zero),  'csc')
        self.phiy = sps.hstack((zero, phi),  'csc')
        self.dphixdx = sps.hstack((dphidx, zero),  'csc')
        self.dphixdy = sps.hstack((dphidy, zero),  'csc')
        self.dphiydx = sps.hstack((zero, dphidx),  'csc')
        self.dphiydy = sps.hstack((zero, dphidy),  'csc')

    def ShapeFunctionsAtGridPoints(self, xi, eta):
        """ xi and eta are the 1d points 
        This method computes the basis functions on the mesh-grid point 
        obtained from the 1d vector points xi and eta 
        """

        phi_xi, dphi_xi = global_basisfuns(
            self.degree[0], self.knotVect[0], xi)
        phi_eta, dphi_eta = global_basisfuns(
            self.degree[1], self.knotVect[1], eta)

        phi = sps.kron(phi_eta,  phi_xi,  'csc')
        dphidxi = sps.kron(phi_eta,  dphi_xi,  'csc')
        dphideta = sps.kron(dphi_eta,  phi_xi,  'csc')

        P = self.Get_P()

        """ Jacobian elements"""
        dxdxi = dphidxi.dot(P[:, 0])
        dxdeta = dphideta.dot(P[:, 0])
        dydxi = dphidxi.dot(P[:, 1])
        dydeta = dphideta.dot(P[:, 1])
        detJ = dxdxi*dydeta - dydxi*dxdeta
        """ Spatial derivatives """
        dphidx = sps.diags(dydeta/detJ).dot(dphidxi) + \
            sps.diags(-dydxi/detJ).dot(dphideta)
        dphidy = sps.diags(-dxdeta/detJ).dot(dphidxi) + \
            sps.diags(dxdxi/detJ).dot(dphideta)
        # Univariate basis functions if needed
        # Nxi  = phi_xi
        # Neta = phi_eta
        N = phi
        return N, dphidx, dphidy

    def GetBoundaryIndices(self):
        P = self.Get_P()
        left = np.where(np.abs(P[:, 0] - np.min(P[:, 0])) <= 1.e-12)[0]
        right = np.where(np.abs(P[:, 0] - np.max(P[:, 0])) <= 1.e-12)[0]
        bottom = np.where(np.abs(P[:, 1] - np.min(P[:, 1])) <= 1.e-12)[0]
        top = np.where(np.abs(P[:, 1] - np.max(P[:, 1])) <= 1.e-12)[0]
        sleft = np.size(left)
        stop = np.size(top)
        sbottom = np.size(bottom)
        sright = np.size(right)
        return bottom, top, right, left, sbottom, stop, sright, sleft

    def PlaneWave(self, T):
        V = np.zeros(self.ndof)
        V[self.conn[:, 0]] = np.cos(self.n[:, 1] / T * 2 * np.pi)
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


def GaussLegendre(n):
    #     [nodes,weigths]=GaussLegendre(n)
    #
    # Generates the abscissa and weights for a Gauss-Legendre quadrature.
    # Reference:  Numerical Recipes in Fortran 77, Cornell press.

    # Preallocations.
    xg = np.zeros(n)
    wg = np.zeros(n)
    m = (n+1)/2
    #import pdb; pdb.set_trace()
    for ii in range(int(m)):  # for ii=1:m
        # Initial estimate.
        z = np.cos(np.pi*(ii+1-0.25)/(n+0.5))
        z1 = z+1
        while np.abs(z-z1) > np.finfo(np.float64).eps:
            p1 = 1
            p2 = 0
            for jj in range(n):  # for jj = 1:n
                p3 = p2
                p2 = p1
                # The Legendre polynomial.
                p1 = ((2*jj+1)*z*p2-(jj)*p3)/(jj+1)

            # The L.P. derivative.
            pp = n*(z*p1-p2)/(z**2-1)
            z1 = z
            z = z1-p1/pp

        xg[ii] = -z                                   # Build up the abscissas.
        xg[-1-ii] = z
        # Build up the weights.
        wg[ii] = 2/((1-z**2)*(pp**2))
        wg[-1-ii] = wg[ii]
    return xg, wg


def SplineFromROI(roi, dx, degree=[2, 2]):
    """Build a structured FE mesh and a pyxel.camera object from a region
    of interest (ROI) selected in an image f

    Parameters
    ----------
    roi : numpy.array
        The Region of Interest made using  f.SelectROI(), f being a pyxel.Image
    dx : numpy or python array
        dx  = [dx, dy]: average element size (can be scalar) in pixels
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
    m, cam = px.MeshFromROI(roi, [20, 20], 3)
    """
    dbox = roi[1] - roi[0]
    NE = (dbox / dx).astype(int)
    NE = np.max(np.c_[NE, np.ones(2, dtype=int)], axis=1)
    m = Rectangle(roi, NE, degree)
    m.n[:, 1] *= -1
    m.ctrlPts = m.N2CrtlPts()
    cam = Camera(2)
    cam.R[2, 0] = 1.5707963267948966
    return m, cam


def Rectangle(roi, n_elems, degree):
    xmin, ymin, xmax, ymax = roi.ravel()
    # Parametric space properties
    p = 1
    q = 1
    Xi = np.concatenate(
        (np.repeat(0, p+1), np.repeat(1, p+1)))*(xmax-xmin) + xmin
    Eta = np.concatenate(
        (np.repeat(0, q+1), np.repeat(1, q+1)))*(ymax-ymin) + ymin
    # Control points for a recangular plate
    x = np.array([[xmin, xmin],
                  [xmax, xmax]])
    y = np.array([[ymin, ymax],
                  [ymin, ymax]])
    ctrlPts = np.array([x, y])
    knot_vector = [Xi, Eta]
    m = BSplinePatch(ctrlPts, np.array([p, q]), knot_vector)
    # Degree elevation
    m.DegreeElevation(degree)
    # Knot refinement
    ubar = [None]*2
    ubar[0] = 1/n_elems[0] * np.arange(1, n_elems[0]) * (xmax-xmin) + xmin
    ubar[1] = 1/n_elems[1] * np.arange(1, n_elems[1]) * (ymax-ymin) + ymin
    m.KnotInsertion(ubar)
    # Building connectivity
    m.Connectivity()
    return m
