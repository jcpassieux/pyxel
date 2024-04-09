# -*- coding: utf-8 -*-
"""
Finite Element Digital Image Correlation method 

@author: JC Passieux, INSA Toulouse, 2021

pyxel

PYthon library for eXperimental mechanics using Finite ELements

"""

import numpy as np

class Camera:
    def __init__(self, p):
        self.set_p(p)

    def set_p(self, p):
        """
        Set manually the camera parameters.

        Parameters
        ----------
        p : NUMPY.ARRAY
            np.array([f, tx, ty, rz])
            f: focal length (scaling)
            tx: x-translation
            ty: y-translation
            rz: z-rotation

        """
        self.f = float(p[0])
        self.tx = float(p[1])
        self.ty = float(p[2])
        self.rz = float(p[3])

    def get_p(self):
        """
        Returns the vector of parameters of the camera model

        Returns
        -------
        NUMPY.ARRAY

        """
        return np.array([self.f, self.tx, self.ty, self.rz])

    def SubSampleCopy(self, nscale):
        """
        Camera model copy with subsampling (of bining) for multiscale initialization
        
        Parameters
        ----------
        NSCALE : INT
            number of scales, such that the scaling is 2**NSCALE

        Returns
        -------
        CAM : 1D PYXEL.CAMERA
            A new camera model that maps the same mesh coord. sys. to the  
            same image but with a bining of NSCALE. 
        """
        p = self.get_p()
        p[0] /= 2 ** nscale
        return Camera(p)

    def P(self, X, Y):
        """
        Camera model projection. Maps a point of the mesh to a point
        in the image plane

        Parameters
        ----------
        X : NUMPY.ARRAY
            X coordinate in the mesh system of the points to map
        Y : NUMPY.ARRAY
            Y coordinate in the mesh system of the points to map

        Returns
        -------
        u : 1D NUMPY.ARRAY
            u coordinate of the corresponding point in the image system
        v : 1D NUMPY.ARRAY
            v coordinate of the corresponding point in the image system

        """
        u = -self.f * (-np.sin(self.rz) * X + np.cos(self.rz) * Y + self.ty)
        v = self.f * (np.cos(self.rz) * X + np.sin(self.rz) * Y + self.tx)
        return u, v

    def Pinv(self, u, v):
        """
        Inverse of the Camera model. Maps a point in the image to a point
        in the mesh coordinate sys.
        (Explicit version for 2D camera model)

        Parameters
        ----------
        u : 1D NUMPY.ARRAY
            u coordinate in the image system of the points to map
        v : 1D NUMPY.ARRAY
            v coordinate in the image system of the points to map

        Returns
        -------
        X : 1D NUMPY.ARRAY
            X coordinate of the corresponding position in the mesh system
        Y : 1D NUMPY.ARRAY
            Y coordinate of the corresponding position in the mesh system

        """
        X = -np.sin(self.rz) * (-u / self.f - self.ty) + np.cos(self.rz) * (
            v / self.f - self.tx
        )
        Y = np.cos(self.rz) * (-u / self.f - self.ty) + np.sin(self.rz) * (
            v / self.f - self.tx
        )
        return X, Y

    def PinvNL(self, u, v):
        """
        Inverse of the Camera model. Maps a point in the image to a point
        in the mesh coordinate sys.
        (General version for any possibly NL camera models)

        Parameters
        ----------
        u : 1D NUMPY.ARRAY
            u coordinate in the image system of the points to map
        v : 1D NUMPY.ARRAY
            v coordinate in the image system of the points to map

        Returns
        -------
        X : 1D NUMPY.ARRAY
            X coordinate of the corresponding position in the mesh system
        Y : 1D NUMPY.ARRAY
            Y coordinate of the corresponding position in the mesh system

        """
        X = np.zeros(len(u))
        Y = np.zeros(len(u))
        for ii in range(10):
            pnx, pny = self.P(X, Y)
            resx = u - pnx
            resy = v - pny
            dPxdX, dPxdY, dPydX, dPydY = self.dPdX(X, Y)
            detJ = dPxdX * dPydY - dPxdY * dPydX
            dX = dPydY / detJ * resx - dPxdY / detJ * resy
            dY = -dPydX / detJ * resx + dPxdX / detJ * resy
            X += dX
            Y += dY
            res = np.linalg.norm(dX) + np.linalg.norm(dY)
            if res < 1e-4:
                break
        return X, Y

    def dPdX(self, X, Y):
        """
        Derivative of the Camera model wrt physical position X, Y

        Parameters
        ----------
        X : 1D NUMPY.ARRAY
            X coordinate of the current position in the mesh system
        Y : 1D NUMPY.ARRAY
            Y coordinate of the current position in the mesh system

        Returns
        -------
        dudx : NUMPY.ARRAY
            Derivative of coord. u wrt to X
        dudy : NUMPY.ARRAY
            Derivative of coord. u wrt to Y
        dvdx : NUMPY.ARRAY
            Derivative of coord. v wrt to X
        dvdy : NUMPY.ARRAY
            Derivative of coord. v wrt to Y

        """
        dudx = self.f * np.sin(self.rz) * np.ones(X.shape[0])
        dudy = -self.f * np.cos(self.rz) * np.ones(X.shape[0])
        dvdx = self.f * np.cos(self.rz) * np.ones(X.shape[0])
        dvdy = self.f * np.sin(self.rz) * np.ones(X.shape[0])
        return dudx, dudy, dvdx, dvdy

    def dPdp(self, X, Y):
        """
        First order derivative of the Camera model wrt camera parameters p

        Parameters
        ----------
        X : 1D NUMPY.ARRAY
            X coordinate of the current position in the mesh system
        Y : 1D NUMPY.ARRAY
            Y coordinate of the current position in the mesh system

        Returns
        -------
        dudp : NUMPY.ARRAY
            Derivative of coord. u wrt to p
        dvdp : NUMPY.ARRAY
            Derivative of coord. v wrt to p

        """
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
        """
        Second order derivative of the Camera model wrt camera parameters p

        Parameters
        ----------
        X : 1D NUMPY.ARRAY
            X coordinate of the current position in the mesh system
        Y : 1D NUMPY.ARRAY
            Y coordinate of the current position in the mesh system

        Returns
        -------
        d2udp2 : NUMPY.ARRAY
            second order derivative of coord. u wrt to p
        d2vdp2 : NUMPY.ARRAY
            second order derivative of coord. v wrt to p

        """
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

class CameraNL:
    def __init__(self, p):
        self.set_p(p)

    def set_p(self, p):
        """
        Set manually the camera parameters.

        Parameters
        ----------
        p : NUMPY.ARRAY
            np.array([f, tx, ty, rz])
            f: focal length (scaling)
            tx: x-translation
            ty: y-translation
            rz: z-rotation
            u0: image center coord x, default Npix_x/2
            v0: image center coord y, default Npix_y/2
            r1: first order radial dist param, default 0
        """
        self.f = p[0]
        self.tx = p[1]
        self.ty = p[2]
        self.rz = p[3]
        self.u0 = p[4]
        self.v0 = p[5]
        self.r1 = p[6]
        self.r2 = p[7]
        self.r3 = p[8]

    def get_p(self):
        """
        Returns the vector of parameters of the camera model

        Returns
        -------
        NUMPY.ARRAY

        """
        return np.array([self.f, self.tx, self.ty, self.rz])

    def SubSampleCopy(self, nscale):
        """
        Camera model copy with subsampling (of bining) for multiscale initialization
        
        Parameters
        ----------
        NSCALE : INT
            number of scales, such that the scaling is 2**NSCALE

        Returns
        -------
        CAM : 1D PYXEL.CAMERA
            A new camera model that maps the same mesh coord. sys. to the  
            same image but with a bining of NSCALE. 
        """
        p = self.get_p()
        p[0] /= 2 ** nscale
        return CameraNL(p)

    def P(self, X, Y):
        """
        Camera model projection. Maps a point of the mesh to a point
        in the image plane

        Parameters
        ----------
        X : NUMPY.ARRAY
            X coordinate in the mesh system of the points to map
        Y : NUMPY.ARRAY
            Y coordinate in the mesh system of the points to map

        Returns
        -------
        u : 1D NUMPY.ARRAY
            u coordinate of the corresponding point in the image system
        v : 1D NUMPY.ARRAY
            v coordinate of the corresponding point in the image system

        """
        u = - self.f * (-np.sin(self.rz) * X + np.cos(self.rz) * Y + self.ty)
        v = self.f * (np.cos(self.rz) * X + np.sin(self.rz) * Y + self.tx)
       
        rho2 = (u-self.u0)**2 + (v-self.v0)**2
        ud = (u-self.u0) * (1 + self.r1*rho2 + self.r2*rho2**2 + self.r3*rho2**3) + self.u0
        vd = (v-self.v0) * (1 + self.r1*rho2 + self.r2*rho2**2 + self.r3*rho2**3) + self.v0

        return ud, vd

    def Pinv(self, u, v):
        """
        Inverse of the Camera model. Maps a point in the image to a point
        in the mesh coordinate sys.
        (General version for any possibly NL camera models)

        Parameters
        ----------
        u : 1D NUMPY.ARRAY
            u coordinate in the image system of the points to map
        v : 1D NUMPY.ARRAY
            v coordinate in the image system of the points to map

        Returns
        -------
        X : 1D NUMPY.ARRAY
            X coordinate of the corresponding position in the mesh system
        Y : 1D NUMPY.ARRAY
            Y coordinate of the corresponding position in the mesh system

        """
        X = np.zeros(len(u))
        Y = np.zeros(len(u))
        for ii in range(10):
            pnx, pny = self.P(X, Y)
            resx = u - pnx
            resy = v - pny
            dPxdX, dPxdY, dPydX, dPydY = self.dPdX(X, Y)
            detJ = dPxdX * dPydY - dPxdY * dPydX
            dX = dPydY / detJ * resx - dPxdY / detJ * resy
            dY = -dPydX / detJ * resx + dPxdX / detJ * resy
            X += dX
            Y += dY
            res = np.linalg.norm(dX) + np.linalg.norm(dY)
            if res < 1e-4:
                break
        return X, Y

    def dPdX(self, X, Y):
        """
        Derivative of the Camera model wrt physical position X, Y

        Parameters
        ----------
        X : 1D NUMPY.ARRAY
            X coordinate of the current position in the mesh system
        Y : 1D NUMPY.ARRAY
            Y coordinate of the current position in the mesh system

        Returns
        -------
        dudx : NUMPY.ARRAY
            Derivative of coord. u wrt to X
        dudy : NUMPY.ARRAY
            Derivative of coord. u wrt to Y
        dvdx : NUMPY.ARRAY
            Derivative of coord. v wrt to X
        dvdy : NUMPY.ARRAY
            Derivative of coord. v wrt to Y

        """
        # import sympy as sp
        # f, tx, ty, rz, u0, v0, r1, X, Y = sp.symbols('f, tx, ty, rz, u0, v0, r1, X, Y', real=True)
        # u = f * (sp.sin(rz) * X - sp.cos(rz) * Y - ty)
        # v = f * (sp.cos(rz) * X + sp.sin(rz) * Y + tx)
        # rho2 = (u-u0)**2 + (v-v0)**2
        # ud = (u - u0) * (1 + r1*rho2) + u0 
        # vd = (v - v0) * (1 + r1*rho2) + v0
        # list_nulle = (sp.diff(ud, X), sp.diff(ud, Y), sp.diff(vd, X), sp.diff(vd, Y))
        # variable_namer = sp.numbered_symbols('v')
        # replacements, reduced = sp.cse(list_nulle, symbols=variable_namer)
        # for key, val in replacements:
        #     print(key, '=', sp.pycode(val))
        # for i, r in enumerate(reduced):
        #     print('deriv[{}]'.format(i), '=', sp.pycode(r))
        f = self.f
        tx = self.tx
        ty = self.ty
        rz = self.rz
        u0 = self.u0
        v0 = self.v0
        r1 = self.r1
        v0 = np.sin(rz)
        v1 = np.cos(rz)
        v2 = f*(X*v1 + Y*v0 + tx) - v0
        v3 = f*(X*v0 - Y*v1 - ty) - u0
        v4 = f*(r1*(v2**2 + v3**2) + 1)
        v5 = v0*v4
        v6 = 2*f
        v7 = v3*v6
        v8 = v0*v7 + v1*v2*v6
        v9 = v1*v4
        v10 = 2*f*v0*v2 - v1*v7
        v11 = r1*v2
        dudx = r1*v3*v8 + v5
        dudy = r1*v10*v3 - v9
        dvdx = v11*v8 + v9
        dvdy = v10*v11 + v5
        # dudx = f*(r1*((f*(X*np.sin(rz) - Y*np.cos(rz) - ty) - u0)**2 + (f*(X*np.cos(rz) + Y*np.sin(rz) + tx) - v0)**2) + 1)*np.sin(rz) + r1*(f*(X*np.sin(rz) - Y*np.cos(rz) - ty) - u0)*(2*f*(f*(X*np.sin(rz) - Y*np.cos(rz) - ty) - u0)*np.sin(rz) + 2*f*(f*(X*np.cos(rz) + Y*np.sin(rz) + tx) - v0)*np.cos(rz))
        # dudy = -f*(r1*((f*(X*np.sin(rz) - Y*np.cos(rz) - ty) - u0)**2 + (f*(X*np.cos(rz) + Y*np.sin(rz) + tx) - v0)**2) + 1)*np.cos(rz) + r1*(f*(X*np.sin(rz) - Y*np.cos(rz) - ty) - u0)*(-2*f*(f*(X*np.sin(rz) - Y*np.cos(rz) - ty) - u0)*np.cos(rz) + 2*f*(f*(X*np.cos(rz) + Y*np.sin(rz) + tx) - v0)*np.sin(rz))
        # dvdx = f*(r1*((f*(X*np.sin(rz) - Y*np.cos(rz) - ty) - u0)**2 + (f*(X*np.cos(rz) + Y*np.sin(rz) + tx) - v0)**2) + 1)*np.cos(rz) + r1*(f*(X*np.cos(rz) + Y*np.sin(rz) + tx) - v0)*(2*f*(f*(X*np.sin(rz) - Y*np.cos(rz) - ty) - u0)*np.sin(rz) + 2*f*(f*(X*np.cos(rz) + Y*np.sin(rz) + tx) - v0)*np.cos(rz))
        # dvdy = f*(r1*((f*(X*np.sin(rz) - Y*np.cos(rz) - ty) - u0)**2 + (f*(X*np.cos(rz) + Y*np.sin(rz) + tx) - v0)**2) + 1)*np.sin(rz) + r1*(f*(X*np.cos(rz) + Y*np.sin(rz) + tx) - v0)*(-2*f*(f*(X*np.sin(rz) - Y*np.cos(rz) - ty) - u0)*np.cos(rz) + 2*f*(f*(X*np.cos(rz) + Y*np.sin(rz) + tx) - v0)*np.sin(rz))
        return dudx, dudy, dvdx, dvdy

    def dPdp(self, X, Y):
        """
        First order derivative of the Camera model wrt camera parameters p

        Parameters
        ----------
        X : 1D NUMPY.ARRAY
            X coordinate of the current position in the mesh system
        Y : 1D NUMPY.ARRAY
            Y coordinate of the current position in the mesh system

        Returns
        -------
        dudp : NUMPY.ARRAY
            Derivative of coord. u wrt to p
        dvdp : NUMPY.ARRAY
            Derivative of coord. v wrt to p

        """
        # import sympy as sp
        # f, tx, ty, rz, u0, v0, r1, X, Y = sp.symbols('f, tx, ty, rz, u0, v0, r1, X, Y', real=True)
        # u = sp.sin(rz) * X - sp.cos(rz) * Y - ty
        # v = sp.cos(rz) * X + sp.sin(rz) * Y + tx
        # rho2 = (u-u0)**2 + (v-v0)**2
        # ud = f * u * (1 + r1*rho2) 
        # vd = f * v * (1 + r1*rho2) 
        # sp.pycode(sp.diff(vd, r1))
        f = self.f
        tx = self.tx
        ty = self.ty
        rz = self.rz
        u0 = self.u0
        v0 = self.v0
        r1 = self.r1
        
        dudf = (r1*((X*np.sin(rz) - Y*np.cos(rz) - ty - u0)**2 + (X*np.cos(rz) + Y*np.sin(rz) + tx - v0)**2) + 1)*(X*np.sin(rz) - Y*np.cos(rz) - ty)
        dudtx = f*r1*(X*np.sin(rz) - Y*np.cos(rz) - ty)*(2*X*np.cos(rz) + 2*Y*np.sin(rz) + 2*tx - 2*v0)       
        dudty = f*r1*(X*np.sin(rz) - Y*np.cos(rz) - ty)*(-2*X*np.sin(rz) + 2*Y*np.cos(rz) + 2*ty + 2*u0) - f*(r1*((X*np.sin(rz) - Y*np.cos(rz) - ty - u0)**2 + (X*np.cos(rz) + Y*np.sin(rz) + tx - v0)**2) + 1)
        dudrz = f*r1*((-2*X*np.sin(rz) + 2*Y*np.cos(rz))*(X*np.cos(rz) + Y*np.sin(rz) + tx - v0) + (2*X*np.cos(rz) + 2*Y*np.sin(rz))*(X*np.sin(rz) - Y*np.cos(rz) - ty - u0))*(X*np.sin(rz) - Y*np.cos(rz) - ty) + f*(X*np.cos(rz) + Y*np.sin(rz))*(r1*((X*np.sin(rz) - Y*np.cos(rz) - ty - u0)**2 + (X*np.cos(rz) + Y*np.sin(rz) + tx - v0)**2) + 1)
        dudu0 = f*r1*(X*np.sin(rz) - Y*np.cos(rz) - ty)*(-2*X*np.sin(rz) + 2*Y*np.cos(rz) + 2*ty + 2*u0)
        dudv0 = f*r1*(X*np.sin(rz) - Y*np.cos(rz) - ty)*(-2*X*np.cos(rz) - 2*Y*np.sin(rz) - 2*tx + 2*v0)
        dudr1 = f*((X*np.sin(rz) - Y*np.cos(rz) - ty - u0)**2 + (X*np.cos(rz) + Y*np.sin(rz) + tx - v0)**2)*(X*np.sin(rz) - Y*np.cos(rz) - ty)
        
        dvdf = (r1*((X*np.sin(rz) - Y*np.cos(rz) - ty - u0)**2 + (X*np.cos(rz) + Y*np.sin(rz) + tx - v0)**2) + 1)*(X*np.cos(rz) + Y*np.sin(rz) + tx)
        dvdtx = f*r1*(X*np.cos(rz) + Y*np.sin(rz) + tx)*(2*X*np.cos(rz) + 2*Y*np.sin(rz) + 2*tx - 2*v0) + f*(r1*((X*np.sin(rz) - Y*np.cos(rz) - ty - u0)**2 + (X*np.cos(rz) + Y*np.sin(rz) + tx - v0)**2) + 1)
        dvdty = f*r1*(X*np.cos(rz) + Y*np.sin(rz) + tx)*(-2*X*np.sin(rz) + 2*Y*np.cos(rz) + 2*ty + 2*u0)
        dvdrz = f*r1*((-2*X*np.sin(rz) + 2*Y*np.cos(rz))*(X*np.cos(rz) + Y*np.sin(rz) + tx - v0) + (2*X*np.cos(rz) + 2*Y*np.sin(rz))*(X*np.sin(rz) - Y*np.cos(rz) - ty - u0))*(X*np.cos(rz) + Y*np.sin(rz) + tx) + f*(-X*np.sin(rz) + Y*np.cos(rz))*(r1*((X*np.sin(rz) - Y*np.cos(rz) - ty - u0)**2 + (X*np.cos(rz) + Y*np.sin(rz) + tx - v0)**2) + 1)
        dvdu0 = f*r1*(X*np.cos(rz) + Y*np.sin(rz) + tx)*(-2*X*np.sin(rz) + 2*Y*np.cos(rz) + 2*ty + 2*u0)
        dvdv0 = f*r1*(X*np.cos(rz) + Y*np.sin(rz) + tx)*(-2*X*np.cos(rz) - 2*Y*np.sin(rz) - 2*tx + 2*v0)
        dvdr1 = f*((X*np.sin(rz) - Y*np.cos(rz) - ty - u0)**2 + (X*np.cos(rz) + Y*np.sin(rz) + tx - v0)**2)*(X*np.cos(rz) + Y*np.sin(rz) + tx)
        return np.c_[dudf, dudtx, dudty, dudrz, dudu0, dudv0, dudr1], np.c_[dvdf, dvdtx, dvdty, dvdrz, dvdu0, dvdv0, dvdr1]

    def d2Pdp2(self, X, Y):
        """
        Second order derivative of the Camera model wrt camera parameters p

        Parameters
        ----------
        X : 1D NUMPY.ARRAY
            X coordinate of the current position in the mesh system
        Y : 1D NUMPY.ARRAY
            Y coordinate of the current position in the mesh system

        Returns
        -------
        d2udp2 : NUMPY.ARRAY
            second order derivative of coord. u wrt to p
        d2vdp2 : NUMPY.ARRAY
            second order derivative of coord. v wrt to p

        """

        return None

    def ImageFiles(self, fname, imnums):
        self.fname = fname
        self.imnums = imnums

#%%
# import sympy as sp
# X, Y, Z, f, rx, ry, rz, tx, ty, tz = sp.symbols('X, Y, Z, f, rx, ry, rz, tx, ty, tz')
# T = np.array([[1, 0, 0, tx],
#               [0, 1, 0, ty],
#               [0, 0, 1, tz],
#               [0, 0, 0, 1]])
# R1 = np.array([[sp.cos(rz),-sp.sin(rz), 0, 0],
#                [sp.sin(rz), sp.cos(rz), 0, 0],
#                [0, 0, 1, 0],
#                [0, 0, 0, 1]])
# R2 = np.array([[1, 0, 0, 0],
#                [0, sp.cos(rx),-sp.sin(rx), 0],
#                [0, sp.sin(rx), sp.cos(rx), 0],
#                [0, 0, 0, 1]])
# R3 = np.array([[sp.cos(ry), 0 ,-sp.sin(ry), 0],
#                [0, 1, 0, 0],
#                [sp.sin(ry), 0, sp.cos(ry), 0],
#                [0, 0, 0, 1]])
# v = np.array([X, Y, Z, 1])
# uvw = f*T@R1@R2@R3@v
# u = uvw[0]
# v = uvw[1]
# w = uvw[2]

# print(sp.pycode(u))
# print(sp.pycode(v))
# print(sp.pycode(w))

# print(sp.pycode(u.diff(X, 1)))
# print(sp.pycode(u.diff(Y, 1)))
# print(sp.pycode(u.diff(Z, 1)))

# print(sp.pycode(v.diff(X, 1)))
# print(sp.pycode(v.diff(Y, 1)))
# print(sp.pycode(v.diff(Z, 1)))

# print(sp.pycode(w.diff(X, 1)))
# print(sp.pycode(w.diff(Y, 1)))
# print(sp.pycode(w.diff(Z, 1)))

# print(sp.pycode(u.diff(f, 1)))
# print(sp.pycode(u.diff(tx, 1)))
# print(sp.pycode(u.diff(ty, 1)))
# print(sp.pycode(u.diff(tz, 1)))
# print(sp.pycode(u.diff(rx, 1)))
# print(sp.pycode(u.diff(ry, 1)))
# print(sp.pycode(u.diff(rz, 1)))

# print(sp.pycode(v.diff(f, 1)))
# print(sp.pycode(v.diff(tx, 1)))
# print(sp.pycode(v.diff(ty, 1)))
# print(sp.pycode(v.diff(tz, 1)))
# print(sp.pycode(v.diff(rx, 1)))
# print(sp.pycode(v.diff(ry, 1)))
# print(sp.pycode(v.diff(rz, 1)))

# print(sp.pycode(w.diff(f, 1)))
# print(sp.pycode(w.diff(tx, 1)))
# print(sp.pycode(w.diff(ty, 1)))
# print(sp.pycode(w.diff(tz, 1)))
# print(sp.pycode(w.diff(rx, 1)))
# print(sp.pycode(w.diff(ry, 1)))
# print(sp.pycode(w.diff(rz, 1)))




class CameraVol:
    def __init__(self, p):
        self.set_p(p)

    def set_p(self, p):
        """
        Set manually the camera parameters.

        Parameters
        ----------
        p : NUMPY.ARRAY
            np.array([f, tx, ty, tz, rx, ry, rz])
            f: focal length (scaling)
            tx: x-translation
            ty: y-translation
            tz: z-translation
            rx: x-rotation
            ry: y-rotation
            rz: z-rotation

        """
        self.f = float(p[0])
        self.tx = float(p[1])
        self.ty = float(p[2])
        self.tz = float(p[3])
        self.rx = float(p[4])
        self.ry = float(p[5])
        self.rz = float(p[6])

    def get_p(self):
        """
        Returns the vector of parameters of the camera model

        Returns
        -------
        NUMPY.ARRAY

        """
        return np.array([self.f, self.tx, self.ty, self.tz, self.rx, self.ry, self.rz])

    def SubSampleCopy(self, nscale):
        """
        Camera model copy with subsampling (of bining) for multiscale initialization
        
        Parameters
        ----------
        NSCALE : INT
            number of scales, such that the scaling is 2**NSCALE

        Returns
        -------
        CAM : 1D PYXEL.CAMERA
            A new camera model that maps the same mesh coord. sys. to the  
            same image but with a bining of NSCALE. 
        """
        p = self.get_p()
        p[0] /= 2 ** nscale
        return CameraVol(p)

    def P(self, X, Y, Z):
        """
        Camera model projection. Maps a point of the mesh to a point
        in the image plane

        Parameters
        ----------
        X : NUMPY.ARRAY
            X coordinate in the mesh system of the points to map
        Y : NUMPY.ARRAY
            Y coordinate in the mesh system of the points to map
        Z : NUMPY.ARRAY
            Z coordinate in the mesh system of the points to map

        Returns
        -------
        u : 1D NUMPY.ARRAY
            u coordinate of the corresponding point in the image system
        v : 1D NUMPY.ARRAY
            v coordinate of the corresponding point in the image system
        w : 1D NUMPY.ARRAY
            w coordinate of the corresponding point in the image system

        """
        f = self.f
        tx = self.tx
        ty = self.ty
        tz = self.tz
        rx = self.rx
        ry = self.ry
        rz = self.rz
        u = X*(f*np.sin(rx)*np.sin(ry)*np.sin(rz) + f*np.cos(ry)*np.cos(rz)) - \
            Y*f*np.sin(rz)*np.cos(rx) + Z*(f*np.sin(rx)*np.sin(rz)*np.cos(ry) - \
                                                 f*np.sin(ry)*np.cos(rz)) + f*tx
        v = X*(-f*np.sin(rx)*np.sin(ry)*np.cos(rz) + f*np.sin(rz)*np.cos(ry)) + \
            Y*f*np.cos(rx)*np.cos(rz) + Z*(-f*np.sin(rx)*np.cos(ry)*np.cos(rz) - \
                                               f*np.sin(ry)*np.sin(rz)) + f*ty
        w = X*f*np.sin(ry)*np.cos(rx) + Y*f*np.sin(rx) + \
            Z*f*np.cos(rx)*np.cos(ry) + f*tz
        return u, v, w

    # import sympy as sp
    # dPxdX, dPxdY, dPxdZ, dPydX, dPydY, dPydZ, dPzdX, dPzdY, dPzdZ = \
    #    sp.symbols('dPxdX, dPxdY, dPxdZ, dPydX, dPydY, dPydZ, dPzdX, dPzdY, dPzdZ')    
    # J = sp.Matrix([[dPxdX, dPxdY, dPxdZ],
    #                [dPydX, dPydY, dPydZ],
    #                [dPzdX, dPzdY, dPzdZ]])
    # detJ = sp.det(J)
    # print(sp.pycode(detJ))
    # Jinv = J.inv()*detJ
    # print(Jinv[0,0])
    # print(Jinv[0,1])
    # print(Jinv[0,2])
    # print(Jinv[1,0])
    # print(Jinv[1,1])
    # print(Jinv[1,2])    
    # print(Jinv[2,0])
    # print(Jinv[2,1])
    # print(Jinv[2,2])

    def Pinv(self, u, v, w):
        """
        Inverse of the Camera model. Maps a point in the image to a point
        in the mesh coordinate sys.
        (General version for any possibly NL camera models)

        Parameters
        ----------
        u : 1D NUMPY.ARRAY
            u coordinate in the image system of the points to map
        v : 1D NUMPY.ARRAY
            v coordinate in the image system of the points to map
        w : 1D NUMPY.ARRAY
            w coordinate in the image system of the points to map

        Returns
        -------
        X : 1D NUMPY.ARRAY
            X coordinate of the corresponding position in the mesh system
        Y : 1D NUMPY.ARRAY
            Y coordinate of the corresponding position in the mesh system
        Z : 1D NUMPY.ARRAY
            Z coordinate of the corresponding position in the mesh system

        """
        X = np.zeros_like(u)
        Y = np.zeros_like(u)
        Z = np.zeros_like(u)
        for ii in range(10):
            pnx, pny, pnz = self.P(X, Y, Z)
            resx = u - pnx
            resy = v - pny
            resz = w - pnz
            dPxdX, dPxdY, dPxdZ, dPydX, dPydY, dPydZ, dPzdX, dPzdY, dPzdZ = self.dPdX(X, Y, Z)
            detJ = dPxdX*dPydY*dPzdZ - dPxdX*dPydZ*dPzdY - dPxdY*dPydX*dPzdZ + \
                dPxdY*dPydZ*dPzdX + dPxdZ*dPydX*dPzdY - dPxdZ*dPydY*dPzdX
            dX = ((dPydY*dPzdZ - dPydZ*dPzdY) * resx + \
                (-dPxdY*dPzdZ + dPxdZ*dPzdY) * resy + \
                 (dPxdY*dPydZ - dPxdZ*dPydY) * resz) / detJ
            dY = ((-dPydX*dPzdZ + dPydZ*dPzdX) * resx + \
                   (dPxdX*dPzdZ - dPxdZ*dPzdX) * resy + \
                  (-dPxdX*dPydZ + dPxdZ*dPydX) * resz ) / detJ
            dZ = ((dPydX*dPzdY - dPydY*dPzdX) * resx + \
                 (-dPxdX*dPzdY + dPxdY*dPzdX) * resy + \
                  (dPxdX*dPydY - dPxdY*dPydX) * resz ) / detJ
            X += dX
            Y += dY
            Z += dZ
            res = np.linalg.norm(dX) + np.linalg.norm(dY) + np.linalg.norm(dZ)
            if res < 1e-4:
                break
        return X, Y, Z

    def dPdX(self, X, Y, Z):
        """
        Derivative of the Camera model wrt physical position X, Y

        Parameters
        ----------
        X : 1D NUMPY.ARRAY
            X coordinate of the current position in the mesh system
        Y : 1D NUMPY.ARRAY
            Y coordinate of the current position in the mesh system
        Z : 1D NUMPY.ARRAY
            Z coordinate of the current position in the mesh system

        Returns
        -------
        dudx : NUMPY.ARRAY
            Derivative of coord. u wrt to X
        dudy : NUMPY.ARRAY
            Derivative of coord. u wrt to Y
        dudz : NUMPY.ARRAY
            Derivative of coord. u wrt to Z
        dvdx : NUMPY.ARRAY
            Derivative of coord. v wrt to X
        dvdy : NUMPY.ARRAY
            Derivative of coord. v wrt to Y
        dvdz : NUMPY.ARRAY
            Derivative of coord. v wrt to Z
        dwdx : NUMPY.ARRAY
            Derivative of coord. w wrt to X
        dwdy : NUMPY.ARRAY
            Derivative of coord. w wrt to Y
        dwdz : NUMPY.ARRAY
            Derivative of coord. w wrt to Z

        """
        f = self.f
        rx = self.rx
        ry = self.ry
        rz = self.rz
        ones = np.ones(X.shape[0])
        dudx = (f*np.sin(rx)*np.sin(ry)*np.sin(rz) + f*np.cos(ry)*np.cos(rz)) * ones
        dudy = -f*np.sin(rz)*np.cos(rx)*ones
        dudz = (f*np.sin(rx)*np.sin(rz)*np.cos(ry) - f*np.sin(ry)*np.cos(rz))*ones
        dvdx = (-f*np.sin(rx)*np.sin(ry)*np.cos(rz) + f*np.sin(rz)*np.cos(ry))*ones
        dvdy = f*np.cos(rx)*np.cos(rz)*ones
        dvdz = (-f*np.sin(rx)*np.cos(ry)*np.cos(rz) - f*np.sin(ry)*np.sin(rz))*ones
        dwdx = f*np.sin(ry)*np.cos(rx)*ones
        dwdy = f*np.sin(rx)*ones
        dwdz = f*np.cos(rx)*np.cos(ry)*ones
        return dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz

    def dPdp(self, X, Y, Z):
        """
        First order derivative of the Camera model wrt camera parameters p

        Parameters
        ----------
        X : 1D NUMPY.ARRAY
            X coordinate of the current position in the mesh system
        Y : 1D NUMPY.ARRAY
            Y coordinate of the current position in the mesh system
        Z : 1D NUMPY.ARRAY
            Z coordinate of the current position in the mesh system

        Returns
        -------
        dudp : NUMPY.ARRAY
            Derivative of coord. u wrt to p
        dvdp : NUMPY.ARRAY
            Derivative of coord. v wrt to p
        dwdp : NUMPY.ARRAY
            Derivative of coord. w wrt to p

        """
        f = self.f
        tx = self.tx
        ty = self.ty
        tz = self.tz
        rx = self.rx
        ry = self.ry
        rz = self.rz
        #
        dudf = X*(np.sin(rx)*np.sin(ry)*np.sin(rz) + np.cos(ry)*np.cos(rz)) - \
            Y*np.sin(rz)*np.cos(rx) + Z*(np.sin(rx)*np.sin(rz)*np.cos(ry) - \
                                             np.sin(ry)*np.cos(rz)) + tx
        dudtx = f + 0 * X
        dudty = 0 * X
        dudtz = 0 * X
        dudrx = X*f*np.sin(ry)*np.sin(rz)*np.cos(rx) + Y*f*np.sin(rx)*np.sin(rz) + \
            Z*f*np.sin(rz)*np.cos(rx)*np.cos(ry)
        dudry = X*(f*np.sin(rx)*np.sin(rz)*np.cos(ry) - f*np.sin(ry)*np.cos(rz)) + \
            Z*(-f*np.sin(rx)*np.sin(ry)*np.sin(rz) - f*np.cos(ry)*np.cos(rz))
        dudrz = X*(f*np.sin(rx)*np.sin(ry)*np.cos(rz) - f*np.sin(rz)*np.cos(ry)) - \
            Y*f*np.cos(rx)*np.cos(rz) + Z*(f*np.sin(rx)*np.cos(ry)*np.cos(rz) + \
                                               f*np.sin(ry)*np.sin(rz))
        dvdf = X*(-np.sin(rx)*np.sin(ry)*np.cos(rz) + np.sin(rz)*np.cos(ry)) + \
            Y*np.cos(rx)*np.cos(rz) + Z*(-np.sin(rx)*np.cos(ry)*np.cos(rz) - \
                                             np.sin(ry)*np.sin(rz)) + ty
        # 
        dvdtx = 0 * X
        dvdty = 0 * X + f
        dvdtz = 0 * X
        dvdrx = -X*f*np.sin(ry)*np.cos(rx)*np.cos(rz) - Y*f*np.sin(rx)*np.cos(rz) - \
            Z*f*np.cos(rx)*np.cos(ry)*np.cos(rz)
        dvdry = X*(-f*np.sin(rx)*np.cos(ry)*np.cos(rz) - f*np.sin(ry)*np.sin(rz)) + \
            Z*(f*np.sin(rx)*np.sin(ry)*np.cos(rz) - f*np.sin(rz)*np.cos(ry))
        dvdrz = X*(f*np.sin(rx)*np.sin(ry)*np.sin(rz) + f*np.cos(ry)*np.cos(rz)) - \
            Y*f*np.sin(rz)*np.cos(rx) + Z*(f*np.sin(rx)*np.sin(rz)*np.cos(ry) - \
                                               f*np.sin(ry)*np.cos(rz))
        #                 
        dwdf = X*np.sin(ry)*np.cos(rx) + Y*np.sin(rx) + Z*np.cos(rx)*np.cos(ry) + tz
        dwdtx = 0 * X
        dwdty = 0 * X
        dwdtz = 0 * X + f
        dwdrx = -X*f*np.sin(rx)*np.sin(ry) + Y*f*np.cos(rx) - Z*f*np.sin(rx)*np.cos(ry)
        dwdry = X*f*np.cos(rx)*np.cos(ry) - Z*f*np.sin(ry)*np.cos(rx)
        dwdrz = 0 * X 
        #
        dudp = np.c_[dudf, dudtx, dudty, dudtz, dudrx, dudry, dudrz]
        dvdp = np.c_[dvdf, dvdtx, dvdty, dvdtz, dvdrx, dvdry, dvdrz]
        dwdp = np.c_[dwdf, dwdtx, dwdty, dwdtz, dwdrx, dwdry, dwdrz]        
        return dudp, dvdp, dwdp

    def ImageFiles(self, fname, imnums):
        self.fname = fname
        self.imnums = imnums