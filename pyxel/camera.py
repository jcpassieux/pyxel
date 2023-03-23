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
        self.f = p[0]
        self.tx = p[1]
        self.ty = p[2]
        self.rz = p[3]

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
        # sp.pycode(sp.diff(vd, Y))
        f = self.f
        tx = self.tx
        ty = self.ty
        rz = self.rz
        u0 = self.u0
        v0 = self.v0
        r1 = self.r1
        dudx = f*(r1*((f*(X*np.sin(rz) - Y*np.cos(rz) - ty) - u0)**2 + (f*(X*np.cos(rz) + Y*np.sin(rz) + tx) - v0)**2) + 1)*np.sin(rz) + r1*(f*(X*np.sin(rz) - Y*np.cos(rz) - ty) - u0)*(2*f*(f*(X*np.sin(rz) - Y*np.cos(rz) - ty) - u0)*np.sin(rz) + 2*f*(f*(X*np.cos(rz) + Y*np.sin(rz) + tx) - v0)*np.cos(rz))
        dudy = -f*(r1*((f*(X*np.sin(rz) - Y*np.cos(rz) - ty) - u0)**2 + (f*(X*np.cos(rz) + Y*np.sin(rz) + tx) - v0)**2) + 1)*np.cos(rz) + r1*(f*(X*np.sin(rz) - Y*np.cos(rz) - ty) - u0)*(-2*f*(f*(X*np.sin(rz) - Y*np.cos(rz) - ty) - u0)*np.cos(rz) + 2*f*(f*(X*np.cos(rz) + Y*np.sin(rz) + tx) - v0)*np.sin(rz))
        dvdx = f*(r1*((f*(X*np.sin(rz) - Y*np.cos(rz) - ty) - u0)**2 + (f*(X*np.cos(rz) + Y*np.sin(rz) + tx) - v0)**2) + 1)*np.cos(rz) + r1*(f*(X*np.cos(rz) + Y*np.sin(rz) + tx) - v0)*(2*f*(f*(X*np.sin(rz) - Y*np.cos(rz) - ty) - u0)*np.sin(rz) + 2*f*(f*(X*np.cos(rz) + Y*np.sin(rz) + tx) - v0)*np.cos(rz))
        dvdy = f*(r1*((f*(X*np.sin(rz) - Y*np.cos(rz) - ty) - u0)**2 + (f*(X*np.cos(rz) + Y*np.sin(rz) + tx) - v0)**2) + 1)*np.sin(rz) + r1*(f*(X*np.cos(rz) + Y*np.sin(rz) + tx) - v0)*(-2*f*(f*(X*np.sin(rz) - Y*np.cos(rz) - ty) - u0)*np.cos(rz) + 2*f*(f*(X*np.cos(rz) + Y*np.sin(rz) + tx) - v0)*np.sin(rz))
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