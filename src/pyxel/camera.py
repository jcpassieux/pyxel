# -*- coding: utf-8 -*-
"""
Finite Element Digital Image Correlation method 

@author: JC Passieux, INSA Toulouse, 2021

pyxel

PYthon library for eXperimental mechanics using Finite ELements

"""

import numpy as np
import cv2

class Camera:
    def __init__(self, dim=3):
        self.K = np.eye(3)
        self.D = np.zeros((1, 5))
        self.R = np.zeros((3, 1))
        self.T = np.zeros((3, 1))
        self.T[2, 0] = 1.
        self.dim = dim  # 2 for 2D-DIC and 3 for Stereo-DIC

    def LoadIntrinsic(self, filename):
        params = dict(np.load(filename))
        self.K = params['Intrinsic']
        self.D = params['Distortion']

    def Get_pix2m(self):
        return self.T[-1, 0]/(0.5*(self.K[0,0] + self.K[1,1]))

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
        cam = Camera(self.dim)
        cam.K = self.K / 2 ** nscale
        cam.D = self.D
        cam.R = self.R
        cam.T = self.T
        return cam

    def set_p(self, pvec, p='extrinsic'):
        """
        Sets the parameters of the camera model from a vector

        p : STRING
            'extrinsic' > set extrinsic only (DEFAULT)
            'intrinsic' > set intrinsic only
            'distortion' > set distortion only
             otherwise > set all parameters
        """
        if p == 'extrinsic':
            if self.dim == 2:
                self.R[2, 0] = pvec[0]
            else:
                self.R[:, 0] = pvec[:3]
            self.T[:, 0] = pvec[-3:]
        elif p == 'intrinsic':
            self.K[[0, 1, 0, 1], [0, 1, 2, 2]] = pvec
        elif p == 'distortion':
            self.D = pvec[np.newaxis]
        else:  # deriv wrt all params
            if self.dim == 2:
                self.R[2, 0] = pvec[0]
                self.T[:, 0] = pvec[1:4]
                self.K[[0, 1, 0, 1], [0, 1, 2, 2]] = pvec[4:8]
                self.D = pvec[8:][np.newaxis]
            else:
                self.R[:, 0] = pvec[:3]
                self.T[:, 0] = pvec[3:6]
                self.K[[0, 1, 0, 1], [0, 1, 2, 2]] = pvec[6:10]
                self.D = pvec[10:][np.newaxis]

    def get_p(self, p='extrinsic'):
        """
        Returns the vector of parameters of the camera model

        p : STRING
            'extrinsic' > get extrinsic only (DEFAULT)
            'intrinsic' > get intrinsic only
            'distortion' > get distortion only
             otherwise > get all parameters
        """
        if self.dim == 2:
            pext = np.append(self.R[2], self.T)
        else:
            pext = np.append(self.R, self.T)
        pint = self.K[[0, 1, 0, 1], [0, 1, 2, 2]]
        dist = self.D[0]
        if p == 'extrinsic':
            return pext
        elif p == 'intrinsic':
            return pint
        elif p == 'distortion':
            return dist
        else:  # deriv wrt all params
            return np.hstack((pext, pint, dist))

    def P(self, X, Y, Z=None):
        """
        Projection of a 3D point X, Y (, Z) in the image plane u, v

        Parameters
        ----------
        X : 1D NUMPY.ARRAY
            X coordinate of the current position in the mesh system
        Y : 1D NUMPY.ARRAY
            Y coordinate of the current position in the mesh system
        Z : 1D NUMPY.ARRAY
            Z coordinate of the current position in the mesh system
            if Z is NONE, Z=0

        Returns
        -------
        u : NUMPY.ARRAY
            coord. u in the image plane
        v : NUMPY.ARRAY
            coord. v in the image plane
        """
        if Z is None:
            Z = X*0
        pts = np.c_[X, Y, Z]
        # Rmtx, _ = cv2.Rodrigues(self.R)
        temp, _ = cv2.projectPoints(pts, self.R, self.T, self.K, self.D)
        return temp[:, 0, 0], temp[:, 0, 1]

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
            res = np.linalg.norm(resx) + np.linalg.norm(resy)
            if res < 1e-2:
                break
        return X, Y

    def dPdX(self, X, Y, Z=None):
        """
        Derivative of the Camera model wrt physical position X, Y

        Parameters
        ----------
        X : 1D NUMPY.ARRAY
            X coordinate of the current position in the mesh system
        Y : 1D NUMPY.ARRAY
            Y coordinate of the current position in the mesh system
        Z : 1D NUMPY.ARRAY (OPTIONNAL)
            Z coordinate of the current position in the mesh system

        Returns
        -------
        dudx : NUMPY.ARRAY
            Derivative of coord. u wrt to X
        dudy : NUMPY.ARRAY
            Derivative of coord. u wrt to Y
        dudz : NUMPY.ARRAY   (ONLY IF IN 3D)
            Derivative of coord. u wrt to Y
        dvdx : NUMPY.ARRAY
            Derivative of coord. v wrt to X
        dvdy : NUMPY.ARRAY
            Derivative of coord. v wrt to Y
        dvdz : NUMPY.ARRAY   (ONLY IF IN 3D)
            Derivative of coord. v wrt to Y

        """
        # import sympy as sp
        # X,Y,Z = sp.symbols('X,Y,Z')
        
        # fx,fy,cx,cy = sp.symbols('fx,fy,cx,cy ')
        
        # r00,r01,r02,r10,r11,r12, r20, r21,r22, t0,t1,t2 = sp.symbols('r00,r01,r02,r10,r11,r12, r20, r21,r22, t0,t1,t2')
        # k1,k2,k3, p1,p2 = sp.symbols('k1,k2,k3, p1,p2')
        # Xc, Yc, Zc = sp.symbols('Xc,Yc,Zc')
        # # K = sp.Matrix([[fx,0,cx],[0,fy,cy],[0,0,1]])
        # RT = sp.Matrix([[r00,r01,r02,t0],[r10,r11,r12,t1],[r20,r21,r22,t2]])
        # Xws = sp.Matrix([X,Y,Z,1])
        # Xcs = RT*Xws
        # Xc = Xcs[0]
        # Yc = Xcs[1]
        # Zc = Xcs[2]
        # xprime = Xc/Zc
        # yprime = Yc/Zc
        # r = sp.sqrt(xprime**2 + yprime**2)
        # xpp = xprime*(1+k1*r**2+k2*r**4+k3*r**6) + 2*p1*xprime*yprime+ p2*(r**2+2*xprime**2)
        # ypp = yprime*(1+k1*r**2+k2*r**4+k3*r**6) + 2*p2*xprime*yprime+ p1*(r**2+2*yprime**2)
        # u = fx*xpp + cx
        # v = fy*ypp + cy
        # derivs = [u.diff(X), u.diff(Y), u.diff(Z), v.diff(X), v.diff(Y), v.diff(Z)]
        # variable_namer = sp.numbered_symbols('T')
        # replacements, reduced = sp.cse(derivs, symbols=variable_namer)
        # for key, val in replacements:
        #     print(key, '=', sp.pycode(val))
        # for i, r in enumerate(reduced):
        #     print('deriv[{}]'.format(i), '=', sp.pycode(r))

        if Z is None:
            Z = X*0

        R, _ = cv2.Rodrigues(self.R)
        r00 = R[0, 0]
        r01 = R[0, 1]
        r02 = R[0, 2]
        r10 = R[1, 0]
        r11 = R[1, 1]
        r12 = R[1, 2]
        r20 = R[2, 0]
        r21 = R[2, 1]
        r22 = R[2, 2]

        t0, t1, t2 = self.T
        k1, k2, p1, p2, k3 = self.D[0, :5]
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        # cx = K[0, 2]
        # cy = K[1, 2]

        T0 = X*r10 + Y*r11 + Z*r12 + t1
        T1 = X*r20 + Y*r21 + Z*r22 + t2
        T2 = T1**(-2)
        T3 = 2*T2
        T4 = T3*p1
        T5 = T0*T4
        T6 = X*r00 + Y*r01 + Z*r02 + t0
        T7 = T4*T6
        T8 = T1**(-3)
        T9 = T8*r20
        T10 = 4*T0*T6
        T11 = T10*p1
        T12 = T0**2
        T13 = T12*T9
        T14 = T0*r10
        T15 = -2*T13 + T14*T3
        T16 = T6**2
        T17 = T16*T9
        T18 = T6*r00
        T19 = 6*T2
        T20 = -6*T17 + T18*T19
        T21 = 1/T1
        T22 = T12*T2 + T16*T2
        T23 = T22**2
        T24 = T22**3*k3 + T22*k1 + T23*k2 + 1
        T25 = T21*T24
        T26 = T2*T24
        T27 = T26*T6
        T28 = -2*T17 + T18*T3
        T29 = 4*T2
        T30 = T22*k2
        T31 = -6*T13 + T14*T19
        T32 = T23*k3
        T33 = T30*(-4*T13 + T14*T29 - 4*T17 + T18*T29) + T32*(T20 + T31) + k1*(T15 + T28)
        T34 = T21*T6
        T35 = T8*r21
        T36 = T12*T35
        T37 = T0*r11
        T38 = T3*T37 - 2*T36
        T39 = T16*T35
        T40 = T6*r01
        T41 = T19*T40 - 6*T39
        T42 = T3*T40 - 2*T39
        T43 = T19*T37 - 6*T36
        T44 = T30*(T29*T37 + T29*T40 - 4*T36 - 4*T39) + T32*(T41 + T43) + k1*(T38 + T42)
        T45 = T8*r22
        T46 = T12*T45
        T47 = T0*r12
        T48 = T3*T47 - 2*T46
        T49 = T16*T45
        T50 = T6*r02
        T51 = T19*T50 - 6*T49
        T52 = T3*T50 - 2*T49
        T53 = T19*T47 - 6*T46
        T54 = T30*(T29*T47 + T29*T50 - 4*T46 - 4*T49) + T32*(T51 + T53) + k1*(T48 + T52)
        T55 = T3*p2
        T56 = T0*T55
        T57 = T55*T6
        T58 = T10*p2
        T59 = T0*T26
        T60 = T0*T21
        dudx = fx*(-T11*T9 + T25*r00 - T27*r20 + T33*T34 + T5*r00 + T7*r10 + p2*(T15 + T20))
        dudy = fx*(-T11*T35 + T25*r01 - T27*r21 + T34*T44 + T5*r01 + T7*r11 + p2*(T38 + T41))
        dudz = fx*(-T11*T45 + T25*r02 - T27*r22 + T34*T54 + T5*r02 + T7*r12 + p2*(T48 + T51))
        dvdx = fy*(T25*r10 + T33*T60 + T56*r00 + T57*r10 - T58*T9 - T59*r20 + p1*(T28 + T31))
        dvdy = fy*(T25*r11 - T35*T58 + T44*T60 + T56*r01 + T57*r11 - T59*r21 + p1*(T42 + T43))
        dvdz = fy*(T25*r12 - T45*T58 + T54*T60 + T56*r02 + T57*r12 - T59*r22 + p1*(T52 + T53))
        if self.dim == 2:
            return dudx, dudy, dvdx, dvdy
        else:
            return dudx, dudy, dudz, dvdx, dvdy, dvdz
            
    def dPdp(self, X, Y, Z=None, p='extrinsic'):
        """
        First order derivative of the Camera model wrt camera parameters

        Parameters
        ----------
        X : 1D NUMPY.ARRAY
            X coordinate of the current position in the mesh system
        Y : 1D NUMPY.ARRAY
            Y coordinate of the current position in the mesh system
        Z : 1D NUMPY.ARRAY
            Z coordinate of the current position in the mesh system
            if Z is NONE, Z=0

        Returns
        -------
        dudp : NUMPY.ARRAY
            Derivative of coord. u wrt to params
        dvdp : NUMPY.ARRAY
            Derivative of coord. v wrt to params
        p : STRING
            'extrinsic' > derivative wrt to extrinsic only (DEFAULT)
            'intrinsic' > derivative wrt to intrinsic only
            'distortion' > derivative wrt to distortion only
             otherwise > derivative wrt to all parameters
        """
        if Z is None:
            Z = X*0
        pts = np.c_[X, Y, Z]
        _, jac = cv2.projectPoints(pts, self.R, self.T, self.K, self.D)
        if p == 'extrinsic':
            if self.dim == 2:
                jac_x = jac[0::2, 2:6]
                jac_y = jac[1::2, 2:6]
            else:
                jac_x = jac[0::2, :6]
                jac_y = jac[1::2, :6]
        elif p == 'intrinsic':
            jac_x = jac[0::2, [6, 7, 8, 9]]
            jac_y = jac[1::2, [6, 7, 8, 9]]
        elif p == 'distortion':
            ndist = -np.prod(self.D.shape)
            jac_x = jac[0::2, ndist:]
            jac_y = jac[1::2, ndist:]
        else:  # deriv wrt all params
            if self.dim == 2:
                jac_x = jac[0::2, 2:]
                jac_y = jac[1::2, 2:]
            else:
                jac_x = jac[0::2, :]
                jac_y = jac[1::2, :]
        return jac_x, jac_y


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

    def Get_pix2m(self):
        return 1/self.f
    
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