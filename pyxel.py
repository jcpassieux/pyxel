# -*- coding: utf-8 -*-
"""
Finite Element Digital Image Correlation method 

@author: JC Passieux, INSA Toulouse, 2020

pyxel

Python library for eXperimental mechanics using finite ELements

"""

import numpy as np
import scipy as sp
import scipy.sparse.linalg as splalg
import scipy.interpolate as spi
import matplotlib.pyplot as plt
import PIL.Image as image
import matplotlib.collections as cols
import matplotlib.animation as animation
import os

import vtktools as vtk

#import pdb
#pdb.set_trace()

import time as time
class Timer():
    def __init__(self):
        self.tstart = time.clock()
    def stop(self):
        dt=time.clock() - self.tstart
        print('Elapsed: %f' % dt)
        return dt


class Elem:
    """ Class Element """
    def __init__(self):
        self.pgx=[]
        self.pgy=[]
        self.phi=[]
        self.dphidx=[]
        self.dphidy=[]

def IsPointInElem2d(xn,yn,x,y):
    """ Find if a point (XP,YP) belong to an element with vertices (XN,YN) """
    yes=np.ones(len(x))
    cx=np.mean(xn)
    cy=np.mean(yn)
    for jn in range(0,len(xn)):
         if jn==(len(xn)-1):
             i1=jn
             i2=0
         else:
             i1=jn
             i2=jn+1
         x1=xn[i1]
         y1=yn[i1]
         dpx=xn[i2]-x1
         dpy=yn[i2]-y1
         a=dpy*(x-x1)-dpx*(y-y1)
         b=dpy*(cx-x1)-dpx*(cy-y1)
         yes=yes*(a*b>=0)
    return yes


def isInBox(x,y,b):
    """ Find whether set of points of coords x,y 
         is in the box b=[[xmin,xmax],[ymin,ymax]] """
    if b.shape[1]!=2:
        print("the box [[xmin,xmax],[ymin,ymax]] in isInBox")
    e=1e-6*np.max(np.abs(b.ravel()))+1e-6*np.std(b.ravel())
    return ((b[0,0]-e)<x)*((b[1,0]-e)<y)*(x<(b[0,1]+e))*(y<(b[1,1]+e))

#%%
class Mesh:
    def __init__(self,e,n,dim=2):
        """ Contructor from elems and node arrays """   
        self.e = e
        self.n = n
        self.conn = []
        self.ndof = []
        self.nvm = []
        self.npg = []
        self.pgx = []
        self.pgy = []
        self.phix = []
        self.phiy = []
        self.wdetJ = []
        self.dim=dim
        
    def Copy(self):
        m=Mesh(self.e.copy(),self.n.copy())
        m.conn=self.conn.copy()
        m.ndof=self.ndof
        m.nvm=self.nvm
        m.dim=self.dim
        return m
        
    def Connectivity(self):
        """ Compute connectivity """
        self.conn=-np.ones(self.n.shape[0],dtype=np.int)
        c=0
        self.nvm=0
        for je in range(len(self.e)):
            newnodes,=np.where(self.conn[self.e[je][1:]]<0)
            self.conn[self.e[je][newnodes+1]]=c+np.arange(len(newnodes))
            c+=newnodes.shape[0]               
            self.nvm+=4*(len(self.e[je])-1)**2
        if self.dim==2:
            self.conn=np.c_[self.conn,self.conn+c*(self.conn>=0)]
        else:
            self.conn=np.c_[self.conn,self.conn+c*(self.conn>=0),self.conn+2*c*(self.conn>=0)]
        self.ndof=c*self.dim


    def DICIntegration(self,cam):
        """ Build the integration scheme """
        nzv=0   # nb of non zero values in phix
        repg=0  # nb of integration points
        elem = np.empty(len(self.e), dtype=object)
        for je in range(len(self.e)):
            elem[je] = Elem()
            if self.e[je][0]==3:  # qua4
                elem[je].repx=self.conn[self.e[je][1:],0]
                elem[je].repy=self.conn[self.e[je][1:],1]
                xn=self.n[self.e[je][1:],0]
                yn=self.n[self.e[je][1:],1]            
                u,v=cam.P(xn,yn)
                dist=np.floor(np.sqrt((u[[0,1]]-u[[1,2]])**2+(v[[0,1]]-v[[1,2]])**2))
                xg,yg,wg=SubQuaIso(max([1,dist[0]]),max([1,dist[1]]))
                elem[je].phi = 0.25 * np.c_[(1-xg)*(1-yg),(1+xg)*(1-yg),(1+xg)*(1+yg),(1-xg)*(1+yg)]
                elem[je].pgx=elem[je].phi.dot(xn)
                elem[je].pgy=elem[je].phi.dot(yn)
                elem[je].repg = repg+np.arange(xg.shape[0])
                repg+=xg.shape[0]            
                dN_xi=0.25*np.c_[-(1-yg),1-yg,1+yg,-(1+yg)]
                dN_eta =0.25*np.c_[-(1-xg),-(1+xg),1+xg,1-xg]
                detJ = dN_xi.dot(xn) * dN_eta.dot(yn) - dN_eta.dot(xn)*dN_xi.dot(yn)
                elem[je].wdetJ=wg*abs(detJ)
            elif self.e[je][0]==2: # tri3
                elem[je].repx=self.conn[self.e[je][1:],0]
                elem[je].repy=self.conn[self.e[je][1:],1]
                xn=self.n[self.e[je][1:],0]
                yn=self.n[self.e[je][1:],1]
                u,v=cam.P(xn,yn)
                uu=np.diff(np.append(u,u[0]))
                vv=np.diff(np.append(v,v[0]))
                nn=np.sqrt(uu**2+vv**2)/1.1
                (a,)=np.where(nn==np.max(nn))[0]             # a is the largest triangle side
                nx=max(nn[np.array([2,0,1])[a]].astype('int8'),1)
                ny=max(nn[np.array([1,2,0])[a]].astype('int8'),1)
                xg,yg,wg=SubTriIso(nx,ny)
                if a==0:
                    pp=np.c_[(1-xg-yg),xg,yg].dot(np.array([[0,1],[0,0],[1,0]]))
                    xg=pp[:,0]
                    yg=pp[:,1]
                elif a==2:
                    pp=np.c_[(1-xg-yg),xg,yg].dot(np.array([[1,0],[0,1],[0,0]]))
                    xg=pp[:,0]
                    yg=pp[:,1]
                    # DO NOT NEED TO MODIFY Gauss WEIGHT, BECAUSE JACOBIAN IS CONSTANT=1
                elem[je].phi = np.c_[(1-xg-yg),xg,yg]
                elem[je].pgx=elem[je].phi.dot(xn)
                elem[je].pgy=elem[je].phi.dot(yn)
                elem[je].repg = repg+np.arange(xg.shape[0])
                repg+=xg.shape[0]
                dN_xi= np.c_[-1,1,0]
                dN_eta=np.c_[-1,0,1]
                detJ = dN_xi.dot(xn)*dN_eta.dot(yn) - dN_eta.dot(xn)*dN_xi.dot(yn)
                elem[je].wdetJ=wg*abs(detJ)
            else:
                print("Oops!  That is not a valid element type...")
            nzv+=np.prod(elem[je].phi.shape)
        self.npg=repg
        self.pgx=np.zeros(self.npg)
        self.pgy=np.zeros(self.npg)
        for je in range(len(self.e)):
            self.pgx[elem[je].repg]=elem[je].pgx
            self.pgy[elem[je].repg]=elem[je].pgy
        
        """ Build the FE interpolation """
        self.wdetJ=np.zeros(self.npg)
        row=np.zeros(nzv)
        col=np.zeros(nzv)
        val=np.zeros(nzv)
        #rowe=np.zeros(self.npg) # Elementary Brightness and Contrast Correction
        #cole=np.zeros(self.npg)
        #vale=np.zeros(self.npg)
        nzv=0
        #nzv1=0
        for je in range(len(self.e)):
            self.wdetJ[elem[je].repg]=elem[je].wdetJ
            [repj,repi]=np.meshgrid(elem[je].repx,elem[je].repg)
            rangephi = nzv + np.arange(np.prod(elem[je].phi.shape))
            row[rangephi]=repi.ravel()
            col[rangephi]=repj.ravel()
            val[rangephi]=elem[je].phi.ravel()
            nzv+=np.prod(elem[je].phi.shape)
            #rangeone = nzv1 + np.arange(len(elem[je].repg))
            #rowe[rangeone]=elem[je].repg
            #cole[rangeone]=je
            #vale[rangeone]=1
            #nzv1+=len(elem[je].repg)
        self.phix=sp.sparse.csc_matrix((val, (row, col+0*self.ndof/2)), shape=(self.npg, self.ndof))
        self.phiy=sp.sparse.csc_matrix((val, (row, col+1*self.ndof/2)), shape=(self.npg, self.ndof))
        #self.Me  =sp.sparse.csc_matrix((vale, (rowe, cole)), shape=(self.npg, len(self.e)))       

    def DICIntegrationPixel(self,cam):
        """ Build a pixel integration scheme """
        nzv=0   # nb of non zero values in phix
        repg=0  # nb of integration points
        elem = np.empty(len(self.e), dtype=object)
        for je in range(len(self.e)):
            elem[je] = Elem()
            if self.e[je][0]==3:  # qua4
                elem[je].repx=self.conn[self.e[je][1:],0]
                elem[je].repy=self.conn[self.e[je][1:],1]
                xn=self.n[self.e[je][1:],0]
                yn=self.n[self.e[je][1:],1]
                u,v=cam.P(xn,yn)
                rx=np.arange(np.floor(min(u)),np.ceil(max(u))+1).astype('int')
                ry=np.arange(np.floor(min(v)),np.ceil(max(v))+1).astype('int')
                [ypix,xpix]=np.meshgrid(ry,rx)
                [xg,yg,elem[je].pgx,elem[je].pgy]=GetPixelsQua(u,v,xpix.ravel(),ypix.ravel())
                elem[je].phi = 0.25 * np.c_[(1-xg)*(1-yg),(1+xg)*(1-yg),(1+xg)*(1+yg),(1-xg)*(1+yg)]
                elem[je].repg = repg+np.arange(xg.shape[0])
                repg+=xg.shape[0]
                elem[je].wdetJ=np.ones(xg.shape[0])
            elif self.e[je][0]==2: # tri3
                elem[je].repx=self.conn[self.e[je][1:],0]
                elem[je].repy=self.conn[self.e[je][1:],1]
                xn=self.n[self.e[je][1:],0]
                yn=self.n[self.e[je][1:],1]
                u,v=cam.P(xn,yn)
                rx=np.arange(np.floor(min(u)),np.ceil(max(u))+1).astype('int')
                ry=np.arange(np.floor(min(v)),np.ceil(max(v))+1).astype('int')
                [ypix,xpix]=np.meshgrid(ry,rx)
                [xg,yg,elem[je].pgx,elem[je].pgy]=GetPixelsTri(u,v,xpix.ravel(),ypix.ravel())
                elem[je].phi = np.c_[(1-xg-yg),xg,yg]
                elem[je].repg = repg+np.arange(xg.shape[0])
                repg+=xg.shape[0]
                elem[je].wdetJ=np.ones(xg.shape[0])
            else:
                print("Oops!  That is not a valid element type...")
            nzv+=np.prod(elem[je].phi.shape)
        self.npg=repg
        self.pgx=np.zeros(self.npg)
        self.pgy=np.zeros(self.npg)
        for je in range(len(self.e)):
            self.pgx[elem[je].repg]=elem[je].pgx
            self.pgy[elem[je].repg]=elem[je].pgy

        """ Inverse Mapping ray tracing """
        pgX=np.zeros(self.pgx.shape[0])
        pgY=np.zeros(self.pgx.shape[0])
        for ii in range(10):
            pgx,pgy=cam.P(pgX,pgY)
            resx=self.pgx-pgx
            resy=self.pgy-pgy
            #print(np.linalg.norm(resx)+np.linalg.norm(resy))
            dPxdX,dPxdY,dPydX,dPydY=cam.dPdX(pgX,pgY)
            detJ=dPxdX*dPydY-dPxdY*dPydX
            dX=dPydY/detJ*resx-dPxdY/detJ*resy
            dY=-dPydX/detJ*resx+dPxdX/detJ*resy
            pgX+=dX
            pgY+=dY
            res=np.linalg.norm(dX)+np.linalg.norm(dY)
            if res<1e-4:
                break
        self.pgx=pgX
        self.pgy=pgY
        
        """ Build the FE interpolation """
        self.wdetJ=np.zeros(self.npg)
        row=np.zeros(nzv)
        col=np.zeros(nzv)
        val=np.zeros(nzv)
        nzv=0
        for je in range(len(self.e)):
            self.wdetJ[elem[je].repg]=elem[je].wdetJ
            [repj,repi]=np.meshgrid(elem[je].repx,elem[je].repg)
            rangephi = nzv + np.arange(np.prod(elem[je].phi.shape))
            row[rangephi]=repi.ravel()
            col[rangephi]=repj.ravel()
            val[rangephi]=elem[je].phi.ravel()
            nzv+=np.prod(elem[je].phi.shape)
        self.phix=sp.sparse.csc_matrix((val, (row, col+0*self.ndof/2)), shape=(self.npg, self.ndof))
        self.phiy=sp.sparse.csc_matrix((val, (row, col+1*self.ndof/2)), shape=(self.npg, self.ndof))
        
    def DICIntegrationWithGrad(self,cam):
        """ Build the integration scheme """
        nzv=0   # nb of non zero values in phix
        repg=0  # nb of integration points
        elem = np.empty(len(self.e) , dtype=object)
        for je in range(len(self.e)):
            elem[je] = Elem()
            if self.e[je][0]==3:  # qua4
                elem[je].repx=self.conn[self.e[je][1:],0]
                elem[je].repy=self.conn[self.e[je][1:],1]
                xn=self.n[self.e[je][1:],0]
                yn=self.n[self.e[je][1:],1]
                u,v=cam.P(xn,yn)
                dist=np.floor(np.sqrt((u[[0,1]]-u[[1,2]])**2+(v[[0,1]]-v[[1,2]])**2))
                xg,yg,wg=SubQuaIso(max([1,dist[0]]),max([1,dist[1]]))
                elem[je].phi = 0.25 * np.array([(1-xg)*(1-yg),(1+xg)*(1-yg),(1+xg)*(1+yg),(1-xg)*(1+yg)]).T
                elem[je].pgx=elem[je].phi.dot(xn)
                elem[je].pgy=elem[je].phi.dot(yn)
                elem[je].repg = repg+np.arange(xg.shape[0])
                repg+=xg.shape[0]            
                dN_xi=0.25*np.array([-(1-yg),1-yg,1+yg,-(1+yg)]).T
                dN_eta =0.25*np.array([-(1-xg),-(1+xg),1+xg,1-xg]).T
            elif self.e[je][0]==2: # tri3
                elem[je].repx=self.conn[self.e[je][1:],0]
                elem[je].repy=self.conn[self.e[je][1:],1]
                xn=self.n[self.e[je][1:],0]
                yn=self.n[self.e[je][1:],1]
                u,v=cam.P(xn,yn)
                uu=np.diff(np.append(u,u[0]))
                vv=np.diff(np.append(v,v[0]))
                nn=np.sqrt(uu**2+vv**2)/1.1
                (a,)=np.where(nn==np.max(nn))[0]             # a is the largest triangle side
                nx=max(nn[np.array([2,0,1])[a]].astype('int8'),1)
                ny=max(nn[np.array([1,2,0])[a]].astype('int8'),1)
                xg,yg,wg=SubTriIso(nx,ny)
                if a==0:
                    pp=np.c_[(1-xg-yg),xg,yg].dot(np.array([[0,1],[0,0],[1,0]]))
                    xg=pp[:,0]
                    yg=pp[:,1]
                elif a==2:
                    pp=np.c_[(1-xg-yg),xg,yg].dot(np.array([[1,0],[0,1],[0,0]]))
                    xg=pp[:,0]
                    yg=pp[:,1]
                    # DO NOT NEED TO MODIFY Gauss WEIGHT, BECAUSE JACOBIAN IS CONSTANT=1
                elem[je].phi = np.c_[(1-xg-yg),xg,yg]
                elem[je].pgx=elem[je].phi.dot(xn)
                elem[je].pgy=elem[je].phi.dot(yn)
                elem[je].repg = repg+np.arange(xg.shape[0])
                repg+=xg.shape[0]
                dN_xi= np.c_[-np.ones(len(xg)),np.ones(len(xg)),np.zeros(len(xg))]
                dN_eta=np.c_[-np.ones(len(xg)),np.zeros(len(xg)),np.ones(len(xg))]
                detJ = dN_xi.dot(xn)*dN_eta.dot(yn) - dN_eta.dot(xn)*dN_xi.dot(yn)
                elem[je].wdetJ=wg*abs(detJ)
            else:
                print("Oops!  That is not a valid element type...")
            dxdr=dN_xi.dot(xn)
            dydr=dN_xi.dot(yn)
            dxds=dN_eta.dot(xn)
            dyds=dN_eta.dot(yn)
            detJ=(dxdr*dyds-dydr*dxds)
            elem[je].wdetJ=wg*abs(detJ)
            elem[je].dphidx=sp.sparse.diags(dyds/detJ).dot(dN_xi)+sp.sparse.diags(-dydr/detJ).dot(dN_eta)
            elem[je].dphidy=sp.sparse.diags(-dxds/detJ).dot(dN_xi)+sp.sparse.diags(dxdr/detJ).dot(dN_eta)
            nzv+=np.prod(elem[je].phi.shape)
        self.npg=repg
        self.pgx=np.zeros(self.npg)
        self.pgy=np.zeros(self.npg)
        for je in range(len(self.e)):
            self.pgx[elem[je].repg]=elem[je].pgx
            self.pgy[elem[je].repg]=elem[je].pgy
        
        """ Build the FE interpolation """
        self.wdetJ=np.zeros(self.npg)
        row=np.zeros(nzv)
        col=np.zeros(nzv)
        val=np.zeros(nzv)
        valx=np.zeros(nzv)
        valy=np.zeros(nzv)
        nzv=0
        for je in range(len(self.e)):
            self.wdetJ[elem[je].repg]=elem[je].wdetJ
            [repj,repi]=np.meshgrid(elem[je].repx,elem[je].repg)
            rangephi = np.arange(np.prod(elem[je].phi.shape))
            row[nzv+rangephi]=repi.ravel()
            col[nzv+rangephi]=repj.ravel()
            val[nzv+rangephi]=elem[je].phi.ravel()
            valx[nzv+rangephi]=elem[je].dphidx.ravel()
            valy[nzv+rangephi]=elem[je].dphidy.ravel()
            nzv=nzv+np.prod(elem[je].phi.shape)
        self.phix=sp.sparse.csc_matrix((val, (row, col+0*self.ndof//2)), shape=(self.npg, self.ndof))
        self.phiy=sp.sparse.csc_matrix((val, (row, col+1*self.ndof//2)), shape=(self.npg, self.ndof))
        self.dphixdx=sp.sparse.csc_matrix((valx, (row, col+0*self.ndof//2)), shape=(self.npg, self.ndof))
        self.dphixdy=sp.sparse.csc_matrix((valy, (row, col+0*self.ndof//2)), shape=(self.npg, self.ndof))
        self.dphiydx=sp.sparse.csc_matrix((valx, (row, col+1*self.ndof//2)), shape=(self.npg, self.ndof))
        self.dphiydy=sp.sparse.csc_matrix((valy, (row, col+1*self.ndof//2)), shape=(self.npg, self.ndof))
        
    def GaussIntegration(self):
        """ Build the integration scheme """
        nzv=0   # nb of non zero values in dphixdx
        repg=0  # nb of integration points
        elem = np.empty(len(self.e) , dtype=object)
        for je in range(len(self.e)):
            elem[je] = Elem()
            if self.e[je][0]==3:  # qua4
                elem[je].repx=self.conn[self.e[je][1:],0]
                elem[je].repy=self.conn[self.e[je][1:],1]
                xn=self.n[self.e[je][1:],0]
                yn=self.n[self.e[je][1:],1]            
                xg=np.sqrt(3)/3*np.array([-1,1,-1,1])
                yg=np.sqrt(3)/3*np.array([-1,-1,1,1])
                wg=np.ones(4)
                elem[je].repg = repg+np.arange(xg.shape[0])
                repg+=xg.shape[0]
                dN_xi=0.25*np.array([-(1-yg),1-yg,1+yg,-(1+yg)]).T
                dN_eta =0.25*np.array([-(1-xg),-(1+xg),1+xg,1-xg]).T
                elem[je].phi = 0.25 * np.array([(1-xg)*(1-yg),(1+xg)*(1-yg),(1+xg)*(1+yg),(1-xg)*(1+yg)]).T
                elem[je].pgx=elem[je].phi.dot(xn)
                elem[je].pgy=elem[je].phi.dot(yn)
            elif self.e[je][0]==2: # tri3
                elem[je].repx=self.conn[self.e[je][1:],0]
                elem[je].repy=self.conn[self.e[je][1:],1]
                xn=self.n[self.e[je][1:],0]
                yn=self.n[self.e[je][1:],1]            
                xg=1./3*np.array([1])
                yg=1./3*np.array([1])
                wg=np.array([0.5])
                elem[je].repg = repg+np.arange(xg.shape[0])
                repg+=xg.shape[0]    
                dN_xi=np.array([[-1,1,0]])
                dN_eta=np.array([[-1,0,1]])
                elem[je].phi = np.array([1-xg-yg,xg,yg]).T
                elem[je].pgx=elem[je].phi.dot(xn)
                elem[je].pgy=elem[je].phi.dot(yn)
            elif self.e[je][0] == 9 : #tri6 
                elem[je].repx=self.conn[self.e[je][1:],0]
                elem[je].repy=self.conn[self.e[je][1:],1]
                xn=self.n[self.e[je][1:],0]
                yn=self.n[self.e[je][1:],1]  
                # quadrature using 3 gp 
                # xg = np.array([1./6, 2./3, 1./6])
                # yg = np.array([1./6, 1./6, 2./3])  
                # wg = 1./6*np.ones(3)
                # quadrature using 6 gp 
                a=0.445948490915965; b=0.091576213509771
                xg=np.array([a,1-2*a,a,b,1-2*b,b])
                yg=np.array([a,a,1-2*a,b,b,1-2*b])
                a=0.111690794839005; b=0.054975871827661
                wg=np.array([a,a,a,b,b,b])
                
                elem[je].repg = repg+np.arange(xg.shape[0])
                repg+=xg.shape[0]
                elem[je].phi= np.array([(1-xg-yg)*(2*(1-xg-yg)-1),\
                                         xg*(2*xg-1), \
                                         yg*(2*yg-1),\
                                         4*(1-xg-yg)*xg,\
                                         4*xg*yg,\
                                         4*yg*(1-xg-yg)]).T
                dN_xi = np.array([4*xg+4*yg-3 , \
                                  4*xg-1,\
                                  xg*0,\
                                  4*(1-2*xg-yg),\
                                  4*yg,\
                                  -4*yg] ).T
                dN_eta = np.array([4*xg+4*yg-3,\
                                   xg*0,\
                                   4*yg-1,\
                                   -4*xg,\
                                   4*xg,\
                                   4*(1-xg-2*yg)]).T
                elem[je].pgx=elem[je].phi.dot(xn)
                elem[je].pgy=elem[je].phi.dot(yn)
            elif self.e[je][0]==16: # qua8
                elem[je].repx=self.conn[self.e[je][1:],0]
                elem[je].repy=self.conn[self.e[je][1:],1]
                xn=self.n[self.e[je][1:],0]
                yn=self.n[self.e[je][1:],1]
                # quadrature using 4 gp 
                # xg=np.sqrt(3)/3*np.array([-1,1,-1,1])
                # yg=np.sqrt(3)/3*np.array([-1,-1,1,1])
                # wg=np.ones(4)
                # quadrature using 9 gp 
                a=0.774596669241483
                xg=a*np.array([-1,1,-1,1,0,1,0,-1,0])
                yg=a*np.array([-1,-1,1,1,-1,0,1,0,0])
                wg=np.array([25,25,25,25,40,40,40,40,64])/81
                elem[je].repg = repg+np.arange(xg.shape[0])
                repg+=xg.shape[0]
                dN_xi=np.array([-0.25*(-1+yg)*(2*xg+yg),  \
                                 0.25*(-1+yg)*(yg-2*xg),  \
                                 0.25* (1+yg)*(2*xg+yg),  \
                                -0.25* (1+yg)*(yg-2*xg),  \
                                          -xg*(1-yg),     \
                                -0.5 * (1+yg)*(-1+yg),    \
                                          -xg*(1+yg),     \
                                -0.5 * (1+yg)*(1-yg)]).T
                dN_eta=np.array([-0.25*(-1+xg)*(xg+2*yg), \
                                  0.25* (1+xg)*(2*yg-xg), \
                                  0.25* (1+xg)*(xg+2*yg), \
                                 -0.25*(-1+xg)*(2*yg-xg), \
                                  0.5 * (1+xg)*(-1+xg),   \
                                           -yg*(1+xg),    \
                                 -0.5 * (1+xg)*(-1+xg),   \
                                            yg*(-1+xg) ]).T
                elem[je].phi= np.array([-0.25*(1-xg)*(1-yg)*(1+xg+yg), \
                                        -0.25*(1+xg)*(1-yg)*(1-xg+yg), \
                                        -0.25*(1+xg)*(1+yg)*(1-xg-yg), \
                                        -0.25*(1-xg)*(1+yg)*(1+xg-yg), \
                                          0.5*(1-xg)*(1+xg)*(1-yg),    \
                                          0.5*(1+xg)*(1+yg)*(1-yg),    \
                                          0.5*(1-xg)*(1+xg)*(1+yg),    \
                                          0.5*(1-xg)*(1+yg)*(1-yg)]).T
                elem[je].pgx=elem[je].phi.dot(xn)
                elem[je].pgy=elem[je].phi.dot(yn)
            elif self.e[je][0]==10: # qua9
                elem[je].repx=self.conn[self.e[je][1:],0]
                elem[je].repy=self.conn[self.e[je][1:],1]
                xn=self.n[self.e[je][1:],0]
                yn=self.n[self.e[je][1:],1]
                a=0.774596669241483
                xg=a*np.array([-1,1,-1,1,0,1,0,-1,0])
                yg=a*np.array([-1,-1,1,1,-1,0,1,0,0])
                wg=np.array([25,25,25,25,40,40,40,40,64])/81
                elem[je].repg = repg+np.arange(xg.shape[0])
                repg+=xg.shape[0]
                dN_xi=np.array([(xg-0.5)*(yg*(yg-1)*0.5),  \
                                (xg+0.5)*(yg*(yg-1)*0.5),  \
                                (xg+0.5)*(yg*(yg+1)*0.5),  \
                                (xg-0.5)*(yg*(yg+1)*0.5),  \
                                 (-2*xg)*(yg*(yg-1)*0.5),  \
                                (xg+0.5)*(1-yg**2),        \
                                 (-2*xg)*(yg*(yg+1)*0.5),  \
                                (xg-0.5)*(1-yg**2),        \
                                 (-2*xg)*(1-yg**2)]).T
                dN_eta=np.array([(xg*(xg-1)*0.5)*(yg-0.5), \
                                 (xg*(xg+1)*0.5)*(yg-0.5), \
                                 (xg*(xg+1)*0.5)*(yg+0.5), \
                                 (xg*(xg-1)*0.5)*(yg+0.5), \
                                       (1-xg**2)*(yg-0.5), \
                                 (xg*(xg+1)*0.5)*(-2*yg),  \
                                       (1-xg**2)*(yg+0.5), \
                                 (xg*(xg-1)*0.5)*(-2*yg),  \
                                       (1-xg**2)*(-2*yg)]).T
                elem[je].phi = np.array([(xg*(xg-1)*0.5)*(yg*(yg-1)*0.5),    \
                                         (xg*(xg+1)*0.5)*(yg*(yg-1)*0.5),    \
                                         (xg*(xg+1)*0.5)*(yg*(yg+1)*0.5),    \
                                         (xg*(xg-1)*0.5)*(yg*(yg+1)*0.5),    \
                                               (1-xg**2)*(yg*(yg-1)*0.5),    \
                                         (xg*(xg+1)*0.5)*(1-yg**2),          \
                                               (1-xg**2)*(yg*(yg+1)*0.5),    \
                                         (xg*(xg-1)*0.5)*(1-yg**2),          \
                                               (1-xg**2)*(1-yg**2)]).T
                elem[je].pgx=elem[je].phi.dot(xn)
                elem[je].pgy=elem[je].phi.dot(yn)
            else:
                print("Oops!  That is not a valid element type... "+str(self.e[je][0]))
            dxdr=dN_xi.dot(xn)
            dydr=dN_xi.dot(yn)
            dxds=dN_eta.dot(xn)
            dyds=dN_eta.dot(yn)
            detJ=(dxdr*dyds-dydr*dxds)
            elem[je].wdetJ=wg*abs(detJ)                        
            elem[je].dphidx=sp.sparse.diags(dyds/detJ).dot(dN_xi)+sp.sparse.diags(-dydr/detJ).dot(dN_eta)
            elem[je].dphidy=sp.sparse.diags(-dxds/detJ).dot(dN_xi)+sp.sparse.diags(dxdr/detJ).dot(dN_eta)            
            nzv+=np.prod(elem[je].dphidx.shape)
        self.npg=repg
        self.pgx=np.zeros(self.npg)
        self.pgy=np.zeros(self.npg)
        for je in range(len(self.e)):
            self.pgx[elem[je].repg]=elem[je].pgx
            self.pgy[elem[je].repg]=elem[je].pgy
        
        """ Build the FE interpolation """
        self.wdetJ=np.zeros(self.npg)
        row=np.zeros(nzv)
        col=np.zeros(nzv)
        val=np.zeros(nzv)
        valx=np.zeros(nzv)
        valy=np.zeros(nzv)
        nzv=0
        for je in range(len(self.e)):
            self.wdetJ[elem[je].repg]=elem[je].wdetJ
            [repj,repi]=np.meshgrid(elem[je].repx,elem[je].repg)
            rangephi = np.arange(np.prod(elem[je].dphidx.shape))
            row[nzv+rangephi]=repi.ravel()
            col[nzv+rangephi]=repj.ravel()
            val[nzv+rangephi]=elem[je].phi.ravel()
            valx[nzv+rangephi]=elem[je].dphidx.ravel()
            valy[nzv+rangephi]=elem[je].dphidy.ravel()
            nzv=nzv+np.prod(elem[je].dphidx.shape)
        self.phix=sp.sparse.csc_matrix((val, (row, col+0*self.ndof//2)), shape=(self.npg, self.ndof))
        self.phiy=sp.sparse.csc_matrix((val, (row, col+1*self.ndof//2)), shape=(self.npg, self.ndof))
        self.dphixdx=sp.sparse.csc_matrix((valx, (row, col+0*self.ndof//2)), shape=(self.npg, self.ndof))
        self.dphixdy=sp.sparse.csc_matrix((valy, (row, col+0*self.ndof//2)), shape=(self.npg, self.ndof))
        self.dphiydx=sp.sparse.csc_matrix((valx, (row, col+1*self.ndof//2)), shape=(self.npg, self.ndof))
        self.dphiydy=sp.sparse.csc_matrix((valy, (row, col+1*self.ndof//2)), shape=(self.npg, self.ndof))
    
    def Stiffness(self,hooke):
        """ Assemble Stiffness Operator """ 
        if not hasattr(self,'dphixdx'):
            m=self.Copy()
            m.GaussIntegration()
            wdetJ=sp.sparse.diags(m.wdetJ)
            Bxy=m.dphixdy+m.dphiydx
            K =  hooke[0,0]*m.dphixdx.T.dot(wdetJ.dot(m.dphixdx)) +   \
                 hooke[1,1]*m.dphiydy.T.dot(wdetJ.dot(m.dphiydy)) +   \
                 hooke[2,2]*Bxy.T.dot(wdetJ.dot(Bxy)) + \
                 hooke[0,1]*m.dphixdx.T.dot(wdetJ.dot(m.dphiydy)) +   \
                 hooke[0,2]*m.dphixdx.T.dot(wdetJ.dot(Bxy)) +  \
                 hooke[1,2]*m.dphiydy.T.dot(wdetJ.dot(Bxy)) +  \
                 hooke[1,0]*m.dphiydy.T.dot(wdetJ.dot(m.dphixdx)) +   \
                 hooke[2,0]*Bxy.T.dot(wdetJ.dot(m.dphixdx)) +  \
                 hooke[2,1]*Bxy.T.dot(wdetJ.dot(m.dphiydy))
        else:
            wdetJ=sp.sparse.diags(self.wdetJ)
            Bxy=self.dphixdy+self.dphiydx
            K =  hooke[0,0]*self.dphixdx.T.dot(wdetJ.dot(self.dphixdx)) +   \
                 hooke[1,1]*self.dphiydy.T.dot(wdetJ.dot(self.dphiydy)) +   \
                 hooke[2,2]*Bxy.T.dot(wdetJ.dot(Bxy)) + \
                 hooke[0,1]*self.dphixdx.T.dot(wdetJ.dot(self.dphiydy)) +   \
                 hooke[0,2]*self.dphixdx.T.dot(wdetJ.dot(Bxy)) +  \
                 hooke[1,2]*self.dphiydy.T.dot(wdetJ.dot(Bxy)) +  \
                 hooke[1,0]*self.dphiydy.T.dot(wdetJ.dot(self.dphixdx)) +   \
                 hooke[2,0]*Bxy.T.dot(wdetJ.dot(self.dphixdx)) +  \
                 hooke[2,1]*Bxy.T.dot(wdetJ.dot(self.dphiydy))
        return K

    def Tikhonov(self):
        """ Assemble Tikhonov Operator """ 
        if not hasattr(self,'dphixdx'):
            m=self.Copy()
            m.GaussIntegration()
            wdetJ=sp.sparse.diags(m.wdetJ)
            L = m.dphixdx.T.dot(wdetJ.dot(m.dphixdx)) +   \
                m.dphiydy.T.dot(wdetJ.dot(m.dphiydy)) +   \
                m.dphixdy.T.dot(wdetJ.dot(m.dphixdy)) +   \
                m.dphiydx.T.dot(wdetJ.dot(m.dphiydx))
        else:
            wdetJ=sp.sparse.diags(self.wdetJ)
            L = self.dphixdx.T.dot(wdetJ.dot(self.dphixdx)) +   \
                self.dphiydy.T.dot(wdetJ.dot(self.dphiydy)) +   \
                self.dphixdy.T.dot(wdetJ.dot(self.dphixdy)) +   \
                self.dphiydx.T.dot(wdetJ.dot(self.dphiydx))
        return L

    def Mass(self,rho):
        """ Assemble Mass Matrix """ 
        if not hasattr(self,'dphixdx'):
            m=self.Copy()
            m.GaussIntegration()
            wdetJ=sp.sparse.diags(m.wdetJ)
            M =  rho*m.phix.T.dot(wdetJ.dot(m.phix)) +   \
                 rho*m.phiy.T.dot(wdetJ.dot(m.phiy))
        else:
            wdetJ=sp.sparse.diags(self.wdetJ)
            M =  rho*self.phix.T.dot(wdetJ.dot(self.phix)) +   \
                 rho*self.phiy.T.dot(wdetJ.dot(self.phiy))
        return M
    
    def VTKMesh(self,filename='mesh'):
        nnode=self.n.shape[0]
        if self.n.shape[1]==2:
            new_node=np.append(self.n,np.zeros((nnode,1)),axis=1).ravel()
        else:
            new_node=self.n.ravel() 
        new_conn=np.array([], dtype='int')
        new_offs=np.array([], dtype='int')
        new_type=np.array([], dtype='int')
        new_num=np.arange(len(self.e))
        nelem=len(self.e)
        coffs=0
        for je in range(len(self.e)):
            if self.e[je][0]==3: # quad4
                coffs=coffs+4
                new_type=np.append(new_type,9)
            elif self.e[je][0]==2: #tri3
                coffs=coffs+3
                new_type=np.append(new_type,5)
            elif self.e[je][0]==9: #tri6
                coffs=coffs+6
                new_type=np.append(new_type,22)
            elif self.e[je][0]==16: #quad8
                coffs=coffs+8
                new_type=np.append(new_type,23)
            elif self.e[je][0]==5: #hex8
                coffs=coffs+8
                new_type=np.append(new_type,12)
            else:
                print("Oops!  That is not a valid element type...")
            new_conn=np.append(new_conn,self.e[je][1:])
            new_offs=np.append(new_offs,coffs)
        vtkfile=vtk.VTUWriter(nnode,nelem,new_node,new_conn,new_offs,new_type)
        vtkfile.addCellData('num',1,new_num)
        # Write the VTU file in the VTK dir
        dir0,filename=os.path.split(filename)
        if not os.path.isdir(os.path.join('vtk',dir0)):
            os.makedirs(os.path.join('vtk',dir0))
        vtkfile.write(os.path.join('vtk',dir0,filename))

    def VTKSolSeries(self,filename,UU):        
        for ig in range(UU.shape[1]):
            fname=filename+'_0_'+str(ig)
            self.VTKSol(fname,UU[:,ig])
        self.PVDFile(filename,'vtu',1,UU.shape[1])

    def PVDFile(self,fileName,ext,npart,nstep):
        dir0,fileName=os.path.split(fileName)
        if not os.path.isdir(os.path.join('vtk',dir0)):
            os.makedirs(os.path.join('vtk',dir0))
        vtk.PVDFile(os.path.join('vtk',dir0,fileName),ext,npart,nstep)

    def VTKSol(self,filename,U,E=[],S=[],T=[]):
        nnode=self.n.shape[0]
        new_node=np.append(self.n,np.zeros((nnode,1)),axis=1).ravel()
        # unused nodes displacement
        conn2=self.conn
        (unused,)=np.where(conn2[:,0]==-1)
        conn2[unused,0]=0
        new_u=np.append(U[self.conn],np.zeros((nnode,1)),axis=1).ravel()
        new_conn=np.array([], dtype='int')
        new_offs=np.array([], dtype='int')
        new_type=np.array([], dtype='int')
        nelem=len(self.e)       
        new_num=np.arange(nelem)
        coffs=0
        for je in range(nelem):
            if self.e[je][0]==3: # quad4
                coffs=coffs+4
                new_type=np.append(new_type,9)
            elif self.e[je][0]==2: #tri3
                coffs=coffs+3
                new_type=np.append(new_type,5)
            elif self.e[je][0]==9: #tri6
                coffs=coffs+6
                new_type=np.append(new_type,22)
            elif self.e[je][0]==16: #quad8
                coffs=coffs+8
                new_type=np.append(new_type,23)
            elif self.e[je][0]==5: #hex8
                coffs=coffs+8
                new_type=np.append(new_type,12)
            else:
                print("Oops!  That is not a valid element type...")
            new_conn=np.append(new_conn,self.e[je][1:])
            new_offs=np.append(new_offs,coffs)
        vtkfile=vtk.VTUWriter(nnode,nelem,new_node,new_conn,new_offs,new_type)
        vtkfile.addPointData('displ',3,new_u)
        vtkfile.addCellData('num',1,new_num)
        if len(T)>0:
            vtkfile.addPointData('temp',1,T[self.conn[:,0]])
        
        # Strain
        if len(E)==0:
            Ex,Ey,Exy=self.StrainAtNodes(U)
            E=np.c_[Ex,Ey,Exy/2/np.sqrt(2)]
            C=(Ex+Ey)/2
            R=np.sqrt((Ex-C)**2+Exy**2)
            EP = np.sort(np.c_[C+R,C-R],axis=1)
        new_e=np.c_[E[self.conn[:,0],0],E[self.conn[:,0],1],E[self.conn[:,0],2]].ravel()
        vtkfile.addPointData('strain',3,new_e)
        new_ep=np.c_[EP[self.conn[:,0],0],EP[self.conn[:,0],1]].ravel()
        vtkfile.addPointData('pcpal_strain',2,new_ep)

        # Stress
        if len(S)>0:      
            new_s=np.c_[S[self.conn[:,0],0],S[self.conn[:,0],1],S[self.conn[:,0],2]].ravel()
            vtkfile.addPointData('stress',3,new_s)
        
        # Write the VTU file in the VTK dir
        dir0,filename=os.path.split(filename)
        if not os.path.isdir(os.path.join('vtk',dir0)):
            os.makedirs(os.path.join('vtk',dir0))
        vtkfile.write(os.path.join('vtk',dir0,filename))


    def StrainAtGP(self,U):
        epsx=self.dphixdx.dot(U)
        epsy=self.dphiydy.dot(U)
        epsxy=0.5*self.dphixdy.dot(U)+0.5*self.dphiydx.dot(U)
        return epsx,epsy,epsxy

    def StrainAtNodes(self,UU):
        m=self.Copy()
        m.GaussIntegration()
        wdetJ=sp.sparse.diags(m.wdetJ)
        phi=m.phix[:,:m.ndof//2]
        if not hasattr(self,'Bx'):
            self.Bx=splalg.splu(phi.T.dot(wdetJ.dot(phi)).T)
            #Mi=splalg.inv(phi.T.dot(wdetJ.dot(phi)).T)
            #self.Bx=Mi.dot(phi.T.dot(wdetJ.dot(m.dphixdx)))
            #self.By=Mi.dot(phi.T.dot(wdetJ.dot(m.dphiydy)))
            #self.Bxy=Mi.dot(phi.T.dot(wdetJ.dot(m.dphixdy+m.dphixdy)))*0.5
        epsx  = self.Bx.solve(phi.T.dot(wdetJ.dot(m.dphixdx.dot(UU))))
        epsy  = self.Bx.solve(phi.T.dot(wdetJ.dot(m.dphiydy.dot(UU))))
        epsxy = self.Bx.solve(phi.T.dot(wdetJ.dot(m.dphixdy.dot(UU)+m.dphiydx.dot(UU))))*0.5
        #epsx=self.Bx.dot(UU)
        #epsy=self.By.dot(UU)
        #epsxy=self.Bxy.dot(UU)
        return epsx,epsy,epsxy

    def Elem2Node(self,edata):
        wdetJ=sp.sparse.diags(self.wdetJ)
        phi=self.phix[:,:self.ndof//2]
        M=splalg.splu(phi.T.dot(wdetJ.dot(phi)).T)
        ndata=M.solve(phi.T.dot(wdetJ.dot(edata)))
        return ndata

    def VTKIntegrationPoints(self,cam,f,g,U,filename='IntPts'):
        nnode=self.pgx.shape[0]
        nelem=nnode
        new_node=np.array([self.pgx,self.pgy,0*self.pgx]).T.ravel()
        new_conn=np.arange(nelem)
        new_offs=np.arange(nelem)+1
        new_type=2*np.ones(nelem).astype('int')
        vtkfile=vtk.VTUWriter(nnode,nelem,new_node,new_conn,new_offs,new_type)        
        ''' Reference image '''        
        u,v=cam.P(self.pgx,self.pgy)
        if hasattr(f,'tck')==0:
            f.BuildInterp()
        imref=f.Interp(u,v)
        vtkfile.addCellData('f',1,imref)
        ''' Deformed image '''
        if hasattr(g,'tck')==0:
            g.BuildInterp()
        imdef=g.Interp(u,v)
        vtkfile.addCellData('g',1,imdef)
        ''' Residual Map '''
        pgu=self.phix.dot(U)
        pgv=self.phiy.dot(U)
        pgxu=self.pgx+pgu
        pgyv=self.pgy+pgv
        u,v=cam.P(pgxu,pgyv)
        imdefu=g.Interp(u,v)
        vtkfile.addCellData('gu',1,(imdefu-imref)/f.Dynamic()*100)
        ''' Displacement field '''        
        new_u=np.array([pgu,pgv,0*pgu]).T.ravel()
        vtkfile.addPointData('disp',3,new_u)
        ''' Strain field '''
        epsxx,epsyy,epsxy=self.StrainAtGP(U)
        new_eps=np.array([epsxx,epsyy,epsxy]).T.ravel()
        vtkfile.addCellData('epsilon',3,new_eps)
        
        # Write the VTU file in the VTK dir
        dir0,filename=os.path.split(filename)
        if not os.path.isdir(os.path.join('vtk',dir0)):
            os.makedirs(os.path.join('vtk',dir0))
        vtkfile.write(os.path.join('vtk',dir0,filename))

    def VTKNodes(self,cam,f,g,U,filename='IntPts'):
        nnode=self.n.shape[0]
        nelem=nnode
        new_node=np.array([self.n[:,0],self.n[:,1],0*self.n[:,0]]).T.ravel()
        new_conn=np.arange(nelem)
        new_offs=np.arange(nelem)+1
        new_type=2*np.ones(nelem).astype('int')
        vtkfile=vtk.VTUWriter(nnode,nelem,new_node,new_conn,new_offs,new_type)        
        ''' Reference image '''        
        u,v=cam.P(self.n[:,0],self.n[:,1])
        if hasattr(f,'tck')==0:
            f.BuildInterp()
        imref=f.Interp(u,v)
        vtkfile.addCellData('f',1,imref)
        ''' Deformed image '''
        if hasattr(g,'tck')==0:
            g.BuildInterp()
        imdef=g.Interp(u,v)
        vtkfile.addCellData('g',1,imdef)
        ''' Residual Map '''
        pgu=U[self.conn[:,0]]
        pgv=U[self.conn[:,1]]
        pgxu=self.n[:,0]+pgu
        pgyv=self.n[:,1]+pgv
        u,v=cam.P(pgxu,pgyv)
        imdefu=g.Interp(u,v)
        vtkfile.addCellData('gu',1,(imdefu-imref)/f.Dynamic()*100)
        ''' Displacement field '''        
        new_u=np.array([pgu,pgv,0*pgu]).T.ravel()
        vtkfile.addPointData('disp',3,new_u)
        
        # Write the VTU file in the VTK dir
        # Write the VTU file in the VTK dir
        dir0,filename=os.path.split(filename)
        if not os.path.isdir(os.path.join('vtk',dir0)):
            os.makedirs(os.path.join('vtk',dir0))
        vtkfile.write(os.path.join('vtk',dir0,filename))
        
    def Morphing(self,U):
        self.n+=U[self.conn]
        
    def PreparePlot(self,U=None,coef=1.,n=None, **kwargs):
        """
        Prepare the matplotlib collections for a plot
        """
        edgecolor=kwargs.pop('edgecolor','k')
        facecolor=kwargs.pop('facecolor','none')
        alpha=kwargs.pop('alpha',0.8)        
        #plt.figure()
        ax=plt.gca()
        """ Plot deformed or undeformes Mesh """
        if n is None:
            n=self.n.copy()
        else:
            print('ok')
        if U is not None:
            n+=coef*U[self.conn]
        ### Find the quad elements
        lqua = [ie for ie in self.e if self.e[ie][0] in (3,16,10)]
        ### Find the tri elements
        ltri = [ie for ie in self.e if self.e[ie][0] in (2,9)]
        
        qua = np.zeros((len(lqua),4),dtype='int64')
        tri = np.zeros((len(ltri),3),dtype='int64')
        for ii,ie in enumerate(lqua):
            qua[ii,:] = self.e[ie][1:]
        for ii,ie in enumerate(ltri):
            tri[ii,:] = self.e[ie][1:]
        
        ### Join the 2 lists of vertices
        nn = n[qua].tolist()+n[tri].tolist()
        ### Create the collection
        pc = cols.PolyCollection(nn, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, **kwargs)
        ### Return the matplotlib collection and the list of vertices
        return pc,nn

    def Plot(self,U=None,coef=1,n=None, **kwargs):
        """
        Plot the mesh
        Inputs: 
            -U: displacement fields for a deformed mesh plot
            -coef: amplification coefficient
            -n: nodes coordinates
        """
        ax = plt.gca()
        pc,nn = self.PreparePlot(U,coef,n, **kwargs)
        ax.add_collection(pc)
        ax.autoscale()
        plt.axis('equal')
        plt.show()

    
    def AnimatedPlot(self,U,coef=1,n=None,timeAnim=5,color=('k','b','r','g','c')):
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
        for jj,u in enumerate(U):
            pc[jj],nn[jj] = self.PreparePlot(u[:,0],coef,n,edgecolor=color[jj])
            ax.add_collection(pc[jj])
            ax.autoscale()
            plt.axis('equal')
        
        def updateMesh(ii):
            """
            Function to update the matplotlib collections
            """
            for jj,u in enumerate(U):
                titi,nn = self.PreparePlot(u[:,ii],coef,n)
                pc[jj].set_paths(nn)
            return pc.values()
        
        line_ani = animation.FuncAnimation(fig, updateMesh, range(ntimes),
                                           blit=True,
                                           interval=timeAnim/ntimes*1000)
        return line_ani

    def PlotContourDispl(self,U=None, **kwargs):
        rep,=np.where(self.conn[:,0]>=0)
        """ Plot mesh and field contour """
        plt.figure()
        plt.tricontourf(self.n[rep,0],self.n[rep,1],U[self.conn[rep,0]],20)
        self.Plot(alpha=0.2)
        plt.axis('off')
        plt.title('Ux')
        plt.colorbar()
        plt.figure()
        plt.tricontourf(self.n[rep,0],self.n[rep,1],U[self.conn[rep,1]],20)
        self.Plot(alpha=0.2)
        plt.axis('equal')
        plt.title('Uy')
        plt.axis('off')
        plt.colorbar()
        plt.show()

    def PlotContourStrain(self,U, **kwargs):
        rep,=np.where(self.conn[:,0]>=0)
        EX,EY,EXY=self.StrainAtNodes(U)
        """ Plot mesh and field contour """
        plt.figure()
        plt.tricontourf(self.n[rep,0],self.n[rep,1],EX[self.conn[rep,0]],20)
        self.Plot(alpha=0.2)
        plt.axis('off')
        plt.title('EPS_X')
        plt.colorbar()
        plt.figure()
        plt.tricontourf(self.n[rep,0],self.n[rep,1],EY[self.conn[rep,0]],20)
        self.Plot(alpha=0.2)
        plt.axis('equal')
        plt.title('EPS_Y')
        plt.axis('off')
        plt.colorbar()
        plt.figure()
        plt.tricontourf(self.n[rep,0],self.n[rep,1],EXY[self.conn[rep,0]],20)
        self.Plot(alpha=0.2)
        plt.axis('equal')
        plt.title('EPS_XY')
        plt.axis('off')
        plt.colorbar()
        plt.show()


    
    def PlotNodeLabels(self, **kwargs):
        self.Plot(**kwargs)
        color=kwargs.get('edgecolor',"k")
        plt.plot(self.n[:,0],self.n[:,1],'.',color=color)
        for i in range(len(self.n[:,1])):
            plt.text(self.n[i,0],self.n[i,1],str(i),color=color)

    def PlotElemLabels(self, **kwargs):
        self.Plot( **kwargs)
        color=kwargs.get('edgecolor',"k")
        for ie in range(len(self.e)):
            ce=np.mean(self.n[self.e[ie][1:],:],axis=0)
            plt.text(ce[0],ce[1],str(ie),horizontalalignment='center',verticalalignment='center',color=color)
            
    def FindDOFinBox(self,box):
        dofs=np.zeros((0,2),dtype='int')
        for jn in range(len(self.n)):
            if isInBox(self.n[jn,0],self.n[jn,1],box):
                dofs=np.vstack((dofs,self.conn[jn]))
        return dofs
    
    def RemoveElemsOutsideRoi(self,cam,box):
        newe=dict()
        ne=0
        for ie in range(len(self.e)):
            u,v=cam.P(np.mean(self.n[self.e[ie][1:],0]),np.mean(self.n[self.e[ie][1:],1]))
            if isInBox(u,v,box):
                newe[ne]=self.e[ie]
                ne+=1
        self.e=newe


#%%
def StructuredMeshQ4(roi,dx):
    #roi=np.array([[xmin,ymin],[xmax,ymax]])
    #dx=[dx,dy]: average element size (can be scalar)
    droi=np.diff(roi,axis=0).astype('int')
    NE=np.round(droi/dx)[0].astype('int')
    [X,Y]=np.meshgrid(np.linspace(roi[0,0],roi[1,0],NE[0]+1),np.linspace(roi[0,1],roi[1,1],NE[1]+1))
    n=np.array([X.T.ravel(),Y.T.ravel()]).T
    e=dict()
    for ix in range(NE[0]):
        for iy in range(NE[1]):
            p1=ix*(NE[1]+1)+iy
            p4=ix*(NE[1]+1)+iy+1
            p2=ix*(NE[1]+1)+iy+NE[1]+1
            p3=ix*(NE[1]+1)+iy+NE[1]+2
            e[ix*NE[1]+iy]=np.array([3,p1,p2,p3,p4])
    m=Mesh(e,n)
    return m

def StructuredMeshT3(roi,dx):
    #roi=np.array([[xmin,ymin],[xmax,ymax]])
    #dx=[dx,dy]: average element size (can be scalar)
    droi=np.diff(roi,axis=0).astype('int')
    NE=np.round(droi/dx)[0].astype('int')
    [X,Y]=np.meshgrid(np.linspace(roi[0,0],roi[1,0],NE[0]+1),np.linspace(roi[0,1],roi[1,1],NE[1]+1))
    n=np.array([X.T.ravel(),Y.T.ravel()]).T
    e=dict()
    for ix in range(NE[0]):
        for iy in range(NE[1]):
            p1=ix*(NE[1]+1)+iy
            p4=ix*(NE[1]+1)+iy+1
            p2=ix*(NE[1]+1)+iy+NE[1]+1
            p3=ix*(NE[1]+1)+iy+NE[1]+2
            e[2*(ix*NE[1]+iy)]=np.array([2,p1,p2,p4])
            e[2*(ix*NE[1]+iy)+1]=np.array([2,p4,p3,p2])            
    m=Mesh(e,n)
    return m 

def MeshFromROI(roi,dx,f):
    # roi=np.array([[x0,y0],[x1,y1]]) 2 points definig a box
    # dx=[dx,dy]: average element size (can be scalar)
    droi=np.diff(roi,axis=0)[0]
    xmin=np.min(roi[:,0])
    ymin=f.pix.shape[0]-np.max(roi[:,1])
    roi=np.array([[0,0],droi])
    m=StructuredMeshQ4(roi,dx)
    p=np.array([1.0,xmin,ymin-f.pix.shape[0],0.0])
    cam=Camera(p)
    return m,cam


#%%    
class Image:
    def Load(self):
        """ Load image data """
        if self.fname.split('.')[-1]=='npy':
            self.pix=np.load(self.fname)
        else:
            self.pix = np.asarray(image.open(self.fname)).astype(float)
            #self.pix=image.imread(self.fname).astype(float)
        return self
    def Copy(self):
        newimg=Image('Copy')
        newimg.pix=self.pix.copy()
        return newimg
    def Save(self,fname):
        PILimg = image.fromarray(self.pix.astype('uint8'))
        PILimg.save(fname)
        #image.imsave(fname,self.pix.astype('uint8'),vmin=0,vmax=255,format='tif')
    def __init__(self,fname):
        """ Contructor """  
        self.fname=fname
    def BuildInterp(self):
        """ build bivariate Spline interp """
        x=np.arange(0,self.pix.shape[0])
        y=np.arange(0,self.pix.shape[1])
        self.tck = spi.RectBivariateSpline(x,y,self.pix)
    def Interp(self,x,y):
        """ evaluate interpolator"""
        return self.tck.ev(x,y)
    def InterpGrad(self,x,y):
        return self.tck.ev(x,y,1,0),self.tck.ev(x,y,0,1)
    def InterpHess(self,x,y):
        return self.tck.ev(x,y,2,0),self.tck.ev(x,y,0,2),self.tck.ev(x,y,1,1)    
    def Show(self):
        plt.imshow(self.pix, cmap="gray", interpolation='none',origin='upper') 
        #plt.axis('off')
        #plt.colorbar()
    def Dynamic(self):
        """ Compute image dynamic """
        g=self.pix.ravel()
        return max(g)-min(g)
    def PlotHistogram(self):
        """ Plot Histogram of graylevels """
        plt.hist(self.pix.ravel(), bins=125, range=(0.0, 255), fc='k', ec='k')
        plt.show()
    def SubSample(self,n):
        scale=2**n
        sizeim1=np.array([self.pix.shape[0]//scale, self.pix.shape[1]//scale])
        nn=scale*sizeim1
        im0=np.mean(self.pix[0:nn[0],0:nn[1]].T.reshape(np.prod(nn)//scale,scale),axis=1)
        nn[0]=nn[0]//scale
        im0=np.mean(im0.reshape(nn[1],nn[0]).T.reshape(np.prod(nn)//scale,scale),axis=1)
        nn[1]=nn[1]//scale
        self.pix=im0.reshape(nn)


def GetPixelsQua(xn,yn,xpix,ypix):
    """Finds the pixels that belong to a quadrilateral element and """
    """inverse the mapping to know their corresponding position in """
    """the parent element."""
    wg=IsPointInElem2d(xn,yn,xpix,ypix)
    ind=np.where(wg)
    xpix=xpix[ind]
    ypix=ypix[ind]
    xg=0*xpix
    yg=0*ypix
    res=1
    for k in range(7):
        N=np.array([0.25*(1-xg)*(1-yg),0.25*(1+xg)*(1-yg),0.25*(1+xg)*(1+yg),0.25*(1-xg)*(1+yg)]).T
        N_r=np.array([-0.25*(1-yg),0.25*(1-yg),0.25*(1+yg),-0.25*(1+yg)]).T
        N_s=np.array([-0.25*(1-xg),-0.25*(1+xg),0.25*(1+xg),0.25*(1-xg)]).T
        dxdr=np.dot(N_r,xn)
        dydr=np.dot(N_r,yn)
        dxds=np.dot(N_s,xn)
        dyds=np.dot(N_s,yn)
        detJ=(dxdr*dyds-dydr*dxds)
        invJ=np.array([dyds/detJ,-dxds/detJ,-dydr/detJ,dxdr/detJ]).T
        xp=np.dot(N,xn)
        yp=np.dot(N,yn)
        dxg=invJ[:,0]*(xpix-xp)+invJ[:,1]*(ypix-yp)
        dyg=invJ[:,2]*(xpix-xp)+invJ[:,3]*(ypix-yp)    
        res=np.dot(dxg,dxg)+np.dot(dyg,dyg)
        xg=xg+dxg
        yg=yg+dyg
        if res<1.e-6:
            break
    return [xg,yg,xpix,ypix]

def GetPixelsTri(xn,yn,xpix,ypix):
    """Finds the pixels that belong to a triangle element and      """
    """inverse the mapping to know their corresponding position in """
    """the parent element."""
    wg=IsPointInElem2d(xn,yn,xpix,ypix)
    ind=np.where(wg)
    xpix=xpix[ind]
    ypix=ypix[ind]
    xg=0*xpix
    yg=0*ypix
    res=1
    for k in range(7):
        N=np.array([1-xg-yg,xg,yg]).T
        N_r=np.array([-np.ones(xg.shape),np.ones(xg.shape),np.zeros(xg.shape)]).T
        N_s=np.array([-np.ones(xg.shape),np.zeros(xg.shape),np.ones(xg.shape)]).T
        dxdr=np.dot(N_r,xn)
        dydr=np.dot(N_r,yn)
        dxds=np.dot(N_s,xn)
        dyds=np.dot(N_s,yn)
        detJ=(dxdr*dyds-dydr*dxds)
        invJ=np.array([dyds/detJ,-dxds/detJ,-dydr/detJ,dxdr/detJ]).T
        xp=np.dot(N,xn)
        yp=np.dot(N,yn)
        dxg=invJ[:,0]*(xpix-xp)+invJ[:,1]*(ypix-yp)
        dyg=invJ[:,2]*(xpix-xp)+invJ[:,3]*(ypix-yp)    
        res=np.dot(dxg,dxg)+np.dot(dyg,dyg)
        xg=xg+dxg
        yg=yg+dyg
        if res<1.e-6:
            break
    return [xg,yg,xpix,ypix]


def SubQuaIso(nx,ny):
    px=1./nx
    xi=np.linspace(px-1,1-px,int(nx))
    py=1./ny
    yi=np.linspace(py-1,1-py,int(ny))
    xg,yg=np.meshgrid(xi,yi)
    wg=4./(nx*ny)
    return xg.ravel(),yg.ravel(),wg


def SubTriIso(nx,ny):
    # M1M2 is divided in nx and M1M3 in ny, the meshing being heterogeneous, we 
    # end up with trapezes on the side of hypothenuse, the rest are rectangles    
    px=1/nx
    py=1/ny
    if nx>ny:
        xg=np.zeros(int(np.sum(np.floor(ny*(1-np.arange(1,nx+1)/nx)))+nx))
        yg=xg.copy()
        wg=xg.copy()
        j=1
        for i in range(1,nx+1):
            niy=int(ny*(1-i/nx)) #number of full rectangles in the vertical dir
            v=np.array([[(i-1)*px,niy*py],[(i-1)*px,1-(i-1)*px],[i*px,niy*py],[i*px,1-i*px]])
            neww = ( px*(v[3,1]-v[0,1]) + px*(v[1,1]-v[3,1])/2 )
            newx = ( (v[3,1]-v[0,1])*(v[2,0]+v[0,0])/2 + (v[1,1]-v[3,1])/2*(v[0,0]+px/3) ) * px/neww
            newy = ( (v[3,1]-v[0,1])*(v[0,1]+v[3,1])/2 + (v[1,1]-v[3,1])/2*(v[3,1]+(v[1,1]-v[3,1])/3) ) * px/neww            
            xg[(j-1):j+niy]=np.append((px/2+(i-1)*px)*np.ones(niy),newx)
            yg[(j-1):j+niy]=np.append(py/2+np.arange(niy)*py,newy)
            wg[(j-1):j+niy]=np.append(px*py*np.ones(niy),neww)
            j=j+niy+1
    else:
        xg=np.zeros(int(np.sum(np.floor(nx*(1-np.arange(1,ny+1)/ny)))+ny))
        yg=xg.copy()
        wg=xg.copy()
        j=1
        for i in range(1,ny+1):
            nix=int(nx*(1-i/ny)) #number of full rectangles in the horizontal dir
            v=np.array([[nix*px,(i-1)*py],[nix*px,i*py],[1-(i-1)*py,(i-1)*py],[1-i*py,i*py]])
            neww = ( py*(v[3,0]-v[0,0]) + py*(v[2,0]-v[3,0])/2 )
            newx = ( (v[3,0]-v[0,0])*(v[3,0]+v[0,0])/2 + (v[2,0]-v[3,0])/2*(v[3,0]+(v[2,0]-v[3,0])/3) ) * py/neww
            newy = ( (v[3,0]-v[0,0])*(v[1,1]+v[0,1])/2 + (v[2,0]-v[3,0])/2*(v[0,1]+py/3) ) * py/neww
            xg[(j-1):j+nix]=np.append(px/2+np.arange(nix)*px,newx)
            yg[(j-1):j+nix]=np.append((py/2+(i-1)*py)*np.ones(nix),newy)
            wg[(j-1):j+nix]=np.append(px*py*np.ones(nix),neww)
            j=j+nix+1
    return xg,yg,wg



#%%
class Camera():
    def __init__(self,p):
        self.set_p(p)
    def set_p(self,p):
        self.f=p[0]
        self.tx=p[1]
        self.ty=p[2]
        self.rz=p[3]
    def get_p(self):
        return np.array([self.f,self.tx,self.ty,self.rz])       
    def SubSampleCopy(self,nscale):
        p=self.get_p()
        p[0]/=(2**nscale)
        return Camera(p)
    def P(self,X,Y):
        u=-self.f*(-np.sin(self.rz)*X+np.cos(self.rz)*Y+self.ty)
        v=self.f*(np.cos(self.rz)*X+np.sin(self.rz)*Y+self.tx)
        return u,v
    def Pinv(self,u,v):
        X=-np.sin(self.rz)*(-u/self.f-self.ty)+np.cos(self.rz)*(v/self.f-self.tx)
        Y= np.cos(self.rz)*(-u/self.f-self.ty)+np.sin(self.rz)*(v/self.f-self.tx)
        return X,Y
    def dPdX(self,X,Y):
        dudx = self.f*np.sin(self.rz)*np.ones(X.shape[0])
        dudy =-self.f*np.cos(self.rz)*np.ones(X.shape[0])
        dvdx = self.f*np.cos(self.rz)*np.ones(X.shape[0])
        dvdy = self.f*np.sin(self.rz)*np.ones(X.shape[0])
        return dudx,dudy,dvdx,dvdy
    def dPdp(self,X,Y):
        dudf = -1*(-np.sin(self.rz)*X+np.cos(self.rz)*Y+self.ty)
        dudtx = 0*X
        dudty = 0*X-self.f
        dudrz = self.f*(np.cos(self.rz)*X+np.sin(self.rz)*Y)
        dvdf = np.cos(self.rz)*X+np.sin(self.rz)*Y+self.tx
        dvdtx = 0*X+self.f
        dvdty = 0*X
        dvdrz = self.f*(-np.sin(self.rz)*X+np.cos(self.rz)*Y)
        return np.c_[dudf,dudtx,dudty,dudrz],np.c_[dvdf,dvdtx,dvdty,dvdrz]  
    def d2Pdp2(self,X,Y):
        d2udf2 = 0*X
        d2udtx2 = 0*X
        d2udty2 = 0*X
        d2udrz2 = self.f*(-np.sin(self.rz)*X+np.cos(self.rz)*Y)
        d2udftx = 0*X
        d2udfty = 0*X-1
        d2udfrz = np.cos(self.rz)*X+np.sin(self.rz)*Y
        d2udtxty = 0*X
        d2udtxrz = 0*X
        d2udtyrz = 0*X
        d2vdf2 = 0*X
        d2vdtx2 = 0*X
        d2vdty2 = 0*X
        d2vdrz2 = -self.f*(np.cos(self.rz)*X+np.sin(self.rz)*Y)
        d2vdftx = 0*X+1
        d2vdfty = 0*X
        d2vdfrz = -np.sin(self.rz)*X+np.cos(self.rz)*Y
        d2vdtxty = 0*X
        d2vdtxrz = 0*X
        d2vdtyrz = 0*X        
        d2udp2 = np.c_[d2udf2,d2udtx2,d2udty2,d2udrz2,d2udftx,d2udfty,d2udfrz,d2udtxty,d2udtxrz,d2udtyrz]
        d2vdp2 = np.c_[d2vdf2,d2vdtx2,d2vdty2,d2vdrz2,d2vdftx,d2vdfty,d2vdfrz,d2vdtxty,d2vdtxrz,d2vdtyrz]
        return d2udp2,d2vdp2
    def ImageFiles(self,fname,imnums):
        self.fname=fname
        self.imnums=imnums

def SelectImagePoints(f,n=-1):
    plt.figure()
    f.Show()
    plt.title('Select '+str(n)+' points... and press enter')
    pts1=np.array(plt.ginput(n))
    plt.close()
    return pts1

def SelectImageLine(f):
    plt.figure()
    f.Show()
    plt.title('Select n points of a straight line... and press enter')
    pts1=np.array(plt.ginput(-1))
    plt.close()
    b=pts1.T.dot(np.ones_like(pts1[:,0]))
    A=pts1.T.dot(pts1)
    res=np.linalg.solve(A,b)
    ui=np.arange(0,f.pix.shape[0])
    vi=np.arange(0,f.pix.shape[1])
    [Yi,Xi]=np.meshgrid(vi,ui)
    lvlset=(Xi*res[1]+Yi*res[0]-1)/np.linalg.norm(res)
    #f.Show()
    #a,b=np.where(abs(lvlset)<1)
    #plt.plot(b,a,'y.')
    #plt.plot(pts1[:,0],pts1[:,1],'+w')
    return abs(lvlset)

def SelectImageCircle(f):
    plt.figure()
    f.Show()
    plt.title('Select n points of a circle... and press enter')
    pts1=np.array(plt.ginput(-1))
    plt.close()
    meanu=np.mean(pts1,axis=0)
    pts=pts1-meanu
    pts2=pts**2
    A=pts.T.dot(pts)    
    b=0.5*np.sum(pts.T.dot(pts2),axis=1)
    cpos=np.linalg.solve(A,b)
    R=np.sqrt(np.linalg.norm(cpos)**2+np.sum(pts2)/pts.shape[0])
    cpos+=meanu
    #x=np.arange(0,1000)/500*np.pi
    #f.Show()
    #plt.plot(cpos[0]+R*np.cos(x),cpos[1]+R*np.sin(x))
    ui=np.arange(0,f.pix.shape[0])
    vi=np.arange(0,f.pix.shape[1])
    [Yi,Xi]=np.meshgrid(vi,ui)
    lvlset=abs(np.sqrt((Xi-cpos[1])**2+(Yi-cpos[0])**2)-R)
    #zer=abs(lvlset)<100
    plt.figure
    plt.imshow(lvlset)
    return lvlset#,R

def SelectMeshPoints(m,n=-1):
    plt.figure()
    m.Plot()
    plt.title('Select '+str(n)+' points... and press enter')
    pts1=np.array(plt.ginput(n))
    plt.close()
    return pts1

def SelectMeshNodes(m,n=-1):
    plt.figure()
    m.Plot()
    plt.title('Select '+str(n)+' points... and press enter')
    pts1=np.array(plt.ginput(n))
    plt.close()
    dx=np.kron(np.ones(pts1.shape[0]),m.n[:,[0]]) - np.kron(np.ones((m.n.shape[0],1)),pts1[:,0])
    dy=np.kron(np.ones(pts1.shape[0]),m.n[:,[1]]) - np.kron(np.ones((m.n.shape[0],1)),pts1[:,1])
    nset=np.argmin(np.sqrt(dx**2+dy**2),axis=0)
    m.Plot()
    plt.plot(m.n[nset,0],m.n[nset,1],'ro')    
    return nset


def SelectMeshLine(m):
    plt.figure()
    m.Plot()
    plt.title('Select 2 points of a line... and press enter')
    pts1=np.array(plt.ginput(2))
    plt.close()
    n1=np.argmin(np.linalg.norm(m.n-pts1[0,:],axis=1))
    n2=np.argmin(np.linalg.norm(m.n-pts1[1,:],axis=1))
    v=np.diff(m.n[[n1,n2]],axis=0)[0]
    nv=np.linalg.norm(v)
    v=v/nv
    n=np.array([v[1],-v[0]])
    c=n.dot(m.n[n1,:])
    rep,=np.where(abs(m.n.dot(n)-c)<1e-8)
    c1=v.dot(m.n[n1,:])
    c2=v.dot(m.n[n2,:])
    nrep=m.n[rep,:]
    rep2,=np.where(((nrep.dot(v)-c1)*(nrep.dot(v)-c2))<nv*1e-3)
    nset=rep[rep2]
    m.Plot()
    plt.plot(m.n[nset,0],m.n[nset,1],'ro')    
    return nset

def SelectMeshCircle(m):
    plt.figure()
    m.Plot()
    plt.title('Select 3 points on a circle... and press enter')
    pts1=np.array(plt.ginput(3))
    plt.close()
    n1=np.argmin(np.linalg.norm(m.n-pts1[0,:],axis=1))
    n2=np.argmin(np.linalg.norm(m.n-pts1[1,:],axis=1))
    n3=np.argmin(np.linalg.norm(m.n-pts1[2,:],axis=1))
    pts1=m.n[[n1,n2,n3],:]
    meanu=np.mean(pts1,axis=0)
    pts=pts1-meanu
    pts2=pts**2
    A=pts.T.dot(pts)
    b=0.5*np.sum(pts.T.dot(pts2),axis=1)
    cpos=np.linalg.solve(A,b)
    R=np.sqrt(np.linalg.norm(cpos)**2+np.sum(pts2)/pts.shape[0])
    cpos+=meanu
    nset,=np.where(np.sqrt(abs((m.n[:,0]-cpos[0])**2+(m.n[:,1]-cpos[1])**2-R**2))<(R*1e-2))
    #m.Plot()
    #plt.plot(m.n[nset,0],m.n[nset,1],'ro')
    return nset#,R

def MeshCalibration(f,m,features,cam=0):
    ''' Calibration of a front parallel setting 2D-DIC '''
    # features = [Number of Circles,Number of Lines]
    if cam==0:
        cam=MeshCalibrationInit(f,m)
    # Circles
    lvl=np.zeros_like(f.pix)+1e10
    setall=np.array([],dtype='int64')
    for i in range(features[0]):
        lvl=np.minimum(lvl,SelectImageCircle(f))
        setall=np.append(setall,SelectMeshCircle(m))
    # Lines
    for i in range(features[1]):
        lvl=np.minimum(lvl,SelectImageLine(f))
        setall=np.append(setall,SelectMeshLine(m))
    l=Image('nothing')
    l.pix=lvl
    l.BuildInterp()
    xp=m.n[setall,0]
    yp=m.n[setall,1]
    p=cam.get_p()
    C=np.diag(p)
    for i in range(40):
        up,vp=cam.P(xp,yp)
        lp=l.Interp(up,vp)
        dPudp,dPvdp=cam.dPdp(xp,yp)
        ldxr,ldyr=l.InterpGrad(up,vp)
        dPdl=sp.sparse.diags(ldxr).dot(dPudp)+sp.sparse.diags(ldyr).dot(dPvdp)
        M=C.T.dot(dPdl.T.dot(dPdl.dot(C)))
        b=C.T.dot(dPdl.T.dot(lp))
        dp=C.dot(np.linalg.solve(M,-b))
        p+=0.5*dp
        cam.set_p(p)        
        err=np.linalg.norm(dp)/np.linalg.norm(p)
        print("Iter # %2d | disc/dyn=%2.2f %% | dU/U=%1.2e" % (i+1,np.mean(lp)/max(l.pix.ravel())*100,err))
        if err<1e-5:
            break
    return cam


#%%
class DICEngine():
    def __init__(self):
        self.f=[]
        self.wphiJdf=[]
        self.dyn=[]
        self.mean0=[]
        self.std0=[]
        
    def ComputeLHS(self,f,m,cam):
        """ Compute the FE-DIC matrix and the FE operators """
        if hasattr(f,'tck')==0:
            f.BuildInterp()
        pgu,pgv=cam.P(m.pgx,m.pgy)
        self.f=f.Interp(pgu,pgv)
        fdxr,fdyr=f.InterpGrad(pgu,pgv)
        Jxx,Jxy,Jyx,Jyy=cam.dPdX(m.pgx,m.pgy)
        phiJdf=sp.sparse.diags(fdxr*Jxx+fdyr*Jyx).dot(m.phix)+sp.sparse.diags(fdxr*Jxy+fdyr*Jyy).dot(m.phiy)
        self.wphiJdf=sp.sparse.diags(m.wdetJ).dot(phiJdf)
        self.dyn=np.max(self.f)-np.min(self.f)
        self.mean0=np.mean(self.f)
        self.std0=np.std(self.f)
        self.f-=self.mean0
        return phiJdf.T.dot(self.wphiJdf)
        
    def ComputeLHS_elemBrightness(self,f,m,cam):
        """ Compute the FE-DIC matrix and the FE operators (with Grad F)"""
        if hasattr(f,'tck')==0:
            f.BuildInterp()
        pgu,pgv=cam.P(m.pgx,m.pgy)
        self.f=f.Interp(pgu,pgv)
        fdxr,fdyr=f.InterpGrad(pgu,pgv)
        Jxx,Jxy,Jyx,Jyy=cam.dPdX(m.pgx,m.pgy)
        phiJdf=sp.sparse.diags(fdxr*Jxx+fdyr*Jyx).dot(m.phix)+sp.sparse.diags(fdxr*Jxy+fdyr*Jyy).dot(m.phiy)
        self.wphiJdf=sp.sparse.diags(m.wdetJ).dot(phiJdf)
        self.dyn=np.max(self.f)-np.min(self.f)
        ff=sp.sparse.diags(self.f).dot(m.Me)
        mean0=np.asarray(np.mean(ff, axis=0))[0]
        self.std0=np.asarray(np.sqrt(np.mean(ff.power(2), axis=0)-mean0**2))[0]      
        self.f-=m.Me.dot(mean0.T)
        return phiJdf.T.dot(self.wphiJdf)
        
    def ComputeLHS2(self,f,g,m,cam,U):
        """ Compute the FE-DIC matrix and the FE operators (with Grad G)"""
        if hasattr(f,'tck')==0:
            f.BuildInterp()
        if hasattr(g,'tck')==0:
            g.BuildInterp()
        pgu,pgv=cam.P(m.pgx,m.pgy)
        self.f=f.Interp(pgu,pgv)
        pgu,pgv=cam.P(m.pgx+m.phix.dot(U),m.pgy+m.phiy.dot(U))
        fdxr,fdyr=g.InterpGrad(pgu,pgv)
        Jxx,Jxy,Jyx,Jyy=cam.dPdX(m.pgx,m.pgy)
        phiJdf=sp.sparse.diags(fdxr*Jxx+fdyr*Jyx).dot(m.phix)+sp.sparse.diags(fdxr*Jxy+fdyr*Jyy).dot(m.phiy)
        self.wphiJdf=sp.sparse.diags(m.wdetJ).dot(phiJdf)
        self.dyn=np.max(self.f)-np.min(self.f)
        self.mean0=np.mean(self.f)
        self.std0=np.std(self.f)
        self.f-=self.mean0
        return phiJdf.T.dot(self.wphiJdf)
        
    def ComputeRHS(self,g,m,cam,U=[]):
        """ Compute the FE-DIC right hand side operator
        from a given displacement field U (with Grad F)
        gives in return B and the std of the residual"""
        if hasattr(g,'tck')==0:
            g.BuildInterp()
        if len(U)!=m.ndof:
            U=np.zeros(m.ndof)
        #pgxu=m.pgx+m.phix.dot(U)
        #pgyv=m.pgy+m.phiy.dot(U)
        #u,v=cam.P(pgxu,pgyv)
        u,v=cam.P(m.pgx+m.phix.dot(U),m.pgy+m.phiy.dot(U))
        res=g.Interp(u,v)
        res-=np.mean(res)
        std1 =np.std(res)
        res=self.f-self.std0/std1*res
        B=self.wphiJdf.T.dot(res)
        return B,res

    def ComputeRHS2(self,g,m,cam,U=[]):
        """ Compute the FE-DIC right hand side operator
        from a given displacement field U (with Grad G)
        gives in return B and the std of the residual"""
        if hasattr(g,'tck')==0:
            g.BuildInterp()
        if len(U)!=m.ndof:
            U=np.zeros(m.ndof)
        u,v=cam.P(m.pgx+m.phix.dot(U),m.pgy+m.phiy.dot(U))
        res=g.Interp(u,v)
        res-=np.mean(res)
        std1 =np.std(res)
        res=self.f-self.std0/std1*res
        fdxr,fdyr=g.InterpGrad(u,v)
        Jxx,Jxy,Jyx,Jyy=cam.dPdX(m.pgx,m.pgy)
        wphiJdf=sp.sparse.diags(m.wdetJ*(fdxr*Jxx+fdyr*Jyx)).dot(m.phix)+sp.sparse.diags(m.wdetJ*(fdxr*Jxy+fdyr*Jyy)).dot(m.phiy)
        B=wphiJdf.T.dot(res)
        return B,np.std(res)

    def ComputeRES(self,g,m,cam,U=[]):
        """ Compute the FE-DIC residual
        from a given displacement field U
        gives the residual on each integration point"""
        if hasattr(g,'tck')==0:
            g.BuildInterp()
        if len(U)!=m.ndof:
            U=np.zeros(m.ndof)
        pgxu=m.pgx+m.phix.dot(U)
        pgyv=m.pgy+m.phiy.dot(U)
        u,v=cam.P(pgxu,pgyv)
        res=g.Interp(u,v)
        res-=np.mean(res)
        std1 =np.std(res)
        res=self.f-self.std0/std1*res
        return res

    def ComputeRHS_elemBrightness(self,g,m,cam,U=[]):
        """ Compute the FE-DIC right hand side operator"""
        """ from a given displacement field U """
        """ gives in return B and the std of the residual"""
        if hasattr(g,'tck')==0:
            g.BuildInterp()
        if len(U)!=m.ndof:
            U=np.zeros(m.ndof)
        pgxu=m.pgx+m.phix.dot(U)
        pgyv=m.pgy+m.phiy.dot(U)
        u,v=cam.P(pgxu,pgyv)
        res=g.Interp(u,v)
        ff=sp.sparse.diags(res).dot(m.Me)
        mean0=np.asarray(np.mean(ff, axis=0))[0]
        std0=np.asarray(np.sqrt(np.mean(ff.power(2), axis=0)-mean0**2))[0]
        res-=m.Me.dot(mean0)
        res=self.f-sp.sparse.diags(m.Me.dot(self.std0/std0)).dot(res)
        B=self.wphiJdf.T.dot(res)
        return B,np.std(res)
        
    def Correlate(self,f,g,m,cam,U=[],reg=0):
        if hasattr(m,'conn')==0:
            m.Connectivity()
        if len(m.pgx)==0:
            m.DICIntegration(cam)
        if len(U)==0:
            U=MultiscaleInit(m,f,g,cam,3)
        H=self.ComputeLHS(f,m,cam)
        if reg:
            L=m.Tikhonov()
            l0=(np.max(m.n[:,0])-np.min(m.n[:,0]))/10
            V=np.zeros(m.ndof)
            V[m.conn[:,0]]=np.cos(m.n[:,1]/l0*2*np.pi)
            H0=V.dot(H.dot(V))
            L0=V.dot(L.dot(V))
            l=H0/L0
            H_LU=splalg.splu(H+l*L)
        else:
            H_LU=splalg.splu(H)
            l=0
            L=0*H
        for ik in range(0,30):
            [b,res]=self.ComputeRHS(g,m,cam,U)
            dU=H_LU.solve(b-l*L.dot(U))
            U+=dU
            err=np.linalg.norm(dU)/np.linalg.norm(U)
            print("Iter # %2d | disc/dyn=%2.2f %% | dU/U=%1.2e" % (ik+1,np.std(res)/self.dyn*100,err))
            if err<1e-3:
                break
        return U

#%% 
def PlotMeshImage(f,m,cam,U=None):
    """ Plot Mesh and ROI over and an image """
    n=m.n.copy()
    if U is not None:
        n+=U[m.conn]
    plt.figure()
    f.Show()
    u,v=cam.P(n[:,0],n[:,1])
    m.Plot(n=np.c_[v,u], edgecolor='y', alpha=0.6)
    #plt.xlim([0,f.pix.shape[1]])
    #plt.ylim([f.pix.shape[0],0])
    plt.axis('on')


def MultiscaleInit(m,imf,img,cam,nscale,l0=None):
    if l0 is None:
        n1 = np.array([m.n[m.e[i][1],:] for i in range(len(m.e))])
        n2 = np.array([m.n[m.e[i][2],:] for i in range(len(m.e))])
        l0 = 4*min(np.linalg.norm(n1-n2,axis=1))
        print(l0)
    used_nodes=m.conn[:,0]>0
    L=m.Tikhonov()
    U=np.zeros(m.ndof)
    for js in range(nscale):
        iscale=nscale-js
        print("SCALE %2d" % (iscale))
        f=imf.Copy()
        f.SubSample(iscale)
        g=img.Copy()
        g.SubSample(iscale)
        cam2=cam.SubSampleCopy(iscale)
        m2=m.Copy()
        #px.PlotMeshImage(f,m2,cam2)
        m2.DICIntegration(cam2)
        dic2=DICEngine()
        H=dic2.ComputeLHS(f,m2,cam2)
        V=np.zeros(m.ndof)
        V[m.conn[used_nodes,0]]=np.cos(m.n[used_nodes,1]/(l0*2**iscale)*2*np.pi)            
        H0=V.dot(H.dot(V))
        L0=V.dot(L.dot(V))
        l=H0/L0
        H_LU=splalg.splu(H+l*L)
        for ik in range(0,100):
            [b,res]=dic2.ComputeRHS(g,m2,cam2,U)
            dU=H_LU.solve(b-l*L.dot(U))
            U+=dU
            err=np.linalg.norm(dU)/np.linalg.norm(U)
            print("Iter # %2d | disc/dyn=%2.2f %% | dU/U=%1.2e" % (ik+1,np.std(res)/dic2.dyn*100,err))
            if err<1e-5:
                break
    return U


def MultiscaleInitTime(m,imf,imfile,imnums,cam,scales,UU=0):    
    L=m.Tikhonov()
    conv=np.ones(len(imnums),dtype='int')
    if type(UU)!=np.ndarray:
        UU=np.zeros((m.ndof,len(imnums)))
    for js in range(len(scales)):  # multiscale loop
        #iscale=nscale-js
        iscale=scales[js]        
        f=imf.Copy()
        f.SubSample(iscale)
        cam2=cam.SubSampleCopy(iscale)
        m2=m.Copy()
        m2.DICIntegration(cam2)  # to reduce the nb of integration points
        #m2.InterpolationWithGrad(cam2)  # to reduce the nb of integration points
        dic2=DICEngine()
        PlotMeshImage(f,m2,cam2)
        H=dic2.ComputeLHS(f,m2,cam2)
        # phi=m2.phix[:,:m2.ndof//2]
        # MM=phi.T.dot(phi)
        # MMLU=splalg.splu(MM)
        
        l=100*10**2
        #l0=100*2**(iscale-1)
        #V=np.zeros(m.ndof)
        #V[m.conn[:,0]]=np.cos(m.n[:,1]/l0*2*np.pi)
        #m.VTKSol(V,'PlaneWave')
        #H0=V.dot(dic2.H.dot(V))
        #L0=V.dot(L.dot(V))
        #l=H0/L0
        #print(l)
        m2.PVDFile('MSinit/MS','vtu',1,len(imnums)-1)
        H_LU=splalg.splu(H+l*L)
        m2.VTKSol('MSinit/MS_0_0',UU[:,0])
        for ig in range(1,len(imnums)):    # time loop
            print("### IMAGE %2d ###" % (ig+1))
            imdef=imfile % imnums[ig]
            g=Image(imdef).Load()
            g.SubSample(iscale)
            for ik in range(0,30):          # Gauss Netwon loop
                [b,res]=dic2.ComputeRHS(g,m2,cam2,UU[:,ig])
                #if js>0 and ig>1 and ik==0:
                if ig>1 and ik==0:
                    [b2,res2]=dic2.ComputeRHS(g,m2,cam2,UU[:,ig-1])
                    if res>res2:
                        UU[:,ig]=UU[:,ig-1]
                        b=b2
                        res=res2
                dU=H_LU.solve(b-l*L.dot(UU[:,ig]))
                # newu= m2.dphidx.dot(UU[m2.conn[:,0],ig])*m2.phix.dot(dU) + m2.dphidy.dot(UU[m2.conn[:,0],ig])*m2.phiy.dot(dU)
                # newv= m2.dphidx.dot(UU[m2.conn[:,1],ig])*m2.phix.dot(dU) + m2.dphidy.dot(UU[m2.conn[:,1],ig])*m2.phiy.dot(dU)
                # bb=phi.T.dot(newu)
                # Ux=MMLU.solve(bb)
                # bb=phi.T.dot(newv)
                # Uy=MMLU.solve(bb)
                # dU+=np.append(Ux,Uy)
                UU[:,ig]+=dU                
                err=np.linalg.norm(dU)/np.linalg.norm(UU[:,ig])
                if err<1e-3:
                    if js==0 and ig<len(imnums)-1:
                        UU[:,ig+1]=UU[:,ig]
                    break
            if ik==29:
                conv[ig]=0
            print("Iter # %2d | disc/dyn=%2.2f %% | dU/U=%1.2e" % (ik+1,res/dic2.dyn*100,err))
            m2.VTKSol('MSinit/MS_0_'+str(ig),UU[:,ig])
    return UU, conv



def MultiscaleInitTimeIncr(m,imf,imfile,imnums,cam,scales):    
    L=m.Tikhonov()
    conv=np.ones(len(imnums),dtype='int')
    UU=np.zeros((m.ndof,len(imnums)))
    for js in range(len(scales)):  # multiscale loop
        iscale=scales[js]        
        f=imf.Copy()
        f.SubSample(iscale)
        cam2=cam.SubSampleCopy(iscale)
        m2=m.Copy()
        m2.DICIntegration(cam2)  # to reduce the nb of integration points
        dic2=DICEngine()
        dic2.PlotMeshImage(f,m2,cam2)
        H=dic2.ComputeLHS(f,m2,cam2)
        l=10*10**2
        m2.PVDFile('MSinit/MS','vtu',1,len(imnums)-1)
        H_LU=splalg.splu(H+l*L)
        m2.VTKSol('MSinit/MS_0_0',UU[:,0])
        for ig in range(1,len(imnums)):    # time loop
            print("### IMAGE %2d ###" % (ig+1))
            imdef=imfile % imnums[ig]
            g=Image(imdef).Load()
            g.SubSample(iscale)
            for ik in range(0,30):          # Gauss Netwon loop                
                if ig>1 and ik==0: # not first time step and first GN iteration
                    if js==0: # Highest grid level
                        m3=m.Copy()
                        m3.Morphing(UU[:,ig-1])
                        m3.DICIntegration(cam2)  # to reduce the nb of integration points
                        dic3=DICEngine()
                        imdef=imfile % imnums[ig-1]
                        fi=Image(imdef).Load()
                        fi.SubSample(iscale)
                        H3=dic3.ComputeLHS(fi,m3,cam2)
                        H_LU3=splalg.splu(H3+l*L)
                        U=np.zeros(m3.ndof)
                        for ik in range(0,30):
                            [b3,res3]=dic3.ComputeRHS(g,m3,cam2,U)
                            dU=H_LU3.solve(b3-l*L.dot(U))
                            U+=dU
                            err=np.linalg.norm(dU)/np.linalg.norm(U)                            
                            if err<1e-3:
                                break
                        print("Iter # %2d | disc/dyn=%2.2f %% | dU/U=%1.2e" % (ik+1,res3/dic3.dyn*100,err))
                        UU[:,ig]=UU[:,ig-1]+U
                        [b,res]=dic2.ComputeRHS(g,m2,cam2,UU[:,ig])
                    else:
                        [b,res]=dic2.ComputeRHS(g,m2,cam2,UU[:,ig])
                        [b2,res2]=dic2.ComputeRHS(g,m2,cam2,UU[:,ig-1])
                        if res>res2:
                            UU[:,ig]=UU[:,ig-1]
                            b=b2
                            res=res2 
                else:
                    [b,res]=dic2.ComputeRHS(g,m2,cam2,UU[:,ig])
                dU=H_LU.solve(b-l*L.dot(UU[:,ig]))
                UU[:,ig]+=dU                
                err=np.linalg.norm(dU)/np.linalg.norm(UU[:,ig])
                if err<1e-3:
                    if js==0 and ig<len(imnums)-1:
                        UU[:,ig+1]=UU[:,ig]
                    break
            if ik==29:
                conv[ig]=0
            print("Iter # %2d | disc/dyn=%2.2f %% | dU/U=%1.2e" % (ik+1,res/dic2.dyn*100,err))
            m2.VTKSol('MSinit/MS_0_'+str(ig),UU[:,ig])
    return UU, conv


def MeshCalibrationInit(f,m):
    ptsm=SelectImagePoints(f)
    ptsM=SelectMeshPoints(m)
    cm=np.mean(ptsm,axis=0)
    cM=np.mean(ptsM,axis=0)
    dm=np.linalg.norm(ptsm-cm,axis=1)
    dM=np.linalg.norm(ptsM-cM,axis=1)
    scale=np.mean(dm/dM)
    dmax=np.argmax(dm)
    vm=ptsm[dmax]-cm
    vM=ptsM[dmax]-cM
    angl=np.arccos(vM.dot(vm)/(np.linalg.norm(vm)*np.linalg.norm(vM)))
    tx=cm[0]/scale - np.cos(angl)*cM[0] - np.sin(angl)*cM[1] 
    ty=-cm[1]/scale + np.sin(angl)*cM[0] - np.cos(angl)*cM[1] 
    p0=np.zeros(4)
    p0[0]=scale
    p0[1]=tx
    p0[2]=ty
    p0[3]=angl
    cam=Camera(p0)
    return cam


#%%
def ReadMeshGMSH(fn,dim=2):
    mshfid = open(fn, 'r')
    line = mshfid.readline()
    while(line.find("$Nodes") < 0):
        line = mshfid.readline()
        pass
    line = mshfid.readline()
    nnodes = int(line)
    nodes=np.zeros((nnodes,3))
    for jn in range(nnodes):
        sl = mshfid.readline().split()
        nodes[jn]=np.double(sl[1:])
    while(line.find("$Elements") < 0):
        line = mshfid.readline()
        pass
    line = mshfid.readline()
    nelems = int(line)
    elems=dict()
    ne=0
    for je in range(nelems):
        line=np.int32(mshfid.readline().split())
        if line[1]==3:    #qua4
            elems[ne]=np.append(line[1],line[5:]-1)
            ne+=1
        elif line[1]==2:  #tri3
            elems[ne]=np.append(line[1],line[5:]-1)
            ne+=1
        elif line[1]==9:  #tri6
            elems[ne]=np.append(line[1],line[5:]-1)
            ne+=1
        elif line[1]==16:  #qua8
            elems[ne]=np.append(line[1],line[5:]-1)
            ne+=1
        elif line[1]==10:  #qua9
            elems[ne]=np.append(line[1],line[5:]-1)
            ne+=1
    if dim==2:
        nodes=np.delete(nodes,2,1)
    m=Mesh(elems,nodes,dim)
    return m
    
def ReadMeshINP(fn):
    """ 2D ONLY, CPS4R ONLY """
    mshfid = open(fn, 'r')
    line = mshfid.readline()
    while(line.find("*Node") < 0):
        line = mshfid.readline()
        pass
    nodes=np.zeros((0,2))
    line = mshfid.readline()
    while(line.find("*Element") < 0):
        nodes = np.vstack((nodes,np.double(line.split(',')[1:])))
        line = mshfid.readline()
    #nnodes=nodes.shape[0]
    elems=dict()
    ne=0
    line = mshfid.readline()
    while(line.find("*") < 0):
        sl=np.int32(line.split(','))
        elem_type = np.int(sl[1:].size)-1
        # elem_type = 3 for qua4 and 2 for tri3
        elems[ne]=np.append(np.array([elem_type]),sl[1:]-1)
        line = mshfid.readline()
        ne+=1
    # here certainly other elements types... todo
    m=Mesh(elems,nodes)
    return m
