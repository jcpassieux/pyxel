# -*- coding: utf-8 -*-
"""
Finite Element Digital Image Correlation method 

@author: JC Passieux, INSA Toulouse, 2018

pyxel

Python library for eXperimental mechanics using finite ELements

"""

import numpy as np
import scipy as sp
import scipy.sparse.linalg as splalg
import scipy.interpolate as spi
import matplotlib.pyplot as plt
import PIL.Image as image

import vtktools as vtk

#import pdb
#pdb.set_trace()

class Elem:
    """ Class Element """
    def __init__(self):
        self.pgx=[]
        self.pgy=[]
        self.phi=[]
        self.dphidx=[]
        self.dphidy=[]

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
        nzv=0   
        repg=0  
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
                (a,)=np.where(nn==np.max(nn))[0]
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
        self.phix=sp.sparse.csc_matrix((val, (row, col+0*self.ndof/2)), shape=(self.npg, self.ndof))
        self.phiy=sp.sparse.csc_matrix((val, (row, col+1*self.ndof/2)), shape=(self.npg, self.ndof))     

    def DICIntegrationWithGrad(self,cam):
        """ Build the integration scheme """
        nzv=0  
        repg=0 
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
                (a,)=np.where(nn==np.max(nn))[0]
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
        nzv=0
        repg=0
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
        """ Assemble Stiffness Matrix """ 
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
        rep=filename.rfind('/')+1
        if rep==0:
            dir0=''
        else:
            dir0=filename[:rep]
        import os as os
        if not os.path.isdir('vtk/'+dir0):
            os.mkdir('vtk/'+dir0)
        vtkfile.write('vtk/'+filename)

    def VTKSolSeries(self,filename,UU):        
        for ig in range(UU.shape[1]):
            fname=filename+'_0_'+str(ig)
            self.VTKSol(fname,UU[:,ig])
        self.PVDFile(filename,'vtu',1,UU.shape[1])

    def PVDFile(self,fileName,ext,npart,nstep):
        rep=fileName.rfind('/')+1
        if rep==0:
            dir0=''
        else:
            dir0=fileName[:rep]
        import os as os
        if not os.path.isdir('vtk/'+dir0):
            os.mkdir('vtk/'+dir0)
        vtk.PVDFile('vtk/'+fileName,ext,npart,nstep)

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
            E=np.c_[Ex,Ey,Exy]
        new_e=np.c_[E[self.conn[:,0],0],E[self.conn[:,0],1],E[self.conn[:,0],2]].ravel()
        vtkfile.addPointData('strain',3,new_e)

        # Stress
        if len(S)>0:      
            new_s=np.c_[S[self.conn[:,0],0],S[self.conn[:,0],1],S[self.conn[:,0],2]].ravel()
            vtkfile.addPointData('stress',3,new_s)
        
        # Write the VTU file in the VTK dir
        rep=filename.rfind('/')+1
        if rep==0:
            dir0=''
        else:
            dir0=filename[:rep]
        import os as os
        if not os.path.isdir('vtk/'+dir0):
            os.mkdir('vtk/'+dir0)
        vtkfile.write('vtk/'+filename)

    def StrainAtGP(self,U):
        epsx=self.dphixdx.dot(U)
        epsy=self.dphiydy.dot(U)
        epsxy=0.5*self.dphixdy.dot(U)+0.5*self.dphiydx.dot(U)
        return epsx,epsy,epsxy

    def StrainAtNodes(self,UU):
        if not hasattr(self,'Bx'):
            m=self.Copy()
            m.GaussIntegration()
            wdetJ=sp.sparse.diags(m.wdetJ)
            phi=m.phix[:,:m.ndof//2]
            Mi=splalg.inv(phi.T.dot(wdetJ.dot(phi)).T)
            self.Bx=Mi.dot(phi.T.dot(wdetJ.dot(m.dphixdx)))
            self.By=Mi.dot(phi.T.dot(wdetJ.dot(m.dphiydy)))
            self.Bxy=Mi.dot(phi.T.dot(wdetJ.dot(m.dphixdy+m.dphixdy)))*0.5
        epsx=self.Bx.dot(UU)
        epsy=self.By.dot(UU)
        epsxy=self.Bxy.dot(UU)
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
        #''' Strain field '''
        #epsxx,epsyy,epsxy=self.StrainAtGP(U)
        #new_eps=np.array([epsxx,epsyy,epsxy]).T.ravel()
        #vtkfile.addCellData('epsilon',3,new_eps)
        
        # Write the VTU file in the VTK dir
        rep=filename.rfind('/')+1
        if rep==0:
            dir0=''
        else:
            dir0=filename[:rep]
        import os as os
        if not os.path.isdir('vtk/'+dir0):
            os.mkdir('vtk/'+dir0)
        vtkfile.write('vtk/'+filename)

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
        rep=filename.rfind('/')+1
        if rep==0:
            dir0=''
        else:
            dir0=filename[:rep]
        import os as os
        if not os.path.isdir('vtk/'+dir0):
            os.mkdir('vtk/'+dir0)
        vtkfile.write('vtk/'+filename)        
        
    def Morphing(self,U):
        self.n+=U[self.conn]
    
    def Plot(self,U=None,coef=1):
        """ Plot deformed or undeformes Mesh """
        if U is None:
            n=self.n
        else:
            n=self.n+coef*U[self.conn]
        edges=np.array([[],[]],dtype='int64')
        for ie in range(len(self.e)):
            new=np.array([self.e[ie][[i,i+1]] for i in range(min(len(self.e[ie])-1,4))])
            new[0,0]=new[-1,-1]
            new=np.sort(new,axis=1)
            edges=np.c_[edges,new.T]
        edges=np.unique(edges,axis=1)
        xn=n[edges,0]
        yn=n[edges,1]
        plt.figure()
        plt.plot(xn,yn,'k-',linewidth=1)
        plt.axis('equal')
        plt.show()
            
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
def StructuredMeshQ4(roi,N):
    # roi=np.array([[xmin,ymin],[xmax,ymax]])
    # N=[Nx,Ny]: average element size (can be scalar)
    droi=np.diff(roi,axis=0).astype('int')
    NE=np.round(droi/N)[0].astype('int')
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

def StructuredMeshT3(roi,N):
    # roi=np.array([[xmin,ymin],[xmax,ymax]])
    # N=[Nx,Ny]: average element size (can be scalar)
    droi=np.diff(roi,axis=0).astype('int')
    NE=np.round(droi/N)[0].astype('int')
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
        newimg.pix=self.pix
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
        plt.imshow(self.pix, cmap="gray", interpolation='none') 
        #plt.axis('off')
        plt.colorbar()
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

def SubQuaIso(nx,ny):
    px=1./nx
    xi=np.linspace(px-1,1-px,nx)
    py=1./ny
    yi=np.linspace(py-1,1-py,ny)
    xg,yg=np.meshgrid(xi,yi)
    wg=4./(nx*ny)
    return xg.ravel(),yg.ravel(),wg


def SubTriIso(nx,ny):
    # on divise M1M2 en nx et M1M3 en ny, le découpage étant hétérogène, on a
    # des trapèzes du côté de l'hypothénuse, le reste étant des rectangles    
    px=1/nx
    py=1/ny
    if nx>ny:
        xg=np.zeros(int(np.sum(np.floor(ny*(1-np.arange(1,nx+1)/nx)))+nx))
        yg=xg.copy()
        wg=xg.copy()
        j=1
        for i in range(1,nx+1):
            niy=int(ny*(1-i/nx)) #nombre de cases entiere verticalement
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
            nix=int(nx*(1-i/ny)) #nombre de cases entiere horizontalement
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
    def SubSampleCopy(self,nscale):
        p=self.get_p()
        p[0]/=nscale
        return Camera(p)
    def P(self,X,Y):
        u=-self.f*(-np.sin(self.rz)*X+np.cos(self.rz)*Y+self.ty)
        v=self.f*(np.cos(self.rz)*X+np.sin(self.rz)*Y+self.tx)
        return u,v
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
    def get_p(self):
        return np.array([self.f,self.tx,self.ty,self.rz])        
    def set_p(self,p):
        self.f=p[0]
        self.tx=p[1]
        self.ty=p[2]
        self.rz=p[3]
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
    ui=np.arange(0,f.pix.shape[0])
    vi=np.arange(0,f.pix.shape[1])
    [Yi,Xi]=np.meshgrid(vi,ui)
    lvlset=abs(np.sqrt((Xi-cpos[1])**2+(Yi-cpos[0])**2)-R)
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
    return nset#,R

def MeshCalibration(f,m,features,cam=0):    
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
        self.M=[]
        
    def ComputeLHS(self,f,m,cam):
        """ Compute the modified Gauss-Newton FE-DIC Hessian matrix (using gradient of F)"""
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
        
    def ComputeLHS2(self,f,g,m,cam,U):
        """ Compute the Gauss-Newton FE-DIC Hessian matrix (using gradient G)"""
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
        """ Compute the FE-DIC right hand side operator"""
        """ from a given displacement field U """
        """ gives in return B and the residual vector"""
        if hasattr(g,'tck')==0:
            g.BuildInterp()
        if len(U)!=m.ndof:
            U=np.zeros(m.ndof)
        u,v=cam.P(m.pgx+m.phix.dot(U),m.pgy+m.phiy.dot(U))
        res=g.Interp(u,v)
        res-=np.mean(res)
        std1 =np.std(res)
        res=self.f-self.std0/std1*res
        B=self.wphiJdf.T.dot(res)
        return B,res

    def ComputeRHS2(self,g,m,cam,U=[]):
        """ Compute the FE-DIC right hand side operator"""
        """ from a given displacement field U """
        """ gives in return B and the std of the residual"""
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
        
    def Correlate(self,g,m,cam,U=[]):
        if len(self.m.conn)==0:
            print('Build Connectivity before Correlation...')
        if len(self.m.phix)==0:
            print('Build Integration scheme before Correlation...')
        if len(U)==0:
            U=MultiscaleInit(self.m,self.f,g,self.cam,3)
        self.ComputeLHS(self,self.f,self.m,self.cam)
        M_LU=splalg.splu(self.M)
        for ik in range(0,30):
            [b,res]=self.ComputeRHS(g,self.m,self.cam,U)
            dU=M_LU.solve(b)
            U+=dU
            err=np.linalg.norm(dU)/np.linalg.norm(U)
            print("Iter # %2d | disc/dyn=%2.2f %% | dU/U=%1.2e" % (ik+1,res/self.dyn*100,err))
            if err<1e-3:
                break
        return U
#%% 
       

def PlotMeshImage(f,m,cam,U=None):
    """ Plot Mesh and ROI over and an image """
    if U is None:
        n=m.n
    else:
        n=m.n+U[m.conn]
    plt.figure()
    f.Show()   
    edges=np.array([[],[]],dtype='int64')
    for ie in range(len(m.e)):
        new=np.array([m.e[ie][[i,i+1]] for i in range(min(len(m.e[ie])-1,4))])
        new[0,0]=new[-1,-1]
        new=np.sort(new,axis=1)
        edges=np.c_[edges,new.T]
    edges=np.unique(edges,axis=1)
    xn=n[edges,0]
    yn=n[edges,1]
    u,v=cam.P(xn,yn)
    plt.plot(v,u,'y-',linewidth=1)
    plt.xlim([0,f.pix.shape[1]])
    plt.ylim([f.pix.shape[0],0])
    plt.axis('off')

        

def MultiscaleInit(m,imf,img,cam,nscale,l0=None):
    if l0 is None:
        n1 = np.array([m.n[m.e[i][1],:] for i in range(len(m.e))])
        n2 = np.array([m.n[m.e[i][2],:] for i in range(len(m.e))])
        l0 = 4*min(np.linalg.norm(n1-n2,axis=1))    
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
        cam2=cam.SubSampleCopy(2**iscale)
        m2=m.Copy()
        #PlotMeshImage(f,m2,cam2)
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


def MeshCalibrationInit(f,m):
    ptsm=SelectImagePoints(f,2)
    ptsM=SelectMeshPoints(m,2)
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
    return Camera(p0)


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
        line=np.int64(mshfid.readline().split())
        if line[1]==3:    #qua4
            elems[ne]=np.append(line[1],line[5:]-1)
            ne+=1
        elif line[1]==2:  #tri3
            elems[ne]=np.append(line[1],line[5:]-1)
            ne+=1
        elif line[1]==9:  #tri6
            elems[ne]=np.append(line[1],line[6:]-1)
            ne+=1
        elif line[1]==16:  #qua8
            elems[ne]=np.append(line[1],line[6:]-1)
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
        sl=np.int64(line.split(','))
        elems[ne]=np.append(np.array([3]),sl[1:]-1)
        line = mshfid.readline()
        ne+=1
    # here certainly other elements types... todo
    m=Mesh(elems,nodes)
    return m

