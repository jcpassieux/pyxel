
import numpy as np
import pyxel as px

"""
   Prior to runnning this code, Please download the dataset below

   Lee, Peter, LaVallee, Yan, & Bay, Brian. (2022). 
   Dynamic X-ray CT of Synthetic magma for Digital Volume Correlation analysis (1.0.0) [Data set].
   Zenodo. https://doi.org/10.5281/zenodo.4835668 

   Then downsample the images to run the sample code faster.
"""

# refname = 'dataset_0.npy'
# defname = 'dataset_1.npy'
# f = px.Volume(refname).Load()
# f.SubSample(2)
# np.save('dataset_0_binning2', f.pix)
# g = px.Volume(defname).Load()
# g.SubSample(2)
# np.save('dataset_1_binning2', g.pix)

#%%
refname = 'dataset_0_binning2.npy'
defname = 'dataset_1_binning2.npy'

f = px.Volume(refname).Load()
print(f.pix.shape)
f.Plot()
f.VTKImage('RefVol')

cpos = np.array([50, 155, 153])
m = px.TetraMeshCylinder(cpos[0], cpos[1], cpos[2], 110, 300, 15)
m.n -= cpos[np.newaxis]
m.Plot()

m.Write('mesh.vtk')
cam = px.CameraVol([1, cpos[0], cpos[1], cpos[2], 0, -np.pi/2, 0])
px.PlotMeshImage3d(f, m, cam)

# u, v, w = cam.P(m.n[:,0], m.n[:,1], m.n[:,2])
# m.n = np.c_[u, v, w]
# m.Write('mesh_imageCSys.vtk')

#%% INIT
m.Connectivity()
m.DVCIntegration(5)
f = px.Volume(refname).Load()
g = px.Volume(defname).Load()
f.BuildInterp()
g.BuildInterp()
U0 = px.MultiscaleInit(f, g, m, cam, scales=[3, 2, 1])

#%% RUN
m.GetApproxElementSize(cam, 'min')
m.DVCIntegration(10)
L = m.Tikhonov()
U, res = px.Correlate(f, g, m, cam, U0=U0, l0=15, L=L)

px.PlotMeshImage3d(g, m, cam, U=U)

#%% Export in the Mesh CSYS
m.VTKSol('disp', U) 

#%% Export Solution in the Pixel CSys.
u, v, w = cam.P(m.n[:,0], m.n[:,1], m.n[:,2])
n0 = np.c_[u, v, w]
Un = U[m.conn]
u, v, w = cam.P(m.n[:,0] + Un[:,0], m.n[:,1] + Un[:,1], m.n[:,2] + Un[:,2])
n1 = np.c_[u, v, w]

Un = n1 - n0
Upix = np.zeros(m.ndof)
Upix[m.conn] = Un
mpix = m.Copy()
mpix.n = n0
mpix.VTKSol('disp_pix', Upix)

