
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

# %%
plot_figs = False

refname = 'dataset_0_binning2.npy'
defname = 'dataset_1_binning2.npy'

f = px.Volume(refname).Load()
if plot_figs:
    print(f.pix.shape)
    f.Plot()
    f.VTKImage('RefVol')

m = px.ReadMesh('conform_mesh.vtk', 3)
m.KeepVolElems()

if plot_figs:
    m.Plot()  # or
    m.Write('mesh_pix_csys.vtk')

cam = px.CameraVol([1, 0, 0, 0, 0, 0, 0])

if plot_figs:
    px.PlotMeshImage3d(f, m, cam)

# %% INIT
m.Connectivity()
f = px.Volume(refname).Load()
g = px.Volume(defname).Load()
f.BuildInterp()
g.BuildInterp()
U0 = px.MultiscaleInit(f, g, m, cam, scales=[3, 2, 1])

# %% RUN
m.DVCIntegration(5)
L = m.Laplacian()
U, res = px.Correlate(f, g, m, cam, U0=U0, l0=15, L=L)

if plot_figs:
    px.PlotMeshImage3d(g, m, cam, U=U)
    m.VTKSol('disp', U)

# %% Export Solution in the Pixel CSys.
""" to plot the FE mesh in the image coord sys. """
u, v, w = cam.P(m.n[:, 0], m.n[:, 1], m.n[:, 2])
n0 = np.c_[u, v, w]
Un = U[m.conn]
u, v, w = cam.P(m.n[:, 0] + Un[:, 0], m.n[:, 1] + Un[:, 1], m.n[:, 2] + Un[:, 2])
n1 = np.c_[u, v, w]
Un = n1 - n0
Upix = np.zeros(m.ndof)
Upix[m.conn] = Un
mpix = m.Copy()
mpix.n = n0
if plot_figs:
    m.Write('mesh_pix_csys.vtk')
    mpix.VTKSol('disp_pix_csys', Upix)
