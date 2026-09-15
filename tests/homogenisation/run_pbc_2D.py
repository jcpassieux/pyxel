# %%
import numpy as np
import pyxel as px

# %%
box = np.array([[-1, -1], [1, 1]])
m = px.OpenHolePlateUnstructured(box, 0.5, [0, 0], 0.1, 0.05)

m.Plot()

edges, corners = m.GetPBCPairs()
all_pairs = np.vstack((edges, corners))

# %%   One simulation with PBC

m.Connectivity()
m.GaussIntegration()
C = px.Hooke([1, 0.3], 'isotropic_2D_ps')
K = m.Stiffness(C)

eps = np.array([[0., 0.3], 
                [0.3, 0]])
Kd, Fd = m.ApplyPeriodicBC(K, eps, all_pairs)
U = m.LinearSolver(Kd, 0*Fd, Fd)

m.VTKSol('test', U)

# %%  Computing Homogenisation hooke tensor

def CalcSigHom(eps):
    Kd, Fd = m.ApplyPeriodicBC(K, eps, all_pairs)
    U = m.LinearSolver(Kd, 0*Fd, Fd)

    En, Es = m.StrainAtGP(U)
    Sn, Ss = px.Strain2Stress(C, En, Es)

    V = np.sum(m.wdetJ)
    Snh = (m.wdetJ @ Sn) / V
    Ssh = (m.wdetJ @ Ss) / V
    return np.append(Snh, Ssh[0])

Shx = CalcSigHom(np.array([[1., 0], [0, 0]]))
Shy = CalcSigHom(np.array([[0., 0], [0, 1]]))
Shxy = CalcSigHom(np.array([[0., 1], [1, 0]]))

Ch = np.vstack((Shx, Shy, Shxy))

