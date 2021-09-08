
import numpy as np

def Hooke(p,typc='isotropic'):
    """Compute 2D Hooke tensor from elastic constants

    Parameters
    ----------
    p : Numpy Array
        p = [E, nu] for isotropic material
        p = [E1, E2, nu12, G12] for orthotropic material
    typc : string
        'isotropic' (DEFAULT)
        'orthotropic'  

    Returns
    -------
    Numpy array
        2D Hooke tensor.

    """
    if typc == 'isotropic':
        E = p[0]
        v = p[1]
        return E / (1 - v**2) * np.array([[1, v, 0], [v, 1, 0], [0, 0, (1 - v) / 2]])
    elif typc == 'orthotropic':
        El = p[0]
        Et = p[1]
        vtl = p[2]
        Glt = p[3]
        vlt = vtl * El / Et
        alp = 1 / (1 - vlt * vtl)
        return np.array([[alp * El, alp * vtl * El, 0],
                        [alp * vlt * Et, alp * Et, 0],
                        [0, 0, 2 * Glt]])
    else:
        print('Unknown elastic constitutive regime')