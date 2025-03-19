
import numpy as np


def Hooke(p, typc='isotropic_2D_ps'):
    """Compute Hooke tensor from elastic constants

    Parameters
    ----------
    p : Numpy Array
        p = [E, nu] for isotropic material
        p = [E1, E2, nu12, G12] for orthotropic material
    typc : string
        'isotropic_2D_ps' plane stress (DEFAULT)
        'isotropic_2D_pe' plane strain 
        'isotropic_2D_axi' axisymmetric 
        'orthotropic_2D'  
        'laminate_2D'
        'isotropic_3D'  
    el_set (OPTIONAL) required for heterogeneous samples
    it is a dict with keys as element type (like mesh.e) and value are id
    of the material type which refers to the dict of materials p
    
    Returns
    -------
    Numpy array
        Hooke tensor.

    """
    if typc == 'isotropic_2D_ps':
        E = p[0]
        v = p[1]
        return E / (1 - v**2) * np.array([[1, v, 0], [v, 1, 0], [0, 0, (1 - v) / 2]])
        # return E / (1 - v**2) * np.array([1, 1, 0.5*(1 - v), v, v])
    elif typc == 'isotropic_2D_pe':
        E = p[0]
        v = p[1]
        return E / ((1 + v)*(1 - 2*v)) * np.array([[1-v, v, 0], [v, 1-v, 0], [0, 0, (1 - 2*v) / 2]])
        # return E / ((1 + v)*(1 - 2*v)) * np.array([1-v, 1-v, 0.5*(1-2*v), v, v])
    elif typc == 'isotropic_2D_axi':
        E = p[0]
        v = p[1]
        return E / ((1 + v)*(1 - 2*v)) * np.array([[1-v, v, v, 0],
                                                    [v, 1-v, v, 0],
                                                    [v, v, 1-v, 0],
                                                    [0, 0, 0, (1 - 2*v) / 2]])
        # return E/((1+v)*(1-2*v)) * np.array([1-v, 1-v, 1-v, 0.5*(1-2*v), v, v, v, v, v, v])
    elif typc == 'orthotropic_2D':
        El = p[0]
        Et = p[1]
        vtl = p[2]
        Glt = p[3]
        vlt = vtl * El / Et
        alp = 1 / (1 - vlt * vtl)
        return np.array([[alp * El, alp * vtl * El, 0],
                        [alp * vlt * Et, alp * Et, 0],
                        [0, 0, 2 * Glt]])
    elif typc == 'laminate_2D':
        # mat.type='Laminate';
        # mat.El=165e9;
        # mat.Et=7.69e9;
        # mat.vlt=0.33;
        # mat.vtl=mat.vlt*mat.Et/mat.El;
        # mat.Glt=4.75e9;
        # mat.layup=[0 45 -45 -45 45 0
        #               ones(1,6)*0.1446*1e-3];

        # ep=mat.layup(2,:);
        # alpha=mat.layup(1,:);
        # El=mat.El;
        # Et=mat.Et;
        # vlt=mat.vlt;
        # vtl=mat.vtl;
        # Glt=mat.Glt;

        # % Orthotropic stiffness term in the ply coordinate syst
        # alp=1/(1-vlt*vtl);
        # H=[alp*El     alp*vtl*El 0
        #    alp*vlt*Et alp*Et     0
        #    0          0         2*Glt];

        # % rotation and addition of the ply properties
        # A=0*H;
        # for iply=1:numel(alpha)
        #     a=alpha(iply)*pi/180;
        #     c=cos(a);
        #     s=sin(a);
        #     T=[c^2 s^2  2*c*s
        #        s^2 c^2 -2*c*s
        #       -c*s c*s  (c^2-s^2)];
        #     Q=T*H*T';
        #     A=A+Q*ep(iply);
        # end
        # h=sum(ep);
        # hooke=A./h;
        raise Exception('HOOKE LAMINATE TODO')
    if typc == 'isotropic_3D':
        E = p[0]
        v = p[1]
        C = E / (2*(1+v)*(1-2*v)) * np.array([[2*(1-v), 2*v, 2*v, 0, 0, 0],
                                            [2*v, 2*(1-v), 2*v, 0, 0, 0],
                                            [2*v, 2*v, 2*(1-v), 0, 0, 0],
                                            [0, 0, 0, (1 - 2*v), 0, 0],
                                            [0, 0, 0, 0, (1 - 2*v), 0],
                                            [0, 0, 0, 0, 0, (1 - 2*v)]])
        return C
    else:
        raise Exception('Unknown elastic constitutive regime (3D)')


def BeamProperties(p):
    """
    p: geometric and mechanical parameters:
        3D: p = [E, nu, S, Igz, k, Igy, J]
        2D: p = [E, nu, S, Igz, k]
    k: used to compute shear flexibility 
        k = 6/7 (circular) or 5/6 rectangular (TIMOSHENKO)
        k = None for EULER BERNOULLI
    """
    bp = {'E': p[0], 'v': p[1], 'S': p[2], 'Iz': p[3]}
    bp['G'] = bp['E']/(2*(1+bp['v']))
    if len(p) > 5:
        bp['Iy'] = p[5]
        bp['J'] = p[6]
    k = p[4]
    if k is None:
        # EULER BERNOULLI
        bp['phil2'] = 0
    else:
        # TIMOSHENKO
        # phi = 12 * E * I / (G * S * ky * L²) but L not known at this stage
        bp['phil2'] = 12 * bp['E'] * bp['Iz']/(bp['G']*bp['S']*k)
    return bp


def Strain2Stress(hooke, En, Es):
    if len(hooke) == 3:  # dim 2 plane strain or plane stress
        SXX = En[:, 0] * hooke[0, 0] + En[:, 1] * \
            hooke[0, 1] + 2 * Es[:, 0] * hooke[0, 2]
        SYY = En[:, 0] * hooke[1, 0] + En[:, 1] * \
            hooke[1, 1] + 2 * Es[:, 0] * hooke[1, 2]
        SXY = En[:, 0] * hooke[2, 0] + En[:, 0] * \
            hooke[2, 1] + 2 * Es[:, 0] * hooke[2, 2]
        Sn = np.c_[SXX, SYY]
        Ss = np.c_[SXY, 0*SXY]
    elif len(hooke) == 4:  # 2D axisymetry
        SXX = En[:, 0] * hooke[0, 0] +\
            En[:, 1] * hooke[0, 1] +\
            En[:, 2] * hooke[0, 2] +\
            2 * Es[:, 0] * hooke[0, 3]
        SYY = En[:, 0] * hooke[1, 0] +\
            En[:, 1] * hooke[1, 1] +\
            En[:, 2] * hooke[1, 2] +\
            2 * Es[:, 0] * hooke[1, 3]
        SZZ = En[:, 0] * hooke[2, 0] +\
            En[:, 1] * hooke[2, 1] +\
            En[:, 2] * hooke[2, 2] +\
            2 * Es[:, 0] * hooke[2, 3]
        SXY = En[:, 0] * hooke[3, 0] +\
            En[:, 1] * hooke[3, 1] +\
            En[:, 2] * hooke[3, 2] +\
            2 * Es[:, 0] * hooke[3, 3]
        Sn = np.c_[SXX, SYY, SZZ]
        Ss = np.c_[SXY, 0*SXY, 0*SXY]
    else:  # dim 3
        i = 0
        SXX = En[:, 0] * hooke[i, 0] + En[:, 1] * hooke[i, 1] + \
            En[:, 2] * hooke[i, 2] + 2 * Es[:, 0] * hooke[i, 3] + \
            2 * Es[:, 1] * hooke[i, 4] + 2 * Es[:, 2] * hooke[i, 5]
        i = 1
        SYY = En[:, 0] * hooke[i, 0] + En[:, 1] * hooke[i, 1] +\
            En[:, 2] * hooke[i, 2] + 2 * Es[:, 0] * hooke[i, 3] +\
            2 * Es[:, 1] * hooke[i, 4] + 2 * Es[:, 2] * hooke[i, 5]
        i = 2
        SZZ = En[:, 0] * hooke[i, 0] + En[:, 1] * hooke[i, 1] +\
            En[:, 2] * hooke[i, 2] + 2 * Es[:, 0] * hooke[i, 3] +\
            2 * Es[:, 1] * hooke[i, 4] + 2 * Es[:, 2] * hooke[i, 5]
        i = 3
        SXY = En[:, 0] * hooke[i, 0] + En[:, 1] * hooke[i, 1] +\
            En[:, 2] * hooke[i, 2] + 2 * Es[:, 0] * hooke[i, 3] +\
            2 * Es[:, 1] * hooke[i, 4] + 2 * Es[:, 2] * hooke[i, 5]
        i = 4
        SXZ = En[:, 0] * hooke[i, 0] + En[:, 1] * hooke[i, 1] +\
            En[:, 2] * hooke[i, 2] + 2 * Es[:, 0] * hooke[i, 3] +\
            2 * Es[:, 1] * hooke[i, 4] + 2 * Es[:, 2] * hooke[i, 5]
        i = 5
        SYZ = En[:, 0] * hooke[i, 0] + En[:, 1] * hooke[i, 1] +\
            En[:, 2] * hooke[i, 2] + 2 * Es[:, 0] * hooke[i, 3] +\
            2 * Es[:, 1] * hooke[i, 4] + 2 * Es[:, 2] * hooke[i, 5]
        Sn = np.c_[SXX, SYY, SZZ]
        Ss = np.c_[SXY, SXZ, SYZ]
    return Sn, Ss


# %%
# import sympy as sp
# E, v = sp.symbols('E, v')
# Cinv = sp.Matrix([[1/E, -v/E, -v/E, 0, 0, 0],
#             [-v/E, 1/E, -v/E, 0, 0, 0],
#             [-v/E, -v/E, 1/E, 0, 0, 0],
#             [0, 0, 0, (1+v)/E, 0, 0],
#             [0, 0, 0, 0, (1+v)/E, 0],
#             [0, 0, 0, 0, 0, (1+v)/E]])
# C = Cinv**-1
# sp.pycode(C)

# %%

def voigt(modulus, volfrac):
    """
    Voigt (upper) effective modulus of N phases (arithmetic avg)
    modulus: list or array of N modulus
    volfrac: list or array of N volume fractions in [0, 1]
    """
    if type(modulus) == list:
        modulus = np.array(modulus).astype(float)
    if type(volfrac) == list:
        volfrac = np.array(volfrac).astype(float)
    if volfrac.sum() < 1.-1e-10:
        raise Exception('The total volume fraction should be equal to 1.0')
    return modulus @ volfrac

def reuss(modulus, volfrac):
    """
    Reuss (lower) effective modulus of N phases (harmonic avg)
    modulus: list or array of N modulus
    volfrac: list or array of N volume fractions in [0, 1]
    """
    if type(modulus) == list:
        modulus = np.array(modulus).astype(float)
    if type(volfrac) == list:
        volfrac = np.array(volfrac).astype(float)
    if volfrac.sum() < 1.-1e-10:
        raise Exception('The total volume fraction should be equal to 1.0')
    if np.sum(modulus < 1e-10):
        # if one of the modulus is zero (void) > lower bound = 0
        return 0.0
    else:
        return 1.0 / np.sum(volfrac / modulus)

def hill(modulus, volfrac):
    """
    Hill average effective modulus, of N phases. 
    Defined as the average of the Reuss (lower) and Voigt (upper) bounds.
    modulus: list or array of N modulus
    volfrac: list or array of N volume fractions in [0, 1]
    """
    return 0.5 * voigt(modulus, volfrac) + 0.5 * reuss(modulus, volfrac)


def hashin_shtrikman(bulk, shear, volfrac):
    """
    Hashin-Shtrikman effective modulus of two phases.
    Args:
        volfrac: list or array of volume fractions (must sum to 1.00 or 100%).
        bluk: bulk modulus of constituents (list or array).
        shear: shear modulus of constituents (list or array).
    Returns:
        python dict with 4 bounds
    """
    if type(volfrac) == list:
        volfrac = np.array(volfrac).astype(float)
    if type(bulk) == list:
        bulk = np.array(bulk).astype(float)
    if type(shear) == list:
        shear = np.array(shear).astype(float)
    if volfrac.sum() < 1.-1e-10:
        raise Exception('The total volume fraction should be equal to 1.0')
    def z_bulk(shear):
        return (4/3.) * shear
    def z_shear(bulk, shear):
        return shear * (9 * bulk + 8 * shear) / (bulk + 2 * shear) / 6
    def bound(bulk, volfrac, z):
        return 1 / np.sum(volfrac / (bulk + z)) - z
    z_min_bulk = z_bulk(np.amin(shear))
    if np.sum(bulk < 1e-10) and np.sum(shear < 1e-10):
        z_min_shear = 0.0
    else:
        z_min_shear = z_shear(np.amin(bulk), np.amin(shear))
    z_max_bulk = z_bulk(np.amax(shear))
    z_max_shear = z_shear(np.amax(bulk), np.amax(shear))
    res = {'bulk_lower': bound(bulk, volfrac, z_min_bulk),
           'bulk_upper': bound(bulk, volfrac, z_max_bulk),
           'shear_lower': bound(bulk, volfrac, z_min_shear),
           'shear_upper': bound(bulk, volfrac, z_max_shear)}
    return res
