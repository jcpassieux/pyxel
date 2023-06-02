
import numpy as np

def Hooke(p, typc='isotropic_2D_ps'):
    """Compute 2D Hooke tensor from elastic constants

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

    Returns
    -------
    Numpy array
        Hooke tensor.

    """
    if typc == 'isotropic_2D_ps':
        E = p[0]
        v = p[1]
        return E / (1 - v**2) * np.array([[1, v, 0], [v, 1, 0], [0, 0, (1 - v) / 2]])
    elif typc == 'isotropic_2D_pe':
        E = p[0]
        v = p[1]
        return E / ((1 + v)*(1 - 2*v)) * np.array([[1-v, v, 0], [v, 1-v, 0], [0, 0, (1 - 2*v) / 2]])
    elif typc == 'isotropic_2D_axi':
        E = p[0]
        v = p[1]
        return E / ((1 + v)*(1 - 2*v)) * np.array([[1-v, v, v, 0],
                                                   [v, 1-v, v, 0],
                                                   [v, v, 1-v, 0],
                                                   [0, 0, 0, (1 - 2*v) / 2]])
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
        return E / ((1+v)*(1-2*v)) * np.array([[1-v, v, v, 0, 0, 0], 
                                               [v, 1-v, v, 0, 0, 0], 
                                               [v, v, 1-v, 0, 0, 0], 
                                               [0, 0, 0, (1 - v) / 2, 0, 0],
                                               [0, 0, 0, 0, (1 - v) / 2, 0],
                                               [0, 0, 0, 0, 0, (1 - v) / 2]])
    else:
        raise Exception('Unknown elastic constitutive regime (3D)')
