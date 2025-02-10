import numpy as np


class efield:
    """
    incident electromagnetic field container class

    Defines an incident electric field including information about wavelengths,
    polarizations,  or whatever optional parameter is supported by
    the used field generator.

    """
    def __init__(self, field_type, wavelength, K_kwargs, E_kwargs):
        """
        incident electromagnetic field container class

        Defines an incident electric field including information about wavelengths,
        polarizations,  or whatever optional parameter is supported by
        the used field generator.

        field_type: str
            the type of incident light

        wavelength: Array of float
            Wavelength of incident light

        :param K_kwargs:
        :param E_kwargs:
        """
        self.field_type = field_type
        self.wavelength = wavelength
        self.E_kwargs = np.array(list(E_kwargs.values()))
        self.K_kwargs = np.array(list(K_kwargs.values()))
        self.phi_k = np.arctan2(self.K_kwargs[:, 1], self.K_kwargs[:, 0])
        self.theta_k = np.arccos(self.K_kwargs[:, 2] / np.linalg.norm(self.K_kwargs, axis=1))

        self.phi_e = np.arctan2(self.E_kwargs[:, 1], self.E_kwargs[:, 0])
        self.theta_e = np.arccos(self.E_kwargs[:, 2] / np.linalg.norm(self.E_kwargs, axis=1))

    def __repr__(self):
        out_str = ' ----- incident field -----'
        out_str += '\n' + '   field type: "{}"'.format(self.field_type)
        out_str += '\n' + '   {} wavelength between {} and {} nm'.format(
            len(self.wavelength), self.wavelength.min(),
            self.wavelength.max()
        )
        if len(self.K_kwargs) < 4:
            out_str += '\n' + '   propogation direction phi:{}, theta:{}'.format(self.phi_k
                                                                                 , self.theta_k)
        else:
            out_str += '\n' + '   propogation direction'
            for i, (phi, theta) in enumerate(zip(self.phi_k[:6], self.theta_k[:6])):
                out_str += '\n- {}: phi={}, theta={}'.format(i, phi, theta)

        if len(self.E_kwargs) < 4:
            out_str += '\n' + '   propogation direction phi:{}, theta:{}'.format(self.phi_e
                                                                                 , self.theta_e)
        else:
            out_str += '\n' + '   propogation direction'
            for i, (phi, theta) in enumerate(zip(self.phi_e[:6], self.theta_e[:6])):
                out_str += '\n- {}: phi={}, theta={}'.format(i, phi, theta)
        return out_str


def plan_wave():
    IB = "plane wave"
    return IB, 0


def gaussian(Waist_r, z0):
    IB = "gaussian"
    return IB, Waist_r, z0
