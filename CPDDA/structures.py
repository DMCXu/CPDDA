import numpy as np
from collections import namedtuple


class struct:
    def __init__(self, d, material, occupied, geometry):
        """
        Defines a structure to perform a CPDDA simulation on
        This includes the geometry and the material-specific dielectric function(s) for each dipole


        Parameters
        ----------
        d: float or int
            size of the cubic lattice

        material: instance objects
            use material classes provided by the `materials` module.
            the dipole material properties ,including the refractive index of the particles
            and the corresponding dielectric constant.

        occupied: tuple
            placeholder for each dipole
            core-shell structure is currently supported

        geometry: tuple
            the grid coordinates of particles


        """
        self.d = d
        self.occupied = occupied
        self.geometry = geometry
        self.material = material

    def __repr__(self):
        out_str = '\n'
        out_str += ' ------ nano-struct -------'
        if len(set(self.material)) > 1:
            out_str += '\n' + '   Inhomogeneous object, consisting of {} materials'.format(
                len(set(self.material)))
            n_name = []
            for i, mat in enumerate(self.material):
                c = mat.__name__
                n_name.append(c)
            out_str += '\n' + '   Materials: ' + ', '.join(n_name)
        else:
            out_str += '\n' + '   Homogeneous object. '
            out_str += '\n' + '   material:             "{}"'.format(self.material[0].__name__)
        out_str += '\n' + '   size of discrete lattice :     {}nm'.format(self.d)
        out_str += '\n' + '   the number of lattice:    {}'.format(self.geometry.N)

        return out_str


Geometry = namedtuple('Geometry', ['r_block', 'N', 'Nx', 'Ny', 'Nz', 'X', 'Y', 'Z', 'd_eff', 'Lx', 'Ly', 'Lz', 'V'])
Occupied = namedtuple('Occupied', ['INDEX_IN_ALL', 'INDEX_INSIDE_ALL', 'mask_shell', 'mask_core', 'INDEX_IN'])


def sphere(r_eff, d, L_core=None, s=None):
    if s is None:
        d_eff = r_eff * 2
        Lx = d_eff
        Ly = d_eff
        Lz = d_eff
        V = (4 / 3) * np.pi * pow(r_eff, 3)
        r_block, N, Nx, Ny, Nz, X, Y, Z = coordinate(Lx, Ly, Lz, d)
    else:
        d_eff = L_core + 2 * s
        Lz = L_core + 2 * s
        Lx = L_core + 2 * s
        Ly = L_core + 2 * s
        V = (4 / 3) * np.pi * pow((Lz/2), 3)
        r_block, N, Nx, Ny, Nz, X, Y, Z = coordinate(Lx, Ly, Lz, d)

    return Geometry(r_block, N, Nx, Ny, Nz, X, Y, Z, d_eff, Lx, Ly, Lz, V)


def rod(AR, d, L_core, s=None):
    if s is None:
        Lz = L_core
        Lx = Ly = Lz / AR
        V = np.pi * (Lz - Lx) * ((Lx / 2) ** 2) + 4 / 3 * np.pi * pow(Lx / 2, 3)
        r = pow((3 * V) / (4 * np.pi), 1 / 3)
        d_eff = r * 2
        r_block, N, Nx, Ny, Nz, X, Y, Z = coordinate(Lx, Ly, Lz, d)
    else:
        Lz = L_core + 2 * s
        Lx = Ly = L_core / AR + 2 * s
        V = np.pi * (Lz - Lx) * ((Lx / 2) ** 2) + 4 / 3 * np.pi * pow(Lx / 2, 3)
        r = pow((3 * V) / (4 * np.pi), 1 / 3)
        d_eff = r * 2
        r_block, N, Nx, Ny, Nz, X, Y, Z = coordinate(Lx, Ly, Lz, d)

    return Geometry(r_block, N, Nx, Ny, Nz, X, Y, Z, d_eff, Lx, Ly, Lz, V)


def ellipsoid(d, a, b, c, s=None):
    if s is None:
        Lx = 2 * a
        Ly = 2 * b
        Lz = 2 * c
        V = (4 / 3) * np.pi * a * b * c
        r = pow((3 * V) / (4 * np.pi), 1 / 3)
        d_eff = r * 2
        r_block, N, Nx, Ny, Nz, X, Y, Z = coordinate(Lx, Ly, Lz, d)
    else:
        Lx = 2 * (a + s)
        Ly = 2 * (b + s)
        Lz = 2 * (c + s)
        V = (4 / 3) * np.pi * a * b * c
        r = pow((3 * V) / (4 * np.pi), 1 / 3)
        d_eff = r * 2
        r_block, N, Nx, Ny, Nz, X, Y, Z = coordinate(Lx, Ly, Lz, d)
    return Geometry(r_block, N, Nx, Ny, Nz, X, Y, Z, d_eff, Lx, Ly, Lz, V)


def rec_block(d, a, b, c, s=None):
    if s is None:
        Lx = a
        Ly = b
        Lz = c
        V = Lx * Ly * Lz
        r = pow((3 * V) / (4 * np.pi), 1 / 3)
        d_eff = r * 2
        r_block, N, Nx, Ny, Nz, X, Y, Z = coordinate(Lx, Ly, Lz, d)
    else:
        Lx = a + s * 2
        Ly = b + s * 2
        Lz = c + s * 2
        V = Lx * Ly * Lz
        r = pow((3 * V) / (4 * np.pi), 1 / 3)
        d_eff = r * 2
        r_block, N, Nx, Ny, Nz, X, Y, Z = coordinate(Lx, Ly, Lz, d)
    return Geometry(r_block, N, Nx, Ny, Nz, X, Y, Z, d_eff, Lx, Ly, Lz, V)


def coordinate(Lx, Ly, Lz, d):
    dx = d
    dy = d
    dz = d

    ix = np.arange(-round(Lx / (2 * dx)), round(Lx / (2 * dx)) + 1)
    iy = np.arange(-round(Ly / (2 * dy)), round(Ly / (2 * dy)) + 1)
    iz = np.arange(-round(Lz / (2 * dz)), round(Lz / (2 * dz)) + 1)

    y, x, z = np.meshgrid(iy, ix, iz)
    Ny = len(iy)
    Nz = len(iz)
    Nx = len(ix)

    N = Nx * Ny * Nz

    X = np.reshape(x, [N, 1], order='F') * dx
    Y = np.reshape(y, [N, 1], order='F') * dy
    Z = np.reshape(z, [N, 1], order='F') * dz

    r_block = np.hstack((X, Y, Z))

    return r_block, N, Nx, Ny, Nz, X, Y, Z


def INDEX_in(Np_shape, geometry, L_core=None, s=None, AR=None):
    X0 = 0
    Y0 = 0
    Z0 = 0

    INDEX_IN_ALL, INDEX_INSIDE_ALL, mask_shell, mask_core, INDEX_IN = None, None, None, None, None

    if L_core is None and s is None:
        if Np_shape == "spherical":
            Index_in = np.nonzero(
                np.sqrt((geometry.X - X0) ** 2 + (geometry.Y - Y0) ** 2 + (geometry.Z - Z0) ** 2) <= (geometry.Lx / 2))
            Index_in = np.ravel_multi_index(Index_in, geometry.X.shape) + 1

        elif Np_shape == "rod":
            Index_in = np.nonzero(
                (np.sqrt((geometry.X - X0) ** 2 + (geometry.Y - Y0) ** 2 + (
                        np.abs(geometry.Z - Z0) - geometry.Lz / 2 + geometry.Lx / 2) ** 2) <= (geometry.Lx / 2)
                 ) & (np.abs(geometry.Z - Z0) > (geometry.Lz / 2 - geometry.Lx / 2)) | (
                        np.sqrt((geometry.X - X0) ** 2 + (geometry.Y - Y0) ** 2) <= (geometry.Lx / 2)
                ) & (np.abs(geometry.Z - Z0) <= (geometry.Lz / 2 - geometry.Lx / 2)))
            Index_in = np.ravel_multi_index(Index_in, geometry.X.shape) + 1

        elif Np_shape == "rec_block":
            Index_in = np.nonzero(
                (np.abs(geometry.X - X0) <= geometry.Lx / 2) & (np.abs(geometry.Y - Y0) <= geometry.Ly / 2) & (
                        np.abs(geometry.Z - Z0) <= geometry.Lz / 2))
            Index_in = np.ravel_multi_index(Index_in, geometry.X.shape) + 1

        elif Np_shape == "ellipsoid":
            Index_in = np.nonzero(np.sqrt(
                (geometry.X - X0) ** 2 / ((geometry.Lx / 2) ** 2) + (geometry.Y - Y0) ** 2 / (
                        (geometry.Ly / 2) ** 2) + (geometry.Z - Z0) ** 2 / (
                        (geometry.Lz / 2) ** 2)) <= 1)  # ** 运算符用于执行数组的逐元素幂运算
            Index_in = np.ravel_multi_index(Index_in, geometry.X.shape) + 1

        INDEX_INSIDE_ALL = np.zeros((geometry.N, 1))
        INDEX_INSIDE_ALL[Index_in - 1] = 1
        INDEX_IN_ALL = np.reshape(INDEX_INSIDE_ALL, [geometry.Nx, geometry.Ny, geometry.Nz], order="F")

    elif L_core is not None and s is not None:
        if Np_shape == "spherical":

            R_core = L_core  # 核的半径
            R_shell = s  # 壳的外半径

            distance_from_center = np.sqrt((geometry.X - X0) ** 2 + (geometry.Y - Y0) ** 2 + (geometry.Z - Z0) ** 2)

            # 核的索引
            Index_core = np.nonzero(distance_from_center <= R_core)
            Index_core = np.ravel_multi_index(Index_core, geometry.X.shape) + 1

            # 壳的索引（壳层和核层不重叠）
            Index_shell = np.nonzero((distance_from_center > R_core) & (distance_from_center <= R_shell))
            Index_shell = np.ravel_multi_index(Index_shell, geometry.X.shape) + 1

            # 区域类所有东西的索引
            Index_in = np.nonzero(
                np.sqrt((geometry.X - X0) ** 2 + (geometry.Y - Y0) ** 2 + (geometry.Z - Z0) ** 2) <= (
                        geometry.d_eff / 2))
            Index_in = np.ravel_multi_index(Index_in, geometry.X.shape) + 1

        elif Np_shape == "rod":
            L_core_xy = L_core / AR
            Index_core = np.nonzero(
                (np.sqrt((geometry.X - X0) ** 2 + (geometry.Y - Y0) ** 2 + (
                        np.abs(geometry.Z - Z0) - L_core / 2 + L_core_xy / 2) ** 2) <= L_core_xy / 2)
                & (np.abs(geometry.Z - Z0) > (L_core / 2 - L_core_xy / 2)) | (
                        np.sqrt((geometry.X - X0) ** 2 + (geometry.Y - Y0) ** 2) <= (L_core_xy / 2)
                ) & (np.abs(geometry.Z - Z0) <= (L_core / 2 - L_core_xy / 2)))
            Index_core = np.ravel_multi_index(Index_core, geometry.X.shape) + 1
            # 壳的索引
            Index_shell = np.nonzero(
                ((np.sqrt((geometry.X - X0) ** 2 + (geometry.Y - Y0) ** 2 + (
                        np.abs(geometry.Z - Z0) - geometry.Lz / 2 + geometry.Lx / 2) ** 2) <= (
                          geometry.Lx / 2))
                 & (np.abs(geometry.Z - Z0) > (geometry.Lz / 2 + geometry.Lx / 2))) | (
                        (np.sqrt((geometry.X - X0) ** 2 + (geometry.Y - Y0) ** 2) <= (geometry.Lx / 2)) & (
                        np.abs(geometry.Z - Z0) <= (geometry.Lz / 2 + geometry.Lx / 2))) & ~((np.sqrt(
                    (geometry.X - X0) ** 2 + (geometry.Y - Y0) ** 2 + (
                            np.abs(geometry.Z - Z0) - L_core / 2 + L_core_xy / 2) ** 2) <= L_core_xy / 2) & (np.abs(geometry.Z - Z0) > (
                        L_core / 2 - L_core_xy / 2)) | (np.sqrt(
                    (geometry.X - X0) ** 2 + (geometry.Y - Y0) ** 2) <= (L_core_xy / 2)) & (np.abs(
                    geometry.Z - Z0) <= (L_core / 2 - L_core_xy / 2))))
            Index_shell = np.ravel_multi_index(Index_shell, geometry.X.shape) + 1

            Index_in = np.nonzero(
                (np.sqrt((geometry.X - X0) ** 2 + (geometry.Y - Y0) ** 2 + (
                        np.abs(geometry.Z - Z0) - geometry.Lz / 2 + geometry.Lx / 2) ** 2) <= (geometry.Lx / 2)
                 ) & (np.abs(geometry.Z - Z0) > (geometry.Lz / 2 - geometry.Lx / 2)) | (
                        np.sqrt((geometry.X - X0) ** 2 + (geometry.Y - Y0) ** 2) <= (geometry.Lx / 2)
                ) & (np.abs(geometry.Z - Z0) <= (geometry.Lz / 2 - geometry.Lx / 2)))
            Index_in = np.ravel_multi_index(Index_in, geometry.X.shape) + 1

        INDEX_INSIDE_ALL = np.zeros((geometry.N, 1))
        INDEX_INSIDE_ALL[Index_in - 1] = 1
        INDEX_IN_ALL = np.reshape(INDEX_INSIDE_ALL, [geometry.Nx, geometry.Ny, geometry.Nz], order="F")

        INDEX_INSIDE = np.zeros((geometry.N, 1))
        INDEX_INSIDE[Index_core - 1] = 1  # 核
        INDEX_INSIDE[Index_shell - 1] = 2  # 壳
        INDEX_IN = np.reshape(INDEX_INSIDE, [geometry.Nx, geometry.Ny, geometry.Nz], order="F")

        mask_core = (INDEX_IN == 1)
        mask_shell = (INDEX_IN == 2)

    elif L_core is None and s is not None:
        if Np_shape == "ellipsoid":
            Index_core = np.nonzero(np.sqrt(
                (geometry.X - X0) ** 2 / (((geometry.Lx - 2 * s) / 2) ** 2) + (geometry.Y - Y0) ** 2 / (
                        ((geometry.Ly - 2 * s) / 2) ** 2) + (geometry.Z - Z0) ** 2 / (
                        ((geometry.Lz - 2 * s) / 2) ** 2)) <= 1)
            Index_core = np.ravel_multi_index(Index_core, geometry.X.shape) + 1

            Index_shell = np.nonzero(
                (np.sqrt((geometry.X - X0) ** 2 / ((geometry.Lx / 2) ** 2) + (geometry.Y - Y0) ** 2 /
                         ((geometry.Ly / 2) ** 2) + (geometry.Z - Z0) ** 2 / ((geometry.Lz / 2) ** 2)) <= 1) & ~
                (np.sqrt((geometry.X - X0) ** 2 / (((geometry.Lx - 2 * s) / 2) ** 2) + (geometry.Y - Y0) ** 2 / (
                        ((geometry.Ly - 2 * s) / 2) ** 2) + (geometry.Z - Z0) ** 2 / (
                                 ((geometry.Lz - 2 * s) / 2) ** 2)) <= 1))
            Index_shell = np.ravel_multi_index(Index_shell, geometry.X.shape) + 1

            Index_in = np.nonzero(np.sqrt((geometry.X - X0) ** 2 / ((geometry.Lx / 2) ** 2) + (geometry.Y - Y0) ** 2 /
                                          ((geometry.Ly / 2) ** 2) + (geometry.Z - Z0) ** 2 / (
                                                  (geometry.Lz / 2) ** 2)) <= 1)
            Index_in = np.ravel_multi_index(Index_in, geometry.X.shape) + 1
        elif Np_shape == "rec_block":
            Index_core = np.nonzero(
                (np.abs(geometry.X - X0) <= (geometry.Lx - 2 * s) / 2) & (
                            np.abs((geometry.Ly - 2 * s) - Y0) <= (geometry.Ly - 2 * s) / 2) & (
                        np.abs((geometry.Lz - 2 * s) - Z0) <= (geometry.Lz - 2 * s) / 2))
            Index_core = np.ravel_multi_index(Index_core, geometry.X.shape) + 1

            Index_shell = np.nonzero(
                (np.abs(geometry.X - X0) <= geometry.Lx / 2) & (np.abs(geometry.Ly - Y0) <= geometry.Ly / 2) & (
                        np.abs(geometry.Lz - Z0) <= geometry.Lz / 2) & ~ (
                            (np.abs(geometry.X - X0) <= (geometry.Lx - 2 * s) / 2) & (
                                np.abs((geometry.Ly - 2 * s) - Y0) <= (geometry.Ly - 2 * s) / 2) & (
                                    np.abs((geometry.Lz - 2 * s) - Z0) <= (geometry.Lz - 2 * s) / 2)))
            Index_shell = np.ravel_multi_index(Index_shell, geometry.X.shape) + 1

            Index_in = np.nonzero(
                (np.abs(geometry.X - X0) <= geometry.Lx / 2) & (np.abs(geometry.Ly - Y0) <= geometry.Ly / 2) & (
                        np.abs(geometry.Lz - Z0) <= geometry.Lz / 2))
            Index_in = np.ravel_multi_index(Index_in, geometry.X.shape) + 1

        INDEX_INSIDE_ALL = np.zeros((geometry.N, 1))
        INDEX_INSIDE_ALL[Index_in - 1] = 1
        INDEX_IN_ALL = np.reshape(INDEX_INSIDE_ALL, [geometry.Nx, geometry.Ny, geometry.Nz], order="F")

        INDEX_INSIDE = np.zeros((geometry.N, 1))
        INDEX_INSIDE[Index_core - 1] = 1  # 核
        INDEX_INSIDE[Index_shell - 1] = 2  # 壳
        INDEX_IN = np.reshape(INDEX_INSIDE, [geometry.Nx, geometry.Ny, geometry.Nz], order="F")

        mask_core = (INDEX_IN == 1)
        mask_shell = (INDEX_IN == 2)

    return Occupied(INDEX_IN_ALL, INDEX_INSIDE_ALL, mask_shell, mask_core, INDEX_IN)
