import numpy as np
import cupy as cp
import copy
import os
import time
from CPDDA import post_processing


class simulation:
    def __init__(self, struct, field, file_name):
        """
        Main CPDDA simulation container class

        - struct: class:`.structures.struct`
            - the dipole material properties
            - the grid coordinates of particles
            - placeholder for each dipole

        - field: class:`.fields.efield`
            - incident electric field including information

        - file_name:  class:`.fields.efield`
            - User-defined
        """
        self.struct = struct
        self.field = field
        self.file_name = os.path.splitext(file_name)[0]

    def DDA(self, **kwargs):
        """  wrapper to : func: DDA"""
        return DDA(self, **kwargs)


def DDA(sim, method):
    """
    - sim: simulation object of CPDDA
    - method: Choose between CPU mode and GPU mode
    """
    rkj1 = sim.struct.geometry.r_block[0, 0] - sim.struct.geometry.r_block[:, 0]
    rkj2 = sim.struct.geometry.r_block[0, 1] - sim.struct.geometry.r_block[:, 1]
    rkj3 = sim.struct.geometry.r_block[0, 2] - sim.struct.geometry.r_block[:, 2]

    rk_to_rj = np.column_stack((rkj1, rkj2, rkj3))
    rk_to_rj[0, :] = 1
    RJK = np.sqrt(rk_to_rj[:, 0] ** 2 + rk_to_rj[:, 1] ** 2 + rk_to_rj[:, 2] ** 2)

    rjkrjk = np.column_stack((rkj1 / RJK, rkj2 / RJK, rkj3 / RJK))

    rjkrjk1_I = rjkrjk[:, 0] * rjkrjk[:, 0] - 1
    rjkrjk2_I = rjkrjk[:, 0] * rjkrjk[:, 1]
    rjkrjk3_I = rjkrjk[:, 0] * rjkrjk[:, 2]
    rjkrjk4_I = rjkrjk[:, 1] * rjkrjk[:, 1] - 1
    rjkrjk5_I = rjkrjk[:, 1] * rjkrjk[:, 2]
    rjkrjk6_I = rjkrjk[:, 2] * rjkrjk[:, 2] - 1

    rjkrjk31_I = 3 * rjkrjk[:, 0] * rjkrjk[:, 0] - 1
    rjkrjk32_I = 3 * rjkrjk[:, 0] * rjkrjk[:, 1]
    rjkrjk33_I = 3 * rjkrjk[:, 0] * rjkrjk[:, 2]
    rjkrjk34_I = 3 * rjkrjk[:, 1] * rjkrjk[:, 1] - 1
    rjkrjk35_I = 3 * rjkrjk[:, 1] * rjkrjk[:, 2]
    rjkrjk36_I = 3 * rjkrjk[:, 2] * rjkrjk[:, 2] - 1

    rjkrjk1_I[0] = 0
    rjkrjk2_I[0] = 0
    rjkrjk3_I[0] = 0
    rjkrjk4_I[0] = 0
    rjkrjk5_I[0] = 0
    rjkrjk6_I[0] = 0

    rjkrjk31_I[0] = 0
    rjkrjk32_I[0] = 0
    rjkrjk33_I[0] = 0
    rjkrjk34_I[0] = 0
    rjkrjk35_I[0] = 0
    rjkrjk36_I[0] = 0

    if method == "cupy":
        px, py, pz, PX_vector, PY_vector, PZ_vector,Inverse_Alpha, data = DDA_with_GPU(sim, rjkrjk1_I, rjkrjk2_I, rjkrjk3_I,
                                                                         rjkrjk4_I, rjkrjk5_I, rjkrjk6_I,
                                                                         rjkrjk31_I, rjkrjk32_I, rjkrjk33_I, rjkrjk34_I,
                                                                         rjkrjk35_I, rjkrjk36_I,
                                                                         RJK)

    elif method == "None":
        px, py, pz, PX_vector, PY_vector, PZ_vector,Inverse_Alpha, data = DDA_WITHOUT_GPU(sim, rjkrjk1_I, rjkrjk2_I, rjkrjk3_I,
                                                                            rjkrjk4_I, rjkrjk5_I, rjkrjk6_I,
                                                                            rjkrjk31_I, rjkrjk32_I, rjkrjk33_I,
                                                                            rjkrjk34_I, rjkrjk35_I, rjkrjk36_I,
                                                                            RJK)

    return px, py, pz, PX_vector, PY_vector, PZ_vector, Inverse_Alpha, data


def DDA_with_GPU(sim, rjkrjk1_I, rjkrjk2_I, rjkrjk3_I, rjkrjk4_I, rjkrjk5_I, rjkrjk6_I,
                 rjkrjk31_I, rjkrjk32_I, rjkrjk33_I, rjkrjk34_I, rjkrjk35_I, rjkrjk36_I,
                 RJK):
    """
    GPU mode
    """
    """==================================Send all subsequent parameters to the GPU================================="""
    start_time = time.time()
    mempool = cp.get_default_memory_pool()
    if len(sim.struct.material) == 2:
        shell_eps = sim.struct.material[0].epsilon(sim)
        core_eps = sim.struct.material[1].epsilon(sim)
        ep_nps_eb_shell_G = cp.array(shell_eps)
        ep_nps_eb_core_G = cp.array(core_eps)
        mask_core_G = cp.array(sim.struct.occupied.mask_core)
        mask_shell_G = cp.array(sim.struct.occupied.mask_shell)
    else:
        eps = sim.struct.material[0].epsilon(sim)
        eps_np_eb = cp.array(eps)

    E0 = cp.array(sim.field.E_kwargs[0])
    K0 = cp.array(sim.field.K_kwargs[0])
    k_G = cp.array(2 * np.pi / sim.struct.material[0].wl * sim.struct.material[0].nb)

    ini_INDEX_IN_G = cp.array(sim.struct.occupied.INDEX_IN, dtype="complex64")
    INDEX_IN_ALL_G = cp.array(sim.struct.occupied.INDEX_IN_ALL)
    INDEX_INSIDE_ALL_G = cp.array(sim.struct.occupied.INDEX_INSIDE_ALL)

    r_block_G = cp.array(sim.struct.geometry.r_block)
    RJK_G = cp.array(RJK)
    rjkrjk1_I_G = cp.array(rjkrjk1_I)
    rjkrjk2_I_G = cp.array(rjkrjk2_I)
    rjkrjk3_I_G = cp.array(rjkrjk3_I)
    rjkrjk4_I_G = cp.array(rjkrjk4_I)
    rjkrjk5_I_G = cp.array(rjkrjk5_I)
    rjkrjk6_I_G = cp.array(rjkrjk6_I)

    rjkrjk31_I_G = cp.array(rjkrjk31_I)
    rjkrjk32_I_G = cp.array(rjkrjk32_I)
    rjkrjk33_I_G = cp.array(rjkrjk33_I)
    rjkrjk34_I_G = cp.array(rjkrjk34_I)
    rjkrjk35_I_G = cp.array(rjkrjk35_I)
    rjkrjk36_I_G = cp.array(rjkrjk36_I)

    length = len(sim.field.wavelength)
    CABSLIST = []
    CSCALIST = []
    CEXTLIST = []
    for J in range(length):
        INDEX_IN_G = copy.deepcopy(ini_INDEX_IN_G)
        kvec = k_G[J] * K0
        Exp_ikvec_rjk = cp.exp(1j * cp.linalg.norm(kvec) * RJK_G) / RJK_G

        ikvec_rjk = (1j * cp.linalg.norm(kvec) * RJK_G - 1) / (RJK_G ** 2)
        kr = kvec[0] * r_block_G[:, 0] + kvec[1] * r_block_G[:, 1] + kvec[2] * r_block_G[:, 2]
        if len(sim.struct.material) == 2:
            eps_NP_eb_shell = ep_nps_eb_shell_G[J]
            eps_NP_eb_core = ep_nps_eb_core_G[J]
        elif len(sim.struct.material) == 1:
            eps_NP_eb = eps_np_eb[J]

        """==========================Calculating Incident electric filed components==============================="""

        E_x = None
        E_y = None
        E_z = None

        if sim.field.field_type[0] == "plane wave":

            expikr = cp.exp(1j * kr)
            Ex = E0[0] * expikr * INDEX_INSIDE_ALL_G.flatten()
            Ey = E0[1] * expikr * INDEX_INSIDE_ALL_G.flatten()
            Ez = E0[2] * expikr * INDEX_INSIDE_ALL_G.flatten()

            E_vector = cp.concatenate((Ex.ravel(), Ey.ravel(), Ez.ravel()))[:, cp.newaxis]

            E_x = cp.reshape(Ex, (sim.struct.geometry.Nx, sim.struct.geometry.Ny, sim.struct.geometry.Nz),
                             order='F')
            E_y = cp.reshape(Ey, (sim.struct.geometry.Nx, sim.struct.geometry.Ny, sim.struct.geometry.Nz),
                             order='F')
            E_z = cp.reshape(Ez, (sim.struct.geometry.Nx, sim.struct.geometry.Ny, sim.struct.geometry.Nz),
                             order='F')

        elif sim.field.field_type[0] == "gaussian":

            Waist_r = sim.field.field_type[1]
            z0 = sim.field.field_type[2]
            if K0[0] == 1:
                r = pow((r_block_G[:, 1] ** 2 + r_block_G[:, 2] ** 2), 0.5)
                z = r_block_G[:, 0]
            elif K0[1] == 1:
                r = pow((r_block_G[:, 0] ** 2 + r_block_G[:, 2] ** 2), 0.5)
                z = r_block_G[:, 1]
            elif K0[2] == 1:
                r = pow((r_block_G[:, 0] ** 2 + r_block_G[:, 2] ** 2), 0.5)
                z = r_block_G[:, 1]

            W0 = Waist_r * sim.field.wavelength[J]
            K = 2*np.pi * sim.struct.material[0].nb
            zR = np.pi * pow(W0, 2) * sim.struct.material[0].nb / sim.field.wavelength[J]
            Qz = np.arctan((z - z0)/zR)

            Wz = W0 * (1 + pow(pow(((z - z0) / zR), 2), 0.5))

            Rz_inverse = z/(pow((z - z0), 2) + zR ** 2)

            E = (W0 / Wz) * np.exp(-pow((r / Wz), 2) - 1j * (K * (z - z0) + K * (r ** 2)) * Rz_inverse / 2 - Qz)
            Ex = E0[0] * E * INDEX_INSIDE_ALL_G.flatten()
            Ey = E0[1] * E * INDEX_INSIDE_ALL_G.flatten()
            Ez = E0[2] * E * INDEX_INSIDE_ALL_G.flatten()

            E_x = cp.reshape(Ex, (sim.struct.geometry.Nx, sim.struct.geometry.Ny, sim.struct.geometry.Nz),
                             order='F')
            E_y = cp.reshape(Ey, (sim.struct.geometry.Nx, sim.struct.geometry.Ny, sim.struct.geometry.Nz),
                             order='F')
            E_z = cp.reshape(Ez, (sim.struct.geometry.Nx, sim.struct.geometry.Ny, sim.struct.geometry.Nz),
                             order='F')

        '''===================================== Polarizability ========================================='''
        if len(sim.struct.material) == 2:

            k0 = 2 * np.pi
            b1 = -1.891531
            b2 = 0.1648469
            b3 = -1.7700004
            dcube = sim.struct.d ** 3
            a_hat = kvec / cp.linalg.norm(kvec)
            e_hat = E0 / cp.linalg.norm(E0)

            S = 0
            for i in range(3):
                S = S + (a_hat[i] * e_hat[i]) ** 2
            S = (a_hat[0] * e_hat[0]) ** 2 + (a_hat[1] * e_hat[1]) ** 2 + (a_hat[2] * e_hat[2]) ** 2

            a_CM_shell = 3 * dcube / (4 * cp.pi) * (eps_NP_eb_shell - 1) / (
                    eps_NP_eb_shell + 2)  # Clausius-Mossotti of shell
            a_CM_core = 3 * dcube / (4 * cp.pi) * (eps_NP_eb_core - 1) / (
                    eps_NP_eb_core + 2)  # Clausius-Mossotti of core

            anr_shell = a_CM_shell / (
                    1 + (a_CM_shell / dcube) * (b1 + eps_NP_eb_shell * b2 + eps_NP_eb_shell * b3 * S) * (
                    (cp.linalg.norm(kvec) * sim.struct.d) ** 2))
            anr_core = a_CM_core / (
                    1 + (a_CM_core / dcube) * (b1 + eps_NP_eb_shell * b2 + eps_NP_eb_shell * b3 * S) * (
                    (cp.linalg.norm(kvec) * sim.struct.d) ** 2))

            aLDR_shell = (anr_shell / (
                    1 - 2 / 3 * 1j * (anr_shell / dcube) * ((cp.linalg.norm(kvec) * sim.struct.d) ** 3)))
            aLDR_core = (
                    anr_core / (1 - 2 / 3 * 1j * (anr_core / dcube) * ((cp.linalg.norm(kvec) * sim.struct.d) ** 3)))

            INDEX_IN_G[mask_core_G] *= 1 / aLDR_core
            INDEX_IN_G[mask_shell_G] *= 1 / (2 * aLDR_shell)

            Inverse_Alpha = INDEX_IN_G
        elif len(sim.struct.material) == 1:
            k0 = 2 * np.pi
            b1 = -1.891531
            b2 = 0.1648469
            b3 = -1.7700004
            dcube = sim.struct.d ** 3
            a_hat = kvec / np.linalg.norm(kvec)
            e_hat = E0 / np.linalg.norm(E0)
            S = 0

            for i in range(3):
                S = S + (a_hat[i] * e_hat[i]) ** 2
            S = (a_hat[0] * e_hat[0]) ** 2 + (a_hat[1] * e_hat[1]) ** 2 + (a_hat[2] * e_hat[2]) ** 2

            a_CM = 3 * dcube / (4 * np.pi) * (eps_NP_eb - 1) / (eps_NP_eb + 2)  # Clausius-Mossotti

            anr = a_CM / (
                    1 + (a_CM / dcube) * (b1 + eps_NP_eb * b2 + eps_NP_eb * b3 * S) * (
                    (np.linalg.norm(kvec) * sim.struct.d) ** 2))

            aLDR = (anr / (1 - 2 / 3 * 1j * (anr / dcube) * ((cp.linalg.norm(kvec) * sim.struct.d) ** 3)))

            Inverse_Alpha = 1 / aLDR * INDEX_IN_ALL_G

        ''' =============================== Interaction Matrix ========================================== '''
        A1 = (Exp_ikvec_rjk * ((cp.linalg.norm(kvec) ** 2) * rjkrjk1_I_G + ikvec_rjk * rjkrjk31_I_G))
        Axx = cp.reshape(A1, (sim.struct.geometry.Nx, sim.struct.geometry.Ny, sim.struct.geometry.Nz), order='F')
        Axx[0, 0, 0] = 0

        A2 = (Exp_ikvec_rjk * ((cp.linalg.norm(kvec) ** 2) * rjkrjk2_I_G + ikvec_rjk * rjkrjk32_I_G))
        Axy = cp.reshape(A2, (sim.struct.geometry.Nx, sim.struct.geometry.Ny, sim.struct.geometry.Nz), order='F')
        Axy[0, 0, 0] = 0

        A3 = (Exp_ikvec_rjk * ((cp.linalg.norm(kvec) ** 2) * rjkrjk3_I_G + ikvec_rjk * rjkrjk33_I_G))
        Axz = cp.reshape(A3, (sim.struct.geometry.Nx, sim.struct.geometry.Ny, sim.struct.geometry.Nz), order='F')
        Axz[0, 0, 0] = 0

        A4 = (Exp_ikvec_rjk * ((cp.linalg.norm(kvec) ** 2) * rjkrjk4_I_G + ikvec_rjk * rjkrjk34_I_G))
        Ayy = cp.reshape(A4, (sim.struct.geometry.Nx, sim.struct.geometry.Ny, sim.struct.geometry.Nz), order='F')
        Ayy[0, 0, 0] = 0

        A5 = (Exp_ikvec_rjk * ((cp.linalg.norm(kvec) ** 2) * rjkrjk5_I_G + ikvec_rjk * rjkrjk35_I_G))
        Ayz = cp.reshape(A5, (sim.struct.geometry.Nx, sim.struct.geometry.Ny, sim.struct.geometry.Nz), order='F')
        Ayz[0, 0, 0] = 0

        A6 = (Exp_ikvec_rjk * ((cp.linalg.norm(kvec) ** 2) * rjkrjk6_I_G + ikvec_rjk * rjkrjk36_I_G))
        Azz = cp.reshape(A6, (sim.struct.geometry.Nx, sim.struct.geometry.Ny, sim.struct.geometry.Nz), order='F')
        Azz[0, 0, 0] = 0
        Exp_ikvec_rjk = None
        mempool.free_all_blocks()

        AXX = cp.zeros(
            (2 * sim.struct.geometry.Nx - 1, 2 * sim.struct.geometry.Ny - 1, 2 * sim.struct.geometry.Nz - 1),
            dtype="complex64")
        AXY = cp.zeros(
            (2 * sim.struct.geometry.Nx - 1, 2 * sim.struct.geometry.Ny - 1, 2 * sim.struct.geometry.Nz - 1),
            dtype="complex64")
        AXZ = cp.zeros(
            (2 * sim.struct.geometry.Nx - 1, 2 * sim.struct.geometry.Ny - 1, 2 * sim.struct.geometry.Nz - 1),
            dtype="complex64")
        AYY = cp.zeros(
            (2 * sim.struct.geometry.Nx - 1, 2 * sim.struct.geometry.Ny - 1, 2 * sim.struct.geometry.Nz - 1),
            dtype="complex64")
        AYZ = cp.zeros(
            (2 * sim.struct.geometry.Nx - 1, 2 * sim.struct.geometry.Ny - 1, 2 * sim.struct.geometry.Nz - 1),
            dtype="complex64")
        AZZ = cp.zeros(
            (2 * sim.struct.geometry.Nx - 1, 2 * sim.struct.geometry.Ny - 1, 2 * sim.struct.geometry.Nz - 1),
            dtype="complex64")

        AXX[0:sim.struct.geometry.Nx, 0:sim.struct.geometry.Ny, 0:sim.struct.geometry.Nz] = Axx
        AXY[0:sim.struct.geometry.Nx, 0:sim.struct.geometry.Ny, 0:sim.struct.geometry.Nz] = Axy
        AXZ[0:sim.struct.geometry.Nx, 0:sim.struct.geometry.Ny, 0:sim.struct.geometry.Nz] = Axz
        AYY[0:sim.struct.geometry.Nx, 0:sim.struct.geometry.Ny, 0:sim.struct.geometry.Nz] = Ayy
        AYZ[0:sim.struct.geometry.Nx, 0:sim.struct.geometry.Ny, 0:sim.struct.geometry.Nz] = Ayz
        AZZ[0:sim.struct.geometry.Nx, 0:sim.struct.geometry.Ny, 0:sim.struct.geometry.Nz] = Azz

        # ====================== expending matrix to 2N-1 =====================
        AXX[sim.struct.geometry.Nx:2 * sim.struct.geometry.Nx - 1, 0:sim.struct.geometry.Ny,
        0:sim.struct.geometry.Nz] = Axx[sim.struct.geometry.Nx - 1:0:-1, 0:sim.struct.geometry.Ny,
                                    0:sim.struct.geometry.Nz]

        AXY[sim.struct.geometry.Nx:2 * sim.struct.geometry.Nx - 1, 0:sim.struct.geometry.Ny,
        0:sim.struct.geometry.Nz] = -Axy[sim.struct.geometry.Nx - 1:0:-1, 0:sim.struct.geometry.Ny,
                                     0:sim.struct.geometry.Nz]

        AXZ[sim.struct.geometry.Nx:2 * sim.struct.geometry.Nx - 1, 0:sim.struct.geometry.Ny,
        0:sim.struct.geometry.Nz] = -Axz[sim.struct.geometry.Nx - 1:0:-1, 0:sim.struct.geometry.Ny,
                                     0:sim.struct.geometry.Nz]

        AYY[sim.struct.geometry.Nx:2 * sim.struct.geometry.Nx - 1, 0:sim.struct.geometry.Ny,
        0:sim.struct.geometry.Nz] = Ayy[sim.struct.geometry.Nx - 1:0:-1, 0:sim.struct.geometry.Ny,
                                    0:sim.struct.geometry.Nz]

        AYZ[sim.struct.geometry.Nx:2 * sim.struct.geometry.Nx - 1, 0:sim.struct.geometry.Ny,
        0:sim.struct.geometry.Nz] = Ayz[sim.struct.geometry.Nx - 1:0:-1, 0:sim.struct.geometry.Ny,
                                    0:sim.struct.geometry.Nz]

        AZZ[sim.struct.geometry.Nx:2 * sim.struct.geometry.Nx - 1, 0:sim.struct.geometry.Ny,
        0:sim.struct.geometry.Nz] = Azz[sim.struct.geometry.Nx - 1:0:-1, 0:sim.struct.geometry.Ny,
                                    0:sim.struct.geometry.Nz]

        # Calculating FFT in X-direction #
        AXX[:, 0:sim.struct.geometry.Ny, 0:sim.struct.geometry.Nz] = cp.fft.fft(
            AXX[:, 0:sim.struct.geometry.Ny, 0:sim.struct.geometry.Nz], axis=0)

        AXY[:, 0:sim.struct.geometry.Ny, 0:sim.struct.geometry.Nz] = cp.fft.fft(
            AXY[:, 0:sim.struct.geometry.Ny, 0:sim.struct.geometry.Nz], axis=0)

        AXZ[:, 0:sim.struct.geometry.Ny, 0:sim.struct.geometry.Nz] = cp.fft.fft(
            AXZ[:, 0:sim.struct.geometry.Ny, 0:sim.struct.geometry.Nz], axis=0)

        AYY[:, 0:sim.struct.geometry.Ny, 0:sim.struct.geometry.Nz] = cp.fft.fft(
            AYY[:, 0:sim.struct.geometry.Ny, 0:sim.struct.geometry.Nz], axis=0)

        AYZ[:, 0:sim.struct.geometry.Ny, 0:sim.struct.geometry.Nz] = cp.fft.fft(
            AYZ[:, 0:sim.struct.geometry.Ny, 0:sim.struct.geometry.Nz], axis=0)

        AZZ[:, 0:sim.struct.geometry.Ny, 0:sim.struct.geometry.Nz] = cp.fft.fft(
            AZZ[:, 0:sim.struct.geometry.Ny, 0:sim.struct.geometry.Nz], axis=0)

        # Expanding dimension of each line in Y-direction to 2Ny-1 #
        AXX[:, sim.struct.geometry.Ny:2 * sim.struct.geometry.Ny - 1, 0:sim.struct.geometry.Nz] = AXX[:,
                                                                                                  sim.struct.geometry.Ny - 1:0:-1,
                                                                                                  0:sim.struct.geometry.Nz]

        AXY[:, sim.struct.geometry.Ny:2 * sim.struct.geometry.Ny - 1, 0:sim.struct.geometry.Nz] = -AXY[:,
                                                                                                   sim.struct.geometry.Ny - 1:0:-1,
                                                                                                   0:sim.struct.geometry.Nz]

        AXZ[:, sim.struct.geometry.Ny:2 * sim.struct.geometry.Ny - 1, 0:sim.struct.geometry.Nz] = AXZ[:,
                                                                                                  sim.struct.geometry.Ny - 1:0:-1,
                                                                                                  0:sim.struct.geometry.Nz]

        AYY[:, sim.struct.geometry.Ny:2 * sim.struct.geometry.Ny - 1, 0:sim.struct.geometry.Nz] = AYY[:,
                                                                                                  sim.struct.geometry.Ny - 1:0:-1,
                                                                                                  0:sim.struct.geometry.Nz]

        AYZ[:, sim.struct.geometry.Ny:2 * sim.struct.geometry.Ny - 1, 0:sim.struct.geometry.Nz] = -AYZ[:,
                                                                                                   sim.struct.geometry.Ny - 1:0:-1,
                                                                                                   0:sim.struct.geometry.Nz]

        AZZ[:, sim.struct.geometry.Ny:2 * sim.struct.geometry.Ny - 1, 0:sim.struct.geometry.Nz] = AZZ[:,
                                                                                                  sim.struct.geometry.Ny - 1:0:-1,
                                                                                                  0:sim.struct.geometry.Nz]

        # ========== Calculating FFT in Y-direction
        AXX[:, :, 0:sim.struct.geometry.Nz] = cp.fft.fft(AXX[:, :, 0:sim.struct.geometry.Nz], axis=1)
        AXY[:, :, 0:sim.struct.geometry.Nz] = cp.fft.fft(AXY[:, :, 0:sim.struct.geometry.Nz], axis=1)
        AXZ[:, :, 0:sim.struct.geometry.Nz] = cp.fft.fft(AXZ[:, :, 0:sim.struct.geometry.Nz], axis=1)
        AYY[:, :, 0:sim.struct.geometry.Nz] = cp.fft.fft(AYY[:, :, 0:sim.struct.geometry.Nz], axis=1)
        AYZ[:, :, 0:sim.struct.geometry.Nz] = cp.fft.fft(AYZ[:, :, 0:sim.struct.geometry.Nz], axis=1)
        AZZ[:, :, 0:sim.struct.geometry.Nz] = cp.fft.fft(AZZ[:, :, 0:sim.struct.geometry.Nz], axis=1)

        # Expanding dimension of each line in Z-direction to 2Nz-1#
        AXX[:, :, sim.struct.geometry.Nz:2 * sim.struct.geometry.Nz - 1] = AXX[:, :, sim.struct.geometry.Nz - 1:0:-1]

        AXY[:, :, sim.struct.geometry.Nz:2 * sim.struct.geometry.Nz - 1] = AXY[:, :, sim.struct.geometry.Nz - 1:0:-1]

        AXZ[:, :, sim.struct.geometry.Nz:2 * sim.struct.geometry.Nz - 1] = -AXZ[:, :, sim.struct.geometry.Nz - 1:0:-1]

        AYY[:, :, sim.struct.geometry.Nz:2 * sim.struct.geometry.Nz - 1] = AYY[:, :, sim.struct.geometry.Nz - 1:0:-1]

        AYZ[:, :, sim.struct.geometry.Nz:2 * sim.struct.geometry.Nz - 1] = -AYZ[:, :, sim.struct.geometry.Nz - 1:0:-1]

        AZZ[:, :, sim.struct.geometry.Nz:2 * sim.struct.geometry.Nz - 1] = AZZ[:, :, sim.struct.geometry.Nz - 1:0:-1]

        # ========== Calculating FFT in Y-direction
        FFT_AXX = cp.fft.fft(AXX[:, :, :], axis=2)
        AXX = None
        FFT_AXY = cp.fft.fft(AXY[:, :, :], axis=2)
        AXY = None
        FFT_AXZ = cp.fft.fft(AXZ[:, :, :], axis=2)
        AXZ = None
        FFT_AYY = cp.fft.fft(AYY[:, :, :], axis=2)
        AYY = None
        FFT_AYZ = cp.fft.fft(AYZ[:, :, :], axis=2)
        AYZ = None
        FFT_AZZ = cp.fft.fft(AZZ[:, :, :], axis=2)
        AZZ = None
        mempool.free_all_blocks()

        '''==============================Biconjugate_Gradient============================'''
        px = 0
        py = 0
        pz = 0

        Apkx = 0
        Apky = 0
        Apkz = 0

        # ============================ rk=E1-A*Pk;
        rkx = (E_x - Apkx)
        rky = (E_y - Apky)
        rkz = (E_z - Apkz)

        # ============================= qk=rk;
        qkx = rkx
        qky = rky
        qkz = rkz
        # ============================= qk_bar=conj(qk);

        Error = 1
        CT = 10 ** (-5)

        while Error > CT:
            FFT_qkx_Y = cp.fft.fft(qkx, 2 * sim.struct.geometry.Ny - 1, axis=1)
            FFT_qky_Y = cp.fft.fft(qky, 2 * sim.struct.geometry.Ny - 1, axis=1)
            FFT_qkz_Y = cp.fft.fft(qkz, 2 * sim.struct.geometry.Ny - 1, axis=1)

            # Calculate FFT in x-direction of the vectors in previous step
            FFT_qkx_x = np.fft.fft(FFT_qkx_Y, 2 * sim.struct.geometry.Nx - 1, axis=0)
            FFT_qkx_Y = None
            FFT_qky_x = np.fft.fft(FFT_qky_Y, 2 * sim.struct.geometry.Nx - 1, axis=0)
            FFT_qkx_Y = None
            FFT_qkz_x = np.fft.fft(FFT_qkz_Y, 2 * sim.struct.geometry.Nx - 1, axis=0)
            FFT_qkx_Y = None

            # Calculate FFT in z-direction of the vectors in previous step
            FFT_qkx_z = np.fft.fft(FFT_qkx_x, 2 * sim.struct.geometry.Nz - 1, axis=2)
            FFT_qkx_x = None
            FFT_qky_z = np.fft.fft(FFT_qky_x, 2 * sim.struct.geometry.Nz - 1, axis=2)
            FFT_qky_x = None
            FFT_qkz_z = np.fft.fft(FFT_qkz_x, 2 * sim.struct.geometry.Nz - 1, axis=2)
            FFT_qkz_x = None

            # Element-wise multiplicaition of FFT of matrix and FFT of vector
            FFT_APX = FFT_AXX * FFT_qkx_z + FFT_AXY * FFT_qky_z + FFT_AXZ * FFT_qkz_z
            FFT_APY = FFT_AXY * FFT_qkx_z + FFT_AYY * FFT_qky_z + FFT_AYZ * FFT_qkz_z
            FFT_APZ = FFT_AXZ * FFT_qkx_z + FFT_AYZ * FFT_qky_z + FFT_AZZ * FFT_qkz_z
            # del FFT_qkx_z,FFT_qky_z,FFT_qkz_z
            FFT_qkx_z = None
            FFT_qky_z = None
            FFT_qkz_z = None

            # IFFT IN Z DIRECTION
            IFFT_APX_Z = np.fft.ifft(FFT_APX, axis=2)
            FFT_APX = None
            IFFT_APY_Z = np.fft.ifft(FFT_APY, axis=2)
            FFT_APY = None
            IFFT_APZ_Z = np.fft.ifft(FFT_APZ, axis=2)
            FFT_Apz = None

            # IFFT IN X DIRCTION
            IFFT_APX_X = np.fft.ifft(IFFT_APX_Z[0:2 * sim.struct.geometry.Nx - 1, 0:2 * sim.struct.geometry.Ny - 1,
                                     0:sim.struct.geometry.Nz], axis=0)
            IFFT_APX_Z = None
            IFFT_APY_X = np.fft.ifft(IFFT_APY_Z[0:2 * sim.struct.geometry.Nx - 1, 0:2 * sim.struct.geometry.Ny - 1,
                                     0:sim.struct.geometry.Nz], axis=0)
            IFFT_APY_Z = None
            IFFT_APZ_X = np.fft.ifft(IFFT_APZ_Z[0:2 * sim.struct.geometry.Nx - 1, 0:2 * sim.struct.geometry.Ny - 1,
                                     0:sim.struct.geometry.Nz], axis=0)
            IFFT_APZ_Z = None

            # IFFT IN Y DIRCTION
            IFFT_APX = np.fft.ifft(
                IFFT_APX_X[0:sim.struct.geometry.Nx, 0:2 * sim.struct.geometry.Ny - 1, 0:sim.struct.geometry.Nz],
                axis=1)
            IFFT_APX_X = None
            IFFT_APY = np.fft.ifft(
                IFFT_APY_X[0:sim.struct.geometry.Nx, 0:2 * sim.struct.geometry.Ny - 1, 0:sim.struct.geometry.Nz],
                axis=1)
            IFFT_APY_X = None
            IFFT_APZ = np.fft.ifft(
                IFFT_APZ_X[0:sim.struct.geometry.Nx, 0:2 * sim.struct.geometry.Ny - 1, 0:sim.struct.geometry.Nz],
                axis=1)
            IFFT_APZ_X = None

            # Negating contribution of the dipoles outside the NPs boundary
            Aqkx = IFFT_APX[0:sim.struct.geometry.Nx, 0:sim.struct.geometry.Ny,
                   0:sim.struct.geometry.Nz] * INDEX_IN_ALL_G + Inverse_Alpha * qkx * INDEX_IN_ALL_G
            IFFT_APX = None
            Aqky = IFFT_APY[0:sim.struct.geometry.Nx, 0:sim.struct.geometry.Ny,
                   0:sim.struct.geometry.Nz] * INDEX_IN_ALL_G + Inverse_Alpha * qky * INDEX_IN_ALL_G
            IFFT_APY = None
            Aqkz = IFFT_APZ[0:sim.struct.geometry.Nx, 0:sim.struct.geometry.Ny,
                   0:sim.struct.geometry.Nz] * INDEX_IN_ALL_G + Inverse_Alpha * qkz * INDEX_IN_ALL_G
            IFFT_APZ = None
            mempool.free_all_blocks()

            rkx_vector = cp.reshape(rkx, sim.struct.geometry.N, order='F')
            rky_vector = cp.reshape(rky, sim.struct.geometry.N, order='F')
            rkz_vector = cp.reshape(rkz, sim.struct.geometry.N, order='F')

            Aqkx_vector = cp.reshape(Aqkx, sim.struct.geometry.N, order='F')
            Aqky_vector = cp.reshape(Aqky, sim.struct.geometry.N, order='F')
            Aqkz_vector = cp.reshape(Aqkz, sim.struct.geometry.N, order='F')

            qkx_vector = cp.reshape(qkx, sim.struct.geometry.N, order='F')
            qky_vector = cp.reshape(qky, sim.struct.geometry.N, order='F')
            qkz_vector = cp.reshape(qkz, sim.struct.geometry.N, order='F')

            alphak = (cp.dot(rkx_vector, rkx_vector) + cp.dot(rky_vector, rky_vector) + cp.dot(rkz_vector,
                                                                                               rkz_vector)) / (
                             cp.dot(qkx_vector, Aqkx_vector) + cp.dot(qky_vector, Aqky_vector) + cp.dot(qkz_vector,
                                                                                                        Aqkz_vector))

            # del Aqkx_vector,Aqky_vector,Aqkz_vector

            Aqkx_vector = None
            Aqky_vector = None
            Aqkz_vector = None

            rk0x = rkx
            rkx = None
            rk0y = rky
            rky = None
            rk0z = rkz
            rkz = None

            rk0x_vector = rkx_vector
            rkx_vector = None
            rk0y_vector = rky_vector
            rky_vector = None
            rk0z_vector = rkz_vector
            rkz_vector = None

            rkx = rk0x - alphak * Aqkx
            rk0x = None
            Aqkx = None
            rky = rk0y - alphak * Aqky
            rk0y = None
            Aqky = None
            rkz = rk0z - alphak * Aqkz
            rk0z = None
            Aqkz = None

            rkx_vector = cp.reshape(rkx, sim.struct.geometry.N, order='F')
            rky_vector = cp.reshape(rky, sim.struct.geometry.N, order='F')
            rkz_vector = cp.reshape(rkz, sim.struct.geometry.N, order='F')

            px = (px + alphak * qkx) * INDEX_IN_ALL_G
            py = (py + alphak * qky) * INDEX_IN_ALL_G
            pz = (pz + alphak * qkz) * INDEX_IN_ALL_G

            rk_vector = cp.hstack((rkx_vector, rky_vector, rkz_vector))
            Error = cp.linalg.norm(rk_vector) / cp.linalg.norm(E_vector)
            rk_vector = None

            betak = (cp.dot(rkx_vector, rkx_vector) + cp.dot(rky_vector, rky_vector) + cp.dot(rkz_vector,
                                                                                              rkz_vector)) / (
                            cp.dot(rk0x_vector, rk0x_vector) + cp.dot(rk0y_vector, rk0y_vector) + cp.dot(
                        rk0z_vector, rk0z_vector))

            # del rkx_vector,rky_vector,rkz_vector,rk0x_vector,rk0y_vector,rk0z_vector

            rkx_vector = None
            rky_vector = None
            rkz_vector = None
            rk0x_vector = None
            rk0y_vector = None
            rk0z_vector = None
            mempool.free_all_blocks()

            qkx = (rkx + betak * qkx) * INDEX_IN_ALL_G
            qky = (rky + betak * qky) * INDEX_IN_ALL_G
            qkz = (rkz + betak * qkz) * INDEX_IN_ALL_G

            px = px * INDEX_IN_ALL_G
            py = py * INDEX_IN_ALL_G
            pz = pz * INDEX_IN_ALL_G

        PX_vector = cp.reshape(px, sim.struct.geometry.N, order='F')
        PY_vector = cp.reshape(py, sim.struct.geometry.N, order='F')
        PZ_vector = cp.reshape(pz, sim.struct.geometry.N, order='F')
        Inv_Alpha = cp.reshape(Inverse_Alpha, sim.struct.geometry.N, order='F')
        Inv_Alpha_vec = cp.hstack((Inv_Alpha, Inv_Alpha, Inv_Alpha))

        E_vector = E_vector.flatten()
        P_vector = cp.hstack((PX_vector, PY_vector, PZ_vector))

        # computation time
        mid_time = time.time()
        time_end = mid_time - start_time
        print(f"{time_end:.6f}")

        Cabs, Cext, Csca = post_processing.section(P_vector, E_vector, Inv_Alpha_vec, kvec, sim)
        CABSLIST.append(Cabs)
        CEXTLIST.append(Cext)
        CSCALIST.append(Csca)

    data = np.column_stack((CEXTLIST, CABSLIST, CSCALIST))

    ## send someting to the CPU for post-processing

    px = cp.asnumpy(px)
    py = cp.asnumpy(py)
    pz = cp.asnumpy(pz)
    PX_vector = cp.asnumpy(PX_vector[0])
    PY_vector = cp.asnumpy(PY_vector[0])
    PZ_vector = cp.asnumpy(PZ_vector[0])

    return px, py, pz, PX_vector, PY_vector, PZ_vector, Inverse_Alpha, data


def DDA_WITHOUT_GPU(sim, rjkrjk1_I, rjkrjk2_I, rjkrjk3_I, rjkrjk4_I, rjkrjk5_I, rjkrjk6_I,
                    rjkrjk31_I, rjkrjk32_I, rjkrjk33_I, rjkrjk34_I, rjkrjk35_I, rjkrjk36_I,
                    RJK):
    start_time = time.time()

    if len(sim.struct.material) == 2:
        shell_eps = sim.struct.material[0].epsilon(sim)
        core_eps = sim.struct.material[1].epsilon(sim)

        mask_core_G = sim.struct.occupied.mask_core
        mask_shell_G = sim.struct.occupied.mask_shell
    else:
        eps = sim.struct.material[0].epsilon(sim)
        eps_np_eb = eps

    length = len(sim.field.wavelength)
    CABSLIST = []
    CSCALIST = []
    CEXTLIST = []

    ini_INDEX_IN = sim.struct.occupied.INDEX_IN
    for J in range(length):
        INDEX_IN = copy.deepcopy(ini_INDEX_IN)
        if len(sim.struct.material) == 2:
            eps_NP_eb_shell = shell_eps[J]
            eps_NP_eb_core = core_eps[J]
        elif len(sim.struct.material) == 1:
            eps_NP_eb = eps_np_eb[J]

        K0 = sim.field.K_kwargs[0]
        E0 = sim.field.E_kwargs[0]
        k = 2 * np.pi / sim.struct.material[0].wl * sim.struct.material[0].nb
        kvec = k[J] * K0

        Exp_ikvec_rjk = np.exp(1j * np.linalg.norm(kvec) * RJK) / RJK

        ikvec_rjk = (1j * np.linalg.norm(kvec) * RJK - 1) / (RJK ** 2)

        kr = kvec[0] * sim.struct.geometry.r_block[:, 0] + kvec[1] * sim.struct.geometry.r_block[:, 1] + kvec[
            2] * sim.struct.geometry.r_block[:, 2]

        '''=====================Incident_Field======================================'''

        if sim.field.field_type == "plane wave":

            expikr = np.exp(1j * kr)  # √
            Ex = E0[0] * expikr * sim.struct.occupied.INDEX_IN_ALL.flatten()
            Ey = E0[1] * expikr * sim.struct.occupied.INDEX_IN_ALL.flatten()
            Ez = E0[2] * expikr * sim.struct.occupied.INDEX_IN_ALL.flatten()

            E_vector = np.concatenate((Ex.ravel(), Ey.ravel(), Ez.ravel()))[:, np.newaxis]

            E_x = np.reshape(Ex, (sim.struct.geometry.Nx, sim.struct.geometry.Ny, sim.struct.geometry.Nz),
                             order='F')
            E_y = np.reshape(Ey, (sim.struct.geometry.Nx, sim.struct.geometry.Ny, sim.struct.geometry.Nz),
                             order='F')
            E_z = np.reshape(Ez, (sim.struct.geometry.Nx, sim.struct.geometry.Ny, sim.struct.geometry.Nz),
                             order='F')

        elif sim.field.field_type[0] == "gaussian":

            Waist_r = sim.field.field_type[1]
            z0 = sim.field.field_type[2]
            if K0[0] == 1:
                r = pow((sim.struct.geometry.r_block[:, 1] ** 2 + sim.struct.geometry.r_block[:, 2] ** 2), 0.5)
                z = sim.struct.geometry.r_block[:, 0]
            elif K0[1] == 1:
                r = pow((sim.struct.geometry.r_block[:, 0] ** 2 + sim.struct.geometry.r_block[:, 2] ** 2), 0.5)
                z = sim.struct.geometry.r_block_G[:, 1]
            elif K0[2] == 1:
                r = pow((sim.struct.geometry.r_block[:, 0] ** 2 + sim.struct.geometry.r_block[:, 2] ** 2), 0.5)
                z = sim.struct.geometry.r_block[:, 1]

            W0 = Waist_r * sim.field.wavelength[J]
            K = 2 * np.pi * sim.struct.material[0].nb
            zR = np.pi * pow(W0, 2) * sim.struct.material[0].nb / sim.field.wavelength[J]
            Qz = np.arctan((z - z0) / zR)

            Wz = W0 * (1 + pow(pow(((z - z0) / zR), 2), 0.5))

            Rz_inverse = z / (pow((z - z0), 2) + zR ** 2)

            E = (W0 / Wz) * np.exp(-pow((r / Wz), 2) - 1j * (K * (z - z0) + K * (r ** 2)) * Rz_inverse / 2 - Qz)
            Ex = E0[0] * E * sim.struct.occupied.INDEX_INSIDE_ALL.flatten()
            Ey = E0[1] * E * sim.struct.occupied.INDEX_INSIDE_ALL.flatten()
            Ez = E0[2] * E * sim.struct.occupied.INDEX_INSIDE_ALL.flatten()

            E_x = cp.reshape(Ex, (sim.struct.geometry.Nx, sim.struct.geometry.Ny, sim.struct.geometry.Nz),
                             order='F')
            E_y = cp.reshape(Ey, (sim.struct.geometry.Nx, sim.struct.geometry.Ny, sim.struct.geometry.Nz),
                             order='F')
            E_z = cp.reshape(Ez, (sim.struct.geometry.Nx, sim.struct.geometry.Ny, sim.struct.geometry.Nz),
                             order='F')

        '''=============================Polarizability=================================='''
        if len(sim.struct.material) == 1:
            k0 = 2 * np.pi
            b1 = -1.891531
            b2 = 0.1648469
            b3 = -1.7700004
            dcube = sim.struct.d ** 3
            a_hat = kvec / np.linalg.norm(kvec)
            e_hat = E0 / np.linalg.norm(E0)
            S = 0

            for i in range(3):
                S = S + (a_hat[i] * e_hat[i]) ** 2
            S = (a_hat[0] * e_hat[0]) ** 2 + (a_hat[1] * e_hat[1]) ** 2 + (a_hat[2] * e_hat[2]) ** 2

            a_CM = 3 * dcube / (4 * np.pi) * (eps_NP_eb - 1) / (eps_NP_eb + 2)  # Clausius-Mossotti

            anr = a_CM / (
                    1 + (a_CM / dcube) * (b1 + eps_NP_eb * b2 + eps_NP_eb * b3 * S) * (
                    (np.linalg.norm(kvec) * sim.struct.d) ** 2))

            aLDR = (anr / (1 - 2 / 3 * 1j * (anr / dcube) * ((np.linalg.norm(kvec) * sim.struct.d) ** 3)))

            Inverse_Alpha = 1 / aLDR * sim.struct.occupied.INDEX_IN_ALL
        elif len(sim.struct.material) == 2:
            k0 = 2 * np.pi
            b1 = -1.891531
            b2 = 0.1648469
            b3 = -1.7700004
            dcube = sim.struct.d ** 3
            a_hat = kvec / np.linalg.norm(kvec)
            e_hat = E0 / np.linalg.norm(E0)

            S = 0
            for i in range(3):
                S = S + (a_hat[i] * e_hat[i]) ** 2
            S = (a_hat[0] * e_hat[0]) ** 2 + (a_hat[1] * e_hat[1]) ** 2 + (a_hat[2] * e_hat[2]) ** 2

            a_CM_shell = 3 * dcube / (4 * cp.pi) * (eps_NP_eb_shell - 1) / (
                    eps_NP_eb_shell + 2)  # Clausius-Mossotti of shell
            a_CM_core = 3 * dcube / (4 * cp.pi) * (eps_NP_eb_core - 1) / (
                    eps_NP_eb_core + 2)  # Clausius-Mossotti of core

            anr_shell = a_CM_shell / (
                    1 + (a_CM_shell / dcube) * (b1 + eps_NP_eb_shell * b2 + eps_NP_eb_shell * b3 * S) * (
                    (cp.linalg.norm(kvec) * sim.struct.d) ** 2))
            anr_core = a_CM_core / (
                    1 + (a_CM_core / dcube) * (b1 + eps_NP_eb_core * b2 + eps_NP_eb_core * b3 * S) * (
                    (cp.linalg.norm(kvec) * sim.struct.d) ** 2))

            aLDR_shell = (anr_shell / (
                    1 - 2 / 3 * 1j * (anr_shell / dcube) * ((cp.linalg.norm(kvec) * sim.struct.d) ** 3)))
            aLDR_core = (
                    anr_core / (1 - 2 / 3 * 1j * (anr_core / dcube) * ((cp.linalg.norm(kvec) * sim.struct.d) ** 3)))

            INDEX_IN[mask_core_G] *= 1 / aLDR_core
            INDEX_IN[mask_shell_G] *= 1 / (2 * aLDR_shell)

            Inverse_Alpha = INDEX_IN


        '''============================Interaction_Matrix==========================='''

        A1 = (Exp_ikvec_rjk * ((np.linalg.norm(kvec) ** 2) * rjkrjk1_I + ikvec_rjk * rjkrjk31_I))
        Axx = np.reshape(A1, [sim.struct.geometry.Nx, sim.struct.geometry.Ny, sim.struct.geometry.Nz], order='F')
        Axx[0, 0, 0] = 0
        A1 = None

        A2 = (Exp_ikvec_rjk * ((np.linalg.norm(kvec) ** 2) * rjkrjk2_I + ikvec_rjk * rjkrjk32_I))
        Axy = np.reshape(A2, [sim.struct.geometry.Nx, sim.struct.geometry.Ny, sim.struct.geometry.Nz], order='F')
        Axy[0, 0, 0] = 0
        A2 = None

        A3 = (Exp_ikvec_rjk * ((np.linalg.norm(kvec) ** 2) * rjkrjk3_I + ikvec_rjk * rjkrjk33_I))
        Axz = np.reshape(A3, [sim.struct.geometry.Nx, sim.struct.geometry.Ny, sim.struct.geometry.Nz], order='F')
        Axz[0, 0, 0] = 0
        A3 = None

        A4 = (Exp_ikvec_rjk * ((np.linalg.norm(kvec) ** 2) * rjkrjk4_I + ikvec_rjk * rjkrjk34_I))
        Ayy = np.reshape(A4, [sim.struct.geometry.Nx, sim.struct.geometry.Ny, sim.struct.geometry.Nz], order='F')
        A4 = None

        A5 = (Exp_ikvec_rjk * ((np.linalg.norm(kvec) ** 2) * rjkrjk5_I + ikvec_rjk * rjkrjk35_I))
        Ayz = np.reshape(A5, [sim.struct.geometry.Nx, sim.struct.geometry.Ny, sim.struct.geometry.Nz], order='F')
        Ayz[0, 0, 0] = 0
        A5 = None

        A6 = (Exp_ikvec_rjk * ((np.linalg.norm(kvec) ** 2) * rjkrjk6_I + ikvec_rjk * rjkrjk36_I))
        Azz = np.reshape(A6, [sim.struct.geometry.Nx, sim.struct.geometry.Ny, sim.struct.geometry.Nz], order='F')
        Azz[0, 0, 0] = 0
        A6 = None
        Exp_ikvec_rjk = None

        '''
        ====================FFT_interaction================
        '''

        AXX = np.zeros(
            (2 * sim.struct.geometry.Nx - 1, 2 * sim.struct.geometry.Ny - 1, 2 * sim.struct.geometry.Nz - 1),
            dtype="complex64")
        AXY = np.zeros(
            (2 * sim.struct.geometry.Nx - 1, 2 * sim.struct.geometry.Ny - 1, 2 * sim.struct.geometry.Nz - 1),
            dtype="complex64")
        AXZ = np.zeros(
            (2 * sim.struct.geometry.Nx - 1, 2 * sim.struct.geometry.Ny - 1, 2 * sim.struct.geometry.Nz - 1),
            dtype="complex64")
        AYY = np.zeros(
            (2 * sim.struct.geometry.Nx - 1, 2 * sim.struct.geometry.Ny - 1, 2 * sim.struct.geometry.Nz - 1),
            dtype="complex64")
        AYZ = np.zeros(
            (2 * sim.struct.geometry.Nx - 1, 2 * sim.struct.geometry.Ny - 1, 2 * sim.struct.geometry.Nz - 1),
            dtype="complex64")
        AZZ = np.zeros(
            (2 * sim.struct.geometry.Nx - 1, 2 * sim.struct.geometry.Ny - 1, 2 * sim.struct.geometry.Nz - 1),
            dtype="complex64")

        AXX[0:sim.struct.geometry.Nx, 0:sim.struct.geometry.Ny, 0:sim.struct.geometry.Nz] = Axx
        AXY[0:sim.struct.geometry.Nx, 0:sim.struct.geometry.Ny, 0:sim.struct.geometry.Nz] = Axy
        AXZ[0:sim.struct.geometry.Nx, 0:sim.struct.geometry.Ny, 0:sim.struct.geometry.Nz] = Axz
        AYY[0:sim.struct.geometry.Nx, 0:sim.struct.geometry.Ny, 0:sim.struct.geometry.Nz] = Ayy
        AYZ[0:sim.struct.geometry.Nx, 0:sim.struct.geometry.Ny, 0:sim.struct.geometry.Nz] = Ayz
        AZZ[0:sim.struct.geometry.Nx, 0:sim.struct.geometry.Ny, 0:sim.struct.geometry.Nz] = Azz

        # Expanding dimension of each line in X-direction to 2Nx-1#
        # Axx.resize((2 * Nx - 1, 2 * Ny - 1, 2 * Nz - 1), refcheck=False)
        AXX[sim.struct.geometry.Nx:2 * sim.struct.geometry.Nx - 1, 0:sim.struct.geometry.Ny,
        0:sim.struct.geometry.Nz] = Axx[sim.struct.geometry.Nx - 1:0:-1, 0:sim.struct.geometry.Ny,
                                    0:sim.struct.geometry.Nz]

        AXY[sim.struct.geometry.Nx:2 * sim.struct.geometry.Nx - 1, 0:sim.struct.geometry.Ny,
        0:sim.struct.geometry.Nz] = -Axy[sim.struct.geometry.Nx - 1:0:-1, 0:sim.struct.geometry.Ny,
                                     0:sim.struct.geometry.Nz]

        AXZ[sim.struct.geometry.Nx:2 * sim.struct.geometry.Nx - 1, 0:sim.struct.geometry.Ny,
        0:sim.struct.geometry.Nz] = -Axz[sim.struct.geometry.Nx - 1:0:-1, 0:sim.struct.geometry.Ny,
                                     0:sim.struct.geometry.Nz]

        AYY[sim.struct.geometry.Nx:2 * sim.struct.geometry.Nx - 1, 0:sim.struct.geometry.Ny,
        0:sim.struct.geometry.Nz] = Ayy[sim.struct.geometry.Nx - 1:0:-1, 0:sim.struct.geometry.Ny,
                                    0:sim.struct.geometry.Nz]

        AYZ[sim.struct.geometry.Nx:2 * sim.struct.geometry.Nx - 1, 0:sim.struct.geometry.Ny,
        0:sim.struct.geometry.Nz] = Ayz[sim.struct.geometry.Nx - 1:0:-1, 0:sim.struct.geometry.Ny,
                                    0:sim.struct.geometry.Nz]

        AZZ[sim.struct.geometry.Nx:2 * sim.struct.geometry.Nx - 1, 0:sim.struct.geometry.Ny,
        0:sim.struct.geometry.Nz] = Azz[sim.struct.geometry.Nx - 1:0:-1, 0:sim.struct.geometry.Ny,
                                    0:sim.struct.geometry.Nz]

        # Calculating FFT in X-direction#

        AXX[:, 0:sim.struct.geometry.Ny, 0:sim.struct.geometry.Nz] = np.fft.fft(
            AXX[:, 0:sim.struct.geometry.Ny, 0:sim.struct.geometry.Nz], axis=0)

        AXY[:, 0:sim.struct.geometry.Ny, 0:sim.struct.geometry.Nz] = np.fft.fft(
            AXY[:, 0:sim.struct.geometry.Ny, 0:sim.struct.geometry.Nz], axis=0)

        AXZ[:, 0:sim.struct.geometry.Ny, 0:sim.struct.geometry.Nz] = np.fft.fft(
            AXZ[:, 0:sim.struct.geometry.Ny, 0:sim.struct.geometry.Nz], axis=0)

        AYY[:, 0:sim.struct.geometry.Ny, 0:sim.struct.geometry.Nz] = np.fft.fft(
            AYY[:, 0:sim.struct.geometry.Ny, 0:sim.struct.geometry.Nz], axis=0)

        AYZ[:, 0:sim.struct.geometry.Ny, 0:sim.struct.geometry.Nz] = np.fft.fft(
            AYZ[:, 0:sim.struct.geometry.Ny, 0:sim.struct.geometry.Nz], axis=0)

        AZZ[:, 0:sim.struct.geometry.Ny, 0:sim.struct.geometry.Nz] = np.fft.fft(
            AZZ[:, 0:sim.struct.geometry.Ny, 0:sim.struct.geometry.Nz], axis=0)

        # Expanding dimension of each line in Y-direction to 2Ny-1#
        AXX[:, sim.struct.geometry.Ny:2 * sim.struct.geometry.Ny - 1, 0:sim.struct.geometry.Nz] = AXX[:,
                                                                                                  sim.struct.geometry.Ny - 1:0:-1,
                                                                                                  0:sim.struct.geometry.Nz]

        AXY[:, sim.struct.geometry.Ny:2 * sim.struct.geometry.Ny - 1, 0:sim.struct.geometry.Nz] = -AXY[:,
                                                                                                   sim.struct.geometry.Ny - 1:0:-1,
                                                                                                   0:sim.struct.geometry.Nz]

        AXZ[:, sim.struct.geometry.Ny:2 * sim.struct.geometry.Ny - 1, 0:sim.struct.geometry.Nz] = AXZ[:,
                                                                                                  sim.struct.geometry.Ny - 1:0:-1,
                                                                                                  0:sim.struct.geometry.Nz]

        AYY[:, sim.struct.geometry.Ny:2 * sim.struct.geometry.Ny - 1, 0:sim.struct.geometry.Nz] = AYY[:,
                                                                                                  sim.struct.geometry.Ny - 1:0:-1,
                                                                                                  0:sim.struct.geometry.Nz]

        AYZ[:, sim.struct.geometry.Ny:2 * sim.struct.geometry.Ny - 1, 0:sim.struct.geometry.Nz] = -AYZ[:,
                                                                                                   sim.struct.geometry.Ny - 1:0:-1,
                                                                                                   0:sim.struct.geometry.Nz]

        AZZ[:, sim.struct.geometry.Ny:2 * sim.struct.geometry.Ny - 1, 0:sim.struct.geometry.Nz] = AZZ[:,
                                                                                                  sim.struct.geometry.Ny - 1:0:-1,
                                                                                                  0:sim.struct.geometry.Nz]

        # ========== Calculating FFT in Y-direction
        AXX[:, :, 0:sim.struct.geometry.Nz] = np.fft.fft(AXX[:, :, 0:sim.struct.geometry.Nz], axis=1)
        AXY[:, :, 0:sim.struct.geometry.Nz] = np.fft.fft(AXY[:, :, 0:sim.struct.geometry.Nz], axis=1)
        AXZ[:, :, 0:sim.struct.geometry.Nz] = np.fft.fft(AXZ[:, :, 0:sim.struct.geometry.Nz], axis=1)
        AYY[:, :, 0:sim.struct.geometry.Nz] = np.fft.fft(AYY[:, :, 0:sim.struct.geometry.Nz], axis=1)
        AYZ[:, :, 0:sim.struct.geometry.Nz] = np.fft.fft(AYZ[:, :, 0:sim.struct.geometry.Nz], axis=1)
        AZZ[:, :, 0:sim.struct.geometry.Nz] = np.fft.fft(AZZ[:, :, 0:sim.struct.geometry.Nz], axis=1)

        # Expanding dimension of each line in Z-direction to 2Nz-1#
        AXX[:, :, sim.struct.geometry.Nz:2 * sim.struct.geometry.Nz - 1] = AXX[:, :, sim.struct.geometry.Nz - 1:0:-1]

        AXY[:, :, sim.struct.geometry.Nz:2 * sim.struct.geometry.Nz - 1] = AXY[:, :, sim.struct.geometry.Nz - 1:0:-1]

        AXZ[:, :, sim.struct.geometry.Nz:2 * sim.struct.geometry.Nz - 1] = -AXZ[:, :, sim.struct.geometry.Nz - 1:0:-1]

        AYY[:, :, sim.struct.geometry.Nz:2 * sim.struct.geometry.Nz - 1] = AYY[:, :, sim.struct.geometry.Nz - 1:0:-1]

        AYZ[:, :, sim.struct.geometry.Nz:2 * sim.struct.geometry.Nz - 1] = -AYZ[:, :, sim.struct.geometry.Nz - 1:0:-1]

        AZZ[:, :, sim.struct.geometry.Nz:2 * sim.struct.geometry.Nz - 1] = AZZ[:, :, sim.struct.geometry.Nz - 1:0:-1]

        FFT_AXX = np.fft.fft(AXX[:, :, :], axis=2)
        AXX = None
        FFT_AXY = np.fft.fft(AXY[:, :, :], axis=2)
        AXY = None
        FFT_AXZ = np.fft.fft(AXZ[:, :, :], axis=2)
        AXZ = None
        FFT_AYY = np.fft.fft(AYY[:, :, :], axis=2)
        AYY = None
        FFT_AYZ = np.fft.fft(AYZ[:, :, :], axis=2)
        AYZ = None
        FFT_AZZ = np.fft.fft(AZZ[:, :, :], axis=2)
        AZZ = None

        '''==============================Biconjugate_Gradient============================'''
        px = 0
        py = 0
        pz = 0

        Apkx = 0
        Apky = 0
        Apkz = 0

        # ============================ rk=E1-A*Pk;
        rkx = (E_x - Apkx)
        rky = (E_y - Apky)
        rkz = (E_z - Apkz)

        # ============================= qk=rk;
        qkx = rkx
        qky = rky
        qkz = rkz
        # ============================= qk_bar=conj(qk);

        Error = 1
        CT = 10 ** (-5)

        while Error > CT:
            '''
                ===================================Ifft===================================
            '''

            FFT_qkx_Y = np.fft.fft(qkx, 2 * sim.struct.geometry.Ny - 1, axis=1)
            FFT_qky_Y = np.fft.fft(qky, 2 * sim.struct.geometry.Ny - 1, axis=1)
            FFT_qkz_Y = np.fft.fft(qkz, 2 * sim.struct.geometry.Ny - 1, axis=1)

            # Calculate FFT in x-direction of the vectors in previous step
            FFT_qkx_x = np.fft.fft(FFT_qkx_Y, 2 * sim.struct.geometry.Nx - 1, axis=0)
            # del FFT_qkx_Y
            FFT_qky_x = np.fft.fft(FFT_qky_Y, 2 * sim.struct.geometry.Nx - 1, axis=0)
            # del FFT_qky_Y
            FFT_qkz_x = np.fft.fft(FFT_qkz_Y, 2 * sim.struct.geometry.Nx - 1, axis=0)
            # del FFT_qkz_Y

            # Calculate FFT in z-direction of the vectors in previous step
            FFT_qkx_z = np.fft.fft(FFT_qkx_x, 2 * sim.struct.geometry.Nz - 1, axis=2)
            # del FFT_qkx_x
            FFT_qky_z = np.fft.fft(FFT_qky_x, 2 * sim.struct.geometry.Nz - 1, axis=2)
            # del FFT_qky_x
            FFT_qkz_z = np.fft.fft(FFT_qkz_x, 2 * sim.struct.geometry.Nz - 1, axis=2)
            # del FFT_qkz_x

            # Element-wise multiplicaition of FFT of matrix and FFT of vector
            FFT_APX = FFT_AXX * FFT_qkx_z + FFT_AXY * FFT_qky_z + FFT_AXZ * FFT_qkz_z
            FFT_APY = FFT_AXY * FFT_qkx_z + FFT_AYY * FFT_qky_z + FFT_AYZ * FFT_qkz_z
            FFT_APZ = FFT_AXZ * FFT_qkx_z + FFT_AYZ * FFT_qky_z + FFT_AZZ * FFT_qkz_z
            # del FFT_qkx_z,FFT_qky_z,FFT_qkz_z

            # IFFT IN Z DIRECTION
            IFFT_APX_Z = np.fft.ifft(FFT_APX, axis=2)
            # del FFT_APX
            IFFT_APY_Z = np.fft.ifft(FFT_APY, axis=2)
            # del FFT_APY
            IFFT_APZ_Z = np.fft.ifft(FFT_APZ, axis=2)
            # del FFT_APZ

            # IFFT IN X DIRCTION
            IFFT_APX_X = np.fft.ifft(IFFT_APX_Z[0:2 * sim.struct.geometry.Nx - 1, 0:2 * sim.struct.geometry.Ny - 1,
                                     0:sim.struct.geometry.Nz], axis=0)
            # del IFFT_APX_Z
            IFFT_APY_X = np.fft.ifft(IFFT_APY_Z[0:2 * sim.struct.geometry.Nx - 1, 0:2 * sim.struct.geometry.Ny - 1,
                                     0:sim.struct.geometry.Nz], axis=0)
            # del IFFT_APY_Z
            IFFT_APZ_X = np.fft.ifft(IFFT_APZ_Z[0:2 * sim.struct.geometry.Nx - 1, 0:2 * sim.struct.geometry.Ny - 1,
                                     0:sim.struct.geometry.Nz], axis=0)
            # del IFFT_APZ_Z

            # IFFT IN Y DIRCTION
            IFFT_APX = np.fft.ifft(
                IFFT_APX_X[0:sim.struct.geometry.Nx, 0:2 * sim.struct.geometry.Ny - 1, 0:sim.struct.geometry.Nz],
                axis=1)
            # del IFFT_APX_X
            IFFT_APY = np.fft.ifft(
                IFFT_APY_X[0:sim.struct.geometry.Nx, 0:2 * sim.struct.geometry.Ny - 1, 0:sim.struct.geometry.Nz],
                axis=1)
            # del IFFT_APY_X
            IFFT_APZ = np.fft.ifft(
                IFFT_APZ_X[0:sim.struct.geometry.Nx, 0:2 * sim.struct.geometry.Ny - 1, 0:sim.struct.geometry.Nz],
                axis=1)
            # del IFFT_APZ_X

            # Negating contribution of the dipoles outside the NPs boundary
            Aqkx = IFFT_APX[0:sim.struct.geometry.Nx, 0:sim.struct.geometry.Ny,
                   0:sim.struct.geometry.Nz] * sim.struct.occupied.INDEX_IN_ALL + Inverse_Alpha * qkx * sim.struct.occupied.INDEX_IN_ALL
            # del IFFT_APX
            Aqky = IFFT_APY[0:sim.struct.geometry.Nx, 0:sim.struct.geometry.Ny,
                   0:sim.struct.geometry.Nz] * sim.struct.occupied.INDEX_IN_ALL + Inverse_Alpha * qky * sim.struct.occupied.INDEX_IN_ALL
            # del IFFT_APY
            Aqkz = IFFT_APZ[0:sim.struct.geometry.Nx, 0:sim.struct.geometry.Ny,
                   0:sim.struct.geometry.Nz] * sim.struct.occupied.INDEX_IN_ALL + Inverse_Alpha * qkz * sim.struct.occupied.INDEX_IN_ALL
            # del IFFT_APZ

            rkx_vector = np.reshape(rkx, sim.struct.geometry.N, order='F')
            rky_vector = np.reshape(rky, sim.struct.geometry.N, order='F')
            rkz_vector = np.reshape(rkz, sim.struct.geometry.N, order='F')

            Aqkx_vector = np.reshape(Aqkx, sim.struct.geometry.N, order='F')
            Aqky_vector = np.reshape(Aqky, sim.struct.geometry.N, order='F')
            Aqkz_vector = np.reshape(Aqkz, sim.struct.geometry.N, order='F')

            qkx_vector = np.reshape(qkx, sim.struct.geometry.N, order='F')
            qky_vector = np.reshape(qky, sim.struct.geometry.N, order='F')
            qkz_vector = np.reshape(qkz, sim.struct.geometry.N, order='F')

            alphak = (np.dot(rkx_vector, rkx_vector) + np.dot(rky_vector, rky_vector) + np.dot(rkz_vector,
                                                                                               rkz_vector)) / (
                             np.dot(qkx_vector, Aqkx_vector) + np.dot(qky_vector, Aqky_vector) + np.dot(qkz_vector,
                                                                                                        Aqkz_vector))

            # del Aqkx_vector,Aqky_vector,Aqkz_vector

            rk0x = rkx
            # del rkx
            rk0y = rky
            # del rky
            rk0z = rkz
            # del rkz

            rk0x_vector = rkx_vector
            # del rkx_vector
            rk0y_vector = rky_vector
            # del rky_vector
            rk0z_vector = rkz_vector
            # del rkz_vector

            rkx = rk0x - alphak * Aqkx
            # del rk0x,Aqkx
            rky = rk0y - alphak * Aqky
            # del rk0y,Aqky
            rkz = rk0z - alphak * Aqkz
            # del rk0z,Aqkz

            rkx_vector = np.reshape(rkx, sim.struct.geometry.N, order='F')
            rky_vector = np.reshape(rky, sim.struct.geometry.N, order='F')
            rkz_vector = np.reshape(rkz, sim.struct.geometry.N, order='F')

            px = (px + alphak * qkx) * sim.struct.occupied.INDEX_IN_ALL
            py = (py + alphak * qky) * sim.struct.occupied.INDEX_IN_ALL
            pz = (pz + alphak * qkz) * sim.struct.occupied.INDEX_IN_ALL

            rk_vector = np.hstack((rkx_vector, rky_vector, rkz_vector))
            Error = np.linalg.norm(rk_vector) / np.linalg.norm(E_vector)
            # del rk_vector

            betak = (np.dot(rkx_vector, rkx_vector) + np.dot(rky_vector, rky_vector) + np.dot(rkz_vector,
                                                                                              rkz_vector)) / (
                            np.dot(rk0x_vector, rk0x_vector) + np.dot(rk0y_vector, rk0y_vector) + np.dot(
                        rk0z_vector, rk0z_vector))

            # del rkx_vector,rky_vector,rkz_vector,rk0x_vector,rk0y_vector,rk0z_vector

            qkx = (rkx + betak * qkx) * sim.struct.occupied.INDEX_IN_ALL
            qky = (rky + betak * qky) * sim.struct.occupied.INDEX_IN_ALL
            qkz = (rkz + betak * qkz) * sim.struct.occupied.INDEX_IN_ALL

            px = px * sim.struct.occupied.INDEX_IN_ALL
            py = py * sim.struct.occupied.INDEX_IN_ALL
            pz = pz * sim.struct.occupied.INDEX_IN_ALL

        PX_vector = np.reshape(px, sim.struct.geometry.N, order='F')
        PY_vector = np.reshape(py, sim.struct.geometry.N, order='F')
        PZ_vector = np.reshape(pz, sim.struct.geometry.N, order='F')
        Inv_Alpha = np.reshape(Inverse_Alpha, sim.struct.geometry.N, order='F')
        Inv_Alpha_vec = np.hstack((Inv_Alpha, Inv_Alpha, Inv_Alpha))

        E_vector = E_vector.flatten()
        P_vector = np.hstack((PX_vector, PY_vector, PZ_vector))

        # computation time
        mid_time = time.time()
        time_end = mid_time - start_time
        print(f"{time_end:.6f}")

        Cabs, Cext, Csca = post_processing.section(P_vector, E_vector, Inv_Alpha_vec, kvec, sim)
        CABSLIST.append(Cabs)
        CEXTLIST.append(Cext)
        CSCALIST.append(Csca)

    data = np.column_stack((CEXTLIST, CABSLIST, CSCALIST))

    return px, py, pz, PX_vector, PY_vector, PZ_vector,Inverse_Alpha, data
