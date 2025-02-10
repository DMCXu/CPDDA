import numpy as np
import cupy as cp
import scipy


def section(Pvector, Evector, InvAlphavec, kvec, sim):
    """
     ## Calculate the section of the particle ##

    sim: simulation object of CPDDA

    :return: section
    """

    if isinstance(Pvector, cp.ndarray):
        Pvector = cp.asnumpy(Pvector)
        Evector = cp.asnumpy(Evector)
        InvAlphavec = cp.asnumpy(InvAlphavec)
        kvec = cp.asnumpy(kvec)
    else:
        pass

    E0 = sim.field.E_kwargs[0]
    Cabs = 4 * np.pi * np.linalg.norm(kvec) / sum(abs(E0 ** 2)) * ((
            np.imag(np.vdot((np.conj(Pvector)), np.conj(Pvector * InvAlphavec))) - 2 / 3 * np.linalg.norm(
        kvec) ** 3 * (np.linalg.norm(Pvector) ** 2)))

    Cext = 4 * np.pi * np.linalg.norm(kvec) / sum(abs(E0 ** 2)) * np.imag(np.vdot(Evector, Pvector))

    Csca = Cext - Cabs

    return Cabs, Cext, Csca


def efficiency(Cabs, Cext, Cscat, sim):
    """
        ## Calculate the efficiency of the particle ##

        data: array of section

        :return: efficiency
    """

    Q_ABS = Cabs / (np.pi * ((sim.struct.geometry.d_eff / 2) ** 2))
    Q_EXT = Cext / (np.pi * ((sim.struct.geometry.d_eff / 2) ** 2))
    Q_SCA = Cscat / (np.pi * ((sim.struct.geometry.d_eff / 2) ** 2))

    return Q_EXT, Q_ABS, Q_SCA


def vol_coe(data, sim):
    """
    ## Calculate the volume factor of the particle ##

    data: array of section

    sim: simulation object of CPDDA

    :return: volume coefficient
    """
    Cext = data[:, 0]
    Cabs = data[:, 1]
    Csca = data[:, 2]
    V_rod = np.pi * (sim.struct.geometry.Lz - sim.struct.geometry.Lx) * (
                (sim.struct.geometry.Lx / 2) ** 2) + 4 / 3 * np.pi * pow(sim.struct.geometry.Lx / 2, 3)
    Aabs = Cabs / V_rod
    Asca = Csca / V_rod
    Aext = Cext / V_rod

    return Aext, Aabs, Asca


def enhancement(sim, PX_vector, PY_vector, PZ_vector, Inverse_Alpha, px, py, pz, plane_2D, IB, Np_shape):
    """
    ## Calculating enhanced E-field of particle ##

    More recommended for single wavelength

    ==============Not fully tested==================

    """
    k = 2 * np.pi / sim.struct.material[0].wl * sim.struct.material[0].nb
    E0 = sim.field.E_kwargs[0]
    K0 = sim.field.K_kwargs[0]
    kvec = k * K0
    Nx_target = sim.struct.geometry.Nx
    Ny_target = sim.struct.geometry.Ny
    Nz_target = sim.struct.geometry.Nz
    Xrectan = sim.struct.geometry.X
    Yrectan = sim.struct.geometry.Y
    Zrectan = sim.struct.geometry.Z
    """=========Two_D_plane========="""
    dx_ = dy_ = dz_ = 0.5
    if plane_2D == "xy":
        X1 = np.arange(-round(0.5 * sim.struct.geometry.Lx / dx_ + sim.struct.geometry.Lx / (2 * dx_)),
                       round(0.5 * sim.struct.geometry.Lx / dx_ + sim.struct.geometry.Lx / (2 * dx_)) + 1)
        X_extend = X1 * dx_
        Y1 = np.arange(-round(0.5 * sim.struct.geometry.Ly / dy_ + sim.struct.geometry.Ly / (2 * dy_)),
                       round(0.5 * sim.struct.geometry.Ly / dy_ + sim.struct.geometry.Ly / (2 * dy_)) + 1)
        Y_extend = Y1 * dy_
        Z1 = np.arange(-round(sim.struct.geometry.Lz / (2 * dz_)), round(sim.struct.geometry.Lz / (2 * dz_)) + 1)
        Z_extend = Z1 * dz_

        Yex, Xex = np.meshgrid(Y_extend, X_extend)

        SIZE = Xex.shape
        Zex = np.zeros((SIZE[0], SIZE[1]))
        N_plane = SIZE[0] * SIZE[1]

        x_plane = np.reshape(Xex, [N_plane, 1], order='F')
        y_plane = np.reshape(Yex, [N_plane, 1], order='F')
        z_plane = np.reshape(Zex, [N_plane, 1], order='F')

        y_rec, x_rec, z_rec = np.meshgrid(Y_extend, X_extend, Z_extend)

    elif plane_2D == "xz":
        X1 = np.arange(-round(0.5 * sim.struct.geometry.Lx / dx_ + sim.struct.geometry.Lx / (2 * dx_)),
                       round(0.5 * sim.struct.geometry.Lx / dx_ + sim.struct.geometry.Lx / (2 * dx_)) + 1)
        X_extend = X1 * dx_
        Y1 = np.arange(-round(sim.struct.geometry.Ly / (2 * dy_)), round(sim.struct.geometry.Ly / (2 * dy_)) + 1)
        Y_extend = Y1 * dy_
        Z1 = np.arange(-round(0.5 * sim.struct.geometry.Lz / dz_ + sim.struct.geometry.Lz / (2 * dz_)),
                       round(0.5 * sim.struct.geometry.Lz / dz_ + sim.struct.geometry.Lz / (2 * dz_)) + 1)
        Z_extend = Z1 * dz_

        Zex, Xex = np.meshgrid(Z_extend, X_extend)
        SIZE = Xex.shape

        Yex = np.zeros((SIZE[0], SIZE[1]))
        N_plane = SIZE[0] * SIZE[1]

        x_plane = np.reshape(Xex, [N_plane, 1], order='F')
        y_plane = np.reshape(Yex, [N_plane, 1], order='F')
        z_plane = np.reshape(Zex, [N_plane, 1], order='F')

        y_rec, x_rec, z_rec = np.meshgrid(Y_extend, X_extend, Z_extend)

    r_plane = np.hstack((x_plane, y_plane, z_plane))
    Nx = len(X_extend)
    Ny = len(Y_extend)
    Nz = len(Z_extend)
    N = Nx * Ny * Nz
    X = np.reshape(x_rec, [N, 1], order='F')
    Y = np.reshape(y_rec, [N, 1], order='F')
    Z = np.reshape(z_rec, [N, 1], order='F')

    r_block = np.hstack((X, Y, Z))

    if sim.field.field_type[0] == "plane wave":
        kr = kvec[0] * r_plane[:, 0] + kvec[1] * r_plane[:, 1] + kvec[2] * r_plane[:, 2]
        expikr = np.exp(1j * kr)

        Ex_incident = E0[0] * expikr
        Ey_incident = E0[1] * expikr
        Ez_incident = E0[2] * expikr

    elif sim.field.field_type[0] == "gaussian":
        Waist_r = sim.field.field_type[1]
        z0 = sim.field.field_type[2]
        if K0[0] == 1:
            r = pow((r_block[:, 1] ** 2 + r_block[:, 2] ** 2), 0.5)
            z = r_block[:, 0]
        elif K0[1] == 1:
            r = pow((r_block[:, 0] ** 2 + r_block[:, 2] ** 2), 0.5)
            z = r_block[:, 1]
        elif K0[2] == 1:
            r = pow((r_block[:, 0] ** 2 + r_block[:, 2] ** 2), 0.5)
            z = r_block[:, 1]

        W0 = Waist_r * sim.field.wavelength
        K = 2 * np.pi * sim.struct.material[0].nb
        zR = np.pi * pow(W0, 2) * sim.struct.material[0].nb / sim.field.wavelength
        Qz = np.arctan((z - z0) / zR)

        Wz = W0 * (1 + pow(pow(((z - z0) / zR), 2), 0.5))

        Rz_inverse = z / (pow((z - z0), 2) + zR ** 2)

        E = (W0 / Wz) * np.exp(-pow((r / Wz), 2) - 1j * (K * (z - z0) + K * (r ** 2)) * Rz_inverse / 2 - Qz)
        Ex_incident = E0[0] * E
        Ey_incident = E0[1] * E
        Ez_incident = E0[2] * E


    qkx = np.zeros((Nx, Ny, Nz), dtype=complex)
    qky = np.zeros((Nx, Ny, Nz), dtype=complex)
    qkz = np.zeros((Nx, Ny, Nz), dtype=complex)
    INDEX_IN_extended = np.zeros((Nx, Ny, Nz), dtype=complex)
    Inverse_Alpha_extended = np.zeros((Nx, Ny, Nz), dtype=complex)

    if plane_2D == "xy":
        index_x1 = int((Nx - Nx_target) / 2)
        index_x2 = int((Nx + Nx_target) / 2)
        index_y1 = int((Ny - Ny_target) / 2)
        index_y2 = int((Ny + Ny_target) / 2)

        Inverse_Alpha_extended[index_x1: index_x2, index_y1: index_y2, 0:Nz_target] = Inverse_Alpha
        Inverse_Alpha = None
        Inverse_Alpha = Inverse_Alpha_extended

        INDEX_IN_extended[index_x1: index_x2, index_y1: index_y2, :Nz_target] = sim.struct.occupied.INDEX_IN_ALL
        INDEX_IN = None
        INDEX_IN = 1

        qkx[index_x1: index_x2, index_y1: index_y2, :Nz_target] = px
        qky[index_x1: index_x2, index_y1: index_y2, :Nz_target] = py
        qkz[index_x1: index_x2, index_y1: index_y2, :Nz_target] = pz

    if plane_2D == "xz":
        index_x1 = int((Nx - Nx_target) / 2)
        index_x2 = int((Nx + Nx_target) / 2)
        index_z1 = int((Nz - Nz_target) / 2)
        index_z2 = int((Nz + Nz_target) / 2)

        Inverse_Alpha_extended[index_x1: index_x2, 0: Ny_target, index_z1:index_z2] = Inverse_Alpha
        Inverse_Alpha = None
        Inverse_Alpha = Inverse_Alpha_extended

        INDEX_IN_extended[index_x1: index_x2, 0: Ny_target, index_z1:index_z2] = INDEX_IN
        INDEX_IN = None
        INDEX_IN = 1

        qkx[index_x1: index_x2, 0: Ny_target, index_z1:index_z2] = px
        qky[index_x1: index_x2, 0: Ny_target, index_z1:index_z2] = py
        qkz[index_x1: index_x2, 0: Ny_target, index_z1:index_z2] = pz

    Outside_Index = Excluding_Nps(x_plane, y_plane, z_plane, N_plane, Np_shape, sim)

    rkj1 = r_block[0, 0] - r_block[:, 0]
    rkj2 = r_block[0, 1] - r_block[:, 1]
    rkj3 = r_block[0, 2] - r_block[:, 2]

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
    Exp_ikvec_rjk = np.exp(1j * np.linalg.norm(kvec) * RJK) / RJK
    ikvec_rjk = (1j * np.linalg.norm(kvec) * RJK - 1) / (RJK ** 2)

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

    AXX = np.zeros((2 * Nx - 1, 2 * Ny - 1, 2 * Nz - 1), dtype=complex)
    AXY = np.zeros((2 * Nx - 1, 2 * Ny - 1, 2 * Nz - 1), dtype=complex)
    AXZ = np.zeros((2 * Nx - 1, 2 * Ny - 1, 2 * Nz - 1), dtype=complex)
    AYY = np.zeros((2 * Nx - 1, 2 * Ny - 1, 2 * Nz - 1), dtype=complex)
    AYZ = np.zeros((2 * Nx - 1, 2 * Ny - 1, 2 * Nz - 1), dtype=complex)
    AZZ = np.zeros((2 * Nx - 1, 2 * Ny - 1, 2 * Nz - 1), dtype=complex)

    AXX[0:Nx, 0:Ny, 0:Nz] = Axx
    AXY[0:Nx, 0:Ny, 0:Nz] = Axy
    AXZ[0:Nx, 0:Ny, 0:Nz] = Axz
    AYY[0:Nx, 0:Ny, 0:Nz] = Ayy
    AYZ[0:Nx, 0:Ny, 0:Nz] = Ayz
    AZZ[0:Nx, 0:Ny, 0:Nz] = Azz

    # Expanding dimension of each line in X-direction to 2Nx-1#
    # Axx.resize((2 * Nx - 1, 2 * Ny - 1, 2 * Nz - 1), refcheck=False)
    AXX[Nx:2 * Nx - 1, 0:Ny, 0:Nz] = Axx[Nx - 1:0:-1, 0:Ny, 0:Nz]
    # del Axx
    Axx = None
    AXY[Nx:2 * Nx - 1, 0:Ny, 0:Nz] = -Axy[Nx - 1:0:-1, 0:Ny, 0:Nz]
    # del Axy
    Axy = None
    AXZ[Nx:2 * Nx - 1, 0:Ny, 0:Nz] = -Axz[Nx - 1:0:-1, 0:Ny, 0:Nz]
    # del Axz
    Axz = None
    AYY[Nx:2 * Nx - 1, 0:Ny, 0:Nz] = Ayy[Nx - 1:0:-1, 0:Ny, 0:Nz]
    # del Ayy

    AYZ[Nx:2 * Nx - 1, 0:Ny, 0:Nz] = Ayz[Nx - 1:0:-1, 0:Ny, 0:Nz]
    Ayz = None
    AZZ[Nx:2 * Nx - 1, 0:Ny, 0:Nz] = Azz[Nx - 1:0:-1, 0:Ny, 0:Nz]
    Azz = None

    # Calculating FFT in X-direction#

    AXX[:, 0:Ny, 0:Nz] = np.fft.fft(AXX[:, 0:Ny, 0:Nz], axis=0)  # axis=0意味着沿着第一维度(即行)来计算fft
    AXY[:, 0:Ny, 0:Nz] = np.fft.fft(AXY[:, 0:Ny, 0:Nz], axis=0)
    AXZ[:, 0:Ny, 0:Nz] = np.fft.fft(AXZ[:, 0:Ny, 0:Nz], axis=0)
    AYY[:, 0:Ny, 0:Nz] = np.fft.fft(AYY[:, 0:Ny, 0:Nz], axis=0)
    AYZ[:, 0:Ny, 0:Nz] = np.fft.fft(AYZ[:, 0:Ny, 0:Nz], axis=0)
    AZZ[:, 0:Ny, 0:Nz] = np.fft.fft(AZZ[:, 0:Ny, 0:Nz], axis=0)

    # Expanding dimension of each line in Y-direction to 2Ny-1#
    AXX[:, Ny:2 * Ny - 1, 0:Nz] = AXX[:, Ny - 1:0:-1, 0:Nz]
    AXY[:, Ny:2 * Ny - 1, 0:Nz] = -AXY[:, Ny - 1:0:-1, 0:Nz]
    AXZ[:, Ny:2 * Ny - 1, 0:Nz] = AXZ[:, Ny - 1:0:-1, 0:Nz]
    AYY[:, Ny:2 * Ny - 1, 0:Nz] = AYY[:, Ny - 1:0:-1, 0:Nz]
    AYZ[:, Ny:2 * Ny - 1, 0:Nz] = -AYZ[:, Ny - 1:0:-1, 0:Nz]
    AZZ[:, Ny:2 * Ny - 1, 0:Nz] = AZZ[:, Ny - 1:0:-1, 0:Nz]

    # ========== Calculating FFT in Y-direction
    AXX[:, :, 0:Nz] = np.fft.fft(AXX[:, :, 0:Nz], axis=1)
    AXY[:, :, 0:Nz] = np.fft.fft(AXY[:, :, 0:Nz], axis=1)
    AXZ[:, :, 0:Nz] = np.fft.fft(AXZ[:, :, 0:Nz], axis=1)
    AYY[:, :, 0:Nz] = np.fft.fft(AYY[:, :, 0:Nz], axis=1)
    AYZ[:, :, 0:Nz] = np.fft.fft(AYZ[:, :, 0:Nz], axis=1)
    AZZ[:, :, 0:Nz] = np.fft.fft(AZZ[:, :, 0:Nz], axis=1)

    # Expanding dimension of each line in Z-direction to 2Nz-1#
    AXX[:, :, Nz:2 * Nz - 1] = AXX[:, :, Nz - 1:0:-1]
    AXY[:, :, Nz:2 * Nz - 1] = AXY[:, :, Nz - 1:0:-1]
    AXZ[:, :, Nz:2 * Nz - 1] = -AXZ[:, :, Nz - 1:0:-1]
    AYY[:, :, Nz:2 * Nz - 1] = AYY[:, :, Nz - 1:0:-1]
    AYZ[:, :, Nz:2 * Nz - 1] = -AYZ[:, :, Nz - 1:0:-1]
    AZZ[:, :, Nz:2 * Nz - 1] = AZZ[:, :, Nz - 1:0:-1]

    FFT_AXX = np.fft.fft(AXX[:, :, :], axis=2)
    del AXX
    FFT_AXY = np.fft.fft(AXY[:, :, :], axis=2)
    del AXY
    FFT_AXZ = np.fft.fft(AXZ[:, :, :], axis=2)
    del AXZ
    FFT_AYY = np.fft.fft(AYY[:, :, :], axis=2)
    del AYY
    FFT_AYZ = np.fft.fft(AYZ[:, :, :], axis=2)
    del AYZ
    FFT_AZZ = np.fft.fft(AZZ[:, :, :], axis=2)
    del AZZ

    '''
       ===================================Ifft===================================
    '''

    FFT_qkx_Y = np.fft.fft(qkx, 2 * Ny - 1, axis=1)
    FFT_qky_Y = np.fft.fft(qky, 2 * Ny - 1, axis=1)
    FFT_qkz_Y = np.fft.fft(qkz, 2 * Ny - 1, axis=1)

    # Calculate FFT in x-direction of the vectors in previous step
    FFT_qkx_x = np.fft.fft(FFT_qkx_Y, 2 * Nx - 1, axis=0)
    del FFT_qkx_Y
    FFT_qky_x = np.fft.fft(FFT_qky_Y, 2 * Nx - 1, axis=0)
    del FFT_qky_Y
    FFT_qkz_x = np.fft.fft(FFT_qkz_Y, 2 * Nx - 1, axis=0)
    del FFT_qkz_Y

    # Calculate FFT in z-direction of the vectors in previous step
    FFT_qkx_z = np.fft.fft(FFT_qkx_x, 2 * Nz - 1, axis=2)
    del FFT_qkx_x
    FFT_qky_z = np.fft.fft(FFT_qky_x, 2 * Nz - 1, axis=2)
    del FFT_qky_x
    FFT_qkz_z = np.fft.fft(FFT_qkz_x, 2 * Nz - 1, axis=2)
    del FFT_qkz_x

    # Element-wise multiplicaition of FFT of matrix and FFT of vector
    FFT_APX = FFT_AXX * FFT_qkx_z + FFT_AXY * FFT_qky_z + FFT_AXZ * FFT_qkz_z
    FFT_APY = FFT_AXY * FFT_qkx_z + FFT_AYY * FFT_qky_z + FFT_AYZ * FFT_qkz_z
    FFT_APZ = FFT_AXZ * FFT_qkx_z + FFT_AYZ * FFT_qky_z + FFT_AZZ * FFT_qkz_z
    del FFT_qkx_z, FFT_qky_z, FFT_qkz_z

    # IFFT IN Z DIRECTION
    IFFT_APX_Z = np.fft.ifft(FFT_APX, axis=2)
    del FFT_APX
    IFFT_APY_Z = np.fft.ifft(FFT_APY, axis=2)
    del FFT_APY
    IFFT_APZ_Z = np.fft.ifft(FFT_APZ, axis=2)
    del FFT_APZ

    # IFFT IN X DIRCTION
    IFFT_APX_X = np.fft.ifft(IFFT_APX_Z[0:2 * Nx - 1, 0:2 * Ny - 1, 0:Nz], axis=0)
    del IFFT_APX_Z
    IFFT_APY_X = np.fft.ifft(IFFT_APY_Z[0:2 * Nx - 1, 0:2 * Ny - 1, 0:Nz], axis=0)
    del IFFT_APY_Z
    IFFT_APZ_X = np.fft.ifft(IFFT_APZ_Z[0:2 * Nx - 1, 0:2 * Ny - 1, 0:Nz], axis=0)
    del IFFT_APZ_Z

    # IFFT IN Y DIRCTION
    IFFT_APX = np.fft.ifft(IFFT_APX_X[0:Nx, 0:2 * Ny - 1, 0:Nz], axis=1)
    del IFFT_APX_X
    IFFT_APY = np.fft.ifft(IFFT_APY_X[0:Nx, 0:2 * Ny - 1, 0:Nz], axis=1)
    del IFFT_APY_X
    IFFT_APZ = np.fft.ifft(IFFT_APZ_X[0:Nx, 0:2 * Ny - 1, 0:Nz], axis=1)
    del IFFT_APZ_X

    # Negating contribution of the dipoles outside the NPs boundary
    Aqkx = IFFT_APX[0:Nx, 0:Ny, 0:Nz] * INDEX_IN + Inverse_Alpha * qkx * INDEX_IN
    del IFFT_APX
    Aqky = IFFT_APY[0:Nx, 0:Ny, 0:Nz] * INDEX_IN + Inverse_Alpha * qky * INDEX_IN
    del IFFT_APY
    Aqkz = IFFT_APZ[0:Nx, 0:Ny, 0:Nz] * INDEX_IN + Inverse_Alpha * qkz * INDEX_IN
    del IFFT_APZ

    if plane_2D == "xy":
        Apkx = Aqkx[0:Nx, 0:Ny, Nz // 2]
        Apky = Aqky[0:Nx, 0:Ny, Nz // 2]
        Apkz = Aqkx[0:Nx, 0:Ny, Nz // 2]

    elif plane_2D == "xz":
        print("hi")

    Apx = np.reshape(Apkx, [N_plane, 1], order='F')
    Apy = np.reshape(Apky, [N_plane, 1], order='F')
    Apz = np.reshape(Apkz, [N_plane, 1], order='F')

    Ex_scat = -(Apx)
    Ey_scat = -(Apy)
    Ez_scat = -(Apz)

    Axx, Axy, Axz, Ayy, Ayz, Azz = None

    Ex_out = (Ex_incident + Ex_scat) * Outside_Index
    Ey_out = (Ey_incident + Ey_scat) * Outside_Index
    Ez_out = (Ez_incident + Ez_scat) * Outside_Index

    E_out = np.sqrt(Ex_out * Ex_out + Ey_out * Ey_out + Ez_out * Ez_out)
    Et_out = abs(np.reshape(E_out, [SIZE[0], SIZE[1]]))
    Et_out = scipy.medfilt2(Et_out)
    I_out = pow(Et_out, 2)

    if plane_2D == "xy":
        Index_in = np.nonzero(Zrectan == 0)

    elif plane_2D == "xz":
        Index_in = np.nonzero(Yrectan == 0)

    Xtarget = Xrectan[Index_in]
    Ytarget = Yrectan[Index_in]
    Ztarget = Zrectan[Index_in]

    Px_in = PX_vector[Index_in]
    Py_in = PY_vector[Index_in]
    Pz_in = PZ_vector[Index_in]
    Inverse_Polar = Inverse_Alpha[Index_in]

    Ex_in = Inverse_Polar * Px_in
    Ey_in = Inverse_Polar * Py_in
    Ez_in = Inverse_Polar * Pz_in
    E_in = pow((Ex_in * Ex_in + Ey_in * Ey_in + Ez_in * Ez_in), 0.5)

    return plane_2D, E_in, Nx_target, Nz_target, Ny_target, Xex, Yex, Zex, Et_out


def Excluding_Nps(x_plane, y_plane, z_plane, N_plane, Np_shape, sim):
    X0 = 0
    Y0 = 0
    Z0 = 0
    if Np_shape == "spherical":
        Index_in = np.nonzero(
            np.sqrt((x_plane - X0) ** 2 + (y_plane - Y0) ** 2 + (z_plane - Z0) ** 2) <= (sim.struct.geometry.Lx / 2))
        Index_in = np.ravel_multi_index(Index_in, x_plane.shape) + 1
    elif Np_shape == "rod":
        Index_in = np.nonzero(
            (np.sqrt((x_plane - X0) ** 2 + (y_plane - Y0) ** 2 + (
                        np.abs(z_plane - Z0) - sim.struct.geometry.Lz / 2 + sim.struct.geometry.Lx / 2) ** 2) <= (
                     sim.struct.geometry.Lx / 2)
             ) & (np.abs(z_plane - Z0) > (sim.struct.geometry.Lz / 2 - sim.struct.geometry.Lx / 2)) | (
                    np.sqrt((x_plane - X0) ** 2 + (y_plane - Y0) ** 2) <= (sim.struct.geometry.Lx / 2)
            ) & (np.abs(z_plane - Z0) <= (sim.struct.geometry.Lz / 2 - sim.struct.geometry.Lx / 2)))
        Index_in = np.ravel_multi_index(Index_in, x_plane.shape) + 1

    Multiply_Nps = np.zeros((N_plane, 1))
    Multiply_Nps[Index_in - 1] = 1

    Mult = np.ones((N_plane, 1))
    Outside_Index = Mult - Multiply_Nps
    Multiply_Nps = None
    Index_in = None

    return Outside_Index
