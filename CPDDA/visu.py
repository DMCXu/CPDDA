import matplotlib.pyplot as plt
import numpy as np


def visu_NPs(sim):
    """
    ## Visualization of particles ##

    sim:simulation object of CPDDA

    :return:fig

    """
    voxels = sim.struct.occupied.INDEX_IN_ALL > 0
    # 绘制立方体
    fig = plt.figure(figsize=(15 / 2.54, 10 / 2.54))
    ax = fig.add_subplot(111, projection='3d')
    # 关闭坐标显示
    ax.set_axis_off()
    ax.voxels(voxels, facecolors='red', edgecolor='k', alpha=0.8)

    plt.savefig("fig111.tif", dpi=300, bbox_inches='tight')
    plt.show()


def visu_section(data, sim):
    """
        ## Visualize volume factors and save data ##

        sim: simulation object of CPDDA

        Cext: extinction section
        Cabs: absorption section
        Csca: scattering section

        :return:fig
        """
    file_name = f"{sim.file_name}.txt"
    np.savetxt(file_name, data, fmt='%.6e', header='CEX, CABS, CSCA', comments='')

    Cext = data[:, 0]
    Cabs = data[:, 1]
    Csca = data[:, 2]

    plt.plot(sim.struct.material[0].wl, Cext, label="Cext", color="blue", linestyle="-", linewidth=2)
    plt.plot(sim.struct.material[0].wl, Cabs, label="Cabs", color="red", linestyle="-", linewidth=2)
    plt.plot(sim.struct.material[0].wl, Csca, label="Csca", color="green", linestyle="-", linewidth=2)

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("C (nm^2)")

    plt.savefig("5optical_properties_plot.png", dpi=300, bbox_inches="tight")  # 保存为高分辨率图像
    plt.show()


def visu_vol_coe(sim, Aext, Aabs, Asca):
    """
    ## Visualize volume factors and save data ##

    sim: simulation object of CPDDA

    Aext: volume extinction coefficient
    Aabs: volume absorption coefficient
    Asca: volume scattering coefficient

    :return:fig
    """
    file_name = f"{sim.file_name}.txt"
    data_ = np.column_stack((Aext, Aabs, Asca))
    np.savetxt(file_name, data_, fmt='%.6e', header='Aext, Aabs, Asca', comments='')

    plt.plot(sim.struct.material[0].wl, Aext, label="Aext", color="blue", linestyle="-", linewidth=2)
    plt.plot(sim.struct.material[0].wl, Aabs, label="Aabs", color="red", linestyle="-", linewidth=2)
    plt.plot(sim.struct.material[0].wl, Asca, label="Asca", color="green", linestyle="-", linewidth=2)

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("A (1/nm)")

    plt.savefig("ABS/.png", dpi=300, bbox_inches="tight")  # 保存为高分辨率图像
    plt.show()


def vis_efficiency(sim, Q_EXT, Q_ABS, Q_SCA):
    """
        ## Visualize volume factors and save data ##

        sim: simulation object of CPDDA

        Qext: extinction efficiency
        Qabs: absorption efficiency
        Qsca: scattering efficiency

        :return:fig
        """
    file_name = f"{sim.file_name}.txt"
    data_ = np.column_stack((Q_EXT, Q_ABS, Q_SCA))
    np.savetxt(file_name, data_, fmt='%.6e', header='Aext, Aabs, Asca', comments='')

    plt.plot(sim.struct.material[0].wl, Q_EXT, label="Aext", color="blue", linestyle="-", linewidth=2)
    plt.plot(sim.struct.material[0].wl, Q_ABS, label="Aabs", color="red", linestyle="-", linewidth=2)
    plt.plot(sim.struct.material[0].wl, Q_SCA, label="Asca", color="green", linestyle="-", linewidth=2)

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Q")

    plt.savefig("efficiency.png", dpi=300, bbox_inches="tight")  # 保存为高分辨率图像
    plt.show()


def vis_enhance(plane_2D, E_in, Nx_target, Nz_target, Ny_target, Xex, Yex, Zex, Et_out):
    """
        ## Visualize enhanced E-field of particle ##

        More recommended for single wavelength

        ==============Not fully tested==================

        """
    if plane_2D == "xy":

        Et_in = np.abs(np.reshape(E_in, [Nx_target, Ny_target]))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(Xex, Yex, Et_out, cmap='viridis', edgecolor='none')
        data_xy = np.column_stack((Xex, Yex, Et_out))
        np.savetxt('enhance.txt', data_xy, fmt='%.6e', header='Xex Yex Et_out', comments='')
        # 设置平滑插值和光照效果
        ax.plot_surface(Xex, Yex, Et_out, cmap='viridis', edgecolor='none', rstride=1, cstride=1, antialiased=True)
        # 颜色条显示
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        # 设置等比例轴
        ax.set_box_aspect([1, 1, 0.5])  # 调整 X, Y, Z 轴比例
        plt.show()

    elif plane_2D == "xz":
        Et_in = np.abs(np.reshape(E_in, [Nx_target, Nz_target]))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(Xex, Zex, Et_out, cmap='viridis', edgecolor='none')
        data_xz = np.column_stack((Xex, Zex, Et_out))
        np.savetxt('enhance.txt', data_xz, fmt='%.6e', header='Xex Yex Et_out', comments='')
        # 设置平滑插值和光照效果
        ax.plot_surface(Xex, Zex, Et_out, cmap='viridis', edgecolor='none', rstride=1, cstride=1, antialiased=True)
        # 颜色条显示
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        # 设置等比例轴
        ax.set_box_aspect([1, 1, 0.5])  # 调整 X, Y, Z 轴比例
        plt.show()

    elif plane_2D == "yz":
        Et_in = np.abs(np.reshape(E_in, [Ny_target, Nz_target]))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(Yex, Zex, Et_out, cmap='viridis', edgecolor='none')
        data_yz = np.column_stack((Yex, Zex, Et_out))
        np.savetxt('enhance.txt', data_yz, fmt='%.6e', header='Xex Yex Et_out', comments='')
        # 设置平滑插值和光照效果
        ax.plot_surface(Yex, Zex, Et_out, cmap='viridis', edgecolor='none', rstride=1, cstride=1, antialiased=True)
        # 颜色条显示
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        # 设置等比例轴
        ax.set_box_aspect([1, 1, 0.5])  # 调整 X, Y, Z 轴比例
        plt.show()
