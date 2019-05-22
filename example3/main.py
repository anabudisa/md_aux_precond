import numpy as np
from tabulate import tabulate

import sys
sys.path.insert(0, '/home/anci/Dropbox/new_porepy/porepy/src/porepy/')  # robust point in polyhedron
sys.path.insert(0, '../src/')  # common

import data


def test_mesh_size():
    file_name = "network_geiger_3d.csv"
    alpha = 1.

    param = {"tol": 1e-6,
             "km": 1.,
             "kf_t": 1.,
             "kf_n": 1.,
             "aperture": 1.,
             "alpha": 1.
             }

    table_h = []
    for k in np.arange(2, 7):
        mesh_size = 1./(2. ** k)

        it = data.solve_(file_name, mesh_size, alpha, param)

        table_h.append([mesh_size, it])

    np.savetxt("mesh_size_iter.csv", table_h, fmt="%d")

    return tabulate(table_h, headers=["h", "iter"])

# ---------------------------------------------------------------------------- #


def test_alpha():
    file_name = "network_geiger_3d.csv"
    mesh_size = 1./16

    param = {"tol": 1e-6,
             "km": 1.,
             "kf_t": 1.,
             "kf_n": 1.,
             "aperture": 1.,
             "alpha": 1.
             }

    table_alpha = []
    for k in np.arange(5):
        alpha = 1./(10. ** k)

        it = data.solve_(file_name, mesh_size, alpha, param)

        table_alpha.append([alpha, it])

    np.savetxt("alpha_iter.csv", table_alpha, fmt="%d")

    return tabulate(table_alpha, headers=["eps", "iter"])

# ---------------------------------------------------------------------------- #


def main():
    file_name = "network_geiger_3d.csv"
    mesh_size = 1./16
    alpha = 1.

    param = {"tol": 1e-6,
             "km": 1.,
             "kf_t": 1.,
             "kf_n": 1.,
             "aperture": 1.,
             "alpha": 1.
             }

    data.solve_(file_name, mesh_size, alpha, param)


# ---------------------------------------------------------------------------- #

if __name__ == "__main__":
    main()
    test_alpha()
    test_mesh_size()
