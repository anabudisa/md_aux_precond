import numpy as np
from tabulate import tabulate

import sys

# sys.path.insert(0, '/home/anci/Dropbox/new_porepy/porepy/src/porepy/')  #
# robust point in polyhedron
sys.path.insert(0, '../src/')  # common

import data
from logger import logger

# ---------------------------------------------------------------------------- #


def test_alpha():
    file_name = "fracture_network.csv"
    mesh_size = 1. / 32

    param = {"tol": 1e-6,
             "km": 1.,
             "kf_t": 1.,
             "kf_n": 1.,
             "aperture": 1.,
             }

    table_alpha = []
    for k in np.arange(-4, 5, 2):
        alpha = 10. ** k
        table_K = []
        for l in np.arange(1):
            param["kf_t"] = 10. ** l
            param["kf_n"] = 10. ** l

            logger.info(
                "Parameters: K_t = " + str(param["kf_t"]) + "; K_n = " + str(
                    param["kf_n"]) + "; alpha = " + str(alpha))

            if alpha < 10. ** (-l):
                table_K.append(0)
            else:
                it = data.solve_(file_name, mesh_size, alpha, param)
                table_K.append(it)

        table_alpha.append(table_K)

    print(table_alpha)
    np.savetxt("alpha_iter.csv", table_alpha, fmt="%d")


# ---------------------------------------------------------------------------- #


def main():
    file_name = "fracture_network_withdomain.csv"
    mesh_size = 1500. / 8
    alpha = 1e6

    param = {"tol": 1e-6,
             "km": 1.,
             "kf_t": 1.,
             "kf_n": 1.,
             "aperture": 1.,
             }

    data.solve_(file_name, mesh_size, alpha, param)


# ---------------------------------------------------------------------------- #



if __name__ == "__main__":
    main()
    # test_alpha()
