import numpy as np
from tabulate import tabulate
import random

import sys
sys.path.insert(0, '/home/anci/Dropbox/new_porepy/porepy/src/porepy/')  # robust point in polyhedron
sys.path.insert(0, '../src/')  # common

import data
from A_reg import A_reg


def main():
    file_name = "network_geiger.csv"
    mesh_size = 1./8
    alpha = 1e0

    param = {"tol": 1e-6,
             "km": 1.,
             "kf_t": 1.,
             "kf_n": 1.,
             "aperture": 1.,
             "alpha": 1e0
             }

    gb = data.make_mesh(file_name, mesh_size)
    Areg = A_reg(gb)

    A_reg_div = Areg.A_reg_div()

    return


# ---------------------------------------------------------------------------- #

if __name__ == "__main__":
    main()