import numpy as np
import scipy.sparse as sps
import time
import porepy as pp

import sys; sys.path.insert(0, "../src/")

from logger import logger


def bc_flag(g, tol):

    b_faces = g.get_boundary_faces()
    b_face_centers = g.face_centers[:, b_faces]

    # define the labels and values for the boundary faces
    labels = np.array(["dir"] * b_faces.size)
    bc_val = np.zeros(g.num_faces)

    # Pressure on boundary is p(x,y) = 1 - x
    bc_val[b_faces] = 1. - b_face_centers[0, :]

    return labels, bc_val

# ------------------------------------------------------------------------------#


def make_mesh(mesh_size, file_name, plot=False):

    # define the domain
    domain = {"xmin": 0, "xmax": 700, "ymin": 0, "ymax": 600}

    # Import fractures coordinates from file
    network_2d = pp.fracture_importer.network_2d_from_csv(file_name,
                                                          domain=domain)

    # Generate a mixed-dimensional mesh and geometry
    gb = network_2d.mesh({"mesh_size_frac": mesh_size,
                          "mesh_size_min": mesh_size,
                          "mesh_size_bound": mesh_size})

    if plot:
        pp.plot_grid(gb, alpha=0, info="all")

    return gb

# ------------------------------------------------------------------------------#


def solve_direct(M, f, bmat=False):

    logger.info("Solve system with direct solver in Python")

    # if M in bmat form
    if bmat:
        M = sps.bmat(M, format="csc")
        f = np.concatenate(tuple(f))

    start_time = time.time()
    upl = sps.linalg.spsolve(M, f)
    t = time.time() - start_time

    logger.info("Elapsed time of direct solver: " + str(t))
    logger.info("Done")

    return upl