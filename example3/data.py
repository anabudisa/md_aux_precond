import numpy as np
import porepy as pp
from tabulate import tabulate

import sys; sys.path.insert(0, "../src/")

from logger import logger
from solver_hazmath import Solver
from flow_setup import Flow


def bc_flag(g, tol):

    b_faces = g.get_boundary_faces()
    b_face_centers = g.face_centers[:, b_faces]

    # define the labels and values for the boundary faces
    labels = np.array(["dir"] * b_faces.size)
    bc_val = np.zeros(g.num_faces)

    # Pressure on boundary is p(x,y) = 1 - x
    bc_val[b_faces] = 1. - b_face_centers[0, :]

    return labels, bc_val

# ---------------------------------------------------------------------------- #


def make_mesh(file_name, mesh_size, plot=False):

    # mesh arguments
    kwargs = {"mesh_size_frac": mesh_size,
              "mesh_size_min": mesh_size,
              "mesh_size_bound": mesh_size}

    # Import fractures coordinates from file
    network_3d = pp.fracture_importer.network_3d_from_csv(file_name)

    # Generate a mixed-dimensional mesh and geometry
    gb = network_3d.mesh(kwargs)

    if plot:
        pp.plot_grid(gb, alpha=0, info="all")

    return gb

# ---------------------------------------------------------------------------- #


def solve_(file_name, mesh_size, alpha, param):
    # create mixed-dimensional grids
    gb = make_mesh(file_name, mesh_size)

    # set parameters and boundary conditions
    folder = "solution"
    darcy_flow = Flow(gb, folder)
    darcy_flow.set_data(param, bc_flag)

    # get matrix and rhs
    A, M, b, block_dof, full_dof = darcy_flow.matrix_rhs()

    # set up solver
    solver = Solver(gb, darcy_flow.discr)
    solver.setup_system(A, M, b, block_dof, full_dof)

    # solve with hazmath library solvers
    x_haz, iters = solver.solve_hazmath(alpha)
    logger.info("Hazmath iters: " + str(iters))
    print(tabulate(solver.cpu_time, headers=["Process", "Time"]))
    # iters = 0
    # solve with direct python solver
    # x_dir = solver.solve_direct()

    # compute error
    # error = np.linalg.norm(x_dir - x_haz) / np.linalg.norm(x_dir)
    # logger.info("Error: " + str(error))

    # extract variables and export hazmath solution
    # darcy_flow.extract_solution(x_haz, block_dof, full_dof)
    # darcy_flow.export_solution(folder, "sol_hazmath")

    # extract variables and export direct solution
    # darcy_flow.extract_solution(x_dir, block_dof, full_dof)
    # darcy_flow.export_solution(folder, "sol_direct")

    return iters
