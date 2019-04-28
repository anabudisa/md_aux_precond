import sys
sys.path.insert(0, '/home/anci/Dropbox/porepy/src/')

import numpy as np
import time
import scipy.sparse as sps
from tabulate import tabulate

import porepy as pp

from data import Data
from solver_hazmath import SolverHazmath
from Hcurl import Hcurl
from porepy.utils import comp_geom as cg

# export solution to .vtk
def visualize(x, variable, gb, assembler, discr, block_dof, full_dof,
              file_name="sol_pressure", folder="solution_rt0"):
    # extract solution
    assembler.distribute_variable(gb, x, block_dof, full_dof)

    for g, d in gb:
        d["pressure"] = discr.extract_pressure(g, d[variable])

    # save solution as vtk
    save = pp.Exporter(gb, file_name, folder=folder)
    save.write_vtk(["pressure"])

    return


# solve with direct python solver (for comparison only)
def solve_direct(M, f, bmat=False):
    # direct solve of system
    #       Mx = f

    # if M in bmat form
    if bmat:
        M = sps.bmat(M, format="csc")
        f = np.concatenate(tuple(f))

    start_time = time.time()

    upl = sps.linalg.spsolve(M, f)

    print("Elapsed time direct solver: ", time.time() - start_time)

    return upl


# main test function
# NOTE: you can comment out the lines related to direct solver if no comparison
# with direct solution is needed;
# for example, line 106, 110, 115, 116, 118 and 119
def main(file_name, mesh_size=0.0625, alpha=1.):
    file_name = file_name + ".csv"

    # set parameters
    param = {
        "mesh_size": mesh_size,
        "aperture": 1.,
        "km": 1.,
        "kf": 1.,
        "kn": 1.,
        "tol": 1e-8,
        "folder": "solution_rt0",
        "pressure": "pressure",
        "flux": "darcy_flux",
        "mortar_flux": "lambda_flux_pressure",
    }

    # create grids
    data = Data(file_name, "flow", param)

    # choose discretization
    discr = pp.RT0("flow")
    coupling = pp.RobinCoupling("flow", discr)

    # define the dof and discretization for the grids
    variable = "flux_pressure"
    flux_id = "flux"

    for g, d in data.gb:
        d[pp.PRIMARY_VARIABLES] = {variable: {"cells": 1, "faces": 1}}
        d[pp.DISCRETIZATION] = {variable: {flux_id: discr}}

    for e, d in data.gb.edges():
        g_slave, g_master = data.gb.nodes_of_edge(e)
        d[pp.PRIMARY_VARIABLES] = {param["mortar_flux"]: {"cells": 1}}
        d[pp.COUPLING_DISCRETIZATION] = {
            variable: {
                g_slave: (variable, flux_id),
                g_master: (variable, flux_id),
                e: (param["mortar_flux"], coupling)
            }
        }

    # assembling darcy problem
    assembler = pp.Assembler()
    A, b, block_dof, full_dof = assembler.assemble_matrix_rhs(data.gb)

    # choose linear solver
    solver_hazmath = SolverHazmath(data.gb, discr)

    # get the block structure of M, f
    solver_hazmath.setup_system(A, b, full_dof, block_dof)

    # solve \alpha*(u, v) + (div u, div v) = (f, v) with HAZMATH
    x_hazmath, iters = solver_hazmath.solve(alpha=alpha)

    # solve (directly)
    x_direct = solve_direct(solver_hazmath.M, solver_hazmath.f, bmat=True)

    # permute solutions
    y_hazmath = solver_hazmath.P.T * x_hazmath
    y_direct = solver_hazmath.P.T * x_direct

    # write to vtk - check this if correct still!
    visualize(y_hazmath, variable, data.gb, assembler, discr, block_dof,
              full_dof, file_name="sol_hazmath")
    visualize(y_direct, variable, data.gb, assembler, discr, block_dof,
              full_dof, file_name="sol_direct")

    error = np.linalg.norm(x_direct - x_hazmath) / np.linalg.norm(x_direct)
    print("Error: ", error)

    return iters


# ---------------------------------------------------------------------------- #


# run the solver for different mesh sizes and \alpha = 1
def test_mesh_size(name_):

    table_h = []
    for k in np.arange(1, 7):
        mesh_size = 1./(2. ** k)

        it = main(name_, mesh_size=mesh_size)

        table_h.append([mesh_size, it])

    np.savetxt("tables/"+name_+"_mesh_size_iter.csv", table_h)

    return tabulate(table_h, headers=["h", "iter"])


# ---------------------------------------------------------------------------- #


# run the solver for different \alpha coefficient and h = 1/16
def test_alpha(name_):
    # mesh_size = 1./16

    table_alpha = []
    for k in np.arange(5):
        alpha = 1./(10. ** k)

        it = main(name_, alpha=alpha)

        table_alpha.append([alpha, it])

    np.savetxt("tables/"+name_+"_alpha_iter.csv", table_alpha)

    return tabulate(table_alpha, headers=["eps", "iter"])


# ---------------------------------------------------------------------------- #

def test(file_name):
    file_name = file_name + ".csv"

    # set parameters
    param = {
        "mesh_size": 0.5,
        "aperture": 1.,
        "km": 1.,
        "kf": 1.,
        "kn": 1.,
        "tol": 1e-8,
        "folder": "solution_rt0",
        "pressure": "pressure",
        "flux": "darcy_flux",
        "mortar_flux": "lambda_flux_pressure",
    }

    # create grids
    data = Data(file_name, "flow", param)

    # choose discretization
    discr = pp.RT0("flow")
    coupling = pp.RobinCoupling("flow", discr)

    # define the dof and discretization for the grids
    variable = "flux_pressure"
    flux_id = "flux"

    for g, d in data.gb:
        d[pp.PRIMARY_VARIABLES] = {variable: {"cells": 1, "faces": 1}}
        d[pp.DISCRETIZATION] = {variable: {flux_id: discr}}

    for e, d in data.gb.edges():
        g_slave, g_master = data.gb.nodes_of_edge(e)
        d[pp.PRIMARY_VARIABLES] = {param["mortar_flux"]: {"cells": 1}}
        d[pp.COUPLING_DISCRETIZATION] = {
            variable: {
                g_slave: (variable, flux_id),
                g_master: (variable, flux_id),
                e: (param["mortar_flux"], coupling)
            }
        }

    # assembling darcy problem
    assembler = pp.Assembler()
    A, b, block_dof, full_dof = assembler.assemble_matrix_rhs(data.gb)

    # choose linear solver
    solver_hazmath = SolverHazmath(data.gb, discr)

    # get the block structure of M, f
    solver_hazmath.setup_system(A, b, full_dof, block_dof)

    # mixed-dim div
    B = -solver_hazmath.M[1, 0]

    # save = pp.Exporter(data.gb, "mesh", folder="mesh")
    # save.write_vtk()

    # Hcurl - curl dof, curl operator, projections etc.
    hcurl = Hcurl(data.gb)
    # mixed-dim curl
    C = hcurl.curl()
    # projection from H1 to H(div)
    Pi_div = hcurl.Pi_div_h()

    # div(curl) should be zero
    np.set_printoptions(linewidth=1000)
    print("Div: \n", B.shape)
    print("Curl: \n", C.shape)

    div_curl = B * C

    print("Div(curl): \n", div_curl)

    # div-free continuous function
    fun = lambda x: x - x + 1.

    # interpolate div-free function to nodes of grids
    f = np.empty(shape=(data.gb.num_graph_nodes(),), dtype=np.object)

    for g, d_g in data.gb:
        nn_g = d_g['node_number']
        if g.dim == 3:
            f_x = fun(g.nodes[0, :]).T
            f_y = fun(g.nodes[1, :]).T
            f_z = fun(g.nodes[2, :]).T
            f[nn_g] = np.concatenate((f_x, f_y, f_z))
        elif g.dim == 2:
            R = cg.project_plane_matrix(g.nodes, check_planar=False)
            fun_proj = np.dot(R, fun(g.nodes))
            f_x = fun_proj[0, :].T
            f_y = fun_proj[1, :].T
            f[nn_g] = np.concatenate((f_x, f_y))
        elif g.dim == 1:
            R = cg.project_line_matrix(g.nodes)
            fun_proj = np.dot(R, fun(g.nodes))
            f[nn_g] = fun_proj[0, :].T
        else:
            f[nn_g] = np.array([0.])
    # import pdb; pdb.set_trace()
    f = np.concatenate(f)

    # div(Pi_div_h(fun)) should be 0
    print("div(Pi(f)): \n", np.linalg.norm(B * Pi_div * f))

    return


if __name__ == "__main__":
    # grid config file
    # name = "no_fracture_3d"
    # name = "two_fractures"
    # name = "one_fracture_3d"
    name = "network_geiger_3d"
    # name = "test_network"
    # name = "gmsh_frac_file.geo"

    test(name)

    # main(name)
    # table_h = test_mesh_size(name)
    # print(table_h)

    # table_alpha = test_alpha(name)
    # print(table_alpha)
