import numpy as np
import scipy.sparse as sps

import porepy as pp

from logger import logger


class Data(object):

    def __init__(self, gb, folder):
        self.keyword = "flow"
        self.gb = gb
        self.param = None

        # discretization operator name
        self.discr_name = "flux"
        self.discr = pp.RT0(self.keyword)

        self.mass_name = "mass"
        self.mass = pp.MixedMassMatrix(self.keyword)

        self.coupling_name = self.discr_name + "_coupling"
        self.coupling = pp.RobinCoupling(self.keyword, self.discr)

        self.source_name = "source"
        self.source = pp.DualScalarSource(self.keyword)

        # master variable name
        self.variable = "flow_variable"
        self.mortar = "lambda_" + self.variable

        # post process variables
        self.pressure = "pressure"
        self.flux = "darcy_flux"  # it has to be this one
        self.P0_flux = "P0_darcy_flux"

        # tolerance
        self.tol = 1e-6

        # exporter
        self.save = None

    # ------------------------------------------------------------------------ #

    def set_data(self, param, bc_flag):

        logger.info("Set problem data")
        self.param = param
        self.gb.add_node_props(["param", "is_tangential"])

        for g, d in self.gb:
            param = {}

            unity = np.ones(g.num_cells)
            zeros = np.zeros(g.num_cells)
            empty = np.empty(0)

            d["is_tangential"] = True
            d["tol"] = self.tol

            # Tangential permeability
            if g.dim == self.gb.dim_max():
                kxx = self.param["km"] * unity
                perm = pp.SecondOrderTensor(g.dim, kxx=kxx, kyy=kxx, kzz=1)
            else:
                kxx = self.param["kf_t"] * unity
                perm = pp.SecondOrderTensor(g.dim, kxx=kxx, kyy=1, kzz=1)

            param["second_order_tensor"] = perm

            # Aperture
            aperture = np.power(self.param["aperture"], self.gb.dim_max() -
                                g.dim)
            param["aperture"] = aperture * unity
            param["mass_weight"] = unity

            # Source term
            param["source"] = zeros

            # Boundaries
            b_faces = g.get_boundary_faces()
            if b_faces.size:
                labels, bc_val = bc_flag(g, param, self.tol)
                param["bc"] = pp.BoundaryCondition(g, b_faces, labels)
            else:
                bc_val = np.zeros(g.num_faces)
                param["bc"] = pp.BoundaryCondition(g, empty, empty)

            param["bc_values"] = bc_val

            d[pp.PARAMETERS] = pp.Parameters(g, self.keyword, param)

        # Normal permeability
        for e, d in self.gb.edges():
            g_l = self.gb.nodes_of_edge(e)[0]

            mg = d["mortar_grid"]
            check_P = mg.slave_to_mortar_avg()

            aperture = self.gb.node_props(g_l, pp.PARAMETERS)[self.keyword]["aperture"]
            gamma = check_P * aperture
            kn = param["kf_n"] * np.ones(mg.num_cells) / gamma
            param = {"normal_diffusivity": kn}

            d[pp.PARAMETERS] = pp.Parameters(e, self.keyword, param)

        # Set now the discretization
        # set the discretization for the grids
        for g, d in self.gb:
            d[pp.PRIMARY_VARIABLES] = {self.variable: {"cells": 1, "faces": 1}}
            d[pp.DISCRETIZATION] = {self.variable: {self.discr_name: self.discr,
                                                    self.mass_name: self.mass,
                                                    self.source_name: self.source}}

        # define the interface terms to couple the grids
        for e, d in self.gb.edges():
            g_slave, g_master = self.gb.nodes_of_edge(e)
            d[pp.PRIMARY_VARIABLES] = {self.mortar: {"cells": 1}}
            d[pp.COUPLING_DISCRETIZATION] = {
                self.coupling_name: {
                    g_slave: (self.variable, self.discr_name),
                    g_master: (self.variable, self.discr_name),
                    e: (self.mortar, self.coupling),
                }
            }

    # ------------------------------------------------------------------------ #

    def matrix_rhs(self):

        # Empty the matrices
        for g, d in self.gb:
            d[pp.DISCRETIZATION_MATRICES] = {self.keyword: {}}

        for e, d in self.gb.edges():
            d[pp.DISCRETIZATION_MATRICES] = {self.keyword: {}}

        # Solution of the darcy problem
        assembler = pp.Assembler()

        logger.info("Assemble the flow problem")
        block_A, block_b, block_dof, full_dof = assembler.assemble_matrix_rhs(self.gb)

        # unpack the matrices just computed
        coupling_name = self.coupling_name + (
            "_" + self.mortar + "_" + self.variable + "_" + self.variable
        )
        discr_name = self.discr_name + "_" + self.variable
        mass_name = self.mass_name + "_" + self.variable
        source_name = self.source_name + "_" + self.variable

        # need a sign for the convention of the conservation equation
        M = - block_A[mass_name]
        A = M + block_A[discr_name] + block_A[coupling_name]
        b = block_b[discr_name] + block_b[coupling_name] + block_b[source_name]
        logger.info("Done")

        return A, M, b, block_dof, full_dof

    # ------------------------------------------------------------------------ #

    def extract_solution(self, x, block_dof, full_dof):

        logger.info("Variable post-process")
        assembler = pp.Assembler()
        assembler.distribute_variable(self.gb, x, block_dof, full_dof)
        for g, d in self.gb:
            d[self.pressure] = self.discr.extract_pressure(g, d[self.variable], d)
            d[self.flux] = self.discr.extract_flux(g, d[self.variable], d)

        # export the P0 flux reconstruction
        pp.project_flux(self.gb, self.discr, self.flux, self.P0_flux, self.mortar)
        logger.info("Done")

    # ------------------------------------------------------------------------------#

    def export_solution(self, sol_folder, sol_file_name="solution"):

        logger.info("Export variables")
        self.save = pp.Exporter(self.gb, sol_file_name, folder=sol_folder)
        self.save.write_vtk([self.pressure, self.P0_flux])
        logger.info("Done")

    # ------------------------------------------------------------------------------#
