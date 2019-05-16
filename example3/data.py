import sys
sys.path.insert(0, '/home/anci/Dropbox/porepy/src/')

import numpy as np
import porepy as pp
from tabulate import tabulate


class Data(object):

    def __init__(self, file_name, model_data, param):

        self.gb = None
        self.domain = None
        self.model_data = model_data
        self.param = param

        self.tol = 1e-8

        self.create_gb(file_name)
        self.add_data()

    # ------------------------------------------------------------------------ #

    def create_gb(self, file_name):
        mesh_kwargs = {}
        mesh_kwargs = {"mesh_size": self.param["mesh_size"],
                       'mesh_size_frac': self.param["mesh_size"],
                       'mesh_size_min': self.param["mesh_size"],
                       'mesh_size_bound': self.param["mesh_size"]}

        # self.domain = {'xmin': 0, 'xmax': 1, 'ymin': 0, 'ymax': 1, 'zmin': 0,
        #                'zmax': 1}

        # make fracture network
        network_3d = pp.fracture_importer.network_3d_from_csv(file_name)

        # Generate a mixed-dimensional mesh
        gb = network_3d.mesh(mesh_kwargs)

        pp.plot_grid(gb, alpha=0, info="all")

        self.gb.compute_geometry()
        self.gb.assign_node_ordering()

        self.up_dof()

    # ------------------------------------------------------------------------ #

    def add_data(self):

        keyword = "flow"
        self.gb.add_node_props(["param", "is_tangential"])

        for g, d in self.gb:
            param = {}
            d["is_tangential"] = True

            unity = np.ones(g.num_cells)
            zeros = np.zeros(g.num_cells)
            empty = np.empty(0)

            # Tangential permeability
            if g.dim == 3:
                kxx = self.param["km"] * unity
                perm = pp.SecondOrderTensor(g.dim, kxx=kxx, kyy=kxx, kzz=kxx)
            elif g.dim == 2:
                kxx = self.param["kf"] * unity
                perm = pp.SecondOrderTensor(g.dim, kxx=kxx, kyy=kxx, kzz=1)
            else:
                kxx = self.param["kf"] * unity
                perm = pp.SecondOrderTensor(g.dim, kxx=kxx, kyy=1, kzz=1)

            param["second_order_tensor"] = perm

            # Aperture
            aperture = np.power(self.param["aperture"], self.gb.dim_max() -
                                g.dim)
            param["aperture"] = aperture * unity

            # Source term
            param["source"] = zeros

            # Boundary data
            bound_faces = g.get_boundary_faces()
            bc_val = np.zeros(g.num_faces)
            if bound_faces.size == 0:
                param["bc"] = pp.BoundaryCondition(g, empty, empty)
            else:
                bound_face_centers = g.face_centers[:, bound_faces]

                # right = bound_face_centers[0, :] > self.domain["xmax"] - \
                #                                    self.tol
                # left = bound_face_centers[0, :] < self.domain['xmin'] +
                # self.tol

                labels = np.array(["dir"] * bound_faces.size)
                # labels[right] = "dir"
                bc_val[bound_faces] = bound_face_centers[0, :]
                # bc_val[bound_faces[right]] = 1
                # bc_val[bound_faces[left]] = -aperture * g.face_areas[
                #     bound_faces[left]]

                param["bc"] = pp.BoundaryCondition(g, bound_faces, labels)

            param["bc_values"] = bc_val

            d[pp.PARAMETERS] = pp.Parameters(g, keyword, param)
            d[pp.DISCRETIZATION_MATRICES] = {keyword: {}}

        # Normal permeability
        for e, d in self.gb.edges():
            g_l = self.gb.nodes_of_edge(e)[0]
            mg = d['mortar_grid']
            check_P = mg.slave_to_mortar_avg()

            if g_l.dim == 2:
                kxx = self.param["kn"]
            else:
                kxx = self.param["kf"]

            aperture = self.gb.node_props(g_l, pp.PARAMETERS)[keyword][
                "aperture"]
            gamma = check_P * np.power(aperture, 1. / (self.gb.dim_max() -
                                                       g_l.dim))
            kn = kxx * np.ones(mg.num_cells) / gamma

            param = {"normal_diffusivity": kn}

            d[pp.PARAMETERS] = pp.Parameters(e, keyword, param)
            d[pp.DISCRETIZATION_MATRICES] = {keyword: {}}

    # ------------------------------------------------------------------------ #

    def up_dof(self):
        # add attribute to all node and edge grids - number of dof for each
        # variable
        self.gb.add_node_props(['dof_u', 'dof_p'])
        for g, d in self.gb:
            d['dof_u'] = g.num_faces
            d['dof_p'] = g.num_cells

        self.gb.add_edge_props('dof_lmbd')
        for _, d in self.gb.edges():
            d['dof_lmbd'] = d['mortar_grid'].num_cells

    # ------------------------------------------------------------------------ #

    def print_setup(self):
        print(" ------------------------------------------------------------- ")
        print(" -------------------- PROBLEM SETUP -------------------------- ")

        table = [["Mesh size", self.param["mesh_size"]],
                 ["Aperture", self.param["aperture"]],
                 ["Km", self.param["km"]],
                 ["Kf", self.param["kf"]],
                 ["Kn", self.param["kn"]]]

        print(tabulate(table, headers=["Parameter", "Value"]))

        print(" ------------------------------------------------------------- ")

    # ------------------------------------------------------------------------ #
