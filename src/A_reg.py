import numpy as np
import scipy.sparse as sps
import time

from porepy.utils import setmembership
from porepy.utils import comp_geom as cg


class A_reg(object):

    def __init__(self, gb):
        # Grid bucket
        self.gb = gb
        self.tol = 1e-12
        self.cpu_time = []

    # ------------------------------------------------------------------------ #

    def A_reg_div(self):
        A = np.empty(shape=(self.gb.num_graph_nodes(), self.gb.num_graph_nodes()),
                     dtype=np.object)

        # interior H1 matrices
        for g, d in self.gb:
            if g.dim == 0:
                A[nn, nn] = self.local_stiff_matrix(g)
            else:
                local_matrix = self.local_stiff_matrix(g) + \
                               self.local_mass_matrix(g)

                nn = d["node_number"]
                A[nn, nn] = sps.block_diag([local_matrix]*g.dim)

        # boundary H1 matrices
        for e, de in self.gb.edges():
            # get lower and higher dimensional grids on this interface
            g_down, g_up = self.gb.nodes_of_edge(e)
            nn_g_up = self.gb.graph.node[g_up]["node_number"]
            mg = de["mortar_grid"]

            for side in np.arange(mg.num_sides()):
                cells_per_side = mg.num_cells / mg.num_sides()
                local_slice = slice(side * cells_per_side, (side + 1) * cells_per_side)
                
                # find all lower-dim cells and higher-dim faces that match on
                # this interface
                cells, faces, data = sps.find(
                    mg.slave_to_mortar_int()[local_slice, :].T * mg.master_to_mortar_int()[local_slice, :])

                # nodes of higher-dim grid
                nodes_up, _, _ = sps.find(g_up.face_nodes[:, faces])
                nodes_up = np.unique(nodes_up)
                nodes_up_coord = g_up.nodes[:, nodes_up]

                # nodes of lower-dim grid
                nodes_down, _, _ = sps.find(g_down.cell_nodes()[:, cells])
                nodes_down = np.unique(nodes_down)
                nodes_down_coord = g_down.nodes[:, nodes_down]

                # TODO: fix this bug below
                """ 
                The problem with matching nodes in this way is that the 
                fracture grid is not cut into two separate grids when it is 
                intersected with another fracture grid. Therefore, 
                at intersections, we have duplicate nodes with same 
                coordinates and one intersection grid in between.
                The problem is that both of those nodes are in the same grid!
                This goes in all dimensions, 3d->2d, 2d->1d and 1D->0d.
                Going through sides of mortar grid, we still cannot avoid 
                this double counting of nodes and matching coordinates won't 
                give us the right mapping we need.
                
                See figure for easier understanding what's happening in the 
                Geiger 2D case: 
                https://www.dropbox.com/s/5rl7bbnfcbmdf83/20200206_165846.jpg?dl=0                
                """
                nodes_map = self.dof_matcher(nodes_down_coord, nodes_up_coord)

                # find face normals of interface faces of higher-dim grid
                cf_up = g_up.cell_faces.tocsr()
                # outward or inward?
                outward = cf_up.data[cf_up.indptr[faces[0]]:
                                    cf_up.indptr[faces[0] + 1]][0]
                face_normal = g_up.face_normals[:, 0] * outward

                # map normal to reference domain
                R = cg.project_plane_matrix(g_up.nodes, check_planar=False)
                face_normal = np.dot(R, face_normal)

                Tr = self.trace_op(g_up.dim, face_normal, nodes_map, nodes_up,
                                nodes_down, g_up, g_down)

                A_bdry = self.local_stiff_matrix(g_down) + \
                        self.local_mass_matrix(g_down)

                A[nn, nn] += Tr.T * A_bdry * Tr

        return sps.bmat(A, format='csr')

    # ------------------------------------------------------------------------ #

    def A_reg_curl(self):
        # only 2D and 3D
        grids_23 = self.gb.get_grids(lambda g: g.dim >= 2)
        n_grids_23 = grids_23.size
        A = np.empty(shape=(n_grids_23, n_grids_23), dtype=np.object)

        for g, d in self.gb:
            nn = d["node_number"]
            local_matrix = self.local_stiff_matrix(g) + \
                           self.local_mass_matrix(g)
            if g.dim == 3:
                A[nn, nn] = sps.block_diag([local_matrix] * 3)
            else:
                A[nn, nn] = local_matrix

        for e, de in self.gb.edges():
            # get lower and higher dimensional grids on this interface
            g_down, g_up = self.gb.nodes_of_edge(e)
            nn_g_up = self.gb.graph.node[g_up]["node_number"]
            mg = de["mortar_grid"]

            # find all lower-dim cells and higher-dim faces that match on
            # this interface
            cells, faces, data = sps.find(
                mg.slave_to_mortar_int().T * mg.master_to_mortar_int())

            # nodes of higher-dim grid
            nodes_up = np.unique(g_up.face_nodes()[:, faces])
            nodes_up_coord = g_up.nodes[:, nodes_up]

            # nodes of lower-dim grid
            nodes_down = np.unique(g_down.cell_nodes()[:, cells])
            nodes_down_coord = g_down.nodes[:, nodes_down]

            nodes_map = self.dof_matcher(nodes_down, nodes_up)

            # find face normals of interface faces of higher-dim grid
            cf_up = g_up.cell_faces.tocsr()
            # outward or inward?
            outward = cf_up.data[cf_up.indptr[faces[0]]:
                                 cf_up.indptr[faces[0] + 1]][0]
            face_normal = g_up.face_normals[:, 0] * outward

            Tr = self.trace_op(g_up.dim, face_normal)
            A_bdry = self.local_stiff_matrix(g_down) + \
                     self.local_mass_matrix(g_down)

            A[nn, nn] += Tr.T * A_bdry * Tr

        return sps.bmat(A, format='csr')

    # ------------------------------------------------------------------------ #

    def local_stiff_matrix(self, g):
        """ Return the matrix for a H1 stiffness matrix using P1 elements.

        Parameters
        ----------
        g: grid, or a subclass, with geometry fields computed.

        Returns
        ------
        matrix: sparse csr (g.num_nodes, g.num_nodes)
            Matrix obtained from the discretization.
        """

        # If a 0-d grid is given then we return an identity matrix
        if g.dim == 0:
            M = sps.csr_matrix((g.num_nodes, g.num_nodes))
            return M

        # Map the domain to a reference geometry (i.e. equivalent to compute
        # surface coordinates in 1d and 2d)
        _, _, _, _, _, node_coords = cg.map_grid(g)

        # Allocate the data to store matrix entries, that's the most efficient
        # way to create a sparse matrix.
        size = np.power(g.dim + 1, 2) * g.num_cells
        I = np.empty(size, dtype=np.int)
        J = np.empty(size, dtype=np.int)
        dataIJ = np.empty(size)
        idx = 0

        cell_nodes = g.cell_nodes()
        nodes, cells, _ = sps.find(cell_nodes)

        for c in np.arange(g.num_cells):
            # For the current cell retrieve its nodes
            loc = slice(cell_nodes.indptr[c], cell_nodes.indptr[c + 1])

            nodes_loc = nodes[loc]
            coord_loc = node_coords[:, nodes_loc]

            # Compute the stiff-H1 local matrix
            A = self.stiffH1(
                g.cell_volumes[c],
                coord_loc,
                g.dim,
            )

            # Save values for stiff-H1 local matrix in the global structure
            cols = np.tile(nodes_loc, (nodes_loc.size, 1))
            loc_idx = slice(idx, idx + cols.size)
            I[loc_idx] = cols.T.ravel()
            J[loc_idx] = cols.ravel()
            dataIJ[loc_idx] = A.ravel()
            idx += cols.size

        # Construct the global matrices
        M = sps.csr_matrix((dataIJ, (I, J)))

        return M

    # ------------------------------------------------------------------------ #

    def local_mass_matrix(self, g):
        """ Return the matrix for a H1 mass matrix using P1 elements.

        Parameters
        ----------
        g: grid, or a subclass, with geometry fields computed.

        Returns
        ------
        matrix: sparse csr (g.num_nodes, g.num_nodes)
            Matrix obtained from the discretization.
        """

        # If a 0-d grid is given then we return an identity matrix
        if g.dim == 0:
            M = sps.csr_matrix((g.num_nodes, g.num_nodes))
            return M

        # Allocate the data to store matrix entries, that's the most efficient
        # way to create a sparse matrix.
        size = np.power(g.dim + 1, 2) * g.num_cells
        I = np.empty(size, dtype=np.int)
        J = np.empty(size, dtype=np.int)
        dataIJ = np.empty(size)
        idx = 0

        cell_nodes = g.cell_nodes()
        nodes, cells, _ = sps.find(cell_nodes)

        for c in np.arange(g.num_cells):
            # For the current cell retrieve its nodes
            loc = slice(cell_nodes.indptr[c], cell_nodes.indptr[c + 1])

            nodes_loc = nodes[loc]

            # Compute the stiff-H1 local matrix
            A = self.massH1(
                g.cell_volumes[c],
                g.dim,
            )

            # Save values for stiff-H1 local matrix in the global structure
            cols = np.tile(nodes_loc, (nodes_loc.size, 1))
            loc_idx = slice(idx, idx + cols.size)
            I[loc_idx] = cols.T.ravel()
            J[loc_idx] = cols.ravel()
            dataIJ[loc_idx] = A.ravel()
            idx += cols.size

        # Construct the global matrices
        M = sps.csr_matrix((dataIJ, (I, J)))

        return M

    # ------------------------------------------------------------------------ #

    def stiffH1(self, c_volume, coord, dim):
        """ Compute the local stiffness H1 matrix using the P1 approach.
        Parameters
        ----------
        c_volume : scalar
            Cell volume.
        Return
        ------
        out: ndarray (num_faces_of_cell, num_faces_of_cell)
            Local mass H1 matrix.
        """

        Q = np.hstack((np.ones((dim + 1, 1)), coord.T))
        dphi = np.linalg.inv(Q)[1:, :]

        return c_volume * np.dot(dphi.T, dphi)

    # ------------------------------------------------------------------------ #

    def massH1(self, c_volume, dim):
        """ Compute the local stiffness H1 matrix using the P1 approach.
        Parameters
        ----------
        c_volume : scalar
            Cell volume.
        Return
        ------
        out: ndarray (num_faces_of_cell, num_faces_of_cell)
            Local mass H1 matrix.
        """

        M = np.ones((dim + 1, dim + 1)) + np.identity(dim + 1)

        return c_volume * M / ((dim + 1) * (dim + 2))

    # ------------------------------------------------------------------------ #

    def trace_op(self, dim, normal, nodes_map, nodes_up, nodes_down, g_up,
                 g_down):
        I = nodes_down[nodes_map]
        J = nodes_up
        dataIJ = np.ones(nodes_up.size)

        R = sps.csr_matrix((dataIJ, (I, J)), shape=(g_down.num_nodes,
                                                  g_up.num_nodes))

        R_d = [R * normal[d] for d in range(dim)]

        return sps.hstack(R_d, format='csr')

    # ------------------------------------------------------------------------ #

    @staticmethod
    def dof_matcher(xy_1, xy_2):
        """
        Returns the indices such that xy_1[mapping[i]] = xy_2[i]
        """

        dim = np.min((xy_1.shape[0], xy_2.shape[0]))

        try:
            xy_dict = {}
            for i, xy in enumerate(xy_1.T):
                xy_dict[str(xy[:dim])] = i

            mapping = np.zeros(xy_2.shape[1])
            for i, xy in enumerate(xy_2.T):
                mapping[i] = xy_dict[str(xy[:dim])]

        except:
            longlist = np.hstack((xy_1[:dim, :], xy_2[:dim, :]))
            _, _, mapping = setmembership.unique_columns_tol(longlist)

            mapping = mapping[xy_1.shape[1]:]

        import pdb; pdb.set_trace()

        assert np.all(np.sort(mapping) == np.arange(len(mapping)))

        return mapping