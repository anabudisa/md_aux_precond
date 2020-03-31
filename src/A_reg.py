import numpy as np
import scipy.sparse as sps
import time

from porepy.utils import comp_geom as cg


class A_reg(object):
    """
    Class of mixed-dimensional regular (H1) inner products for functions in
    mixed-dimensional H(div) and H(curl). Using vector and scalar P1 elements.
    Parameters
    ----------
    :param gb: graph
        grid bucket as graph of grids
    :param tol: scalar
        set tolerance for matching floats
    :param cpu_time: list
        measuring CPU time for assembling inner product matrices
    """

    def __init__(self, gb):
        # Grid bucket
        self.gb = gb
        self.tol = 1e-12
        self.cpu_time = []
        # local matrices
        self.mass_matrices = np.empty(shape=(self.gb.num_graph_nodes(),) ,
                                      dtype=np.object)
        self.stiff_matrices = np.empty(shape=(self.gb.num_graph_nodes(),) ,
                                       dtype=np.object)
        self.div_Tr = np.empty(shape=(self.gb.num_graph_edges(),2) ,
                                      dtype=np.object)
        self.curl_Tr = np.empty(shape=(self.gb.num_graph_edges(),2) ,
                                      dtype=np.object)
        # Set local matrices
        self.set_local_matrices()
        self.set_traces()

    # ------------------------------------------------------------------------ #

    def reg_div(self):
        """
        Assemble the H1 mass and stiffness matrix for the regularized
        div functions by P1 elements.
        --------
        :return:
        scipy.sparse.csr_matrix
            of (num_grids, num_grids) blocks,
            where every block is scipy.sparse.csr_matrix of local grid mass
            matrices
        scipy.sparse.csr_matrix
            of (num_grids, num_grids) blocks,
            where every block is scipy.sparse.csr_matrix of local grid stiffness
            matrices
        """
        start_time = time.time()
        A_mass = np.empty(shape=(self.gb.num_graph_nodes(),
                                 self.gb.num_graph_nodes()), dtype=np.object)
        A_stiff = np.empty(shape=(self.gb.num_graph_nodes(),
                                  self.gb.num_graph_nodes()), dtype=np.object)

        # interior H1 matrices
        for g, d in self.gb:
            nn = d["node_number"]
            if g.dim == 0:
                A_mass[nn, nn] = self.mass_matrices[nn]
                A_stiff[nn, nn] = self.stiff_matrices[nn]
            else:
                A_mass[nn, nn] = sps.block_diag([self.mass_matrices[nn]]*g.dim)
                A_stiff[nn, nn] = sps.block_diag([self.stiff_matrices[nn]]*g.dim)

        # boundary H1 matrices
        for e, de in self.gb.edges():
            # get lower and higher dimensional grids on this interface
            g_down, g_up = self.gb.nodes_of_edge(e)
            nn_g_up = self.gb.graph.node[g_up]["node_number"]
            nn_g_down = self.gb.graph.node[g_down]["node_number"]
            mg = de["mortar_grid"]
            nn_e = de["edge_number"]

            for side in np.arange(mg.num_sides()):
                # Get div trace operator
                Tr = self.div_Tr[nn_e, side]
                if Tr is None:
                    import pdb; pdb.set_trace()
                # local boundary H1 matrix
                A_mass[nn_g_up, nn_g_up] += Tr.T * self.mass_matrices[nn_g_down] * Tr
                A_stiff[nn_g_up, nn_g_up] += Tr.T * self.stiff_matrices[nn_g_down] * Tr

        t = time.time() - start_time
        self.cpu_time.append(["Reg div", str(t)])

        return sps.bmat(A_mass, format='csr'), sps.bmat(A_stiff, format='csr')

    # ------------------------------------------------------------------------ #

    # TODO: still missing traces of 2 dimensions down
    def reg_curl(self):
        """
        Assemble the H1 mass and stiffness matrix for the regularized curl
        functions by P1 elements.
        --------
        :return:
        scipy.sparse.csr_matrix
            of (num_grids_of_dim_2&3, num_grids_of_dim_2&3) blocks,
            where every block is scipy.sparse.csr_matrix of local grid
            mass matrices
        scipy.sparse.csr_matrix
            of (num_grids_of_dim_2&3, num_grids_of_dim_2&3) blocks,
            where every block is scipy.sparse.csr_matrix of local grid
            stiffness matrices
        """
        start_time = time.time()

        # only 2D and 3D
        grids_23 = self.gb.get_grids(lambda g: g.dim >= 2)
        n_grids_23 = grids_23.size
        A_mass = np.empty(shape=(n_grids_23, n_grids_23), dtype=np.object)
        A_stiff = np.empty(shape=(n_grids_23, n_grids_23), dtype=np.object)

        for g in grids_23:
            d = self.gb.graph.node[g]
            nn = d["node_number"]

            if g.dim == 3:
                A_mass[nn, nn] = sps.block_diag([self.mass_matrices[nn]] * 3)
                A_stiff[nn, nn] = sps.block_diag([self.stiff_matrices[nn]] * 3)
            else:
                A_mass[nn, nn] = self.mass_matrices[nn]
                A_stiff[nn, nn] = self.stiff_matrices[nn]

        # boundary H1 matrices
        for e, de in self.gb.edges():
            # get lower and higher dimensional grids on this interface
            g_down, g_up = self.gb.nodes_of_edge(e)
            nn_g_up = self.gb.graph.node[g_up]["node_number"]
            nn_g_down = self.gb.graph.node[g_down]["node_number"]
            mg = de["mortar_grid"]
            nn_e = de["edge_number"]

            # only 3d and 2d grids are considered
            if g_up.dim >= 2:
                for side in np.arange(mg.num_sides()):
                    # Get curl trace operator
                    Tr = self.curl_Tr[nn_e, side]

                    # local boundary H1 matrix
                    if g_up.dim == 3:
                        A_bdry_mass = sps.block_diag([self.mass_matrices[nn_g_down]] * 3)
                        A_bdry_stiff = sps.block_diag([self.stiff_matrices[nn_g_down]] * 3)
                    else:
                        A_bdry_mass = self.mass_matrices[nn_g_down]
                        A_bdry_stiff = self.stiff_matrices[nn_g_down]

                    A_mass[nn_g_up, nn_g_up] += Tr.T * A_bdry_mass * Tr
                    A_stiff[nn_g_up, nn_g_up] += Tr.T * A_bdry_stiff * Tr

        t = time.time() - start_time
        self.cpu_time.append(["Reg curl", str(t)])

        return sps.bmat(A_mass, format='csr'), sps.bmat(A_stiff, format='csr')

    # ------------------------------------------------------------------------ #
    # @profile
    def set_traces(self):
        """
        Set restriction operators from nodes of higher dimension grid to
        nodes of lower dimension grid

        :return:
        """
        start_time = time.time()

        # boundary H1 matrices
        for e, de in self.gb.edges():
            # get lower and higher dimensional grids on this interface
            g_down, g_up = self.gb.nodes_of_edge(e)
            nn_g_up = self.gb.graph.node[g_up]["node_number"]
            nn_g_down = self.gb.graph.node[g_down]["node_number"]
            mg = de["mortar_grid"]
            nn_e = de["edge_number"]

            for side in np.arange(mg.num_sides()):
                cells_per_side = mg.num_cells / mg.num_sides()
                local_slice = slice(side * cells_per_side,
                                    (side + 1) * cells_per_side)

                # find all lower-dim cells and higher-dim faces that
                # match on
                # this interface
                cells, faces, data = sps.find(
                    mg.slave_to_mortar_int()[local_slice,
                    :].T * mg.master_to_mortar_int()[local_slice, :])

                # restriction operator
                size = g_up.dim * np.size(data)
                I = np.empty(size, dtype=np.int)
                J = np.empty(size, dtype=np.int)
                dataIJ = np.zeros(size)

                count = 0
                for i in np.arange(np.size(data)):
                    face = faces[i]
                    cell = cells[i]

                    # nodes of higher-dim grid
                    nodes_up = g_up.face_nodes[:, face].tocoo().row
                    nodes_up = np.unique(nodes_up)
                    nodes_up_coord = g_up.nodes[:, nodes_up]

                    # nodes of lower-dim grid
                    cell_nodes = g_down.face_nodes * np.abs(g_down.cell_faces[:, cell])
                    nodes_down = cell_nodes.tocoo().row
                    nodes_down = np.unique(nodes_down)
                    nodes_down_coord = g_down.nodes[:, nodes_down]

                    # how many nodes
                    nodes_count = np.size(nodes_up)

                    # nodes_down_coord[ind_match] = nodes_up_coord
                    idx_match = self.match_coordinates(nodes_up_coord,
                                                       nodes_down_coord)

                    # update restriction matrix
                    idx_track = slice(count, count + nodes_count)
                    I[idx_track] = np.ravel(nodes_down[idx_match])
                    J[idx_track] = np.ravel(nodes_up)
                    dataIJ[idx_track] = np.ravel(
                        np.full(nodes_count, True))

                    count += nodes_count

                # Restriction of nodes from higher dim grid to lower
                # dim grid
                Restriction = sps.csr_matrix((dataIJ, (I, J)), shape=(
                    g_down.num_nodes, g_up.num_nodes))

                # find face normals of interface faces of higher-dim
                # grid
                cf_up = g_up.cell_faces.tocsr()
                # outward or inward?
                outward = cf_up.data[cf_up.indptr[faces[0]]:
                                     cf_up.indptr[faces[0] + 1]][0]
                face_normal = g_up.face_normals[:, 0] * outward
                face_normal /= np.linalg.norm(face_normal)

                # map normal to reference domain
                if g_up.dim == self.gb.dim_max():
                    R = np.identity(3)
                elif g_up.dim == 2:
                    R = cg.project_plane_matrix(g_up.nodes,
                                                check_planar=False)
                else:
                    R = cg.project_line_matrix(g_up.nodes)

                face_normal = np.dot(R, face_normal)

                # Set curl trace operator
                if g_up.dim > 1:
                    self.curl_Tr[nn_e, side] = self.trace_op_curl(g_up.dim, face_normal, Restriction)
                # Set div trace operator
                self.div_Tr[nn_e, side] = self.trace_op_div(g_up.dim, face_normal, Restriction)

        t = time.time() - start_time
        self.cpu_time.append(["Set traces", str(t)])

    # ------------------------------------------------------------------------ #

    def set_local_matrices(self):
        """ Setup local H1 stiffness and mass matrices for every grid in the
        gridbucket.

        """
        start_time = time.time()

        for g, d in self.gb:
            nn = d["node_number"]
            self.stiff_matrices[nn], self.mass_matrices[nn] = self.local_stiff_and_mass(g)

        t = time.time() - start_time
        self.cpu_time.append(["Set local matrices", str(t)])

    # ------------------------------------------------------------------------ #
    def local_stiff_and_mass(self, g):
        """ Return the H1 stiffness matrix local to a grid using P1 elements.
        Parameters
        ----------
        g: grid, or a subclass, with geometry fields computed.
        Returns
        ------
        matrix: sparse csr (g.num_nodes, g.num_nodes)
            Stiffness matrix obtained from the discretization.
        matrix: sparse csr (g.num_nodes, g.num_nodes)
            Mass matrix obtained from the discretization.
        """

        # If a 0-d grid is given then we return an identity matrix
        if g.dim == 0:
            M = sps.identity(1)
            return M

        # Map the domain to a reference geometry (i.e. equivalent to compute
        # surface coordinates in 1d and 2d)
        _, _, _, _, _, node_coords = cg.map_grid(g)

        # Allocate the data to store matrix entries, that's the most efficient
        # way to create a sparse matrix.
        size = np.power(g.dim + 1, 2) * g.num_cells
        I = np.empty(size, dtype=np.int)
        J = np.empty(size, dtype=np.int)
        data_stiff = np.empty(size)
        data_mass = np.empty(size)
        idx = 0

        cell_nodes = g.cell_nodes()

        for c in np.arange(g.num_cells):
            # For the current cell retrieve its nodes
            loc = slice(cell_nodes.indptr[c], cell_nodes.indptr[c + 1])

            nodes_loc = cell_nodes.indices[loc]
            coord_loc = node_coords[:, nodes_loc]

            # Compute the stiff-H1 local matrix
            A_local = self.stiffH1(
                g.cell_volumes[c],
                coord_loc,
                g.dim,
            )
            M_local = self.massH1(
                g.cell_volumes[c],
                g.dim,
            )

            # Save values for stiff-H1 local matrix in the global structure
            cols = np.tile(nodes_loc, (nodes_loc.size, 1))
            loc_idx = slice(idx, idx + cols.size)
            I[loc_idx] = cols.T.ravel()
            J[loc_idx] = cols.ravel()
            data_stiff[loc_idx] = A_local.ravel()
            data_mass[loc_idx] = M_local.ravel()
            idx += cols.size

        # Construct the global matrices
        A = sps.csr_matrix((data_stiff, (I, J)))
        M = sps.csr_matrix((data_mass, (I, J)))

        return A, M

    # ------------------------------------------------------------------------ #

    def stiffH1(self, c_volume, coord, dim):
        """ Compute the H1 stiffness matrix local to a cell using the P1
        approach.
        Parameters
        ----------
        c_volume : scalar
            Cell volume.
        coord : ndarray (dim, num_nodes_of_cell)
            Coordinates of cell nodes
        dim : scalar
            Grid dimension
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
        """ Compute the H1 mass matrix local to a cell using the P1 approach.
        Parameters
        ----------
        c_volume : scalar
            Cell volume.
        dim : scalar
            Grid dimension
        Return
        ------
        out: ndarray (num_faces_of_cell, num_faces_of_cell)
            Local mass H1 matrix.
        """

        M = np.ones((dim + 1, dim + 1)) + np.identity(dim + 1)

        return c_volume * M / ((dim + 1) * (dim + 2))

    # ------------------------------------------------------------------------ #

    def trace_op_div(self, dim, normal, R):
        """
        Assemble the trace operator mapping normal trace of current grid to
        interface (lower-dim) grids.
        ----------
        :param dim: scalar
            Grid dimension.
        :param normal: ndarray (3, 1)
            Unit outward normal on grid boundary.
        :param R: ndarray (g_down.num_nodes, g_up.num_nodes)
            Restriction to lower-dim interface grid (already mapped to
            reference domain).
        :return:
        scipy.sparse.csr_matrix (g_down.num_nodes, dim * g_up.num_nodes)
            Trace operator.
        """

        R_d = [R * normal[d] for d in range(dim)]

        return sps.hstack(R_d, format='csr')

    # ------------------------------------------------------------------------ #

    def trace_op_curl(self, dim, normal, R):
        """
        Assemble the trace operator mapping "cross" trace of current grid to
        interface (lower-dim) grids.
        ----------
        :param dim: scalar
            Grid dimension.
        :param normal: ndarray (3, 1)
            Unit outward normal on grid boundary.
        :param R: ndarray (g_down.num_nodes, g_up.num_nodes)
            Restriction to lower-dim interface grid (already mapped to
            reference domain).
        :return:
        scipy.sparse.csr_matrix (g_down.num_nodes, dim * g_up.num_nodes)
            Trace operator.
        """

        if dim == 3:
            R_d = [[None, R * normal[2], -R * normal[1]],
                   [-R * normal[2], None, R * normal[0]],
                   [R * normal[1], -R * normal[0], None]]
            return sps.bmat(R_d, format='csr')
        else:
            return R

    # ------------------------------------------------------------------------ #

    @staticmethod
    def match_coordinates(a, b):
        # TODO: check how slow this is (WB: Still the fastest I can think of)
        # compare and match columns of a and b
        # return: ind s.t. b[:, ind] = a
        # Note: we assume that all columns will match
        #       and a and b match in shape
        ind = np.empty((b.shape[1],), dtype=int)
        for i in np.arange(a.shape[1]):
            for j in np.arange(b.shape[1]):
                if np.linalg.norm(a[:, i] - b[:, j]) < 1e-12:
                    ind[i] = j
                    break

        return ind
