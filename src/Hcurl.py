import numpy as np
import scipy.sparse as sps

import sys; sys.path.insert(0, '/home/anci/Dropbox/porepy/src/')
import porepy as pp

# ---------------------------------------------------------------------------- #
# Class of P1 functions on 2D grids


class Hcurl(object):

    def __init__(self, gb):
        # Grid bucket
        self.gb = gb
        self.tol = 1e-10

    # ------------------------------------------------------------------------ #

    def ndof(self, g):
        if isinstance(g, pp.Grid) or isinstance(g, pp.MortarGrid):
            return g.num_nodes
        else:
            raise ValueError

    # ------------------------------------------------------------------------ #

    def curl_jump(self, e, mg):
        # jump part of mixed-dim curl operator

        # lower and higher-dim grid adjacent to the mortar grid mg
        g_down, g_up = self.gb.nodes_of_edge(e)

        # jump
        J = np.zeros(shape=(g_down.num_faces, g_up.num_nodes))

        # find which higher-dim faces and lower-dim faces (cells) relate to
        # each other
        cells, faces, data = sps.find(
            mg.slave_to_mortar_int.T * mg.master_to_mortar_int)

        for i in np.arange(np.size(data)):
            face = faces[i]
            cell = cells[i]

            # find face normals of interface faces of higher-dim grid
            cf_up = g_up.cell_faces.tocsr()
            # outward or inward?
            outward = cf_up.data[cf_up.indptr[face]:cf_up.indptr[face+1]][0]
            face_normal = g_up.face_normals[:2, face] * outward
            # rotate the normal
            face_normal_rot = np.array([-face_normal[1], face_normal[0]])

            # nodes of that face
            fn_up = g_up.face_nodes
            nodes_up = fn_up.indices[fn_up.indptr[face]:fn_up.indptr[face+1]]

            # corresponding nodes in lower-dim grid
            cf_down = g_down.cell_faces
            nodes_down = cf_down.indices[cf_down.indptr[cell]:cf_down.indptr[
                cell+1]]

            # find their normals (orientation)
            node_normals_down = g_down.face_normals[:2, nodes_down]
            # orientation identifying
            # we say that orientations align if the rotated mortar
            # normal corresponds to the normal of the
            # lower-dimensional face
            orientations = np.sign(np.dot(face_normal_rot, node_normals_down))

            # check the coordinates! swap the nodes if coordinates don't match
            down_xy = g_down.face_centers[:, nodes_down]
            up_xy = g_up.nodes[:, nodes_up]

            if np.linalg.norm(down_xy[:, 0] - up_xy[:, 0]) < self.tol:
                J[nodes_down[0], nodes_up[0]] = orientations[0]
                J[nodes_down[1], nodes_up[1]] = orientations[1]
            else:
                J[nodes_down[0], nodes_up[1]] = orientations[1]
                J[nodes_down[1], nodes_up[0]] = orientations[0]

        return sps.csc_matrix(J, dtype='d')

    # ------------------------------------------------------------------------ #

    def curl_grid(self, g):
        # local curl matrix (not yet decomposed into interior
        # and mortar)
        C_g = sps.csc_matrix(g.face_nodes, dtype='d')

        # Curl mapping from nodes to faces in 2D
        for face in np.arange(g.num_faces):
            # find indices for nodes of this face
            loc = C_g.indices[C_g.indptr[face]:C_g.indptr[face + 1]]
            # xy(z) coordinates of two nodes
            nodes_xy = g.nodes[:, loc]
            # face normal
            face_normal = g.face_normals[:, face]

            face_tangent = nodes_xy[:2, 0] - nodes_xy[:2, 1]
            face_normal_rot = np.array([-face_normal[1], face_normal[
                0]])
            # if the face tangent and the rotated normal are of same
            # orientation, then we say the curl of first node follows
            # that orientation and the second node doesn't (hence, *(-1))
            # vice versa if they have different orientation
            if np.dot(face_normal_rot, face_tangent) > 0:
                C_g[loc[1], face] = -1.
            else:
                C_g[loc[0], face] = -1.

        # transpose! we need nodes_faces mapping
        return C_g.T.tocsr()

    # ------------------------------------------------------------------------ #

    def curl(self):

        # mixed-dimensional curl operator
        # curl : H(curl) -> H(div)
        # we loop over all curl dofs, which are (for now) all nodes of 2D grids
        # in grid bucket self.gb
        # then we take curl of those dofs to land in all div dofs, which are
        # all faces of 2D grids (triangle edges) and 1D grids (line segment end
        # nodes) and
        # all cells of 1D mortar grids (line segments) and 0D mortar grids (
        # intersection nodes)
        # this means that the curl operator has three parts:
        # (1) standard curl from interior curl dofs to interior div dofs of each
        # 2D grid
        # (2) jump operator from boundary curl dofs of 2D grid to div dofs of
        # all neighbouring 1D grids
        # (3) restriction operator (transpose of extension operator) from
        # boundary curl dofs of 2D grid to neighbouring mortar div dofs (
        # because div dofs are decomposed into u_0 + R\lambda )

        # global curl matrix
        curl_all_grids = np.empty(shape=(self.gb.size(),
                                         len(self.gb.grids_of_dimension(2))),
                                         dtype=np.object)

        # init
        for g in self.gb.grids_of_dimension(2):
            d = self.gb.graph.nodes[g]

            # node number of this grid
            nn_g = d['node_number']
            num_nodes_g = g.num_nodes

            for gg, d_gg in self.gb:
                nn_gg = d_gg["node_number"]

                curl_all_grids[nn_gg, nn_g] = \
                    sps.csr_matrix((gg.num_faces, num_nodes_g), dtype='d')

            for e, d_e in self.gb.edges():
                mg = d_e['mortar_grid']
                nn_mg = d_e['edge_number'] + self.gb.num_graph_nodes()

                curl_all_grids[nn_mg, nn_g] = \
                    sps.csr_matrix((mg.num_cells, num_nodes_g), dtype='d')

        # loop over all 2D grids
        for g in self.gb.grids_of_dimension(2):
            # grid data
            d = self.gb.graph.nodes[g]

            # node number of this grid
            nn_g = d['node_number']

            # local curl matrix for this grid
            C_g = self.curl_grid(g)

            curl_all_grids[nn_g, nn_g] += C_g

            # decompose local curl matrix to interior and mortar (and add jump)
            # loop over adjacent 1D mortar grids
            for e, d_e in self.gb.edges_of_node(g):
                mg = d_e['mortar_grid']
                if mg.dim == g.dim - 1:
                    # splitting of curl to interior and mortar part

                    # Recover the orientation information
                    faces_h, _, sign_h = sps.find(g.cell_faces)
                    _, ind_faces_h = np.unique(faces_h, return_index=True)
                    sign_h = sign_h[ind_faces_h]

                    Signs = sps.diags(sign_h)

                    # face to mortar cell mapping
                    P_mg = mg.master_to_mortar_int
                    nn_mg = d_e['edge_number'] + self.gb.num_graph_nodes()

                    # Extract mortar (1D) dofs and remove from interior (2D)
                    curl_all_grids[nn_mg, nn_g] += P_mg * Signs * C_g
                    curl_all_grids[nn_g, nn_g] -= P_mg.T * P_mg * C_g

                    # jump curl part (maps 2D nodes to 1D nodes)
                    J_g = self.curl_jump(e, mg)

                    g_down, _ = self.gb.nodes_of_edge(e)
                    nn_g_down = self.gb.graph.node[g_down]['node_number']

                    curl_all_grids[nn_g_down, nn_g] += J_g

                    # splitting of jump to interior and mortar part

                    # loop over 0D mortar grids connected to the child g_down
                    for e_down, d_e_down in self.gb.edges_of_node(g_down):

                        mg_down = d_e_down['mortar_grid']

                        if mg_down.dim == g_down.dim - 1:
                            nn_mg_down = d_e_down['edge_number'] + \
                                         self.gb.num_graph_nodes()

                            # Recover whether the normals have the same sign
                            faces_h, _, sign_h = sps.find(g_down.cell_faces)
                            _, ind_faces_h = np.unique(faces_h, return_index=True)
                            sign_h = sign_h[ind_faces_h]

                            # Velocity degree of freedom matrix
                            Signs = sps.diags(sign_h)

                            Pi_mg = mg_down.master_to_mortar_int

                            # Extract mortar (0D) dofs and remove from
                            # interior (1D)
                            curl_all_grids[nn_mg_down, nn_g] += Pi_mg * Signs\
                                                                * J_g
                            curl_all_grids[nn_g_down, nn_g] -= Pi_mg.T * \
                                                               Pi_mg * J_g


        return sps.bmat(curl_all_grids, format='csr')

    # ------------------------------------------------------------------------ #


class Projections(object):

    def __init__(self, gb):
        # Grid bucket
        self.gb = gb
        self.tol = 1e-10

    # ------------------------------------------------------------------------ #

    def Pi_div_h(self):
        # global Pi matrix
        Pi = np.empty(shape=(self.gb.size(), self.gb.size()), dtype=np.object)

        # assume 2D case
        for g, d in self.gb:
            nn_g = d['node_number']
            if g.dim == 2:
                Pi_gx = np.zeros(shape=(g.num_faces, g.num_nodes))
                Pi_gy = np.zeros(shape=(g.num_faces, g.num_nodes))

                for face in np.arange(g.num_faces):
                    fn = g.face_nodes
                    nodes = fn.indices[fn.indptr[face]:fn.indptr[face+1]]
                    face_normal = g.face_normals[:, face]
                    normal_norm = np.linalg.norm(face_normal)
                    face_area = g.face_areas[face]

                    Pi_gx[face, nodes] = face_normal[0] * face_area \
                                         / (2 * normal_norm)
                    Pi_gy[face, nodes] = face_normal[1] * face_area \
                                         / (2 * normal_norm)

                Pi_g = sps.csr_matrix(np.hstack((Pi_gx, Pi_gy)))

            else:
                Pi_g = sps.identity(g.num_nodes, format='csr')

            Pi[nn_g, nn_g] = Pi_g

            # loop over mortar grids of lower dimension
            for e, d_e in self.gb.edges_of_node(g):
                mg = d_e['mortar_grid']
                if mg.dim == g.dim - 1:
                    # splitting of curl to interior and mortar part

                    # face to mortar cell mapping
                    P_mg = mg.master_to_mortar_int
                    nn_mg = d_e['edge_number'] + self.gb.num_graph_nodes()

                    # Recover the orientation information
                    faces_h, _, sign_h = sps.find(g.cell_faces)
                    _, ind_faces_h = np.unique(faces_h, return_index=True)
                    sign_h = sign_h[ind_faces_h]

                    Signs = sps.diags(sign_h)

                    Pi[nn_mg, nn_g] = P_mg * Signs * Pi_g
                    Pi[nn_g, nn_g] -= P_mg.T * P_mg * Pi_g

        return sps.bmat(Pi, format='csr')

    # ------------------------------------------------------------------------ #
