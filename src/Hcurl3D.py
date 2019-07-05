import numpy as np
import scipy.sparse as sps
import time

from porepy.utils import setmembership
from porepy.utils import comp_geom as cg

# ---------------------------------------------------------------------------- #
# Class of Hcurl functions on 3D grids


class Hcurl(object):

    def __init__(self, gb):
        # Grid bucket
        self.gb = gb
        self.tol = 1e-12
        self.cpu_time = []

        self.num_edges = np.zeros(shape=(self.gb.size(),), dtype=int)
        self.face_edges = np.empty(shape=(self.gb.size(),), dtype=np.object)
        self.edge_nodes = np.empty(shape=(self.gb.size(),), dtype=np.object)
        
        self.edge_centers = np.empty(shape=(self.gb.size(),), dtype=np.object)
        self.edge_tangents = np.empty(shape=(self.gb.size(),), dtype=np.object)

        self.compute_edges()

    # ------------------------------------------------------------------------ #

    def curl_jump_2d(self, e, mg):
        start_time = time.time()
        # jump part of mixed-dim curl operator

        # lower and higher-dim grid adjacent to the mortar grid mg
        g_down, g_up = self.gb.nodes_of_edge(e)

        # Mapping to the right plane
        R = cg.project_plane_matrix(g_up.nodes, check_planar=False)

        # jump
        J = sps.lil_matrix((g_down.num_faces, g_up.num_nodes))

        # find which higher-dim faces and lower-dim faces (cells) relate to
        # each other
        cells, faces, data = sps.find(
            mg.slave_to_mortar_int().T * mg.master_to_mortar_int())

        # Precompute CSR version of cell_faces for fast lookup
        cf_up = g_up.cell_faces.tocsr()

        for i in np.arange(np.size(data)):
            face = faces[i]
            cell = cells[i]

            # find face normals of interface faces of higher-dim grid
            # outward or inward?
            outward = cf_up.data[cf_up.indptr[face]:cf_up.indptr[face+1]][0]
            face_normal = np.dot(R, g_up.face_normals[:, face] * outward)
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
            node_normals_down = np.dot(R, g_down.face_normals[:, nodes_down])
            # orientation identifying
            # we say that orientations align if the rotated mortar
            # normal corresponds to the normal of the
            # lower-dimensional face
            orientations = np.sign(np.dot(face_normal_rot,
                                          node_normals_down[:2, :]))

            # check the coordinates! swap the nodes if coordinates don't match
            down_xy = g_down.face_centers[:, nodes_down]
            up_xy = g_up.nodes[:, nodes_up]

            if np.linalg.norm(down_xy[:, 0] - up_xy[:, 0]) < self.tol:
                J[nodes_down[0], nodes_up[0]] = orientations[0]
                J[nodes_down[1], nodes_up[1]] = orientations[1]
            else:
                J[nodes_down[0], nodes_up[1]] = orientations[1]
                J[nodes_down[1], nodes_up[0]] = orientations[0]

        # Jump maps to zero at fracture tips
        J[g_down.tags['tip_faces'], :] = 0.

        t = time.time() - start_time
        self.cpu_time.append(["Curl jump 2d", str(t)])

        return sps.csc_matrix(J, dtype='d')

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

        return ind

    # ------------------------------------------------------------------------ #

    def curl_jump_3d(self, e, mg):
        start_time = time.time()
        # jump part of mixed-dim curl operator

        # lower and higher-dim grid adjacent to the mortar grid mg
        g_down, g_up = self.gb.nodes_of_edge(e)
        nn_g_up = self.gb.graph.node[g_up]["node_number"]
        num_edges_up = self.num_edges[nn_g_up]

        # jump
        # mapping 3D interface edges to 2D faces (edges of the triangles)
        J = sps.lil_matrix((g_down.num_faces, num_edges_up))

        # find which higher-dim faces and lower-dim faces (cells) relate to
        # each other
        cells, faces, data = sps.find(
            mg.slave_to_mortar_int().T * mg.master_to_mortar_int())

        # Precompute CSR version of cell_faces for fast lookup
        cf_up = g_up.cell_faces.tocsr()
        
        for i in np.arange(np.size(data)):
            face = faces[i]
            cell = cells[i]

            # find face normals of interface faces of higher-dim grid
            # outward or inward?
            outward = cf_up.data[cf_up.indptr[face]:cf_up.indptr[face + 1]][0]
            # outward face normal
            face_normal = g_up.face_normals[:, face] * outward

            # edges of that face
            fe_up = self.face_edges[nn_g_up]
            edges_up = fe_up.indices[
                       fe_up.indptr[face]:fe_up.indptr[face + 1]]

            # corresponding faces in lower-dim grid (2d faces = triangle
            # edges!)
            cf_down = g_down.cell_faces
            faces_down = cf_down.indices[
                         cf_down.indptr[cell]:cf_down.indptr[cell + 1]]

            # find their normals (orientation)
            face_normals_down = g_down.face_normals[:, faces_down]

            # Extract edges centers and tangents
            edges_centers_xyz = self.edge_centers[nn_g_up][:, edges_up]
            edges_tangents    = self.edge_tangents[nn_g_up][:, edges_up]

            # faces centers of the corresponding lower-dim cell
            down_xyz = g_down.face_centers[:, faces_down]
            # down_xyz[ind_match] = edges_centers_xyz
            ind_match = self.match_coordinates(edges_centers_xyz, down_xyz)

            # orientation identifying
            # we say that orientations align if the outer product of the face
            # normal and edge tangent in 3D match the face normal in 2D;
            # first get outer product of face_normal and edge tangents in 3D
            # ( check if we can do cross(vector, array_of_vectors)! )
            outer = -np.cross(face_normal, edges_tangents, axisb=0).T
            # these vectors for outer product should match face normals in 2d
            # cell (up to the permutation ind_match and direction);
            # that means that dot product of corresponding (ind_match) vectors
            # is +/-1, which defines our orientations
            orientations = np.zeros(3)
            for i in np.arange(3):
                dot_i = np.dot(outer[:, i], face_normals_down[:, ind_match[i]])
                orientations[i] = np.sign(dot_i)

            # also remember to match the appropriate edges (3d) and faces (2d)
            # with ind_match
            J[faces_down[ind_match], edges_up] = orientations

        import pdb; pdb.set_trace()
        # Jump maps to zero at fracture tips
        J[g_down.tags['tip_faces'], :] = 0.

        t = time.time() - start_time
        self.cpu_time.append(["Curl jump 3d", str(t)])

        return sps.csc_matrix(J, dtype='d')

    # ------------------------------------------------------------------------ #

    def curl_grid(self, g):
        start_time = time.time()
        # local curl matrix (not yet decomposed into interior
        # and mortar)
        if g.dim == 3:
            nn_g = self.gb.graph.node[g]["node_number"]
            C_g = self.face_edges[nn_g]
        else:
            C_g = sps.csc_matrix(g.face_nodes, dtype='d')

            # Mapping to the right plane
            R = cg.project_plane_matrix(g.nodes, check_planar=False)

            # Curl mapping from nodes to faces in 2D
            for face in np.arange(g.num_faces):
                # find indices for nodes of this face
                loc = C_g.indices[C_g.indptr[face]:C_g.indptr[face + 1]]
                # xy(z) coordinates of two nodes
                nodes_xy = np.dot(R, g.nodes[:, loc])
                # face normal
                face_normal = np.dot(R, g.face_normals[:, face])

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
        t = time.time() - start_time
        self.cpu_time.append(["Curl grid", str(t)])

        return C_g.T.tocsr()

    # ------------------------------------------------------------------------ #

    def curl(self):
        start_time = time.time()
        # mixed-dimensional curl operator
        # curl : H(curl) -> H(div)
        # we loop over all curl dof, which are all nodes of 3D grids
        # in grid bucket self.gb
        # then we take curl of those dof to land in all div dof
        # this means that the curl operator has three parts:
        # (1) standard curl from interior curl dof to interior div dof of each
        #     3D grid
        # (2) jump operator from boundary curl dof of 3D grid to div dof of
        #     all neighbouring 2D grids
        # (3) restriction operator (transpose of extension operator) from
        #     boundary curl dof of 3D grid to neighbouring mortar div dof (
        #     because div dof are decomposed into u_0 + R\lambda )

        # global curl matrix
        grids_23 = self.gb.get_grids(lambda g: g.dim >= 2)
        n_grids_23 = grids_23.size
        curl_all_grids = np.empty(shape=(self.gb.size(), n_grids_23),
                                  dtype=np.object)

        # init
        for g in grids_23:
            d = self.gb.graph.node[g]

            # node number of this grid
            nn_g = d['node_number']
            if g.dim == 3:
                num_edges_g = self.num_edges[nn_g]
            else:
                num_edges_g = g.num_nodes

            for gg, d_gg in self.gb:
                nn_gg = d_gg["node_number"]

                curl_all_grids[nn_gg, nn_g] = \
                    sps.csr_matrix((gg.num_faces, num_edges_g), dtype='d')

            for e, d_e in self.gb.edges():
                mg = d_e['mortar_grid']
                nn_mg = d_e['edge_number'] + self.gb.num_graph_nodes()

                curl_all_grids[nn_mg, nn_g] = \
                    sps.csr_matrix((mg.num_cells, num_edges_g), dtype='d')

        # loop over all 2D and 3D grids
        for g in grids_23:
            # grid data
            d = self.gb.graph.node[g]

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
                    P_mg = mg.master_to_mortar_int()
                    nn_mg = d_e['edge_number'] + self.gb.num_graph_nodes()

                    # Extract mortar (1D) dofs and remove from interior (2D)
                    curl_all_grids[nn_mg, nn_g] += P_mg * Signs * C_g
                    curl_all_grids[nn_g, nn_g] -= P_mg.T * P_mg * C_g

                    # jump curl part (maps 2D nodes to 1D nodes)
                    if g.dim == 3:
                        J_g = self.curl_jump_3d(e, mg)
                    else:
                        J_g = self.curl_jump_2d(e, mg)

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

                            Pi_mg = mg_down.master_to_mortar_int()

                            # Extract mortar (0D) dofs and remove from
                            # interior (1D)
                            curl_all_grids[nn_mg_down, nn_g] += Pi_mg * Signs\
                                                                * J_g
                            curl_all_grids[nn_g_down, nn_g] -= Pi_mg.T * \
                                                               Pi_mg * J_g

        t = time.time() - start_time
        self.cpu_time.append(["Curl", str(t)])

        return sps.bmat(curl_all_grids, format='csr')

    # ------------------------------------------------------------------------ #

    def compute_edges(self):
        # TODO: improve cpu time of this
        start_time = time.time()
        for g, d in self.gb:
            if g.dim == 3:
                ng = d["node_number"]
                # Pre-allocation
                edges = np.ndarray((3*g.num_faces, 2))
                orientations = np.ones(3*g.num_faces)

                for face in np.arange(g.num_faces):
                    # find indices for nodes of this face
                    loc = g.face_nodes.indices[g.face_nodes.indptr[face]:\
                                               g.face_nodes.indptr[face + 1]]
                    # Define edges between each pair of nodes
                    # according to right-hand rule
                    edges[3*face,   :] = [loc[0], loc[1]]
                    edges[3*face+1, :] = [loc[1], loc[2]]
                    edges[3*face+2, :] = [loc[2], loc[0]]

                    # Save orientation of each edge w.r.t. the face
                    orientation_loc = np.sign(loc[[1, 2, 0]] - loc)
                    orientations[3*face:3*face + 3] = orientation_loc

                # Edges are oriented from low to high node indices
                edges.sort(axis=1)
                edges, _, indices = setmembership.unique_rows(edges)
                self.num_edges[ng] = edges.shape[0]
                
                # Calculate edge centers and tangents
                xyz_0 = g.nodes[:, edges[:,0].astype(int)]
                xyz_1 = g.nodes[:, edges[:,1].astype(int)]
                self.edge_centers[ng] = (xyz_0 + xyz_1) / 2
                self.edge_tangents[ng] = xyz_1 - xyz_0

                # Generate edge-node connectivity such that
                # edge_nodes(i, j) = +/- 1:
                # edge j points to/away from node i
                indptr = np.arange(0, edges.size + 1, 2)
                ind = np.ravel(edges)
                data = -(-1)**np.arange(edges.size)
                self.edge_nodes[ng] = sps.csc_matrix((data, ind, indptr))

                # Generate face_edges such that
                # face_edges(i, j) = +/- 1:
                # face j has edge i with same/opposite orientation
                # with the orientation defined according to the right-hand rule
                indptr = np.arange(0, indices.size + 1, 3)
                self.face_edges[ng] = sps.csc_matrix((orientations, indices,
                                                  indptr))

        t = time.time() - start_time
        self.cpu_time.append(["Compute edges", str(t)])

    # ------------------------------------------------------------------------ #

    def Pi_div_h(self):
        start_time = time.time()
        # global Pi matrix
        Pi = np.empty(shape=(self.gb.size(), self.gb.size()), dtype=np.object)

        # assume 3D case
        for g, d in self.gb:
            nn_g = d['node_number']
            if g.dim == 3:
                fn = g.face_nodes
                Pi_gx = fn.copy().asfptype()
                Pi_gy = fn.copy().asfptype()
                Pi_gz = fn.copy().asfptype()

                for face in np.arange(g.num_faces):
                    face_indices = np.arange(fn.indptr[face], fn.indptr[face+1])
                    face_normal = g.face_normals[:, face]
                    normal_norm = np.linalg.norm(face_normal)
                    face_area = g.face_areas[face]

                    Pi_gx.data[face_indices] = face_normal[0] * face_area / (3 * normal_norm)
                    Pi_gy.data[face_indices] = face_normal[1] * face_area / (3 * normal_norm)
                    Pi_gz.data[face_indices] = face_normal[2] * face_area / (3 * normal_norm)

                Pi_g = sps.hstack([Pi_gx.T, Pi_gy.T, Pi_gz.T])

            elif g.dim == 2:
                R = cg.project_plane_matrix(g.nodes, check_planar=False)
                fn = g.face_nodes

                Pi_gx = fn.copy().asfptype()
                Pi_gy = fn.copy().asfptype()

                for face in np.arange(g.num_faces):
                    face_indices = np.arange(fn.indptr[face], fn.indptr[face+1])
                    face_normal = np.dot(R, g.face_normals[:, face])
                    normal_norm = np.linalg.norm(face_normal)
                    face_area = g.face_areas[face]

                    Pi_gx.data[face_indices] = face_normal[0] * face_area / (2 * normal_norm)
                    Pi_gy.data[face_indices] = face_normal[1] * face_area / (2 * normal_norm)

                Pi_g = sps.hstack([Pi_gx.T, Pi_gy.T])
            else:
                normal_norm = np.linalg.norm(g.face_normals, axis=0)
                Pi_g = sps.diags(normal_norm, format='csr')

            Pi[nn_g, nn_g] = Pi_g

            # loop over mortar grids of lower dimension
            for e, d_e in self.gb.edges_of_node(g):
                mg = d_e['mortar_grid']
                if mg.dim == g.dim - 1:
                    # splitting of projection to interior and mortar part

                    # face to mortar cell mapping
                    P_mg = mg.master_to_mortar_int()
                    nn_mg = d_e['edge_number'] + self.gb.num_graph_nodes()

                    # Recover the orientation information
                    faces_h, _, sign_h = sps.find(g.cell_faces)
                    _, ind_faces_h = np.unique(faces_h, return_index=True)
                    sign_h = sign_h[ind_faces_h]

                    Signs = sps.diags(sign_h)

                    Pi[nn_mg, nn_g] = P_mg * Signs * Pi_g
                    Pi[nn_g, nn_g] -= P_mg.T * P_mg * Pi_g

        t = time.time() - start_time
        self.cpu_time.append(["Pi div", str(t)])

        return sps.bmat(Pi, format='csr')

    # ------------------------------------------------------------------------ #

    def Pi_curl_h(self):
        start_time = time.time()
        grids_23 = self.gb.get_grids(lambda g: g.dim >= 2)
        n_grids_23 = grids_23.size
        # global Pi matrix
        Pi = np.empty(shape=(n_grids_23, n_grids_23), dtype=np.object)

        # assume 3D case
        for g in grids_23:
            nn_g = self.gb.graph.node[g]['node_number']
            num_edges = self.num_edges[nn_g]
            en = self.edge_nodes[nn_g]
            if g.dim == 3:
                Pi_gx = en.copy().asfptype()
                Pi_gy = en.copy().asfptype()
                Pi_gz = en.copy().asfptype()

                for edge in np.arange(num_edges):
                    edge_indices = np.arange(en.indptr[edge], en.indptr[edge+1])
                    nodes = en.indices[edge_indices]
                    edge_tangent = g.nodes[:, nodes[1]] - g.nodes[:, nodes[0]]

                    Pi_gx.data[edge_indices] = edge_tangent[0] / 2
                    Pi_gy.data[edge_indices] = edge_tangent[1] / 2
                    Pi_gz.data[edge_indices] = edge_tangent[2] / 2

                Pi_g = sps.hstack([Pi_gx.T, Pi_gy.T, Pi_gz.T])

            else:
                Pi_g = sps.identity(g.num_nodes, format='csr')

            Pi[nn_g, nn_g] = Pi_g

        t = time.time() - start_time
        self.cpu_time.append(["Pi curl", str(t)])

        return sps.bmat(Pi, format='csr')