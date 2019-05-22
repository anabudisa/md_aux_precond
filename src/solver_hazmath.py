import numpy as np
import scipy as sp
import scipy.sparse as sps
import time
import ctypes

import porepy as pp

from Hcurl3D import Hcurl
from logger import logger


class Solver(object):

    def __init__(self, gb, discr):
        # Grid bucket
        self.gb = gb
        # Discretization
        self.discr = discr
        # All degrees of freedom
        self.full_dof = None
        # Block indices to grid-variable combination
        self.block_dof = None
        # Permutation matrix from original porepy order to new 2x2 block order
        self.P = None
        # List of block degrees of freedom: 0 -> (u, lambda), 1 -> p
        self.block_dof_list = None
        # Stiffness matrix in 2x2 block form: 0 -> (u, lambda), 1 -> p
        self.M = None
        # Right hand side in 2x1 block form: : 0 -> (u, lambda), 1 -> p
        self.f = None
        # Curl operator
        self.Curl = None
        # Projection operator nodes to faces dof
        self.Pi_div_h = None
        # Projection operator nodes to edges dof
        self.Pi_curl_h = None
        # Sign fixing - cos for some reason mortar variable gives a negative
        # definite matrix
        self.signs = None

    # ------------------------------------------------------------------------ #

    def solve_direct(self, bmat=True):

        logger.info("Solve the system with direct solver in Python")

        # if M in bmat form
        if bmat:
            Mm = sps.bmat(self.M, format="csc")
            ff = np.concatenate(tuple(self.f))

        start_time = time.time()
        upl = sps.linalg.spsolve(Mm, ff)
        t = time.time() - start_time

        logger.info("Elapsed time of direct solver: " + str(t))
        logger.info("Done")

        # permute solution
        y = self.permute_solution(upl)

        return y

    # ------------------------------------------------------------------------ #

    def solve_hazmath(self, alpha=1., tol=1e-5, maxit=100):
        # solve the system using HAZMATH
        #          M x = f

        # !! NB !!
        # M[0, 0] represents product (u, v) where u,v \in H(div)
        # for \alpha * (u, v) multiply M[0, 0] with alpha
        # e.g. M[0, 0].multiply(alpha)
        #
        # since M represents a stiffness matrix for the flow problem (Darcy +
        # m.c.), then
        # M[1, 0] represents (discrete) -div operator;

        logger.info("Solve the system with hazmath library using auxiliary "
                    "space preconditioners")

        # right hand side
        ff = np.concatenate(tuple(self.f))

        # Mass matrix of pressure
        Mp = self.P0_mass_matrix()
        Mp_diag = Mp.diagonal()

        # ------------------------------------
        # prepare HAZMATH solver
        # ------------------------------------
        # call HAZMATH solver library
        libHAZMATHsolver = ctypes.cdll.LoadLibrary(
            '/home/anci/Dropbox/hazmath2/hazmath/lib/libhazmath.so')
        # libHAZMATHsolver = ctypes.cdll.LoadLibrary(
        # '/home/xiaozhehu/Work/Projects/HAZMATH/hazmath/lib/libhazmath.so')

        # parameters for HAZMATH solver
        prtlvl = ctypes.c_int(3)
        tol = ctypes.c_double(tol)
        maxit = ctypes.c_int(maxit)

        # ------------------------------------
        # convert
        # ------------------------------------
        Muu_size = self.M[0, 0].shape
        nrowp1_uu = Muu_size[0] + 1
        nrow_uu = ctypes.c_int(Muu_size[0])
        ncol_uu = ctypes.c_int(Muu_size[1])
        nnz_uu = ctypes.c_int(self.M[0, 0].nnz)

        Mup_size = self.M[0, 1].shape
        nrowp1_up = Mup_size[0] + 1
        nrow_up = ctypes.c_int(Mup_size[0])
        ncol_up = ctypes.c_int(Mup_size[1])
        nnz_up = ctypes.c_int(self.M[0, 1].nnz)

        # M[1, 0] is (-div) operator
        Mpu_size = self.M[1, 0].shape
        nrowp1_pu = Mpu_size[0] + 1
        nrow_pu = ctypes.c_int(Mpu_size[0])
        ncol_pu = ctypes.c_int(Mpu_size[1])
        nnz_pu = ctypes.c_int(self.M[1, 0].nnz)

        Mpp_size = self.M[1, 1].shape
        nrowp1_pp = Mpp_size[0] + 1
        nrow_pp = ctypes.c_int(Mpp_size[0])
        ncol_pp = ctypes.c_int(Mpp_size[1])
        nnz_pp = ctypes.c_int(self.M[1, 1].nnz)

        # HX preconditioner
        Pidiv_size = self.Pi_div_h.shape
        nrowp1_Pidiv = Pidiv_size[0] + 1
        nrow_Pidiv = ctypes.c_int(Pidiv_size[0])
        ncol_Pidiv = ctypes.c_int(Pidiv_size[1])
        nnz_Pidiv = ctypes.c_int(self.Pi_div_h.nnz)

        Picurl_size = self.Pi_curl_h.shape
        nrowp1_Picurl = Picurl_size[0] + 1
        nrow_Picurl = ctypes.c_int(Picurl_size[0])
        ncol_Picurl = ctypes.c_int(Picurl_size[1])
        nnz_Picurl = ctypes.c_int(self.Pi_curl_h.nnz)

        Curl_size = self.Curl.shape
        nrowp1_Curl = Curl_size[0] + 1
        nrow_Curl = ctypes.c_int(Curl_size[0])
        ncol_Curl = ctypes.c_int(Curl_size[1])
        nnz_Curl = ctypes.c_int(self.Curl.nnz)

        # allocate solution
        nrow = Muu_size[0] + Mpp_size[0]
        nrow_double = ctypes.c_double * nrow
        hazmath_sol = nrow_double()
        numiters = ctypes.c_int(-1)

        # ------------------------------------
        # solve using HAZMATH
        # ------------------------------------
        libHAZMATHsolver.python_wrapper_krylov_mixed_darcy(
            ctypes.byref(nrow_uu), ctypes.byref(ncol_uu),
            ctypes.byref(nnz_uu),
            (ctypes.c_int * nrowp1_uu)(*self.M[0, 0].indptr),
            (ctypes.c_int * self.M[0, 0].nnz)(*self.M[0, 0].indices),
            (ctypes.c_double * self.M[0, 0].nnz)(*self.M[0, 0].data),
            ctypes.byref(nrow_up),
            ctypes.byref(ncol_up), ctypes.byref(nnz_up),
            (ctypes.c_int * nrowp1_up)(*self.M[0, 1].indptr),
            (ctypes.c_int * self.M[0, 1].nnz)(*self.M[0, 1].indices),
            (ctypes.c_double * self.M[0, 1].nnz)(*self.M[0, 1].data),
            ctypes.byref(nrow_pu), ctypes.byref(ncol_pu),
            ctypes.byref(nnz_pu),
            (ctypes.c_int * nrowp1_pu)(*self.M[1, 0].indptr),
            (ctypes.c_int * self.M[1, 0].nnz)(*self.M[1, 0].indices),
            (ctypes.c_double * self.M[1, 0].nnz)(*self.M[1, 0].data),
            ctypes.byref(nrow_pp),
            ctypes.byref(ncol_pp), ctypes.byref(nnz_pp),
            (ctypes.c_int * nrowp1_pp)(*self.M[1, 1].indptr),
            (ctypes.c_int * self.M[1, 1].nnz)(*self.M[1, 1].indices),
            (ctypes.c_double * self.M[1, 1].nnz)(*self.M[1, 1].data),
            ctypes.byref(nrow_Pidiv),
            ctypes.byref(ncol_Pidiv), ctypes.byref(nnz_Pidiv),
            (ctypes.c_int * nrowp1_Pidiv)(*self.Pi_div_h.indptr),
            (ctypes.c_int * self.Pi_div_h.nnz)(*self.Pi_div_h.indices),
            (ctypes.c_double * self.Pi_div_h.nnz)(*self.Pi_div_h.data),
            ctypes.byref(nrow_Curl),
            ctypes.byref(ncol_Curl), ctypes.byref(nnz_Curl),
            (ctypes.c_int * nrowp1_Curl)(*self.Curl.indptr),
            (ctypes.c_int * self.Curl.nnz)(*self.Curl.indices),
            (ctypes.c_double * self.Curl.nnz)(*self.Curl.data),
            (ctypes.c_double * Mpp_size[0])(*Mp_diag),
            (ctypes.c_double * nrow)(*ff),
            ctypes.byref(hazmath_sol), ctypes.byref(tol),
            ctypes.byref(maxit),
            ctypes.byref(prtlvl), ctypes.byref(numiters))

        logger.info("Done")
        # ------------------------------------
        # convert solution
        # ------------------------------------
        x = sp.array(hazmath_sol)
        numiters = sp.array(numiters)

        # permute solution
        y = self.permute_solution(x)

        return y, numiters

    # ------------------------------------------------------------------------ #

    def P0_mass_matrix(self):
        # TODO: deprecate for porepy mass matrix
        matrices = []

        # assemble matrix for each grid in GridBucket
        for g, d in self.gb:
            M_g = sps.diags(g.cell_volumes)
            matrices.append(M_g)

        return sps.block_diag(tuple(matrices), format="csr")

    # ------------------------------------------------------------------------ #

    def permutation_matrix(self, perm):
        self.P = sps.identity(len(perm)).tocoo()

        self.P.row = np.argsort(perm)

    # ------------------------------------------------------------------------ #

    def permute_dofs(self):
        dof_u = np.array([], dtype=int)
        dof_p = np.array([], dtype=int)
        dof_l = np.array([], dtype=int)

        # list of global degrees of freedom per each grid
        dof_count = np.cumsum(np.append(0, np.asarray(self.full_dof)))

        # loop through all block dof
        for pair, bi in self.block_dof.items():
            # node or edge
            g = pair[0]
            # dof for this grid
            local_dof_g = np.arange(dof_count[bi], dof_count[bi + 1])

            # if we're in a node (-> flux and pressure)
            if isinstance(g, pp.Grid):
                # how many flux dof for this grid
                dim_u = g.num_faces
                # assume we first order flux then pressure dof
                dof_u = np.append(dof_u, local_dof_g[:dim_u])
                dof_p = np.append(dof_p, local_dof_g[dim_u:])

            # it we're in an edge (-> mortar)
            else:
                dof_l = np.append(dof_l, local_dof_g)

        perm = np.hstack((dof_u, dof_l, dof_p))
        self.block_dof_list = [np.concatenate((dof_u, dof_l)), dof_p]

        # change sign in mortar equations
        signs_ul = sps.diags(np.concatenate((np.ones(dof_u.size), -np.ones(
            dof_l.size))), format="csr")
        signs_p = sps.diags(np.ones(dof_p.size), format="csr")
        self.signs = np.asarray([signs_ul, signs_p])

        return perm

    # ------------------------------------------------------------------------ #

    def permute_solution(self, x):
        return self.P.T * x

    # ------------------------------------------------------------------------ #

    def setup_system(self, A, b, block_dof, full_dof):
        # set the dof count vector
        self.full_dof = full_dof
        self.block_dof = block_dof

        # get the permutation - this is needed later in post-processing
        logger.info("Permute dof")
        perm = self.permute_dofs()
        self.permutation_matrix(perm)

        # TODO: see if copying is actually necessary
        AA = A.copy()
        bb = np.copy(b)

        # setup block structure; NB - (-div) matrix is exactly M[1, 0]
        # block_dof_list contains the all dof in order (u, lambda, p)
        blocks_no = len(self.block_dof_list)
        self.M = np.empty(shape=(blocks_no, blocks_no), dtype=np.object)
        self.f = np.empty(shape=(blocks_no,), dtype=np.object)

        logger.info("Set up saddle point system")
        # self.signs[row] *
        for row in np.arange(blocks_no):
            for col in np.arange(blocks_no):
                self.M[row, col] = self.signs[row] * AA[self.block_dof_list[row], :].tocsc()[:,self.block_dof_list[col]].tocsr()
            self.f[row] = self.signs[row] * bb[self.block_dof_list[row]]
        logger.info("Done")

        # set up curl operator and projection operators
        logger.info("Get operators for preconditioners")
        hcurl = Hcurl(self.gb)

        self.Curl = hcurl.curl()
        self.Pi_div_h = hcurl.Pi_div_h()

        if self.gb.dim_max() > 2:
            hcurl.compute_edges()
            self.Pi_curl_h = hcurl.Pi_curl_h()
        logger.info("Done")

    # ------------------------------------------------------------------------ #
