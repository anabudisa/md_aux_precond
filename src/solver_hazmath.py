import numpy as np
import scipy as sp
import scipy.sparse as sps
import time
import ctypes
# import suitesparse
from tabulate import tabulate

import porepy as pp

from Hcurl3D import Hcurl
from A_reg import A_reg

from logger import logger


class Solver(object):

    def __init__(self, gb, discr):
        # Grid bucket
        self.gb = gb
        # Discretization
        self.discr = discr
        # solver time in table
        self.cpu_time = []
        # All degrees of freedom
        self.full_dof = None
        # Block indices to grid-variable combination
        self.block_dof = None
        # Permutation matrix from original porepy order to new 2x2 block order
        self.P = None
        # List of block degrees of freedom: 0 -> (u, lambda), 1 -> p
        self.block_dof_list = None
        # Stiffness matrix in 2x2 block form: 0 -> (u, lambda), 1 -> p
        self.A = None
        # Right hand side in 2x1 block form: : 0 -> (u, lambda), 1 -> p
        self.b = None
        # pressure mass matrix
        self.M_p = None
        # Curl operator
        self.Curl = None
        # Projection operator nodes to faces dof
        self.Pi_div_h = None
        # Projection operator nodes to edges dof
        self.Pi_curl_h = None
        # mass part of H1 inner product for regularized div
        self.A_reg_div_mass = None
        # stiffness part of H1 inner product for regularized div
        self.A_reg_div_stiff = None
        # mass part of H1 inner product for regularized curl
        self.A_reg_curl_mass = None
        # stiffness part of H1 inner product for regularized curl
        self.A_reg_curl_stiff = None
        # Sign fixing - cos for some reason mortar variable gives a negative
        # definite matrix
        self.signs = None
        # dimension node tags for div functions (for A_div_grad)
        self.node_tags_div = None
        # dimension node tags for curl functions (for A_curl_grad)
        self.node_tags_curl = None

    # ------------------------------------------------------------------------ #

    def solve_direct(self, bmat=True):

        logger.info("Solve the system with direct solver in Python")

        # if M in bmat form
        if bmat:
            AA = sps.bmat(self.A, format="csc")
            bb = np.concatenate(tuple(self.b))

        start_time = time.time()
        x = sps.linalg.spsolve(AA, bb)
        t = time.time() - start_time

        logger.info("Elapsed time of direct solver: " + str(t))
        logger.info("Done")

        # permute solution
        y = self.permute_solution(x)

        return y

    # ------------------------------------------------------------------------ #

    def solve_hazmath(self, alpha=1., tol=1e-5, maxit=100):
        # solve the system using HAZMATH
        #          A x = b

        # !! NB !!
        # A[0, 0] represents product (u, v) where u,v \in H(div)
        # for \alpha * (u, v) multiply A[0, 0] with alpha
        # e.g. A[0, 0].multiply(alpha)
        #
        # since A represents a stiffness matrix for the flow problem (Darcy +
        # m.c.), then
        # A[1, 0] represents (discrete) -div operator;

        logger.info("Setup hazmath solver")

        start_time = time.time()
        # right hand side
        bb = np.concatenate(tuple(self.b))

        # Mass matrix of pressure
        Mp_diag = self.M_p.diagonal()

        # ------------------------------------
        # prepare HAZMATH solver
        # ------------------------------------
        # call HAZMATH solver library
        # libHAZMATHsolver = ctypes.cdll.LoadLibrary(
        #    '/home/anci/Dropbox/hazmath2/hazmath/lib/libhazmath.so')
        # libHAZMATHsolver = ctypes.cdll.LoadLibrary(
        #    '/home/xiaozhehu/Work/Projects/HAZMATH/hazmath/lib/libhazmath.so')
        libHAZMATHsolver = ctypes.cdll.LoadLibrary(
            '/home/abudis01/hazmath/lib/libhazmath.so')

        # parameters for HAZMATH solver
        prtlvl = ctypes.c_int(3)
        tol = ctypes.c_double(tol)
        maxit = ctypes.c_int(maxit)
        alpha = ctypes.c_double(alpha)

        # ------------------------------------
        # convert
        # ------------------------------------
        Auu_size = self.A[0, 0].shape
        nrowp1_uu = Auu_size[0] + 1
        nrow_uu = ctypes.c_int(Auu_size[0])
        ncol_uu = ctypes.c_int(Auu_size[1])
        nnz_uu = ctypes.c_int(self.A[0, 0].nnz)

        Aup_size = self.A[0, 1].shape
        nrowp1_up = Aup_size[0] + 1
        nrow_up = ctypes.c_int(Aup_size[0])
        ncol_up = ctypes.c_int(Aup_size[1])
        nnz_up = ctypes.c_int(self.A[0, 1].nnz)

        # M[1, 0] is (-div) operator
        Apu_size = self.A[1, 0].shape
        nrowp1_pu = Apu_size[0] + 1
        nrow_pu = ctypes.c_int(Apu_size[0])
        ncol_pu = ctypes.c_int(Apu_size[1])
        nnz_pu = ctypes.c_int(self.A[1, 0].nnz)

        App_size = self.A[1, 1].shape
        nrowp1_pp = App_size[0] + 1
        nrow_pp = ctypes.c_int(App_size[0])
        ncol_pp = ctypes.c_int(App_size[1])
        nnz_pp = ctypes.c_int(self.A[1, 1].nnz)

        ##  HX preconditioner
        # Div to H1 projection
        Pidiv_size = self.Pi_div_h.shape
        nrowp1_Pidiv = Pidiv_size[0] + 1
        nrow_Pidiv = ctypes.c_int(Pidiv_size[0])
        ncol_Pidiv = ctypes.c_int(Pidiv_size[1])
        nnz_Pidiv = ctypes.c_int(self.Pi_div_h.nnz)

        # Curl operator
        Curl_size = self.Curl.shape
        nrowp1_Curl = Curl_size[0] + 1
        nrow_Curl = ctypes.c_int(Curl_size[0])
        ncol_Curl = ctypes.c_int(Curl_size[1])
        nnz_Curl = ctypes.c_int(self.Curl.nnz)

        # H1 mass inner product for regular div
        Ardivmass_size = self.A_reg_div_mass.shape
        nrowp1_divmass = Ardivmass_size[0] + 1
        nrow_divmass = ctypes.c_int(Ardivmass_size[0])
        ncol_divmass = ctypes.c_int(Ardivmass_size[1])
        nnz_divmass = ctypes.c_int(self.A_reg_div_mass.nnz)

        # H1 stiffness inner product for regular div
        Ardivstiff_size = self.A_reg_div_stiff.shape
        nrowp1_divstiff = Ardivstiff_size[0] + 1
        nrow_divstiff = ctypes.c_int(Ardivstiff_size[0])
        ncol_divstiff = ctypes.c_int(Ardivstiff_size[1])
        nnz_divstiff = ctypes.c_int(self.A_reg_div_stiff.nnz)

        # H1 mass inner product for regular curl
        Arcurlmass_size = self.A_reg_curl_mass.shape
        nrowp1_curlmass = Arcurlmass_size[0] + 1
        nrow_curlmass = ctypes.c_int(Arcurlmass_size[0])
        ncol_curlmass = ctypes.c_int(Arcurlmass_size[1])
        nnz_curlmass = ctypes.c_int(self.A_reg_curl_mass.nnz)

        # H1 stiffness inner product for regular curl
        Arcurlstiff_size = self.A_reg_curl_stiff.shape
        nrowp1_curlstiff = Arcurlstiff_size[0] + 1
        nrow_curlstiff = ctypes.c_int(Arcurlstiff_size[0])
        ncol_curlstiff = ctypes.c_int(Arcurlstiff_size[1])
        nnz_curlstiff = ctypes.c_int(self.A_reg_curl_stiff.nnz)

        # allocate solution
        nrow = Auu_size[0] + App_size[0]
        nrow_double = ctypes.c_double * nrow
        hazmath_sol = nrow_double()
        numiters = ctypes.c_int(-1)

        t1 = time.time() - start_time
        # ------------------------------------
        # solve using HAZMATH
        # ------------------------------------
        if self.gb.dim_max() > 2:
            # 3D wrapper
            start_time = time.time()

            # Curl to H1 projection
            Picurl_size = self.Pi_curl_h.shape
            nrowp1_Picurl = Picurl_size[0] + 1
            nrow_Picurl = ctypes.c_int(Picurl_size[0])
            ncol_Picurl = ctypes.c_int(Picurl_size[1])
            nnz_Picurl = ctypes.c_int(self.Pi_curl_h.nnz)

            t2 = time.time() - start_time
            self.cpu_time.append(["Solver setup", str(t1 + t2)])
            logger.info("Elapsed time solver setup: " + str(t1 + t2))
            logger.info("Done")

            logger.info("Solve the system with hazmath library using auxiliary "
                        "space preconditioners")

            libHAZMATHsolver.python_wrapper_krylov_mixed_darcy_HX_3D(
                ctypes.byref(nrow_uu), ctypes.byref(ncol_uu),
                ctypes.byref(nnz_uu),
                (ctypes.c_int * nrowp1_uu)(*self.A[0, 0].indptr),
                (ctypes.c_int * self.A[0, 0].nnz)(*self.A[0, 0].indices),
                (ctypes.c_double * self.A[0, 0].nnz)(*self.A[0, 0].data),
                ctypes.byref(nrow_up),
                ctypes.byref(ncol_up), ctypes.byref(nnz_up),
                (ctypes.c_int * nrowp1_up)(*self.A[0, 1].indptr),
                (ctypes.c_int * self.A[0, 1].nnz)(*self.A[0, 1].indices),
                (ctypes.c_double * self.A[0, 1].nnz)(*self.A[0, 1].data),
                ctypes.byref(nrow_pu), ctypes.byref(ncol_pu),
                ctypes.byref(nnz_pu),
                (ctypes.c_int * nrowp1_pu)(*self.A[1, 0].indptr),
                (ctypes.c_int * self.A[1, 0].nnz)(*self.A[1, 0].indices),
                (ctypes.c_double * self.A[1, 0].nnz)(*self.A[1, 0].data),
                ctypes.byref(nrow_pp),
                ctypes.byref(ncol_pp), ctypes.byref(nnz_pp),
                (ctypes.c_int * nrowp1_pp)(*self.A[1, 1].indptr),
                (ctypes.c_int * self.A[1, 1].nnz)(*self.A[1, 1].indices),
                (ctypes.c_double * self.A[1, 1].nnz)(*self.A[1, 1].data),
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
                ctypes.byref(nrow_Picurl),
                ctypes.byref(ncol_Picurl), ctypes.byref(nnz_Picurl),
                (ctypes.c_int * nrowp1_Picurl)(*self.Pi_curl_h.indptr),
                (ctypes.c_int * self.Pi_curl_h.nnz)(*self.Pi_curl_h.indices),
                (ctypes.c_double * self.Pi_curl_h.nnz)(*self.Pi_curl_h.data),
                (ctypes.c_double * App_size[0])(*Mp_diag),
                (ctypes.c_double * nrow)(*bb),
                ctypes.byref(hazmath_sol),
                ctypes.byref(alpha),
                ctypes.byref(tol), ctypes.byref(maxit),
                ctypes.byref(prtlvl), ctypes.byref(numiters))
        else:
            # 2D wrapper
            self.cpu_time.append(["Solver setup", str(t1)])
            logger.info("Elapsed time solver setup: " + str(t1))
            logger.info("Done")

            logger.info("Solve the system with hazmath library using auxiliary "
                        "space preconditioners")

            libHAZMATHsolver.python_wrapper_krylov_mixed_darcy_HX_2D(
                ctypes.byref(nrow_uu), ctypes.byref(ncol_uu),
                ctypes.byref(nnz_uu),
                (ctypes.c_int * nrowp1_uu)(*self.A[0, 0].indptr),
                (ctypes.c_int * self.A[0, 0].nnz)(*self.A[0, 0].indices),
                (ctypes.c_double * self.A[0, 0].nnz)(*self.A[0, 0].data),
                ctypes.byref(nrow_up),
                ctypes.byref(ncol_up), ctypes.byref(nnz_up),
                (ctypes.c_int * nrowp1_up)(*self.A[0, 1].indptr),
                (ctypes.c_int * self.A[0, 1].nnz)(*self.A[0, 1].indices),
                (ctypes.c_double * self.A[0, 1].nnz)(*self.A[0, 1].data),
                ctypes.byref(nrow_pu), ctypes.byref(ncol_pu),
                ctypes.byref(nnz_pu),
                (ctypes.c_int * nrowp1_pu)(*self.A[1, 0].indptr),
                (ctypes.c_int * self.A[1, 0].nnz)(*self.A[1, 0].indices),
                (ctypes.c_double * self.A[1, 0].nnz)(*self.A[1, 0].data),
                ctypes.byref(nrow_pp),
                ctypes.byref(ncol_pp), ctypes.byref(nnz_pp),
                (ctypes.c_int * nrowp1_pp)(*self.A[1, 1].indptr),
                (ctypes.c_int * self.A[1, 1].nnz)(*self.A[1, 1].indices),
                (ctypes.c_double * self.A[1, 1].nnz)(*self.A[1, 1].data),
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
                (ctypes.c_double * App_size[0])(*Mp_diag),
                (ctypes.c_double * nrow)(*bb),
                ctypes.byref(hazmath_sol),
                ctypes.byref(alpha),
                ctypes.byref(tol), ctypes.byref(maxit),
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

    def setup_system(self, A, M, b, block_dof, full_dof):
        # set the dof count vector
        self.full_dof = full_dof
        self.block_dof = block_dof

        # get the permutation - this is needed later in post-processing
        logger.info("Permute dof")
        start_time = time.time()

        perm = self.permute_dofs()
        self.permutation_matrix(perm)

        t = time.time() - start_time
        self.cpu_time.append(["Perm setup", str(t)])
        logger.info("Elapsed time permutation setup: " + str(t))
        logger.info("Done")

        # setup block structure; NB - (-div) matrix is exactly M[1, 0]
        # block_dof_list contains the all dof in order ((u, lambda), p)
        blocks_no = len(self.block_dof_list)
        self.A = np.empty(shape=(blocks_no, blocks_no), dtype=np.object)
        self.b = np.empty(shape=(blocks_no,), dtype=np.object)

        logger.info("Set up saddle point system")
        start_time = time.time()
        # self.signs[row] *
        for row in np.arange(blocks_no):
            for col in np.arange(blocks_no):
                self.A[row, col] = self.signs[row] * A[self.block_dof_list[row], :].tocsc()[:,self.block_dof_list[col]].tocsr()
            self.b[row] = self.signs[row] * b[self.block_dof_list[row]]
        self.M_p = M[self.block_dof_list[1], :].tocsc()[:, self.block_dof_list[1]].tocsr()

        logger.info("Matrix \t Shape")
        logger.info("A00 \t"+str(self.A[0,0].shape))
        logger.info("A01 \t"+str(self.A[0,1].shape))
        logger.info("A10 \t"+str(self.A[1,0].shape))
        logger.info("A11 \t"+str(self.A[1,1].shape))
        t = time.time() - start_time
        self.cpu_time.append(["Saddle point setup", str(t)])
        logger.info("Elapsed time saddle point system setup: " + str(t))
        logger.info("Done")

        logger.info("Total number of dof: " + str(self.A[0, 0].shape[0] + self.M_p.shape[0]))
        logger.info("Flux number of dof: " + str(self.A[0, 0].shape[0]))
        
        # set up curl operator and projection operators
        logger.info("Get auxiliary operators for preconditioners")
        start_time = time.time()
        hcurl = Hcurl(self.gb)

        self.Curl = hcurl.curl()
        self.Pi_div_h = hcurl.Pi_div_h()
        self.node_tags_div = np.hstack(hcurl.node_tags_div)

        if self.gb.dim_max() > 2:
            self.Pi_curl_h = hcurl.Pi_curl_h()
            self.node_tags_curl = np.hstack(hcurl.node_tags_curl)

        t = time.time() - start_time
        self.cpu_time.append(["Aux operators setup", str(t)])
        logger.info("Elapsed time setup auxiliary operators: " + str(t))
        print(tabulate(hcurl.cpu_time, headers=["hcurl process", "time"]))
        # cum_sum = 0
        # for list_ in hcurl.cpu_time:
        #     if list_[0] not in ["Curl", "Pi div", "Pi curl", "Compute edges"]:
        #         cum_sum += float(list_[1])
        # print("Cum sum curl: ", cum_sum)

        # set up H1 inner product operators for div and curl
        start_time = time.time()
        Areg = A_reg(self.gb)

        self.A_reg_div_mass, self.A_reg_div_stiff = Areg.reg_div()
        self.A_reg_curl_mass, self.A_reg_curl_stiff = Areg.reg_curl()

        t = time.time() - start_time
        self.cpu_time.append(["H1 inner products setup", str(t)])
        print(tabulate(Areg.cpu_time, headers=["Areg process", "time"]))
        logger.info("Elapsed time setup H1 inner products: " + str(t))
        logger.info("Done")

    # ------------------------------------------------------------------------ #

    # def test(self):
    #
    #     print("Shape A_mass: ", self.A[0, 0].shape)
    #     A_div = self.A[0, 0] + self.A[0, 1] * self.M_p * self.A[1, 0]
    #     print("Shape A_div: ", A_div.shape)
    #     print("Shape Pi_div: ", self.Pi_div_h.shape)
    #     print("Shape Pi_curl: ", self.Pi_curl_h.shape)
    #     print("Shape Curl: ", self.Curl.shape)
    #     A_div_grad = self.Pi_div_h.T * A_div * self.Pi_div_h
    #     A_curl_grad = self.Pi_curl_h.T * self.Curl.T * A_div * self.Curl * self.Pi_curl_h
    #     print("Shape A_div_grad: ", A_div_grad.shape)
    #     print("Shape A_curl_grad: ", A_curl_grad.shape)
    #     print("Node tags div shape: ", self.node_tags_div.shape)
    #     print("Node tags curl shape: ", self.node_tags_curl.shape)
    #     print("Number of nodes in gb: ", self.gb.num_nodes())

