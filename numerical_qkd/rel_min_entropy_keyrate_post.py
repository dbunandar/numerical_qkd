"""
rel_min_entropy_keyrate_post
====================
Calculates finite secret-key rate using a general prototypical protocol
that allows for postselection.

Author: Darius Bunandar (dariusb@mit.edu)

Unauthorized use and/or duplication of this material without express and
written permission from the author and/or owner is strictly prohibited.
==============================================================================
"""
import numpy as np
import cvxpy as cvx
from numerical_qkd import PostselectedKeyrate, block_diagonal_stack, \
    np_array_to_matlab


class PostselectedRelativeMinEntropyKeyrate(PostselectedKeyrate):
    """
    PostselectedRelativeMinEntropyKeyrate

    Computes numerical finite key rate of a QKD protocol by solving
    min-entropy: H_{min}(Z|E) where Z is Alice's classical information.

    With post-selection, the Hilbert space of interests are:
    A   = Alice's quantum state (of size dim_A)
    Ab  = Alice's basis
    Av  = Alice's element/value in that basis (of size n_value_A)
    B   = Bob's quantum state (of size dim_B)
    Bb  = Bob's basis
    Bv  = Bob's element/value in that basis (of size n_value_B)
    The basis projection operators project in the Ab and Bb spaces.
    For simplicity, we abbreviate A'B' = ApBp = Ab o Bb o A o B o Av

    Uses the abstract base class: PostselectedKeyrate
    """

    def get_primal_problem(self):

        # matrix sizes for AB and A'B
        n_basis = len(self._basis_info_list)  # number of bases postselected
        mat_size_AB = self.dim_A * self.dim_B
        mat_size_ApBp = n_basis * mat_size_AB * self._n_value_A

        X11 = cvx.Variable((mat_size_ApBp, mat_size_ApBp), complex=True)
        X12 = cvx.Variable((mat_size_ApBp, mat_size_ApBp), complex=True)
        X22 = cvx.Variable((mat_size_ApBp, mat_size_ApBp), complex=True)
        rho_AB = cvx.Variable((mat_size_AB, mat_size_AB), hermitian=True)
        sigma_ApBp = cvx.Variable(
            (mat_size_ApBp, mat_size_ApBp), hermitian=True)

        X = cvx.bmat([[X11, X12], [X12.H, X22]])

        obj = cvx.Maximize(cvx.trace(cvx.real(X12)))

        K_AB = self._basis_info_list.unpack('kraus_op')
        rho_ApBp_ = [k * rho_AB * np.conj(k.T) for k in K_AB]
        rho_ApBp = block_diagonal_stack(rho_ApBp_)

        const_X11 = (X11 << rho_ApBp)
        key_map_povm_ = [np.kron(np.eye(n_basis * mat_size_AB), povm)
                         for povm in self._key_map_povm]
        cq_sigma_ApBp = np.sum(povm * sigma_ApBp *
                               povm for povm in key_map_povm_)
        const_X22 = (X22 << cq_sigma_ApBp)
        const_sigma_AB = (cvx.real(cvx.trace(sigma_ApBp)) <= 1)
        const_X_pos = (X >> 0)

        const_rho_AB = const_rho_AB_ub = const_rho_AB_lb = []
        if self.Gamma_exact is not None:
            const_rho_AB = [(cvx.trace(rho_AB * G) == g)
                            for g, G in zip(self.gamma, self.Gamma_exact)]
        if self.Gamma_inexact is not None:
            const_rho_AB_ub = [(cvx.real(cvx.trace(rho_AB * G)) <= g_ub)
                               for g_ub, G in
                               zip(self.gamma_ub, self.Gamma_inexact)]
            const_rho_AB_lb = [(cvx.real(cvx.trace(rho_AB * G)) >= g_lb)
                               for g_lb, G in
                               zip(self.gamma_lb, self.Gamma_inexact)]
        const_rho_normalized = (cvx.trace(rho_AB) == 1)
        const_sigma_pos = (sigma_ApBp >> 0)
        const_rho_pos = (rho_AB >> 0)

        constraints = [const_X11, const_X22,
                       const_sigma_AB, const_X_pos,
                       const_sigma_pos, const_rho_pos, const_rho_normalized] \
            + const_rho_AB + const_rho_AB_ub + const_rho_AB_lb

        problem = cvx.Problem(obj, constraints)
        return problem

    def get_dual_problem(self):

        # matrix sizes for A'B'
        n_basis = len(self._basis_info_list)  # number of bases postselected
        mat_size_AB = self.dim_A * self.dim_B
        mat_size_ABAv = mat_size_AB * self._n_value_A
        mat_size_ApBp = n_basis * mat_size_ABAv

        Y11_list = [cvx.Variable((mat_size_ABAv, mat_size_ABAv),
                                 hermitian=True) for _ in range(n_basis)]
        Y22 = cvx.Variable((mat_size_ApBp, mat_size_ApBp), hermitian=True)
        y_id = cvx.Variable()
        y = cvx.Variable(nonneg=True)

        obj_exact = obj_inexact = 0.0
        zi_sum = xi_sum = yi_sum = 0.0
        if self.Gamma_exact is not None:
            zi_vec = cvx.Variable(len(self.gamma))
            obj_exact = cvx.sum(zi_vec * self.gamma)
            zi_sum = np.sum(zi * G for zi, G in zip(zi_vec, self.Gamma_exact))

        if self.Gamma_inexact is not None:
            yi_vec = cvx.Variable(len(self.gamma_ub), nonneg=True)
            xi_vec = cvx.Variable(len(self.gamma_lb), nonneg=True)
            obj_inexact = cvx.sum(yi_vec * self.gamma_ub) - \
                cvx.sum(xi_vec * self.gamma_lb)
            yi_sum = np.sum(yi * G for yi, G
                            in zip(yi_vec, self.Gamma_inexact))
            xi_sum = np.sum(xi * G for xi, G
                            in zip(xi_vec, self.Gamma_inexact))

        obj = cvx.Minimize(y + y_id + obj_exact + obj_inexact)

        zero_mat = np.zeros([mat_size_ApBp, mat_size_ApBp])
        Y11 = block_diagonal_stack(Y11_list)
        Y = cvx.bmat([[Y11, zero_mat],
                      [zero_mat, Y22]])
        A = .5 * cvx.bmat([[zero_mat, np.eye(mat_size_ApBp)],
                           [np.eye(mat_size_ApBp), zero_mat]])

        key_map_povm_ = [np.kron(np.eye(n_basis * mat_size_AB), povm)
                         for povm in self._key_map_povm]
        cq_Y22 = np.sum(povm * Y22 * povm for povm in key_map_povm_)
        const_y = (y * np.eye(mat_size_ApBp) >> cq_Y22)
        K_AB = self._basis_info_list.unpack('kraus_op')
        Y11_AB = np.sum(np.conj(kAB.T) @ mat @ kAB for kAB, mat in zip(
            K_AB, Y11_list))

        sum_of_Gammas = zi_sum + yi_sum - xi_sum
        const_Y11 = (sum_of_Gammas + y_id *
                     np.eye(self.dim_A*self.dim_B) >> Y11_AB)
        const_Y = (Y >> A)

        constraints = [const_y, const_Y11, const_Y]

        problem = cvx.Problem(obj, constraints)
        return problem

    def compute_dual_with_matlab(self, matlab_engine, p_pass, trace_val=None,
                                 solver='SEDUMI'):
        """
        Solves the dual problem with MATLAB and CVX (for speed)
        """

        Gamma_exact_ = Gamma_inexact_ = np_array_to_matlab(np.array([]))
        gamma_ = gamma_lb_ = gamma_ub_ = np_array_to_matlab(np.array([]))
        exact_const_exists_ = inexact_const_exists_ = False

        if self.Gamma_exact is not None:
            exact_const_exists_ = True
            Gamma_exact_ = [np_array_to_matlab(G) for G in self.Gamma_exact]
            gamma_ = np_array_to_matlab(self.gamma)

        if self.Gamma_inexact is not None:
            inexact_const_exists_ = True
            Gamma_inexact_ = [np_array_to_matlab(G) for G
                              in self.Gamma_inexact]
            gamma_ub_ = np_array_to_matlab(self.gamma_ub)
            gamma_lb_ = np_array_to_matlab(self.gamma_lb)

        if trace_val is None:
            trace_val_ = float(1)
        else:
            trace_val_ = float(trace_val)

        dims_ = np_array_to_matlab(np.array([self.dim_A, self.dim_B]))
        p_pass_ = float(p_pass)

        n_basis_ = float(len(self._basis_info_list))
        n_value_A_ = float(self._n_value_A)

        key_map_povm_ = [np_array_to_matlab(povm)
                         for povm in self._key_map_povm]

        K_AB = self._basis_info_list.unpack('kraus_op')
        K_AB_ = [np_array_to_matlab(kAB) for kAB in K_AB]

        val = matlab_engine.postselected_rel_min_entropy_keyrate(
            key_map_povm_, n_basis_, n_value_A_,
            Gamma_exact_, gamma_, exact_const_exists_,
            Gamma_inexact_, gamma_ub_, gamma_lb_, inexact_const_exists_,
            K_AB_,
            dims_, p_pass_, trace_val_, solver, self.verbose)

        return val

    def _keyrate_function(self, value, p_pass):
        if value <= 0:
            return 0
        else:
            return -2*np.log2(value) + np.log2(p_pass)
