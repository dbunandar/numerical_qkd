"""
rel_entropy_keyrate
====================
Calculates the secret keyrate by approximately solving the
quantum relative entropy

Author: Darius Bunandar (dariusb@mit.edu)

Unauthorized use and/or duplication of this material without express and
written permission from the author and/or owner is strictly prohibited.
==============================================================================
"""
import sys
import os
import numpy as np
import scipy as sp
import cvxpy as cvx
import matlab
from numerical_qkd import np_array_to_matlab, general_kron, Keyrate, \
    quantum_rel_entropy, matrix_gram_schmidt


class RelativeEntropyKeyrate(Keyrate):
    """
    RelativeEntropyKeyrate

    Computes the numerical key rate of a QKD protocol by solving the relative
    quantum entropy: H(Z|E) where Z is Alice's classical information

    Uses the abstract base class: Keyrate
    """

    @staticmethod
    def leggauss_zero_to_one(m):
        # Legendre-Gauss quadrature points and weights:
        # the numpy method is for interval [-1,1], we want over the interval
        # of [0,1] (simple change of integrand)
        s, w = np.polynomial.legendre.leggauss(m)
        s = .5*(s+1)
        w = w/2
        return s, w

    def get_primal_problem(self, m=2, k=2):

        mat_size = np.prod(self.dims)  # size of rho_AB
        opt_mat_sh = (mat_size**2, mat_size**2)

        # Define cvxpy variables
        M = [cvx.Variable(opt_mat_sh, hermitian=True) for _ in range(k)]
        rho_AB = cvx.Variable((mat_size, mat_size), hermitian=True)
        tau = cvx.Variable()
        T = cvx.Variable(m)

        cq_rho_AB = np.sum(povm @ rho_AB @ povm for povm in self._key_map_povm)

        X = general_kron(rho_AB, np.eye(mat_size))
        Y = cvx.kron(np.eye(mat_size), cvx.conj(cq_rho_AB))
        M.insert(0, Y)  # M[0] = Y
        Z = M[-1]  # M[k] = Z

        # Constraints related to matrix geometric mean cone
        const_geo_mean_cone = []
        for i in range(k):
            M_matrix = cvx.bmat([[M[i], M[i+1]], [M[i+1], X]])
            const_geo_mean_cone.append((M_matrix >> 0))

        # Constraints related to operator relative entropy cone
        e = np.reshape(np.eye(mat_size), (-1, 1), order="C")
        s, w = self.leggauss_zero_to_one(m)

        const_rel_entropy = []
        eXe = e.T @ X @ e
        eX = e.T @ X
        Xe = X @ e
        for j in range(m):
            T_matrix = cvx.bmat([[eXe - s[j]*T[j]/w[j], eX],
                                 [Xe, X+s[j]*(Z-X)]])
            const_rel_entropy.append((T_matrix >> 0))
        const_rel_entropy.append((np.power(2, k)*cvx.sum(T) + tau >= 0))

        # Constraints related to the state rho_AB
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

        # Other constraints
        const_rho_normalized = (cvx.trace(rho_AB) == 1)
        const_rho_pos = (rho_AB >> 0)
        constraints = [const_rho_normalized, const_rho_pos] + \
            const_rho_AB + const_rho_AB_ub + const_rho_AB_lb + \
            const_geo_mean_cone + const_rel_entropy

        obj = cvx.Minimize(tau)
        problem = cvx.Problem(obj, constraints)
        return problem

    def _check_constraints(self, Gamma, check_hermitian=False):
        for i, G in enumerate(Gamma):
            print("checking constraint ", i)
            # Check that G is Hermitian
            if check_hermitian:
                assert(np.allclose(G, G.conj().T)), "G must be Hermitian"
            for Gj in Gamma[i+1:]:
                # Check that G and Gj are orthogonal
                assert(np.isclose(np.trace(G @ Gj.conj().T), 0)
                       ), "Constraints are not orthogonal "

    def _del_rel_entropy_keyrate(self, rho, transpose=False):
        fudge = 1e-16
        sh = rho.shape
        new_rho = (1-fudge) * rho + fudge * np.eye(sh[0])
        term_1 = sp.linalg.logm(new_rho)
        cq_rho = np.sum(povm @ new_rho @ povm for povm in self._key_map_povm)
        term_2 = sp.linalg.logm(cq_rho)
        result = term_1 + term_2
        if transpose:
            result = result.T
        return result  # in nats

    def _rel_entropy_keyrate(self, rho):
        cq_rho = np.sum(povm @ rho @ povm for povm in self._key_map_povm)
        return quantum_rel_entropy(rho, cq_rho) * np.log(2)  # in nats

    def get_near_optimal_attack(self, rho, eps_th=1e-7, max_iter=1000,
                                solver='MOSEK', **solver_kwargs):
        """
        Finds a state that optimizes Eve's attack as much as possible.
        The computation is done by performing QR decomposition of the
        constraint Gamma's and make a guess on the other constraint Omegas
        """

        mat_size = np.prod(self.dims)
        n_basis = mat_size**2  # total number of basis Gamma/Omega vectors

        all_Gamma = [np.eye(mat_size)]
        all_gamma = [1]
        n_Gamma_exact = n_Gamma_inexact = 0
        if self.Gamma_exact is not None:
            all_Gamma += self.Gamma_exact
            n_Gamma_exact = len(self.Gamma_exact)
        if self.Gamma_inexact is not None:
            all_Gamma += self.Gamma_inexact
            n_Gamma_inexact = len(self.Gamma_inexact)

        n_Gamma = n_Gamma_exact + n_Gamma_inexact + 1

        # Obtain the complete basis
        Gamma_orth, Omega, q, _ = matrix_gram_schmidt(
            all_Gamma, normalize=False)
        n_Omega = len(Omega)

        # Check that all the POVM constraints are orthonormal basis vectors
        Constraints_orth = Gamma_orth + Omega
        # self._check_constraints(Gamma_orth, check_hermitian=True)
        # self._check_constraints(Constraints_orth)

        # Confirm that all the basis vectors have been found
        assert(n_Gamma + n_Omega == n_basis), "Missing basis vectors"

        # Solve for the values of gamma in the new orthogonalized bases
        rho_flat = rho.flatten('F')
        gamma_orth = np.linalg.solve(q, rho_flat)

        # Step 1: Finding the optimal Eve attack
        rho_eve_opt = np.sum(
            g * G for g, G in zip(gamma_orth, Constraints_orth))

        # scipy optimize verbosity settings
        disp = 0
        if self.verbose:
            disp = 2

        for it in range(max_iter):
            # Find the direction of minimization
            omega = cvx.Variable(n_Omega)
            delta_rho = np.sum(o * O for o, O in zip(omega, Omega))
            del_f_rho = self._del_rel_entropy_keyrate(rho_eve_opt)

            new_rho = rho_eve_opt + delta_rho
            const_rho_psd = (new_rho >> 0)

            obj_1 = cvx.Minimize(cvx.real(cvx.trace(delta_rho @ del_f_rho)))
            constraints_1 = [const_rho_psd]
            problem_1 = cvx.Problem(obj_1, constraints_1)
            problem_1.solve(solver, **solver_kwargs, verbose=self.verbose)

            if problem_1.status not in ["infeasible", "unbounded"]:
                delta_rho_val = delta_rho.value
            else:
                raise ValueError("Unable to solve for optimal Eve attack")

            # Check if we have achieved threshold
            # A full step in this direction will put us under the threshold
            if abs(np.trace(delta_rho_val @ del_f_rho)) < eps_th:
                break

            # Determine the step size t
            def obj_fun(x):
                return self._rel_entropy_keyrate(rho_eve_opt + x*delta_rho_val)
            t_opt = sp.optimize.fminbound(obj_fun, 0, 1,
                                          xtol=eps_th, maxfun=max_iter,
                                          disp=disp)
            rho_eve_opt += t_opt * delta_rho_val

        if it == (max_iter-1):
            raise UserWarning("Reached max iteration, might not be optimal")

        return rho_eve_opt

    def get_dual_problem(self, rho_eve_opt):
        # Step 2: Obtaining a reliable lower bound
        mat_size = np.prod(self.dims)
        del_f_rho = self._del_rel_entropy_keyrate(rho_eve_opt)

        obj_exact = obj_inexact = 0.0
        zi_sum = xi_sum = yi_sum = 0.0

        if self.Gamma_exact is not None:
            zi_vec = cvx.Variable(len(self.gamma))
            obj_exact = cvx.sum(zi_vec * self.gamma)
            zi_sum = np.sum(zi * G for zi, G in zip(zi_vec, self.Gamma_exact))

        if self.Gamma_inexact is not None:
            yi_vec = cvx.Variable(len(self.gamma_ub), nonneg=True)
            xi_vec = cvx.Variable(len(self.gamma_lb), nonneg=True)
            obj_inexact = - cvx.sum(yi_vec * self.gamma_ub) + \
                cvx.sum(xi_vec * self.gamma_lb)
            yi_sum = np.sum(yi * G for yi, G
                            in zip(yi_vec, self.Gamma_inexact))
            xi_sum = np.sum(xi * G for xi, G
                            in zip(xi_vec, self.Gamma_inexact))

        y_id = cvx.Variable()
        obj_2 = cvx.Maximize(y_id + obj_exact + obj_inexact)
        sum_of_Gammas = zi_sum + xi_sum - yi_sum + y_id * np.eye(mat_size)
        constraints_2 = [(sum_of_Gammas << del_f_rho)]
        problem_2 = cvx.Problem(obj_2, constraints_2)

        return problem_2

    def compute_dual(self, rho, eps_th=1e-7, max_iter=1000,
                     solver='MOSEK', **solver_kwargs):

        rho_eve_opt = self.get_near_optimal_attack(rho, eps_th,
                                                   max_iter, solver,
                                                   **solver_kwargs)
        problem = self.get_dual_problem(rho_eve_opt)
        problem.solve(solver, **solver_kwargs, verbose=self.verbose)

        if problem.status not in ["infeasible", "unbounded"]:
            f_rho = self._rel_entropy_keyrate(rho_eve_opt)
            del_f_rho = self._del_rel_entropy_keyrate(rho_eve_opt)
            f_times_grad_rho = np.trace(rho_eve_opt @ del_f_rho)
            value = f_rho - f_times_grad_rho + problem.value  # in nats
            return np.real(value) / np.log(2)  # convert to bits and real
        else:
            return 0

    def compute_primal_with_matlab(self, matlab_engine,
                                   m=2, k=2,
                                   solver='SEDUMI'):

        # Convert to data types understandable by matlab_engine
        # - List of np.arrays are converted into list of matlab arrays
        # - np.array is converted into a matlab array
        # - Legendre-Gauss quadrature requires float m and k

        key_map_povm_ = [np_array_to_matlab(Z) for Z in self._key_map_povm]
        Gamma_exact_ = Gamma_inexact_ = np_array_to_matlab(np.array([]))
        gamma_ = gamma_lb_ = gamma_ub_ = np_array_to_matlab(np.array([]))

        if self.Gamma_exact is not None:
            Gamma_exact_ = [np_array_to_matlab(G) for G in self.Gamma_exact]
            gamma_ = np_array_to_matlab(self.gamma)

        if self.Gamma_inexact is not None:
            Gamma_inexact_ = [np_array_to_matlab(G) for G
                              in self.Gamma_inexact]
            gamma_ub_ = np_array_to_matlab(self.gamma_ub)
            gamma_lb_ = np_array_to_matlab(self.gamma_lb)
        dims_ = np_array_to_matlab(np.array(self.dims))
        m_ = float(m)
        k_ = float(k)

        val = matlab_engine.rel_entropy_keyrate(key_map_povm_,
                                                Gamma_exact_, gamma_,
                                                Gamma_inexact_,
                                                gamma_ub_, gamma_lb_,
                                                dims_, m_, k_,
                                                solver,
                                                self.verbose)
        return val

    def _keyrate_function(self, value):
        return value / np.log(2)
