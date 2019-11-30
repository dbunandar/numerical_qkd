"""
rel_entropy_keyrate_post
====================
Calculates finite secret-key rate using a general prototypical protocol
that allows for postselection.

Author: Darius Bunandar (dariusb@mit.edu)

Unauthorized use and/or duplication of this material without express and
written permission from the author and/or owner is strictly prohibited.
==============================================================================
"""
import sys
import os
import numpy as np
import cvxpy as cvx
import matlab
from numerical_qkd import np_array_to_matlab, general_kron, PostselectedKeyrate


class PostselectedRelativeEntropyKeyrate(PostselectedKeyrate):
    """
    RelativeEntropyKeyrate

    Computes the postselected numerical key rate of a QKD protocol 
    by solving the relative quantum entropy: 
    H(Z|E) where Z is Alice's classical information

    Uses the abstract base class: PostselectedKeyrate
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

    def _quantum_rel_entropy_problem(self, rho, key_map_povm, m, k):
        mat_size = rho.shape[0]
        opt_mat_sh = (mat_size**2, mat_size**2)
        # Define cvxpy variables
        M = [cvx.Variable(opt_mat_sh, hermitian=True) for _ in range(k)]
        tau = cvx.Variable()
        T = cvx.Variable(m)

        key_map_povm_ = [np.kron(povm, np.eye(self.dim_B))
                         for povm in key_map_povm]
        cq_rho = np.sum(povm @ rho @ povm for povm in key_map_povm_)

        X = general_kron(rho, np.eye(mat_size))
        Y = cvx.kron(np.eye(mat_size), cvx.conj(cq_rho))
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
        constraints = const_geo_mean_cone + const_rel_entropy

        obj = cvx.Minimize(tau)
        problem = cvx.Problem(obj, constraints)
        return problem

    def _measurement_constraints(self, rho_AB):
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
        const_rho_pos = (rho_AB >> 0)
        rho_constraints = [const_rho_normalized, const_rho_pos] + \
            const_rho_AB + const_rho_AB_ub + const_rho_AB_lb
        return rho_constraints

    def _get_entangled_primal_problem(self, m, k):

        mat_size = self.dim_A*self.dim_B  # size of rho_AB
        rho_AB = cvx.Variable((mat_size, mat_size), name="rho", hermitian=True)

        # Compute the problem related to quantum relative entropy
        # and add them together
        p_pass = np.sum(self._basis_info_list.unpack("probability"))
        for i, basis in enumerate(self._basis_info_list):
            # need to transform the density matrix according to efficiency
            K_AB = np.kron(basis.kraus_A_op, basis.kraus_B_op)
            rho_AB_p = K_AB @ rho_AB @ np.conj(K_AB.T)
            # obtain the problem
            prob = self._quantum_rel_entropy_problem(
                rho_AB_p, basis.key_map_povm, m, k)
            p = basis.alice_prob * basis.bob_prob / p_pass
            if i == 0:
                q_rel_problem = p*prob
            else:
                q_rel_problem += p*prob

        rho_constraints = self._measurement_constraints(rho_AB)

        # Define the final problem
        problem = cvx.Problem(q_rel_problem.objective,
                              q_rel_problem.constraints + rho_constraints)
        return problem

    def _get_prepare_and_measure_primal_problem(self, m, k):

        mat_size = self.dim_A*self.dim_B  # size of rho_AB
        rho_AB = cvx.Variable((mat_size, mat_size), name="rho", hermitian=True)

        # Compute the problem related to quantum relative entropy
        # and add them together
        for i, basis in enumerate(self._basis_info_list):
            # need to transform the density matrix according to efficiency
            K_AB = np.kron(basis.kraus_A_op, basis.kraus_B_op)
            rho_AB_p = K_AB @ rho_AB @ np.conj(K_AB.T)

            # Pick up only the relevant constraints and key map povm
            indices = []
            for k in basis.pm_indices:
                indices += list(range(self.dim_B*k, self.dim_B*(k+1)))
            rel_indices = np.ix_(basis.pm_indices, basis.pm_indices)
            rel_indices_B = np.ix_(indices, indices)

            # obtain the problem
            rho = rho_AB_p[rel_indices_B]
            key_map_povm = [povm[rel_indices] for povm in basis.key_map_povm]

            prob = self._quantum_rel_entropy_problem(
                rho, key_map_povm, m, k)
            p = basis.bob_prob
            if i == 0:
                q_rel_problem = p*prob
            else:
                q_rel_problem += p*prob

        rho_constraints = self._measurement_constraints(rho_AB)
        # Define the final problem
        problem = cvx.Problem(q_rel_problem.objective,
                              q_rel_problem.constraints + rho_constraints)
        return problem

    def get_primal_problem(self, m=2, k=2):
        # If prepare_and_measure
        if self.prepare_and_measure:
            problem = self._get_prepare_and_measure_primal_problem(m, k)
        else:
            problem = self._get_entangled_primal_problem(m, k)
        return problem

    def compute_primal_with_matlab(self, matlab_engine, p_pass, trace_val=None,
                                   m=2, k=2,
                                   solver='SEDUMI'):
        """
        Solves the same primal problem but using MATLAB and CVX
        Note: p_pass is not used if self.prepare_and_measure = True
        """

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

        if trace_val is None:
            trace_val_ = float(1)
        else:
            trace_val_ = float(trace_val)

        dims_ = np_array_to_matlab(np.array([self.dim_A, self.dim_B]))
        m_ = float(m)
        k_ = float(k)
        p_pass_ = float(p_pass)
        n_sift_ = float(len(self._basis_info_list))

        if self.prepare_and_measure:
            indices_ = []
            n_povms = []
            key_map_povms_ = []
            kraus_AB_ = []

            for i, basis in enumerate(self._basis_info_list):
                # Pick up only the relevant constraints and key map povm
                indices = []
                for k in basis.pm_indices:
                    indices += list(range(self.dim_B*k, self.dim_B*(k+1)))
                indices_.append(np_array_to_matlab(np.array(indices)+1))

                # Pack the key_map_povms, and count how many povms there are
                rel_indices = np.ix_(basis.pm_indices, basis.pm_indices)
                count = 0
                for povm in basis.key_map_povm:
                    povm_ = povm[rel_indices]
                    povm_ = np_array_to_matlab(
                        np.kron(povm_, np.eye(self.dim_B)))
                    key_map_povms_.append(povm_)
                    count += 1
                n_povms.append(count)

                # Kraus operators related to measurements
                K_AB = np.kron(basis.kraus_A_op, basis.kraus_B_op)
                kraus_AB_.append(np_array_to_matlab(K_AB))

            n_povms_ = np_array_to_matlab(np.array(n_povms))

            bob_prob_ = np_array_to_matlab(
                self._basis_info_list.unpack("bob_prob"))

            val = matlab_engine.pm_postselected_rel_entropy_keyrate(
                key_map_povms_, n_povms_, indices_, bob_prob_,
                kraus_AB_,
                Gamma_exact_, gamma_,
                Gamma_inexact_, gamma_ub_, gamma_lb_,
                trace_val_,
                dims_, n_sift_, m_, k_,
                solver, self.verbose)

        else:
            key_map_povms = self._basis_info_list.unpack("key_map_povm")
            n_povms = []
            key_map_povms_ = []
            for km_povm in key_map_povms:
                count = 0
                for povm in km_povm:
                    povm_ = np_array_to_matlab(
                        np.kron(povm, np.eye(self.dim_B)))
                    key_map_povms_.append(povm_)
                    count += 1
                n_povms.append(count)
            n_povms_ = np_array_to_matlab(np.array(n_povms))

            kraus_AB_ = []
            for basis in self._basis_info_list:
                # Kraus operators related to measurements
                K_AB = np.kron(basis.kraus_A_op, basis.kraus_B_op)
                kraus_AB_.append(np_array_to_matlab(K_AB))

            alice_prob_ = np_array_to_matlab(
                self._basis_info_list.unpack("alice_prob"))
            bob_prob_ = np_array_to_matlab(
                self._basis_info_list.unpack("bob_prob"))
            postselect_prob_ = np_array_to_matlab(
                self._basis_info_list.unpack("probability"))

            val = matlab_engine.ent_postselected_rel_entropy_keyrate(
                key_map_povms_, n_povms_, alice_prob_, bob_prob_,
                kraus_AB_,
                Gamma_exact_, gamma_,
                Gamma_inexact_, gamma_ub_, gamma_lb_,
                trace_val_,
                dims_, n_sift_, p_pass_, m_, k_,
                solver, self.verbose)

        return val

    def _keyrate_function(self, value, p_pass):
        return p_pass * value / np.log(2)
