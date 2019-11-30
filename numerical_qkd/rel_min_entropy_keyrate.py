"""
rel_min_entropy_keyrate
====================
Calculates secret keyrate by solving for Hmin

Author: Darius Bunandar (dariusb@mit.edu)

Unauthorized use and/or duplication of this material without express and
written permission from the author and/or owner is strictly prohibited.
==============================================================================
"""

import numpy as np
import cvxpy as cvx
from numerical_qkd import Keyrate


class RelativeMinEntropyKeyrate(Keyrate):
    """
    RelativeMinEntropyKeyrate

    Computes numerical finite key rate of a QKD protocol by solving
    min-entropy: H_{min}(Z|E) where Z is Alice's classical information.

    Uses the abstract base class: Keyrate
    """

    def get_primal_problem(self):

        mat_size = np.prod(self.dims)  # matrix sizes

        X11 = cvx.Variable((mat_size, mat_size), complex=True)
        X12 = cvx.Variable((mat_size, mat_size), complex=True)
        X22 = cvx.Variable((mat_size, mat_size), complex=True)
        rho_AB = cvx.Variable((mat_size, mat_size), hermitian=True)
        sigma_AB = cvx.Variable((mat_size, mat_size), hermitian=True)

        X = cvx.bmat([[X11, X12], [X12.H, X22]])

        obj = cvx.Maximize(cvx.trace(cvx.real(X12)))

        const_X11 = (X11 << rho_AB)
        cq_sigma_AB = np.sum(povm*sigma_AB*povm for povm in self._key_map_povm)
        const_X22 = (X22 << cq_sigma_AB)
        const_sigma_AB = (cvx.real(cvx.trace(sigma_AB)) <= 1)
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
        const_sigma_pos = (sigma_AB >> 0)
        const_rho_pos = (rho_AB >> 0)
        constraints = [const_X11, const_X22,
                       const_sigma_AB, const_X_pos,
                       const_sigma_pos, const_rho_pos, const_rho_normalized] \
            + const_rho_AB + const_rho_AB_ub + const_rho_AB_lb

        problem = cvx.Problem(obj, constraints)
        return problem

    def get_dual_problem(self):

        mat_size = np.prod(self.dims)  # matrix sizes

        Y11 = cvx.Variable((mat_size, mat_size), hermitian=True)
        Y22 = cvx.Variable((mat_size, mat_size), hermitian=True)
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

        zero_mat = np.zeros([mat_size, mat_size])
        Y = cvx.bmat([[Y11, zero_mat],
                      [zero_mat, Y22]])
        A = .5*cvx.bmat([[zero_mat, np.eye(mat_size)],
                         [np.eye(mat_size), zero_mat]])

        cq_Y22 = np.sum(povm*Y22*povm for povm in self._key_map_povm)
        const_y = (y*np.eye(mat_size) >> cq_Y22)

        sum_of_Gammas = zi_sum + yi_sum - xi_sum
        const_yi = (sum_of_Gammas + y_id * np.eye(mat_size) >> Y11)
        const_Y = (Y >> A)

        constraints = [const_y, const_yi, const_Y]

        problem = cvx.Problem(obj, constraints)
        return problem

    def _keyrate_function(self, value):
        if value <= 0:
            return 0
        else:
            return -2*np.log2(value)
