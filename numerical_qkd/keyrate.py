"""
keyrate
====================
Abstract base class for keyrate calculations

Author: Darius Bunandar (dariusb@mit.edu)

Unauthorized use and/or duplication of this material without express and
written permission from the author and/or owner is strictly prohibited.
==============================================================================
"""

import abc
import numpy as np
import scipy.linalg as linalg
from numerical_qkd import one_hot_unit_vector, Constraints


class Keyrate(object):
    """
    Keyrate

    An abstract base class for computing the different QKD keyrates

    :param key_map_povm: {Z_A}_j
        POVM that maps Alice's quantum information to classical information.
        List of 2-d numpy array of shape [dim_A, dim_A]
    :param exact_constraints:
        A dictionary consisting of
        Gamma: {G_AB}_i
            POVM constraints of the protocol.
            List of 2-d numpy array of shape [dim_A*dim_B, dim_A*dim_B]
        gamma: {g_i}
            Values of the measured POVM constraints.
            List/1-d numpy array. Gamma and gamma must have the same length
    :param inexact_constraints:
        A dictionary consisting of
        Gamma: {G_AB}_i
            POVM constraints of the protocol.
            List of 2-d numpy array of shape [dim_A*dim_B, dim_A*dim_B]
        gamma_lb: {g_lb_i}
            Lower-bound values of the measured POVM constraints.
            List/1-d numpy array. Gamma and gamma_lb must have the same length
        gamma_ub: {g_ub_i}
            Upper-bound values of the measured POVM constraints.
            List/1-d numpy array. Gamma and gamma_ub must have the same length
    :param dims: List of integers [dim_A, dim_B], e.g. BB84 has dims = [2, 2]
        Dimensionality of the Hilbert space for both Alice and Bob.
    :param expand_povm: Boolean
        Process the key map POVMs, which are usually defined for Alice only,
        to have identities for Bob. Set 'False' if your key map POVMs
        are already fully defined for both parties.
    :param verbose: Boolean
    """

    def __init__(self, key_map_povm, dims,
                 exact_constraints=None, inexact_constraints=None,
                 expand_povm=True,
                 verbose=False):
        self.dims = dims
        self.clear_constraints()  # This initiates all the constraints
        self.set_constraints(exact_constraints, inexact_constraints)
        self.expand_povm = expand_povm
        self.key_map_povm = key_map_povm
        self.verbose = verbose

    @property
    def key_map_povm(self):
        return self._key_map_povm

    @key_map_povm.setter
    def key_map_povm(self, key_map_povm):
        # Check if key_map_povm is supplied
        if key_map_povm is None:
            self._key_map_povm = None
            return

        if self.expand_povm:
            # Process key_map_povm, ensure that they match the dimensions
            if len(self.dims) == 2:
                # We expect key_map_povm to be only for A
                dim_B = self.dims[1]
                self._key_map_povm = [np.kron(povm, np.eye(dim_B))
                                      for povm in key_map_povm]
            else:  # no pre-processing needed
                # We expect key_map_povm to be defined for all
                self._key_map_povm = key_map_povm
        else:
            self._key_map_povm = key_map_povm

    def set_constraints(self, reset=True,
                        exact_constraints=None, inexact_constraints=None):
        if reset:
            self.clear_constraints()
        if exact_constraints is not None:
            self.Gamma_exact = exact_constraints['Gamma']
            self.gamma = exact_constraints['gamma']
        if inexact_constraints is not None:
            self.Gamma_inexact = inexact_constraints['Gamma']
            self.gamma_lb = inexact_constraints['gamma_lb']
            self.gamma_ub = inexact_constraints['gamma_ub']

    def clear_constraints(self):
        self.Gamma_exact = None
        self.Gamma_inexact = None
        self.gamma = None
        self.gamma_lb = None
        self.gamma_ub = None

    @abc.abstractmethod
    def _keyrate_function(self, value):
        pass

    @abc.abstractmethod
    def get_primal_problem(self):
        pass

    @abc.abstractmethod
    def get_dual_problem(self):
        pass

    def solve_problem(self, problem, solver='SCS', **solver_kwargs):
        problem.solve(solver, **solver_kwargs, verbose=self.verbose)
        if problem.status not in ["infeasible", "unbounded"]:
            return self._keyrate_function(problem.value)
        else:
            return 0

    def compute_primal(self, solver='SCS', **solver_kwargs):
        problem = self.get_primal_problem()
        value = self.solve_problem(problem, solver, **solver_kwargs)
        return value

    def compute_dual(self, solver='SCS', **solver_kwargs):
        problem = self.get_dual_problem()
        value = self.solve_problem(problem, solver, **solver_kwargs)
        return value


class PostselectedKeyrate(Keyrate):
    """
    PostselectedKeyrate

    An abstract base class for computing ```postselected''' QKD keyrates

    With post-selection, the Hilbert space of interests are:
    A   = Alice's quantum state (of size dim_A)
    Ab  = Alice's basis
    Av  = Alice's element/value in that basis (of size n_value_A)
    B   = Bob's quantum state (of size dim_B)
    Bb  = Bob's basis
    Bv  = Bob's element/value in that basis (of size n_value_B)
    The basis projection operators project in the Ab and Bb spaces.
    For simplicity, we abbreaviate A'B' = ApBp = AAvBBv

    :param key_map_povm: {Z_Av}_j
        POVM that maps Alice's element to keys.
        List of 2-d numpy array of:
            id(dim_A) x id(dim_B) x [n_value_A, n_value_A] x id(n_value_B)
    :param exact_constraints:
        A dictionary consisting of
        Gamma: {G_AB}_i
            POVM constraints of the protocol.
            List of 2-d numpy array of shape [dim_A*dim_B, dim_A*dim_B]
        gamma: {g_i}
            Values of the measured POVM constraints.
            List/1-d numpy array. Gamma and gamma must have the same length
    :param inexact_constraints:
        A dictionary consisting of
        Gamma: {G_AB}_i
            POVM constraints of the protocol.
            List of 2-d numpy array of shape [dim_A*dim_B, dim_A*dim_B]
        gamma_lb: {g_lb_i}
            Lower-bound values of the measured POVM constraints.
            List/1-d numpy array. Gamma and gamma_lb must have the same length
        gamma_ub: {g_ub_i}
            Upper-bound values of the measured POVM constraints.
            List/1-d numpy array. Gamma and gamma_ub must have the same length
    :param dims:
        Dimensionality of the Hilbert space for both Alice and Bob.
        List/1-d numpy array of [dim_A, dim_B].
    :param verbose: Boolean
    """

    def __init__(self, key_map_povm, dims,
                 prepare_and_measure=False,
                 basis_info_list=None,
                 exact_constraints=None, inexact_constraints=None,
                 expand_povm=True,
                 verbose=False):
        super().__init__(key_map_povm, dims, exact_constraints,
                         inexact_constraints, expand_povm, verbose)
        assert(len(self.dims) == 2), "Only bipartite has been implemented"
        self.dim_A, self.dim_B = self.dims
        self.prepare_and_measure = prepare_and_measure
        if basis_info_list is not None:
            self.set_basis_info_list(basis_info_list)
        # if basis_info_list is None:  # setup using Kraus operators
        #     self.set_kraus_operators(kraus_operators, postselectors)
        # else:  # setup using precomputed basis_info_list
        #     self.set_basis_info_list(basis_info_list)

    # def set_kraus_operators(self, kraus_operators, postselectors, rho=None):
    #     """
    #     The postselection procedure must be computed everytime we change
    #     either the Kraus operators or the basis postselectors
    #     """
    #     constraints = Constraints(kraus_operators, postselectors,
    #                               [self.dim_A, self.dim_B])
    #     self._basis_info_list = \
    #         constraints.generate_constraint_povms_and_postselection_probs(rho)
    #     self._n_value_A, _ = constraints.get_n_values()
    #     # self._n_basis_A, self._n_basis_B = constraints.get_n_bases()

    def set_basis_info_list(self, basis_info_list):
        self._basis_info_list = basis_info_list
        # TODO:
        # - Do we use the  values of Gamma and gamma's in basis_info_list?
        # - Figure out a better way of getting n_value_A
        kraus_ops = basis_info_list.unpack('kraus_op')
        self._n_value_A = kraus_ops[0].shape[0] // np.prod(self.dims)

    @abc.abstractmethod
    def _keyrate_function(self, value, p_pass):
        pass

    def solve_problem(self, problem, p_pass, solver='MOSEK', **solver_kwargs):
        problem.solve(solver, **solver_kwargs, verbose=self.verbose)
        if problem.status not in ["infeasible", "unbounded"]:
            return self._keyrate_function(problem.value, p_pass)
        else:
            return 0

    def compute_primal(self, p_pass, solver='SCS', **solver_kwargs):
        problem = self.get_primal_problem()
        value = self.solve_problem(problem, p_pass, solver, **solver_kwargs)
        return value

    def compute_dual(self, p_pass, solver='SCS', **solver_kwargs):
        problem = self.get_dual_problem()
        value = self.solve_problem(problem, p_pass, solver, **solver_kwargs)
        return value
