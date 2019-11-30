"""
constraints
====================
Methodically generate constraints

Author: Darius Bunandar (dariusb@mit.edu)

Unauthorized use and/or duplication of this material without express and
written permission from the author and/or owner is strictly prohibited.
==============================================================================
"""

import numpy as np
import scipy.linalg as linalg
from numerical_qkd import one_hot_unit_vector, \
    BasisInformation, BasisInformationList
import itertools


def generate_simple_constraints(rho, kraus_A, kraus_B, Gamma_A):
    """
    GenerateConstraints

    Given the state rho (on Hilbert space AB), generate POVMs Gamma and
    their respective measurement values gamma.
    :param kraus_A: Alice's Kraus operators
    :param kraus_B: Bob's Kraus operators
    :param Gamma_A: Full constraint that defines Alice's state (for P&M)
    """

    Gamma = [np.kron(kA.povm, kB.povm) for kA in kraus_A for kB in kraus_B]
    Gamma = Gamma + Gamma_A
    gamma = [np.real(np.trace(G @ rho)) for G in Gamma]

    return Gamma, gamma


def generate_sender_constraints(rho):
    """
    generate_sender_constraints

    Generate the constraints that belong to the sender only. The sender, after
    characterizing his/her state, will know these parameters with 100%
    certainty.

    We use the fact that rho = sum_i p_i |psi_i><psi_i|
    to get the relevant eigenvalues (probability) and eigenvectors

    :param rho: Sender's state (do partial trace to obtain this state)
    """
    eigvals, eigvecs = np.linalg.eig(rho)

    n = eigvecs.shape[1]

    Gamma = []
    for i in range(n):
        v = eigvecs[:, i]
        G = np.outer(v, np.conj(v))
        Gamma.append(G)

    gamma = np.real(eigvals)  # density matrices are positive semidefinite

    return Gamma, gamma


class Constraints(object):
    """
    Constraints
    """

    def __init__(self, kraus_operators, postselectors,
                 dims, prepare_and_measure=False):
        self.set_kraus_operators(kraus_operators, postselectors)
        self.dim_A, self.dim_B = dims
        self.pm = prepare_and_measure

    def set_kraus_operators(self, kraus_operators, postselectors):
        """
        The postselection procedure must be computed everytime we change
        either the Kraus operators or the basis postselectors
        """
        self._kraus_A, self._kraus_B = kraus_operators
        self._postselect_A, self._postselect_B = postselectors
        # Perform checks on the Kraus operators and post-selectors
        self._n_basis_A, self._n_value_A = self._check_kraus_operators(
            self._kraus_A)
        self._n_basis_B, self._n_value_B = self._check_kraus_operators(
            self._kraus_B)

    def _check_kraus_operators(self, kraus):
        """
        Check the Kraus operators to ensure that there are not unused basis
        or unnecessary value
        """
        basis = np.unique([k.basis for k in kraus])
        value = np.unique([k.value for k in kraus])

        assert len(basis) == np.amax(basis) + 1, "Unused basis exists"
        assert len(value) == np.amax(value) + 1, "Unused value exists"
        return len(basis), len(value)

    def get_n_values(self):
        return self._n_value_A, self._n_value_B

    def get_n_bases(self):
        return self._n_basis_A, self._n_basis_B

    def generate_constraint_povms_and_postselection_probs(
            self, rho=None, third_state_proj=None):
        """
        Generates the constraint POVMs and postelection probabilities.
        The details are packed into basis_info_list.

        :param rho: 2-D numpy array describing the density matrix
        :param third_state_proj: 2-D numpy array. A projector on the a third
            Hilbert space. This is usually only used in the case of MDI-QKD.
            The three Hilbert spaces would be: A, B, and C/E.
            Note: the numpy array rho must have the dimensions of the three
            Hilbert spaces.
        """
        G_size = self.dim_A * self.dim_B
        basis_info_list = BasisInformationList()
        # Order the Kraus operators according to its basis
        for bA, bB in zip(self._postselect_A, self._postselect_B):
            # find which kraus operators are in this basis
            kraus_A_b = [kA for kA in self._kraus_A if (kA.basis == bA)]
            kraus_B_b = [kB for kB in self._kraus_B if (kB.basis == bB)]

            G_same = np.zeros([G_size, G_size], dtype=np.complex128)
            # get the operators that indicate they have the same value
            for val in range(np.amax([self._n_value_A, self._n_value_B])):
                kraus_A_bv = [kA for kA in kraus_A_b if (kA.value == val)]
                kraus_B_bv = [kB for kB in kraus_B_b if (kB.value == val)]

                if not kraus_A_bv:  # if empty
                    povm_A_bv = np.eye(self.dim_A, dtype=np.complex128)
                elif len(kraus_A_bv) > 1:  # if multiple operators exist
                    raise ValueError("Multiple POVMs give the same result")
                else:
                    povm_A_bv = kraus_A_bv[0].povm

                if not kraus_B_bv:  # if empty
                    povm_B_bv = np.eye(self.dim_B, dtype=np.complex128)
                elif len(kraus_B_bv) > 1:  # if multiple operators exist
                    raise ValueError("Multiple POVMs give the same result")
                else:
                    povm_B_bv = kraus_B_bv[0].povm

                G_same += np.kron(povm_A_bv, povm_B_bv)

            """
            Find the key map POVM for each value in this basis
            TODO: Make this method more general
            We currently assume:
            - The key map POVM belongs to Alice (probably okay)
            - Translates from a key map POVM is diagonal in the standard basis
                of Alice's value (Av) Hilbert space,
                i.e. {|i><i|} for i = [0, ..., n_value_A-1]
            """
            key_map_povm = []
            for kA in kraus_A_b:
                povm = kA.povm / kA.basis_prob
                key_map_povm.append(povm)
            bob_prob = kraus_B_b[0].basis_prob
            alice_prob = kraus_A_b[0].basis_prob

            pm_indices = []
            K_AB_full = np.zeros([self.dim_A * self.dim_B *
                                  self._n_value_A * self._n_value_B,
                                  self.dim_A * self.dim_B],
                                 dtype=np.complex128)
            K_AB = np.zeros([self.dim_A * self.dim_B * self._n_value_A,
                             self.dim_A * self.dim_B], dtype=np.complex128)

            for kA in kraus_A_b:
                assert(np.isclose(kA.basis_prob, alice_prob)
                       ), "Inconsistent basis probabilities for Alice"
                if self.pm:
                    nz = np.transpose(np.nonzero(kA.povm))
                    # Check that there is only one non-zero element
                    assert(len(nz) == 1), "More than one non-zero exists"
                    # Make sure that it's on the diagonal
                    v = nz[0]
                    assert(v[0] == v[1]), "Non-zero element must be on diag"
                    pm_indices.append(v[0])
                for kB in kraus_B_b:
                    assert(np.isclose(kB.basis_prob, bob_prob)
                           ), "Inconsistent basis probabilities for Bob"

                    sqrt_povm_A = linalg.sqrtm(kA.povm).astype(complex)
                    sqrt_povm_B = linalg.sqrtm(kB.povm).astype(complex)
                    vA = one_hot_unit_vector(kA.value, self._n_value_A)
                    vB = one_hot_unit_vector(kB.value, self._n_value_B)

                    # Construct sqrt(povm_A) o sqrt(povm_B) o vA o vB
                    sqrt_povm_AB = np.kron(sqrt_povm_A, sqrt_povm_B)
                    values_AB = np.kron(
                        vA[:, np.newaxis], vB[:, np.newaxis])
                    K_AB_full += np.kron(sqrt_povm_AB, values_AB)
                    K_AB += np.kron(sqrt_povm_AB, vA[:, np.newaxis])

            K_A = np.zeros([self.dim_A, self.dim_A], dtype=np.complex128)
            K_B = np.zeros([self.dim_B, self.dim_B], dtype=np.complex128)

            for kA in kraus_A_b:
                K_A += linalg.sqrtm(kA.povm/kA.basis_prob).astype(complex)
            for kB in kraus_B_b:
                K_B += linalg.sqrtm(kB.povm/kB.basis_prob).astype(complex)

            # Compute postselection probability of this basis
            pr = None
            if rho is not None:
                if third_state_proj is not None:
                    K_AB_full = np.kron(K_AB_full, third_state_proj)
                pr = np.real(np.trace(K_AB_full @ rho @ np.conj(K_AB_full.T)))

            b_info = BasisInformation(basis=(bA, bB), Gamma=G_same,
                                      kraus_A_op=K_A,
                                      kraus_B_op=K_B,
                                      kraus_op=K_AB,
                                      probability=pr,
                                      key_map_povm=key_map_povm,
                                      pm_indices=pm_indices,
                                      alice_prob=alice_prob,
                                      bob_prob=bob_prob)
            basis_info_list.append(b_info)

        # Check that each element of the basis announcement is unique
        basis_list = basis_info_list.unpack('basis')
        assert (len(basis_list) == len(set(basis_list))
                ), "non-unique basis exists"

        return basis_info_list

    def generate_exact_constraint_values(self, rho, third_state_proj=None):
        basis_info_list = \
            self.generate_constraint_povms_and_postselection_probs(
                rho, third_state_proj)
        for p in basis_info_list:
            Gamma = p.Gamma
            if third_state_proj is not None:
                Gamma = np.kron(Gamma, third_state_proj)
            p.gamma = np.real(np.trace(Gamma @ rho))
        return basis_info_list

    def generate_single_photon_constraint_values(self, rho, total_signals_sent,
                                                 error, bounding_method,
                                                 single_photon_simulator,
                                                 third_state_proj=None,
                                                 **sim_kwargs):

        basis_info_list = self.generate_exact_constraint_values(
            rho, third_state_proj)
        basis_info_list = self.single_photon_constraint_bounds(
            basis_info_list, total_signals_sent,
            error, bounding_method, single_photon_simulator, **sim_kwargs)
        return basis_info_list

    @staticmethod
    def single_photon_constraint_bounds(basis_info_list,
                                        total_signals_sent, error,
                                        bounding_method,
                                        single_photon_simulator,
                                        **sim_kwargs):
        gamma = basis_info_list.unpack('gamma')
        postselection_probability = basis_info_list.unpack('probability')

        prob_click = single_photon_simulator(**sim_kwargs)
        total_clicks = prob_click*total_signals_sent
        mi_vec = postselection_probability*total_clicks
        si_vec = gamma*total_clicks

        if np.isscalar(error):
            error = error * np.ones(len(mi_vec))
        delta_si_neg = np.array([bounding_method.delta_lower_bound(
            mi, ei) for mi, ei in zip(mi_vec, error)])
        delta_si_pos = np.array([bounding_method.delta_upper_bound(
            mi, ei) for mi, ei in zip(mi_vec, error)])
        si_lb_vec = np.amax(
            [si_vec + delta_si_neg, np.zeros(len(si_vec))], axis=0)
        si_ub_vec = np.amin([si_vec + delta_si_pos, mi_vec], axis=0)
        gamma_lb = si_lb_vec/total_signals_sent
        gamma_ub = si_ub_vec/total_signals_sent

        basis_info_list.pack('gamma_lb', gamma_lb)
        basis_info_list.pack('gamma_ub', gamma_ub)
        return basis_info_list
