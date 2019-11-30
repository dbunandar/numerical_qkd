"""
utils
====================
Miscellaneous utility functions for working with numpy and cvxpy

Author: Darius Bunandar (dariusb@mit.edu)

Unauthorized use and/or duplication of this material without express and
written permission from the author and/or owner is strictly prohibited.
==============================================================================
"""

import numpy as np
import scipy as sp
import cvxpy
import matlab
import itertools
from cvxpy.expressions.expression import Expression


# Methods related to CVXPY
# ============================================================================
def expr_as_np_array(cvx_expr):
    if cvx_expr.is_scalar():
        return np.array(cvx_expr)
    elif len(cvx_expr.shape) == 1:
        return np.array([v for v in cvx_expr])
    else:
        # then cvx_expr is a 2d array
        rows = []
        for i in range(cvx_expr.shape[0]):
            row = [cvx_expr[i, j] for j in range(cvx_expr.shape[1])]
            rows.append(row)
        arr = np.array(rows)
        return arr


def np_array_as_expr(np_arr):
    aslist = np_arr.tolist()
    expr = cvxpy.bmat(aslist)
    return expr


def general_kron(a, b):
    """
    Returns a CVXPY Expression representing the Kronecker product of a and b.

    At most one of "a" and "b" may be CVXPY Variable objects.

    :param a: 2D numpy ndarray, or a CVXPY Variable with a.ndim == 2
    :param b: 2D numpy ndarray, or a CVXPY Variable with b.ndim == 2
    """
    if not isinstance(a, Expression):
        a = cvxpy.Constant(value=a)
    if not isinstance(b, Expression):
        b = cvxpy.Constant(value=b)
    a_np = expr_as_np_array(a)
    b_np = expr_as_np_array(b)
    c_np = np.kron(a_np, b_np)
    return np_array_as_expr(c_np)


def partial_trace(rho, dims, axis=0):
    """
    Computes partial trace over the subsystem defined by axis.
    Each subsystem must be a square matrix.

    :param rho: 2D numpy ndarray, or a CVXPY Variable with ndim == 2
    :param dims: 1D numpy ndarray, or a list of the dimension of each subsystem
    :param axis: an integer, the index of the subsytem to be traced out
    """
    if not isinstance(rho, Expression):
        rho = cvxpy.Constant(value=rho)
    rho_np = expr_as_np_array(rho)
    traced_rho = np_partial_trace(rho_np, dims, axis)
    traced_rho = np_array_as_expr(traced_rho)
    return traced_rho


def block_diagonal_stack(matrices):
    """
    Make a block diagonal matrix from the list of matrices.
    """

    n = len(matrices)

    if n == 1:
        return matrices[0]
    else:
        full_matrix = []
        for i, mat in enumerate(matrices):
            z = np.zeros(mat.shape)
            preceding_zeros = list(itertools.repeat(z, i))
            following_zeros = list(itertools.repeat(z, n-i-1))
            row_list = preceding_zeros + [mat] + following_zeros
            row = cvxpy.hstack(row_list)
            full_matrix.append(row)
        full_matrix = cvxpy.vstack(full_matrix)
        return full_matrix

# ============================================================================

# Useful general methods in numpy
# ============================================================================


def np_partial_trace(rho, dims, axis=0):
    """
    Takes partial trace over the subsystem defined by 'axis'
    rho: a matrix
    dims: a list containing the dimension of each subsystem
    axis: the index of the subsytem to be traced out
    (We assume that each subsystem is square)
    """
    dims_ = np.array(dims)
    # Reshape the matrix into a tensor with the following shape:
    # [dim_0, dim_1, ..., dim_n, dim_0, dim_1, ..., dim_n]
    # Each subsystem gets one index for its row and another one for its column
    reshaped_rho = np.reshape(rho, np.concatenate((dims_, dims_), axis=None))

    # Move the subsystems to be traced towards the end
    reshaped_rho = np.moveaxis(reshaped_rho, axis, -1)
    reshaped_rho = np.moveaxis(reshaped_rho, len(dims) + axis - 1, -1)

    # Trace over the very last row and column indices
    traced_out_rho = np.trace(reshaped_rho, axis1=-2, axis2=-1)

    # traced_out_rho is still in the shape of a tensor
    # Reshape back to a matrix
    dims_untraced = np.delete(dims_, axis)
    rho_dim = np.prod(dims_untraced)
    return traced_out_rho.reshape([rho_dim, rho_dim])


def move_hilbert_space(rho, dims, axis_from, axis_to):
    dims_ = np.array(dims)

    # Reshape the matrix into a tensor with the following shape:
    # [dim_0, dim_1, ..., dim_n, dim_0, dim_1, ..., dim_n]
    # Each subsystem gets one index for its row and another one for its column
    reshaped_rho = np.reshape(rho, np.concatenate((dims_, dims_), axis=None))

    # Move the subsystems around
    reshaped_rho = np.moveaxis(reshaped_rho, axis_from, axis_to)
    reshaped_rho = np.moveaxis(reshaped_rho, len(
        dims) + axis_from, len(dims) + axis_to)

    # Reshape back into a matrix and return
    rho_dim = np.prod(dims)
    return reshaped_rho.reshape([rho_dim, rho_dim])


def quantum_entropy(rho):
    """
    Computes the von Neumann entropy of a given quantum state.

    :param rho: 2D numpy ndarray
    """
    fudge = 1e-16
    sh = rho.shape
    new_rho = (1 - fudge) * rho + fudge * np.eye(sh[0])
    return -1 * np.trace(np.matmul(rho, sp.linalg.logm(new_rho))) / np.log(2)


def matrix_gram_schmidt(Gamma, normalize=True):
    """
    Performs matrix orthgonalization by performing QR decomposition.

    Assume that each Gamma is an m x m matrix and there are N of them.
    1. We flatten (vectorize) each Gamma into a length M = m^2 vector.
    2. We perform the complete QR decomposition to obtain an M x M matrix
    3. The first N are the orthogonalized Gamma matrices --> Gamma_out
    4. The remaining M-N matrices are the rest of the basis --> Omega_out
    """
    N = len(Gamma)
    sh = Gamma[0].shape

    # Flatten Gamma and stack them column-by-column
    Gamma_flat = [G.flatten('F')[:,np.newaxis] for G in Gamma]
    if normalize:
        Gamma_flat = [G_flat / np.linalg.norm(G_flat) for G_flat in Gamma_flat]
    Gamma_flat = np.hstack(Gamma_flat)

    # Perform Gram-Schmidt with QR decomposition
    q, r = np.linalg.qr(Gamma_flat, mode='complete')

    Gamma_out = []
    Omega_out = []
    for i in range(len(q)):
        G = q[:,i].reshape(sh, order='F')
        if i < N:
            Gamma_out.append(G)
        else:
            Omega_out.append(G)

    return Gamma_out, Omega_out, q, r

def quantum_rel_entropy(rho1, rho2):
    """
    Computes the quantum relative entropy D(rho1 || rho2)

    :param rho1: 2D numpy ndarray
    :param rho2: 2D numpy ndarray
    """

    fudge = 1e-16
    assert(rho1.shape == rho2.shape), "Need two matrices to be the same shape"
    sh = rho2.shape

    new_rho2 = (1-fudge) * rho2 + fudge * np.eye(sh[0])

    A = -1*quantum_entropy(rho1)
    B = np.trace(np.matmul(rho1, sp.linalg.logm(new_rho2))) / np.log(2)

    return A - B


def bin_entropy(x):
    """
    Binary entropy h(x) = -x*log2(x) - (1-x)*log2(1-x)

    :param x: a float with 0<=x<=1
    """
    return classical_entropy(np.array([x, 1-x]))


def classical_entropy(x):
    """
    Classical entropy function

    H(X) = sum_i -x_i * log2(x_i)
    :param x: 1D numpy ndarray, must add up to 1 and non-negative
    """
    assert np.isclose(np.sum(x), 1.0), "probabilities don't add up to 1"
    if np.any(x <= 0) or np.any(x >= 1):
        return 0
    else:
        return np.sum(-x * np.log2(x))


def one_hot_unit_vector(value, dim):
    vector = np.zeros(dim)
    vector[value] = 1
    return vector

# ============================================================================
# MATLAB
# ============================================================================


def np_array_to_matlab(np_arr):
    aslist = np_arr.tolist()

    if np_arr.dtype in [np.float64, np.float32]:
        mat_array = matlab.double(aslist)
    elif np_arr.dtype in [np.int64, np.int32, np.int16, np.int8]:
        mat_array = matlab.int64(aslist)
    elif np_arr.dtype in [np.complex128, np.complex64]:
        mat_array = matlab.double(aslist, is_complex=True)

    return mat_array

# ============================================================================
# Channels
# ============================================================================


def depolarizing_channel(rho, dims, p, conf=None):
    """
    Depolarizing channel on system B with probability p

    :param rho: 2D numpy ndarray, initial density matrix
    :param dims: a list of dimensions [dim_A, dim_B]
    :param p: a float, probability depolarizing occurs
    """
    rho_A = np_partial_trace(rho, dims, axis=1)
    dim_B = dims[1]
    if conf is None:
        conf = dim_B
    sigma_AB = np.kron(rho_A, np.eye(dim_B)/conf)
    return (1 - p) * rho + p * sigma_AB
# ============================================================================


def simulate_single_photons(bob_eff=1, prob_dark_count=0):
    prob_click = 1-(1-2*prob_dark_count)*(1-bob_eff)
    return prob_click


if __name__ == '__main__':
    """
    Test out the partial_trace numpy module by creating a matrix
    rho_ABC = rho_A \otimes rho_B \otimes rho_C
    Each rho_i is normalized, i.e. Tr(rho_i) = 1
    """

    # Generate five test cases
    rho_A = np.random.rand(4, 4) + 1j * np.random.rand(4, 4)
    rho_A /= np.trace(rho_A)
    rho_B = np.random.rand(3, 3) + 1j * np.random.rand(3, 3)
    rho_B /= np.trace(rho_B)
    rho_C = np.random.rand(2, 2) + 1j * np.random.rand(2, 2)
    rho_C /= np.trace(rho_C)
    rho_AB = np.kron(rho_A, rho_B)
    rho_AC = np.kron(rho_A, rho_C)

    # Construct a cvxpy Variable with value equal
    # to rho_A \otimes rho_B \otimes rho_C.
    temp = np.kron(rho_AB, rho_C)
    rho_ABC = cvxpy.Variable(shape=temp.shape, complex=True)
    rho_ABC.value = temp

    # Try to recover simpler tensors products by taking partial traces of
    # more complicated tensors.
    rho_AB_test = partial_trace(rho_ABC, [4, 3, 2], axis=2)
    rho_AC_test = partial_trace(rho_ABC, [4, 3, 2], axis=1)
    rho_A_test = partial_trace(rho_AB_test, [4, 3], axis=1)
    rho_B_test = partial_trace(rho_AB_test, [4, 3], axis=0)
    rho_C_test = partial_trace(rho_AC_test, [4, 2], axis=0)

    # See if the outputs of partial_trace are correct
    print("rho_AB test correct? ", np.allclose(rho_AB_test.value, rho_AB))
    print("rho_AC test correct? ", np.allclose(rho_AC_test.value, rho_AC))
    print("rho_A test correct? ", np.allclose(rho_A_test.value, rho_A))
    print("rho_B test correct? ", np.allclose(rho_B_test.value, rho_B))
    print("rho_C test correct? ", np.allclose(rho_C_test.value, rho_C))
