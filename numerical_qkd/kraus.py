"""
kraus
====================
Defines class that deals with Kraus operators

Author: Darius Bunandar (dariusb@mit.edu)

Unauthorized use and/or duplication of this material without express and
written permission from the author and/or owner is strictly prohibited.
==============================================================================
"""


class KrausOperator(object):
    """
    KrausOperator

    Kraus operator as defined for both Alice's and Bob's measurement.
    dim = dim_A for Alice and dim = dim_B for Bob.

    :param povm: 2-d numpy array POVM of size [dim, dim]
    :param basis: integer value for the basis information of that measurement
    :param value: integer value for the output element when the POVM succeeds
    :param basis_prob: float for the probability of choosing this POVM
    """

    def __init__(self, povm, basis, value, basis_prob):
        self.povm = povm
        self.basis = basis
        self.value = value
        self.basis_prob = basis_prob
