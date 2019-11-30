"""
basis_info
====================
Defines class that deals with a postselected basis information

Author: Darius Bunandar (dariusb@mit.edu)

Unauthorized use and/or duplication of this material without express and
written permission from the author and/or owner is strictly prohibited.
==============================================================================
"""

import numpy as np


class BasisInformation(object):
    """
    BasisInformation
    """

    def __init__(self, basis=(None, None),
                 Gamma=None,
                 kraus_A_op=None,
                 kraus_B_op=None,
                 kraus_op=None,
                 probability=None,
                 key_map_povm=None,
                 pm_indices=None,
                 alice_prob=None,
                 bob_prob=None,
                 gamma=None, gamma_lb=None, gamma_ub=None):
        self.basis = basis
        self.Gamma = Gamma
        self.kraus_A_op = kraus_A_op
        self.kraus_B_op = kraus_B_op
        self.kraus_op = kraus_op
        self.probability = probability
        self.key_map_povm = key_map_povm
        self.pm_indices = pm_indices
        self.alice_prob = alice_prob
        self.bob_prob = bob_prob
        self.gamma = gamma
        self.gamma_lb = gamma_lb
        self.gamma_ub = gamma_ub


class BasisInformationList(list):
    """
    BasisInformationList
    """

    def unpack(self, key):
        values = [getattr(p, key) for p in self]
        if key in ['gamma', 'gamma_lb', 'gamma_ub', 'probability',
                   'alice_prob', 'bob_prob']:
            values = np.array(values)
        return values

    def pack(self, key, values):
        for p, v in zip(self, values):
            setattr(p, key, v)
        return self
