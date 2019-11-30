"""
concentration_ineq
====================
Concentration inequalities

Author: Darius Bunandar (dariusb@mit.edu)

Unauthorized use and/or duplication of this material without express and
written permission from the author and/or owner is strictly prohibited.
==============================================================================
"""

import abc
import numpy as np


class ConcentrationInequality(object):
    """
    ConcentrationInequality

    An abstract base class for the different concentration inequalities
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.reset_counters()

    def reset_counters(self):
        self.upper_bound_counter = 0
        self.lower_bound_counter = 0
        self.upper_bound_failure_prob = 0
        self.lower_bound_failure_prob = 0

    @abc.abstractmethod
    def delta_upper_bound(self, count, error):
        """Gives the (positive) delta for the upper bound"""
        pass

    @abc.abstractmethod
    def delta_lower_bound(self, count, error):
        """Gives the (negative) delta for the lower bound"""
        pass

    @abc.abstractmethod
    def prob_upper(self, error):
        """Probability that the upper bound delta fails"""
        pass

    @abc.abstractmethod
    def prob_lower(self, error):
        """Probability that the lower bound delta fails"""
        pass


class HoeffdingInequality(ConcentrationInequality):
    """
    HoeffdingInequality

    Uses Hoeffding inequality to get the bounds. Simple but not the tightest.
    """
    def __init__(self, dim=2):
        super().__init__()
        self.d = float(dim)

    def delta_upper_bound(self, count, error):
        self.upper_bound_counter += 1
        self.upper_bound_failure_prob += self.prob_upper(error)
        # return np.sqrt(.5*count*np.log(2/error))
        return .5*np.sqrt(count*(2*np.log(1./error) + self.d*np.log(count+1.)))

    def delta_lower_bound(self, count, error):
        self.lower_bound_counter += 1
        self.lower_bound_failure_prob += self.prob_lower(error)
        # return -1.0 * np.sqrt(.5*count*np.log(2/error))
        return -.5*np.sqrt(count*(2*np.log(1./error) + self.d*np.log(count+1.)))

    def prob_upper(self, error):
        return error

    def prob_lower(self, error):
        return error