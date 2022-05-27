
from .pattern import Pattern

from itertools import product
import numpy as np


class Candidates:

    def __init__(self, ST, CT):

        # Get a new set of candidates with their estimated usage
        self._cartesian_product(ST, CT)

        # Retrieve estimated gain from the candidate objects
        self._get_estimated_gain()

        # Sort the candidates
        self._sort()

    def _cartesian_product(self, ST, CT):
        """
        Computes the cartesian product of an array of letters with itself.

        Returns a new set of candidates from the cartesian product of the CT with itself 
        and an array of combined usages for these new candidates, that will be useful for 
        the estimated gain computation.
        """

        # Generators
        candidates = (''.join([p.name for p in patterns])
                      for patterns in product(CT.patterns, repeat=2))
        usage = (i for i in product(CT.usage, repeat=2))

        # Generate Candidate objects
        self.candidates = [Candidate(ST, CT, p, u)
                           for p, u in zip(candidates, usage)]

        return

    def _get_estimated_gain(self):
        """
        Return an array with the estimated gain of the candidates.
        """

        self.estimated_gain = np.array([candidate.gain
                                        for candidate in self.candidates])

        return

    def _sort(self):
        """
        Sort the candidates array in Candidate Order, i.e. by decreasing estimated gain(X).

        This order preferes the most promising candidates in terms of compression gain.
        However, the compression gain is merely an estimation and may not reflect any
        actual gain.
        """

        # Decreasing estimated gain
        sort_indexes = np.argsort(self.estimated_gain)

        # Add a cutoff to remove candidates with inf gain
        cutoff = len(self.estimated_gain[self.estimated_gain < np.inf])

        # Sort candidates
        self.candidates = [self.candidates[i] for i in sort_indexes]
        self.candidates = self.candidates[:cutoff]

        return

    def __getitem__(self, item):
        return self.candidates[item]

    def __len__(self):
        return len(self.candidates)


class Candidate(Pattern):
    """
    Candidate class, inherited from the pattern class.

    A candidate is made out of two patterns, hence
    why self.usage is a tuple.
    """

    # We define the allowed attributes to gain memory space
    __slots__ = [
        'gain'
    ]

    def __init__(self, ST, CT, pattern, usage):

        # Pattern object inheritance
        Pattern.__init__(self, pattern)

        # Usage
        self.usage = usage

        # Estimated gain
        self.gain = self._gain(ST, CT)

    def _gain(self, ST, CT):
        """
        Compute the estimated gain for this candidate.

        Current candidate Z is constructed as Z = X U Y,
        where X and Y are two previous patterns in the CT
        associated during the cartesian product step.
        """

        # We arbitrarily cap the max number of symbols in the candidates
        if len(self.symbols) > 5:
            return np.inf

        # If either X or Y has a usage of 0, the estimated gain is inf
        if np.min(self.usage) == 0:
            return np.inf

        # Otherwise, we can compute the estimated gain of the pattern
        x, y, s = self.usage[0], self.usage[1], np.sum(CT.usage)

        # Case 1: x = y (includes X = Y or equal usage pattern)
        if x == y:
            z = x/2

            # ΔL(D|CT)
            deltaL_D_CT = x*np.log10(x/s) + y*np.log10(y/s) \
                - z*np.log10(z/(s-x+z))

            # ΔL(CT|D)
            deltaL_CT_D = np.log10(
                x/s) + np.log10(y/s) - np.log10(z/(s-x+z))

            for symbol in self.symbols:
                deltaL_CT_D += - \
                    np.log10(ST.usage_dict[symbol]/ST.usage_sum)

            # Total gain ΔL(D, CT)
            return deltaL_D_CT + deltaL_CT_D

        # Case 2: X != Y
        else:
            z = min(x, y)
            delta_xy = max(x, y) - z

            # ΔL(D|CT)
            deltaL_D_CT = x*np.log10(x/s) + y*np.log10(y/s) - delta_xy * \
                np.log10(delta_xy/(s-z)) - \
                z*np.log10(z/(s-z))

            # ΔL(CT|D)
            deltaL_CT_D = np.log10(x/s) + np.log10(y/s) - np.log10(
                delta_xy/(s-z)) - np.log10(z/(s-z))

            for symbol in self.symbols:
                deltaL_CT_D += - \
                    np.log10(ST.usage_dict[symbol]/ST.usage_sum)

            # Total gain ΔL(D, CT)
            return deltaL_D_CT + deltaL_CT_D
