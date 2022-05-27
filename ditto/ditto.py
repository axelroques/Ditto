
from .analysis.analysis import show_cover
from .candidates import Candidates
from .variations import variations
from .cover import Cover
from .prune import prune
from .tree import Tree
from .MDL import MDL
from .ST import ST
from .CT import CT

import pandas as pd
import numpy as np


class Ditto:

    def __init__(self, D):
        """
        D = database to process.
        """

        # Store D as a numpy array
        self.D = self._get_D(D)
        self.size = self.D.shape

        # Create a generalized suffix tree from D
        self.tree = self._get_tree(self.D)

        # Initiate an empty cover
        self.cover = Cover(self.size, self.tree)

        # Initiate the singleton code table (ST)
        self.ST = ST(self.D, self.cover)

        # Initiate the code table (CT)
        self.CT = CT(self.ST)

        # Initiate a MDL object for the encoded length computations
        self.MDL = MDL(self.ST, self.cover)

    def process(self):
        """
        Run the Ditto algorithm, inspired by Bertens et. al. (2016).
        """

        # Generate a set of candidate patterns of all pairwise combinations of
        # singletons and sort the candidates in 'Candidate Order'
        print('\n*** Generating and sorting candidates ***')
        candidates = Candidates(self.ST, self.CT)

        # Loop over all candidates
        print('*** Iterating through potential candidates ***')
        i_candidate = 0
        while i_candidate < len(candidates):

            # Current candidate
            candidate = candidates[i_candidate]

            print('\tCandidate being investigated =', candidate)

            # MDL computations
            self.MDL.compare(self.CT, candidate)

            # If adding this new candidate decreases the total encoded length, we keep it
            if self.MDL.is_beneficial:

                print(f"-> Retained candidate = '{candidate}'")

                # Prune CT
                print('*** Pruning ***')
                prune(self.CT, self.MDL)

                # Test whether to add variations
                print('*** Variations ***')
                variations(self.MDL, self.CT, candidate,
                           candidates=[], candidates_usage=[])

                # Update the candidate set
                print('\n*** Generating and sorting new set of candidates ***')
                candidates = Candidates(self.ST, self.CT)
                # print('Candidate set after sorting =', candidates, '\n')

                print('*** Iterating through potential candidates ***')
                self.CT.has_changed = True
                i_candidate = 0

            else:
                i_candidate += 1

        print('\nNo more gain in the compression with the current set of candidates.')

        return

    def get_results(self):
        """
        Get cover with the final CT.
        Also return a DataFrame with the patterns in the CT 
        in Cover Order.
        """

        # Cover
        self.cover.cover(self.CT)

        # Cover Order
        patterns = [self.CT.patterns[i] for i in self.CT.i_sort]

        data = {
            'Pattern': [pattern.name for pattern in patterns],
            'Usage': [pattern.usage for pattern in patterns],
            'Gaps': [pattern.gaps for pattern in patterns]
        }

        return self.cover.C, pd.DataFrame(data=data)

    def get_cover(self, CT):
        """
        Cover the data with the patterns in a CT.
        """

        self.cover.cover(CT)

        return self.cover.C

    def show_cover(self, id_pattern, letters=False):
        """
        Display the cover and color the pattern specified
        by id_pattern.
        """

        show_cover(self.get_cover(self.CT), self.CT, id_pattern, letters)

        return

    @staticmethod
    def _get_D(D):
        """
        Type checking for input database D.

        Also encodes each symbol as 'symbol_i' where 'i'
        is the index of the time series in D.

        Accepts either:
            - a list of strings.
            - a numpy array of shape (N x n) with N the number 
            of time series and n the length of each time series.
            - a pandas DataFrame. In that case, all columns except
            a potential column 't' are kept.

        Always returns a numpy array of shape (N x n).
        """

        # List
        if isinstance(D, list):
            return np.array([[symbol+str(i) for symbol in l]
                             for i, l in enumerate(D)])

        # Numpy array
        elif isinstance(D, np.ndarray):
            assert D.shape[0] < D.shape[1], 'Expected D of shape (N x n).'
            return np.array([[symbol+str(i) for symbol in l]
                             for i, l in enumerate(D)])

        # Pandas DataFrame
        elif isinstance(D, pd.core.frame.DataFrame):
            return np.array([[symbol+str(i) for i, symbol in enumerate(symbols)]
                             for key, symbols in D.iteritems() if key != 't'])

        else:
            raise RuntimeError('Invalid input type for D.')
            return

    @staticmethod
    def _get_tree(D):
        """
        Constructs a generalized suffix tree from D.

        First converts each row of an array of characters into a string,
        i.e. ['a', 'b', 'c'] --> 'abc'.
        """

        return Tree({f'S{i}': ''.join(symbol for symbol in time_series)
                     for i, time_series in enumerate(D)})
