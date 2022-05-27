
from .pattern import Pattern
import numpy as np


class CT:
    """
    Code table (CT) generated from D and updated
    during the Ditto process.
    """

    def __init__(self, ST):

        # Patterns
        self.patterns = ST.patterns.copy()

        # Usage
        self.usage = ST.usage

        # Handy variable so that we do not re-compute cover the data
        # again if the CT did not change
        self.has_changed = True

        # Pattern id, useful when adding new candidates to the CT
        self.pattern_id = len(self.patterns)

    def sort_cover_order(self):
        """
        Sort the candidates in the CT in Cover Order.
        This order preferes larger and most frequent patterns for which we 
        expect a higher compression gain

        Note that we cannot reorder the CT directly. This is a consequence 
        of how the cover is constructed: each symbol is encoded by their 
        index in the CT. Because we do not want to re-order the CT and 
        update the whole cover with the new indexes everytime, we resort  
        to using the indexes of the pattern in the original unsorted CT.
        """

        # Creates a numpy array of tuples (-len(pattern), pattern)
        temp = np.array([(-len(p), p) for p in self.patterns],
                        dtype=np.dtype([('len', int),
                                        ('pattern', np.unicode_, 32)]))

        # Sorts the previous array by descending length first and
        # then lexicographically
        self.i_sort = np.argsort(temp, kind='stable', order=('len', 'pattern'))

        return

    def update(self):
        """
        Update pattern usage after the covering step.
        """

        self.usage = np.array([pattern.usage for pattern in self.patterns])
        self.gaps = np.array([pattern.gaps for pattern in self.patterns])
        self.fills = np.array([pattern.fills for pattern in self.patterns])

        return

    def add_candidate(self, candidate):
        """
        Add a new candidate to the CT.
        """

        candidate.id = self.pattern_id
        self.patterns.append(candidate)
        self.pattern_id += 1

        return

    def remove_candidate(self, candidate):
        """
        Remove a new candidate to the CT.
        """

        self.patterns.pop(candidate.id)
        self.pattern_id -= 1

        if candidate.id <= len(self.patterns):
            self._update_pattern_id()

        return

    def _update_pattern_id(self):
        """
        Update pattern id (after sorting).
        """

        for id, pattern in enumerate(self.patterns):
            pattern.id = id

        return

    def __getitem__(self, item):
        return self.patterns[item]
