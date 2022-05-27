
from .pattern import Pattern

import numpy as np


class ST():
    """
    Singleton code table (ST) generated from D.
    """

    def __init__(self, D, cover):

        # Find all unique singletons
        self.singletons_list = np.concatenate(
            [np.unique(time_series)
             for time_series in D]
        )

        # Store singletons objects
        self.patterns = [Pattern(p, id)
                         for id, p in enumerate(self.singletons_list)]

        # Get singleton coverage
        _ = cover.cover(self)

    def sort_cover_order(self):
        """
        Mock function for compatibility with Cover object.
        """

        self.i_sort = np.arange(len(self.patterns))

        return

    def update(self):
        """
        Update singleton usage after the covering step.
        """

        self.usage = np.array([pattern.usage for pattern in self.patterns])
        self.usage_dict = {
            s: u for s, u in zip(self.singletons_list, self.usage)
        }
        self.usage_sum = np.sum(self.usage)
        self.gaps = np.array([pattern.gaps for pattern in self.patterns])
        self.fills = np.array([pattern.fills for pattern in self.patterns])

        return

    def __getitem__(self, item):
        return self.patterns[item]
