
from collections import defaultdict
import numpy as np

class Pattern():
    """
    Pattern class.
    """

    # We define the allowed attributes to gain memory space 
    __slots__ = [
        'id', 'name', 'symbols', 't', 'usage', 'gaps', 'fills'
    ]

    def __init__(self, pattern, id=None):
        
        # Basic parameters
        self.id = id
        self.name = pattern

        # Symbols
        self.symbols = [pattern[i:i+2] 
                        for i in range(0, len(pattern), 2)]

        # Duration
        self.t = self._t(self.symbols)
    
    def update(self, counts, gaps, fills):
        """
        Update pattern usage, gaps and fills after the covering step.
        """

        self.usage = counts
        self.gaps = gaps
        self.fills = fills

        return

    @staticmethod
    def _t(symbols):
        """
        Computes the length of a pattern. The length is defined as the number of timesteps.
        Therefore, we can't just do len(pattern), we need to count the number of symbols on each 
        sequence and remember the max. We use a defaultdict to do just that
        """
        lengths = defaultdict(int)
        for symbol in symbols:
            lengths[int(symbol[-1])] += 1
        return np.amax([lengths[key] for key in lengths.keys()])

    def __repr__(self) -> str:
        return self.name

    def __len__(self):
        return len(self.name)

        