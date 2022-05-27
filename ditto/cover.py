
import numpy as np


class Cover:

    def __init__(self, size, tree):

        # Basic parameters
        self.size = size
        self.tree = tree

    def cover(self, CT):
        """
        Cover the data using the patterns in the CT in Cover Order.

        Also computes 3 arrays:
            - usage = number of occurrences of the pattern in the cover.
            - gaps = number of gaps in all occurrences of the pattern.
            - fills = the number of fills (1 per pattern).

        Since the algorithm allows for gaps, we must first find the 
        occurrences of each symbol in the pattern. This is done using 
        the function _find_occurrences which returns a list of tuples.

        Then, all potential occurrences of the pattern can be found using 
        the cartesian product of the arrays containing each symbol's 
        position. However, we can severely reduce the computation cost by 
        restricting the positions in the cartesian product using the two 
        cover rules: 1) the position of the symbol should not overlap with 
        an element already in the cover C, and 2) there is a maximum gap 
        allowed that restricts the potential distance between two symbols. 
        The cartesian product is computed recursively both in _prune_occurrences 
        and in _recursive_prune_occurrences. 
        Using these rules, we can prune the possible occurrences of the pattern.

        The final step is to iterate over all of the available positions for 
        the pattern and cover the data.
        """

        # Initialize an empty cover
        self.C = np.zeros(self.size, dtype="<U6")

        # Sort the patterns
        CT.sort_cover_order()

        # Iterate over the sorted patterns
        for pattern in [CT[i] for i in CT.i_sort]:

            # Find all occurrences of the pattern in the database
            pattern_pos = self._find_occurrences(self.tree, pattern)

            # Prune occurrences using covering rules
            pruned_pos = self._prune_occurrences(
                self.C, pattern_pos, pattern)

            # Cover the data
            self._assign_occurrences(pruned_pos, pattern)

        # Update the CT parameters
        CT.update()

        return self.C

    @staticmethod
    def _find_occurrences(tree, pattern):
        """
        Find all occurences of a specific pattern in a sequence using 
        a suffix tree.

        Return a list of tuples like (id_seq, list(pos)) where pos are 
        the positions of the symbols in the sequence.

        E.g. 
            if the pattern is "a0a0" and "a0" occurs at 
            positions = [0, 1, 2, 4]
            the function will return
            pattern_pos = [
                (0, [0, 1, 2, 4]),
                (0, [0, 1, 2, 4])
            ]
        """

        # Retrieve the occurrences of each symbol in the pattern
        pattern_pos = []
        for symbol in pattern.symbols:

            # Which time series this symbol belongs to
            id_seq = int(symbol[-1])

            # Find occurences of the symbol
            _, positions = tree.find_motifs(symbol)

            # Store the results
            pattern_pos.append((id_seq, [pos // 2 for pos in positions]))

        return pattern_pos

    @staticmethod
    def _prune_occurrences(C, pattern_pos, pattern):
        """
        Finds all valid positions for a pattern based on the individual 
        positions of its symbols.
        Basically computes a reduced version of the cartesian product of 
        a list of arrays with an additional rule based on the maximum 
        distance between two positions (the maximal gap allowed in Bertens 
        et. al. (2016)). 
        The cartesian product is only computed if the symbol does not overlap
        with already covered data in C.

        Here we use a recursive function called recursive pruning to loop 
        over lists of size > 2. 
        _prune_occurrences cannot be used as a recursive function by itself 
        because of a mismatch in shape between the pattern_pos and 
        the returned pruned_pos.

        If we suppose that the cover C is empty, here are two examples.
        E.g. 1:
            if 
            pattern_pos = [
                (0, [0, 1, 2, 4]),
                (0, [0, 1, 2, 4])
            ]
            the function should return
            pruned_pos = [
                [(0, 0), (0, 1)],
                [(0, 0), (0, 2)], # gap between the 1st symbol and 2nd symbol
                [(0, 1), (0, 2)],
                [(0, 2), (0, 4)], # gap between the 1st symbol and 2nd symbol
            ]

        E.g. 2:
            if 
            pattern_pos = [
                (0, [0, 1, 2, 4]),
                (1, [0, 1, 2, 4])
            ]
            the function should return
            pruned_pos = [
                [(0, 0), (1, 0)],
                [(0, 0), (1, 1)],
                [(0, 1), (1, 1)],
                [(0, 1), (1, 2)],
                [(0, 2), (1, 2)]
            ]

        """

        # If there is only one symbol in the pattern, then we have nothing to do
        if len(pattern_pos) == 1:
            return [[(pattern_pos[0][0], pos)] for pos in pattern_pos[0][1]]

        # Otherwise we start to prune the positions
        else:
            pruned_pos = []

            # Iterate over the positions of the first symbol
            for off1 in pattern_pos[0][1]:
                id_off1 = pattern_pos[0][0]

                # Check if the position of the symbol does not overlap with elements in the cover
                if C[id_off1, off1] != '':
                    # If it overlaps, move on to the next position
                    continue

                # Iterate over the positions of the second symbol
                for off2 in pattern_pos[1][1]:
                    id_off2 = pattern_pos[1][0]

                    # Check if the position of the symbol does not overlap with elements in the cover
                    if C[id_off2, off2] != '':
                        # If it overlaps, move on to the next position
                        continue

                    # Assure directed pattern + Maximum gap criteria
                    if (off2-off1 >= 0) and ((off2-off1+1) < 2*pattern.t):

                        # If both symbol are in the same sequence, their position cannot match
                        if id_off1 == id_off2:
                            pruned_pos += [[(id_off1, off1),
                                            (id_off2, off2)]] if off1 != off2 else []
                        else:
                            pruned_pos += [[(id_off1, off1),
                                            (id_off2, off2)]]

        # If there are only two symbols or if we couldn't find any pattern, we stop here
        if (len(pattern_pos) == 2) or (pruned_pos == []):
            return pruned_pos

        # Otherwise we continue with a recursive function
        else:
            return Cover._recursive_prune_occurrences(C, pattern_pos[2:], pruned_pos, pattern)

    @staticmethod
    def _recursive_prune_occurrences(C, pattern_pos, pruned_pos, pattern):
        """
        Actual recursive function used by _prune_occurrences for lists of size > 2.
        See _prune_occurrences for details on the process. 
        """

        if pattern_pos == []:
            return pruned_pos

        else:
            # Iterate over the positions of the pruned list: the position of interest is
            # the position of the last symbol in the pattern

            new_pruned_pos = []

            for _, pruned in enumerate(pruned_pos):

                # Iterate over the second symbol positions
                for off in pattern_pos[0][1]:
                    id_off = pattern_pos[0][0]

                    # Check if the position of the symbol does not overlap with elements in the cover
                    if C[id_off, off] != '':
                        # If it overlaps, move on to the next position
                        continue

                    # Assure directed pattern + maximum gap criteria
                    try:
                        if (off-pruned[1][1] >= 0) and ((off-pruned[0][1]+1) < 2*pattern.t):

                            # If both symbol are in the same sequence, their positions cannot match
                            if id_off == pruned[1][0]:
                                new_pruned_pos += [pruned +
                                                   [(id_off, off)]] if off != pruned[1][1] else []
                            else:
                                new_pruned_pos += [pruned +
                                                   [(id_off, off)]]

                    except:
                        print('Something went wrong here')

            # Recursion
            return Cover._recursive_prune_occurrences(C, pattern_pos[1:], new_pruned_pos, pattern)

    def _assign_occurrences(self, pruned_pos, pattern):
        """
        Iterate over all available positions of the pattern and cover 
        the data (using the index of the pattern in the CT). 

        Iteratively updates the usage, gaps and fills arrays.

        E.g.
            For the first pattern in the CT with:
            pruned_pos = [
                [(0, 0), (0, 1)],
                [(0, 0), (0, 2)],
                [(0, 1), (0, 2)],
                [(0, 2), (0, 4)],
            ]
            Then, the cover for this part of the data is:
            C = ["0", "0", "0", "", "0"]
                |____|    |________|
                1st         2nd        occurrences
        """

        # Keep track of interesting values for the MDL principle
        counts, gap = 0, 0

        for positions in pruned_pos:

            # We can't directly append each position to C because one of the position could
            # overlap with the previous so we need to keep track of the symbol positions and
            # then, once we're sure that the whole pattern can be placed, add the pattern to
            # the cover
            ok_to_append = True
            to_append = []
            symbol_count = 0

            for i, position in positions:

                # If the cover is not empty, we should not cover the data with the pattern
                if self.C[i, position] != "":
                    ok_to_append = False
                    break

                else:
                    # Heren there is a case that is actually pretty common (and took me a while to figure
                    # out that it was a problem...). The idea is that for instance with the pattern a0a1a0
                    # with a layout like this : a0 a0
                    #                           a1
                    # the _recursive_prune_occurrences function, when given
                    # pruned_pos = [[(0, 0), (1, 0)]] has no problem giving a result like
                    # new_pruned_pos = [[(0, 0), (1, 0), (0, 0)]] when adding the last a0 since
                    # it only checks if the two last symbols belong to different time series
                    # This condition prevents just that
                    to_append_limited = []
                    for ii, jj, _ in to_append:
                        to_append_limited.append((ii, jj))
                    if (i, position) in to_append_limited:
                        ok_to_append = False
                        break

                    else:
                        to_append += [(i, position,
                                       pattern.name[symbol_count % (len(pattern))])]
                        symbol_count += 2

            # If we're clear to append
            if ok_to_append == True:
                for i, position, symbol in to_append:
                    self.C[i, position] = str(pattern.id) + '_' + symbol

                # Update interesting variables
                counts += 1
                if len(to_append) > 1:
                    gap += to_append[-1][1] - to_append[0][1] - pattern.t + 1

        # Update pattern parameters
        pattern.update(counts, gap, (pattern.t-1) * counts)

        return
