
from .prune import prune

import numpy as np


def variations(MDL, CT, pattern, candidates=[], candidates_usage=[]):
    """
    Recursively test whether to add variations of the accepted pattern in the CT.

    That is, when a pattern leads to a gain in compression, we consider all ways by which
    we can extend it using events that occur in the gaps of its usages. This way we
    consider a rich set of candidates, plus speed up the search as we are automatically
    directed to patterns that actually exist in the data.

    For example, consider the dataset {a, b, a, b, c, a, c, a} where pattern {a, a} occurs
    twice with a gap of length one. After adding pattern {a, a} to CT we consider the
    patterns {a, b, a} and {a, c, a} for addition to CT.
    """

    print(f'\tVariations on pattern = {pattern}')

    # If the length of the pattern is 1, we can't compute variations
    if pattern.t == 1:
        print('No variations: pattern size is 1.')
        return

    # If the length of the pattern is >= 5 we stop creating variants (UNJUSTIFIED CAP FOR NOW)
    if len(pattern.symbols) >= 5:
        print('Pattern has >= 5 symbols, we do not allow more variations.')
        return

    #################
    # Find variants #
    #################

    else:

        # Find the occurrences in the cover of all symbols in the pattern
        occurrences = np.where(np.char.startswith(
            MDL.C, str(pattern.id), start=0, end=1)
        )

        # print('Occurrences position OG =', occurrences)

        # Reorder the positions (right now it's ordered row by row)
        array1, array2 = occurrences[0], occurrences[1]
        sort_index = np.argsort(array2)
        array1 = array1[sort_index]
        array2 = array2[sort_index]

        # Regroup occurrences together so that each group contains the position of all of the
        # symbols of a single pattern
        occurrences = [(array1[i:i+len(pattern.symbols)], array2[i:i+len(pattern.symbols)])
                       for i in range(0, len(occurrences[0]), len(pattern.symbols))]
        # print('\nRegrouped occurrences', occurrences)

        # For each pattern in the cover
        for occurrence in occurrences:
            # print('\nPositions of the symbols in the cover =', occurrence)

            # Find the limits of the positions of the symbols
            min_row = np.min(occurrence[0])
            max_row = np.max(occurrence[0])
            min_col = np.min(occurrence[1])
            max_col = np.max(occurrence[1])
            # print(f'Start = {min_col}, End = {max_col}')

            # Check if the pattern contains a gap
            # There are two cases :
            #   - If the symbols of the pattern are all on the same line, the formula is easy
            #   - If the symbols are not on the same line, it's more complicated
            gap = True
            # First case
            if max_row-min_row == 0:
                if (max_col-min_col+1) == pattern.t:
                    gap = False
                    # print('No Gap')
                    continue
            # Second case
            else:
                if np.all(np.unique(occurrence[1]) == np.arange(min_col, max_col+1)):
                    gap = False
                    # print('No Gap')
                elif (max_col-min_col+1) == pattern.t:
                    # Not sure if this condition is truly useful, it might be included in the former
                    gap = False
                    # print('No Gap')
                else:
                    pass
                    # print('Gap')

            # If there is a gap, create a variation, otherwise we move on
            if gap:
                # Define the subset of C that contains the pattern
                C_cut = MDL.C[min_row:max_row+1, min_col:max_col+1]
                # print('C_cut =\n', C_cut)

                # Gap symbols are contained in the columns where there are no pattern symbol
                row_diff = [
                    j-i for i, j in zip(occurrence[1][:-1], occurrence[1][1:])]
                gap_symbols = []
                c_cut_col_count = 0
                occurrence_col_count = 0
                for i_diff in row_diff:
                    if i_diff == 1:
                        c_cut_col_count += 1
                    if i_diff == 2:
                        c_cut_id = c_cut_col_count + 1
                        gap_symbols += [(C_cut[:, c_cut_id][symb], occurrence_col_count+1, row)
                                        for row, symb in enumerate(range(len(C_cut[:, c_cut_id])))]
                        c_cut_col_count = 0
                    occurrence_col_count += 1
                # print('Gap symbols =', gap_symbols)

                # Here, if two patterns are intertwined, this could result in problems.
                # For instance if a pattern ends after the start of another, then their
                # occurrences will be regrouped differently leading to one of the pattern
                # being badly cut in C_cut
                # We solve this issue rather lazily: if there is an issue, we just move on
                if len(gap_symbols) == 0:
                    print('Two successive patterns are intertwined, we move on.')
                    continue

                # Gap symbols are saved in gap_symbols as a list of sets
                # (gap_symbol, column in occurrence[1], row in c_cut)
                # We must create a new variant with each gap symbol
                for symbol in gap_symbols:
                    # Variant is start of the pattern + gap symbol + end of pattern
                    variant = pattern[:2*len(occurrence[1][:symbol[1]])] +\
                        symbol[0][-1] + str(np.min(occurrence[0])+symbol[2]) +\
                        pattern[2*len(occurrence[1][:symbol[1]]):]
                    # print('Original pattern =', pattern,
                    #       '; Variant created =', variant)

                    # Check if the variant already exists in the candidate array
                    if variant in candidates:
                        # We update the usage of this variant
                        candidates_usage[candidates == variant] += 1
                    else:
                        # We add it to the array
                        # print("Variant added!")
                        candidates.append(variant)
                        candidates_usage.append(1)

            else:
                continue

    #####################################

    # Now we have a (new or updated) set of candidate variants
    # print('\nSet of candidate variants =', candidates)
    # print('with the following usages =', candidates_usage)

    if candidates == []:
        print('No variants of this pattern in the data.')
        return

    ########################
    # Order the candidates #
    ########################

    i_order = np.argsort(candidates_usage)[::-1]
    candidates = [candidates[i] for i in i_order]
    candidates_usage = [candidates_usage[i] for i in i_order]
    # print('After ordering, variants =', candidates)
    # print('with usages =', candidates_usage)

    ##################
    # Encoded length #
    ##################

    # We don't re-compute the initial length everytime if the CT did not change
    CT.has_changed = True

    # Iterate over each variant
    for i_variant, variant in enumerate(candidates):

        # MDL computations
        MDL.compare(CT, variant)

        # If this variant decreases the encoded lengt we add it to the CT
        if MDL.is_beneficial:

            print(f'{variant} is a good variant, it will be added to the CT.')

            # Prune CT
            prune(CT, MDL)

            # Recursively use the variation algorithm
            variations(MDL, CT, variant, candidates[i_variant+1:],
                       candidates_usage[i_variant+1:])

    return
