
import numpy as np


def prune(CT, MDL):
    """
    After the acceptance of a new pattern in our code table other patterns may have become
    redundant if their role has been overtaken by the newer pattern. Therefore, each
    time a pattern X is successfully added to the code table, we consider removing those
    Y ∈ CT for which the usage decreased and hence the pattern code length increased

    First, we sort the patterns with decreased usage. Then, for each pattern, we check if
    the encoded length is smaller or greater without the pattern. If the encoded length is
    smaller when the pattern is absent, we remove the pattern from CT.
    """

    print('\tCT before pruning =', CT.patterns)

    # Order the CT in Pruning Order :
    # The patterns which usage decreased with the addition of the new pattern are selected
    # and ordered based with their usage increasing
    # If the usage has not decreased, it is set to 0 as it will be skipped further down below
    usage = np.where(CT.usage[:-1] != MDL.base_usage,
                     MDL.base_usage-CT.usage[:-1], 0)
    sort_indexes = np.argsort(usage)[::-1]
    sort_indexes = sort_indexes[:np.size(np.nonzero(usage))]

    print('\tPatterns which had their usage decreased',
          [CT[i] for i in sort_indexes])

    # print(f'Pruning order =, {sort_indexes}, i.e., {[CT[i] for i in sort_indexes]} with usage {[usage[i] for i in sort_indexes]}')

    # We don't re-compute the initial length everytime if the CT did not change
    CT.has_changed = False

    # Iterate over the patterns in 'Prune Order'
    for id_pattern in sort_indexes:
        # print('Pattern =', CT[id_pattern], 'id_pattern =', id_pattern, 'usage =', usage[id_pattern])

        # We force singletons to stay in the CT
        if len(CT[id_pattern]) > 2:

            # If the pattern has a usage of zero, we can't remove it but we don't compute the length
            # The length should be the same with or without it anyway
            if usage[id_pattern] == 0:
                continue

            # Encoded length computations
            usage, sort_indexes = MDL.prune(CT, CT[id_pattern],
                                            usage, sort_indexes)

    print('\tCT after pruning =', CT.patterns)

    return
