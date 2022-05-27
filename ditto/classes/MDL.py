
import numpy as np


class MDL:
    """
    Minimum Description Length (MDL) class to compute the 
    total encoded length.
    """

    def __init__(self, ST, cover):

        # Basic parameters
        self.ST = ST
        self.cover = cover

        # Boolean that determines if the candidate is beneficial
        self.is_beneficial = False

    def compare(self, CT, candidate):
        """
        Compare the total encoded length with the new candidate
        compared to a previously computed length.

        If the CT has changed, first cover the data and compute
        the total encoded length for the new CT.

        Then, cover the data and compute the new encoded length
        for the CT with the new candidate.

        If this new candidate leads to a gain in compression, it
        remains in the CT. Otherwise, it is removed from the CT.
        """

        if CT.has_changed:

            # Cover the data
            self.base_C = self.cover.cover(CT)

            # Compute total encoded length
            self.base_L_D, self.base_L_CT = self._compute(
                self.ST, CT, self.base_C
            )
            self.base_length = self.base_L_D + self.base_L_CT

            # Store base pattern usage
            self.base_usage = CT.usage

        # Add the new candidate to the CT
        CT.add_candidate(candidate)

        # Cover the data
        self.C = self.cover.cover(CT)

        # Compute total encoded length
        self.L_D, self.L_CT = self._compute(
            self.ST, CT, self.C
        )
        self.length = self.L_D + self.L_CT

        # print('\tBase length =', self.base_length)
        # print('\tNew length =', self.length)

        # Check if this new candidate leads to a gain in compression
        if 0.99*self.base_length > self.length:
            self.is_beneficial = True
            CT.has_changed = True

        else:
            self.is_beneficial = False
            CT.remove_candidate(candidate)
            CT.has_changed = False

        return

    def prune(self, CT, candidate, usage, sort_indexes):
        """
        Iteratively assess if pruning patterns from the CT leads
        to a gain in compression.

        The idea is to select the patterns which had their usage 
        decrease after addition of the new pattern to the CT.

        Then, we cover the data with a CT pruned of this pattern 
        and compare the encoded length with the length when this 
        pattern was present in the CT.

        If removing that pattern leads to a gain in compression, it
        is definitely removed from the CT. Otherwise, it is kept.
        """

        if CT.has_changed:

            # Cover the data
            self.base_C = self.cover.cover(CT)

            # Compute total encoded length
            self.base_L_D, self.base_L_CT = self._compute(
                self.ST, CT, self.base_C
            )
            self.base_length = self.base_L_D + self.base_L_CT

            # Store base pattern usage
            self.base_usage = CT.usage

        # Remove this candidate from the CT
        CT.remove_candidate(candidate)

        # Cover the data
        self.C = self.cover.cover(CT)

        # Compute total encoded length
        self.L_D, self.L_CT = self._compute(
            self.ST, CT, self.C
        )
        self.length = self.L_D + self.L_CT

        # print('\tBase length =', self.base_length)
        # print('\tNew length =', self.length)

        # Check if this new candidate leads to a gain in compression
        if 0.99*self.base_length > self.length:

            print(f'Pattern {candidate} will be removed from the CT')
            CT.has_changed = True

            # Modify sort_indexes to take into considerations the changes in CT
            usage = usage[np.arange(len(usage)) != candidate.id]
            sort_indexes[sort_indexes > candidate.id] -= 1

        else:
            CT.add_candidate(candidate)
            CT.has_changed = False

        return usage, sort_indexes

    @staticmethod
    def _compute(ST, CT, C):
        """
        Compute total encoded length.
        """

        if "" in C:
            return np.inf, np.inf

        return MDL._L_D_CT(CT), MDL._L_CT_C(ST, CT)

    @staticmethod
    def _L_D_CT(CT):
        """
        Computes the encoded length of the data using the code 
        table CT.

        This encoded length is the sum of the encoded length of 
        the pattern stream and the gap stream.

        Both terms are computed similarly, using Shannon's entropy.
        """

        ########################################
        # Encoded length of the pattern stream #
        ########################################
        L_code = np.sum(
            CT.usage*(-np.log10(CT.usage/np.sum(CT.usage),
                                where=CT.usage != 0))
        )

        ####################################
        # Encoded length of the gap stream #
        ####################################
        # We should only compute the length of gaps and fills for
        # patterns of size > 1
        nonzero_indexes = np.nonzero(CT.gaps)

        # If no gaps, return
        if nonzero_indexes[0].size == 0:
            return 0

        nz_gaps = CT.gaps[nonzero_indexes]
        nz_fills = CT.fills[nonzero_indexes]

        # Weird but sometimes the gap value is < 0 which causes
        # issues with the log!
        nz_gaps = np.where(nz_gaps > 0, nz_gaps, 0)

        # Careful with log(0)!
        L_gap = np.sum(nz_gaps*(-np.log10(nz_gaps/(nz_gaps+nz_fills),
                                          where=nz_gaps > 0)))

        # print('\n\t\t ** L(D|CT) **')
        # print(f'\tPattern length = {L_code}')
        # print(f'\tGap length = {L_gap}')

        return L_code + L_gap

    @staticmethod
    def _L_CT_C(ST, CT):
        """
        Computes the encoded length of the CT. Encodes:
            - The pattern codes
            - The patterns themselves using the codes associated with the 
            singleton-only code table ST
        """

        #######################
        # Pattern code length #
        #######################
        L_code = np.sum((-np.log10(CT.usage/np.sum(CT.usage),
                                   where=CT.usage != 0)))

        ##################
        # Pattern length #
        ##################
        L_pattern = 0
        for pattern in CT.patterns:

            # If the pattern is used in the cover
            if pattern.usage != 0:

                for symbol in pattern.symbols:
                    L_pattern += -np.log10(ST.usage_dict[symbol]/ST.usage_sum)

        # print('\n\t\t ** L(CT|C) **')
        # print(f'\tPattern code length = {L_code}')
        # print(f'\tPattern length = {L_pattern}')

        return L_code + L_pattern
