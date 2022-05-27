from ditto.toolbox import convert_database, make_ST, t
from collections import defaultdict
from string import ascii_lowercase
import numpy as np


def generate_toy_data(l_seq, n_seq, n_patterns, pattern_size_range, gap_chance,
                      support, n_max_multimodality, alphabet_size_range):
    """
    We generate n_seq random sequences of symbolic data of length l_seq with an alphabet size 
    per sequence in alphabet_size_range. 
    After generation of the data, we plant n_patterns, where each pattern has a size in
    pattern_size_range, a gap chance between each symbol of gap_chance and a support
    such that each pattern spans support% of all events in the dataset.


    TODO GAP CHANCE
    """

    toy_data = []

    #######################
    ### Data generation ###
    #######################
    for seq in range(n_seq):
        alphabet = ascii_lowercase[:np.random.randint(alphabet_size_range[0],
                                                      alphabet_size_range[1]+1)]
        draw = np.random.choice([l for l in alphabet], size=l_seq)
        toy_data.append(''.join(l for l in draw))

    ################
    ### Patterns ###
    ################
    D = convert_database(toy_data)
    ST = make_ST(D)

    # Create n_patterns patterns
    i_pattern = 0
    patterns = []
    while i_pattern < n_patterns:
        pattern = np.random.choice(ST, size=np.random.randint(pattern_size_range[0],
                                                              pattern_size_range[1]+1))
        # Only accept pattern if two conditions are met:
        # - If the pattern doesn't already exist
        # - If the n_max_multimodality value is respected
        if ''.join(l for l in pattern) not in patterns:
            counts = defaultdict(int)
            for symbol in pattern:
                counts[symbol[-1]] += 1
            if len(counts.keys()) < n_max_multimodality:
                patterns.append(''.join(l for l in pattern))
                i_pattern += 1

    # Plant patterns in the data
    pattern_positions = np.empty_like(D)
    for id_pattern, pattern in enumerate(patterns):

        symbols = [pattern[i:i+2] for i in range(0, len(pattern), 2)]
        t_pattern = t(symbols)

        pattern_support = 0
        while pattern_support < support:

            # Random position in D = start of the pattern
            pos = np.random.randint(l_seq-t_pattern)

            # For each symbol in the pattern, check if its position does not overlap
            # with a previous pattern
            ok_to_append = True
            to_append = []
            for relative_pos, symbol in enumerate(symbols):
                if pattern_positions[int(symbols[relative_pos][-1]), pos+relative_pos] != "":
                    ok_to_append = False
                    break
                else:
                    to_append.append(
                        (int(symbols[relative_pos][-1]), pos+relative_pos))

            if ok_to_append:
                for (row, col), symbol in zip(to_append, symbols):
                    pattern_positions[row, col] = str(id_pattern)
                    D[row, col] = symbol

                pattern_support += len(symbols)/l_seq/n_seq

    return toy_data, D, patterns, pattern_positions
