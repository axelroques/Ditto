
import matplotlib.pyplot as plt
import numpy as np


def show_cover(C, CT, id_pattern=0, letters=False):
    """
    Colorify the cover for visual interpretation of the patterns.
    """

    _, ax = plt.subplots(figsize=(15, 0.5*C.shape[0]))

    n_row = C.shape[0]
    n_col = C.shape[1]

    # To plot the legend
    i_label = True

    # CT target pattern id from sorted id
    target_pattern_id = CT.i_sort[id_pattern]

    # Left to right, top to bottom
    for i_row, rows in enumerate(C):
        for i_col, c in enumerate(rows):

            if letters:
                # Plot text
                ax.text(x=(i_col+0.5)/n_col, y=1-(i_row+0.5)/n_row, s=c[-1],
                        ha='center', va='center', transform=ax.transAxes)

            # Color interesting pattern
            if int(c.split('_')[0]) == target_pattern_id:
                ax.axhspan(-i_row/n_row, (-i_row-1)/n_row, i_col/n_col, (i_col+1)/n_col,
                           facecolor='royalblue', alpha=0.8,
                           edgecolor='silver', label=CT[CT.i_sort[id_pattern]] if i_label == True else '',
                           transform=ax.transAxes)
                i_label = False

            # Other patterns stay white
            else:
                ax.axhspan(-i_row/n_row, (-i_row-1)/n_row, i_col/n_col, (i_col+1)/n_col,
                           facecolor='none', alpha=0.8, edgecolor='silver', transform=ax.transAxes)

    # y axis parameters
    ticklabels = np.linspace(0, -1, n_row, endpoint=False) - 0.5/n_row
    ax.set_yticks(ticklabels)
    ax.set_yticklabels([f'S_{i}' for i in range(n_row)])
    ax.set_ylim((-1, 0))

    # x axis parameters
    x_ticks = np.linspace(0, 1, n_col+1) + 1/n_col/2
    ax.set_xticks(x_ticks[:-1])
    ax.set_xticklabels([])

    ax.grid(False)

    ax.legend(loc='upper center', bbox_to_anchor=(
        0.5, -0.05), fancybox=True, shadow=True, ncol=5)

    plt.show()

    return
