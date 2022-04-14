""""
Helper functions for plotting.
"""

import numpy as np
from matplotlib import cm


def colours(num, cmap):
    """
    Get a number of sequential colours in rgb format from given colourmap
    :param num:  number of colours to sample
    :param cmap: colourmap from matplotlib
    :return: List of colours in rgb format
    """
    import matplotlib
    cmap = cm.get_cmap(cmap)
    cols = [cmap(i/float(num)) for i in range(num)]
    return cols


def adjust_ylabels(ax, x_offset=0):
    """
    Scan all ax list and identify the outmost y-axis position.
    Setting all the labels to that position + x_offset.
    """

    xc = 0.0
    for a in ax:
        xc = min(xc, (a.yaxis.get_label()).get_position()[0])

    for a in ax:
        a.yaxis.set_label_coords(xc + x_offset,
                                 (a.yaxis.get_label()).get_position()[1])


def smooth(x, window_len=20):
    """Smooth signal with a rectangular window of given length."""
    w = np.ones(window_len, 'd')/window_len
    return np.convolve(w, x, mode='same')


def plot_violin(ax, pos, data, color=None, showmeans=True):
    """
    Makes violin of data at x position pos in axis object ax.
    - data is an array of values
    - pos is a scalar
    - ax is an axis object

    Kwargs: color (if None default mpl is used) and whether to plot the mean
    """

    parts = ax.violinplot(data, positions=[pos], showmeans=showmeans, widths=0.6)
    if color:
        for pc in parts['bodies']:
            pc.set_color(color)
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
            vp = parts[partname]
            vp.set_edgecolor(color)
            vp.set_linewidth(1)


def get_plane(x, y, z, t1=577, t2=855, t3=711):
    """
    Compute the 3D mesh of a plane based on 3 points specified by time points of a 3d time series.
    - x: x values of time series
    - y: y values
    - z: z values
    - t1, t2, t3: time different time points of the series (can be random points)

    Returns x-y meshgrid and corresponding y values of the plane
    """

    # concatenate data
    xyz = np.array([x, y, z]).T

    # choose 3 points
    p1 = xyz[t1]
    p2 = xyz[t2]
    p3 = xyz[t3]

    # compute normal vector and a point orthogonal to it
    normal = np.cross((p2 - p1), (p3 - p1))
    d = -p1.dot(normal)

    # compute x-y meshgrid and corresponding z values
    xx, yy = np.meshgrid(np.arange(-1.2, 1.21, 0.1), np.arange(-1.2, 1.21, 0.1))
    zz = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]

    return xx, yy, zz
