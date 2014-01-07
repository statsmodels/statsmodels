import numpy as np
import utils

try:
    import matplotlib.transforms as transforms
    if matplotlib.__version__ < '1':
        raise
    have_matplotlib = True
except:
    have_matplotlib = False


def dotplot(points, intervals=None, lines=None, sections=None,
            styles=None, marker_props=None, line_props=None,
            ax=None, split_names=None, order_sections=None,
            order_lines=None, stacked=False, striped=False):
    """
    Produce a dotplot similar in style to those in Cleveland's
    "Visualizing Data" book.  Several extensions to the basic dotplot
    are also implemented: intervals can be plotted along with the
    dots, and multiple points and intervals can be drawn on each line
    of the dotplot.

    Parameters
    ----------
    points : array_like
        The quantitative values to be plotted as markers.

    intervals : array_like

        The intervals to be plotted around the points.  The elements
        of intervals are either scalars or sequences of length 2.  A
        scalar indicates the half width of a symmetric interval.  A
        sequence of length 2 contains the left and right half-widths
        (respectively) of a nonsymmetric interval.  If None, not
        intervals are drawn.

    lines : array_like

        A grouping variable indicating which points/intervals are
        drawn on a common line.  If None, each point/interval appears
        on its own line.

    sections : array_like

        A grouping variable indicating which lines are grouped into
        sections.  If None, everything is drawn in a single section.

    styles : array_like

        A label defining the plotting style of the marker.

    marker_props : dict

        A dictionary mapping style codes (the values in `styles`) to
        dictionaries defining key/value pairs to be passed as keyword
        arguments to `plot` when plotting markers.  Useful keyword
        arguments are "color", "marker", and "ms" (marker size).

    line_props : dict

        A dictionary mapping style codes (the values in `styles`) to
        dictionaries defining key/value pairs to be passed as keyword
        arguments to `plot` when plotting interval lines.  Useful
        keyword arguments are "color", "linestyle", "solid_capstyle",
        and "linewidth".

    ax : matplotlib.axes
        The axes on which the dotplot is drawn.  If None, a new axes
        is created.

    split_names : string
        If not None, this is used to split the values of groupby into
        substrings that are drawn in the left and right margins,
        respectively.

    order_sections : array_like
        The section labels in the order in which they appear in the
        dotplot; can also be used to select a subset of the section
        names.

    order_lines : array_like
        The groupby labels in the order in which they appear in the
        dotplot; can also be used to select a subset of the groupby
        labels.

    stacked : boolean
        If True, when multiple points or intervals are drawn on the
        same line, they are offset vertically from each other.
    striped : boolean
        If True, every other line is enclosed in a shaded box.

    Returns
    -------
    fig : Figure
        The figure given by `ax.figure` or a new instance.

    Notes
    -----
    `points`, `intervals`, `lines`, `sections`, `styles` must all have
    the same length whenever present.

    References
    ----------
      * Cleveland, William S. (1993). "Visualizing Data". Hobart
        Press.
      * Jacoby, William G. (2006) "The Dot Plot: A Graphical Display
        for Labeled Quantitative Values." The Political Methodologist
        14(1): 6-14.
    """

    fig, ax = utils.create_mpl_ax(ax)

    npoint = len(points)

    if lines is None:
        lines = np.arange(npoint)

    if sections is None:
        sections = np.zeros(npoint)

    if styles is None:
        styles = np.zeros(npoint)

    # The vertical space (in inches) for a section title
    section_title_space = 0.5

    # The number of sections
    nsect = len(set(sections))

    # The number of section titles
    nsect_title = nsect if nsect > 1 else 0

    # The total vertical space devoted to section titles.
    section_space_total = section_title_space * nsect_title

    # Add a bit of horizontal room so that points that fall at the
    # axis limits are not cut in half.
    ax.set_xmargin(0.02)

    if order_sections is None:
        lines0 = list(set(sections))
        lines0.sort()
    else:
        lines0 = order_sections

    if order_lines is None:
        lines1 = list(set(lines))
        lines1.sort()
    else:
        lines1 = order_lines

    # A map from (section,line) codes to index positions.
    lines_map = {}
    for i in range(npoint):
        ky = (sections[i], lines[i])
        if ky not in lines_map:
            lines_map[ky] = []
        lines_map[ky].append(i)

    # Get the size of the axes on the parent figure in inches
    bbox = ax.get_window_extent().transformed(
        fig.dpi_scale_trans.inverted())
    awidth, aheight = bbox.width, bbox.height

    # The number of lines in the plot.
    nrows = len(lines_map)

    # The position of the highest guideline in axes coordinates.
    top = 1

    # The position of the lowest guideline in axes coordinates.
    bottom = 0

    # x coordinate is data, y coordinate is axes
    trans = transforms.blended_transform_factory(ax.transData,
                                                 ax.transAxes)

    # Space used for a section title, in axes coordinates
    title_space_axes = section_title_space / aheight

    # Space between lines
    dy = (top - bottom - nsect_title*title_space_axes) /\
        float(nrows)

    # Determine the spacing for stacked points
    # The maximum number of points on one line.
    style_codes = list(set(styles))
    style_codes.sort()
    nval = len(style_codes)
    if nval > 1:
        stackd = dy / (2.5*(float(nval)-1))
    else:
        stackd = 0.

    # Map from style code to its integer position
    style_codes_map = {x: style_codes.index(x) for x in style_codes}

    # Setup default marker styles
    colors = ["r", "g", "b", "y", "k", "purple", "orange"]
    if marker_props is None:
        marker_props = {x: {} for x in style_codes}
    for j in range(nval):
        sc = style_codes[j]
        if "color" not in marker_props[sc]:
            marker_props[sc]["color"] = colors[j]
        if "marker" not in marker_props[sc]:
            marker_props[sc]["marker"] = "o"
        if "ms" not in marker_props[sc]:
            marker_props[sc]["ms"] = 10 if stackd == 0 else 6

    # Setup default line styles
    if line_props is None:
        line_props = {x: {} for x in style_codes}
    for j in range(nval):
        sc = style_codes[j]
        if "color" not in line_props[sc]:
            line_props[sc]["color"] = "grey"
        if "linewidth" not in line_props[sc]:
            line_props[sc]["linewidth"] = 2 if stackd > 0 else 8

    # The vertical position of the first line.
    y = top - dy/2 if nsect == 1 else top

    # Points that have already been labeled
    labeled = set()

    # Positions of the y axis grid lines
    yticks = []

    # Loop through the sections
    for k0 in lines0:

        # Draw a section title
        if nsect_title > 0:

            y0 = y + dy/2 if k0 == lines0[0] else y

            ax.fill_between((0, 1), (y0,y0),
                            (y-0.7*title_space_axes, y-0.7*title_space_axes),
                            color='darkgrey',
                            transform=ax.transAxes,
                            zorder=1)

            txt = ax.text(0.5, y - 0.35*title_space_axes, k0,
                          horizontalalignment='center',
                          verticalalignment='center',
                          transform=ax.transAxes)
            txt.set_fontweight("bold")
            y -= title_space_axes

        jrow = 0
        for k1 in lines1:

            # No data to plot
            if (k0, k1) not in lines_map:
                continue

            # Draw the guideline
            ax.axhline(y, color='grey')

            # Set up the labels
            if split_names is not None:
                us = k1.split(split_names)
                if len(us) >= 2:
                    left_label, right_label = us[0], us[1]
                else:
                    left_label, right_label = k1, None
            else:
                left_label, right_label = k1, None

            # Draw the stripe
            if striped and jrow % 2 == 0:
                ax.fill_between((0, 1), (y-dy/2, y-dy/2),
                                (y+dy/2, y+dy/2),
                                color='lightgrey',
                                transform=ax.transAxes,
                                zorder=0)
            jrow += 1

            # Draw the left margin label
            txt = ax.text(-0.01, y, left_label,
                           horizontalalignment="right",
                           verticalalignment='center',
                           transform=ax.transAxes, family='monospace')

            # Draw the right margin label
            if right_label is not None:
                txt = ax.text(1.01, y, right_label,
                              horizontalalignment="left",
                              verticalalignment='center',
                              transform=ax.transAxes,
                              family='monospace')

            # Save the vertical position so that we can place the
            # tick marks
            yticks.append(y)

            # Loop over the points in one line
            for ji,jp in enumerate(lines_map[(k0,k1)]):

                # Calculate the vertical offset
                yo = 0
                if stacked:
                    yo = -dy/5 + style_codes_map[styles[jp]]*stackd

                pt = points[jp]

                # Plot the interval
                if intervals is not None:

                    # Symmetric interval
                    if np.isscalar(intervals[jp]):
                        lcb, ucb = pt - intervals[jp],\
                            pt + intervals[jp]

                    # Nonsymmetric interval
                    else:
                        lcb, ucb = pt - intervals[jp][0],\
                            pt + intervals[jp][1]

                    # Draw the interval
                    ax.plot([lcb, ucb], [y+yo, y+yo], '-',
                            transform=trans, **line_props[styles[jp]])

                # Plot the point
                sl = styles[jp]
                sll = sl if sl not in labeled else None
                labeled.add(sl)
                ax.plot([pt,], [y+yo,], ls='None', transform=trans,
                        label=sll, **marker_props[sl])

            y -= dy

    # Set up the axis
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("none")
    ax.set_yticklabels([])
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_position(('axes', -0.01))

    ax.set_ylim(0, 1)

    ax.yaxis.set_ticks(yticks)

    ax.autoscale_view(scaley=False, tight=True)

    return fig
