import numpy as np
#import matplotlib.pyplot as plt
import matplotlib.transforms as transforms


def dotplot(vals, ax, left_labels=None, right_labels=None,
            stack=False, stripe=False, props={}):
    """
    Produce a dotplot similar in style to those in Cleveland's
    "Visualizing Data" book.  Several extensions to the basic dotplot
    are also implemented: intervals can be plotted along with the
    dots, and multiple points and intervals can be drawn on each line
    of the dotplot.

    Parameters
    ----------
    vals : array_like
        A list containing the data values to be plotted as points, and
        optionally, information about intervals that are to be plotted
        around the points.  Each element `v` of `vals` contains the
        data to be plotted on a single row of the dotplot.  The
        elements of `v` are either (1) scalars, if only points are to
        be plotted, (2) sequences (x, w) if the interval x-w, x+w is
        to be plotted, or (3) sequences (x, left, right) if the
        interval x-left, x+right is to be plotted.
    ax : matplotlib.axes
        The axes on which the dotplot is drawn.
    left_labels : array_like
        A list of labels to be placed in the left margin.  If None,
        no labels are drawn in the left margin.
    right_labels : array_like
        A list of labels to be placed in the right margin.  If None,
        no labels are drawn in the right margin.
    stack : boolean
        If True, when multiple points or intervals are drawn on the
        same line, they are offset vertically from each other.
    stripe : boolean
        If True, every other line is enclosed in a shaded box.
    props : dictionary
        A dictionary containing parameters that affect the layout of
        the dotplot.  See notes below for information about the
        parameters in `props`.

    Returns
    -------
    rslt : dictionary
        A dictionary of matplotlib elements in the plot.  The `points`
        and `intervals` are lists of lists containing the points and
        intervals that are drawn on each line of the dotplot.

    Notes
    -----
    The general format of `vals` is

    [
      # First line in the plot
      [
        point1, point2, ...
      ]
      ...
      # Last line in the plot
      [
        point1, point2, ...
      ]
    ]

    The elements of `vals` correspond to the lines of the plot.  Each
    element of `vals` contains a sequence of points to be plotted on a
    single line.  The points are represented as sequences of 1, 2, or
    3 numeric values, specifying a point, a symmetric interval, or a
    nonmsymmetric interval.  An element of `vals` that contains only a
    single string defines a section title.

    There are two special cases:

      * If `vals` is a sequence of scalars, these are taken as points
        to be plotted without intervals, with one point on each line.

      * If `vals` is a sequence of sequences of length 2 or 3, these
        are taken to be symmetric or nonsymmetric intervals to be
        plotted, with one interval on each line.

    Plotting properties that may be placed into `props`:

      * top: the distance in inches between the top of the axes and
        the top guide line, defaults to 0.5in.

      * bottom : the distance in inches between the bottom of the axes
        and the bottom guide line, defaults to 0.75in.

      * axis_pos : the distance in inches between the bottom of the
        axes and the axis, defaults to 0.5in.

      * section_space : The space between sections, in inches.

    References
    ----------
      * Cleveland, William S. (1993). "Visualizing Data". Hobart
        Press.
      * Jacoby, William G. (2006) "The Dot Plot: A Graphical Display
        for Labeled Quantitative Values." The Political Methodologist
        14(1): 6-14.
    """

    default_props = {"top": 0.5,
                     "bottom": 0.75,
                     "axis_pos": 0.5,
                     "section_space": 0.5}

    # Fill in any properties not specied by the user with the default.
    for k in default_props:
        if k not in props:
            props[k] = default_props[k]

    rslt = {}

    # The number of sections
    nsect = 0
    for v in vals:
        if np.isscalar(v) and isinstance(v, str):
            nsect += 1

    # The total vertical space devoted to section titles.
    section_space_total = props["section_space"] * nsect

    # Add a bit of horizontal room so that points that fall at the
    # axis limits are not cut in half.
    ax.set_xmargin(0.02)

    # Get the size of these axes on the parent figure in inches
    fig = ax.figure
    bbox = ax.get_window_extent().transformed(
        fig.dpi_scale_trans.inverted())
    awidth, aheight = bbox.width, bbox.height

    # The default colors.  To change the colors, use the `set_color`
    # method of the objects in points.
    colors = ['r', 'b', 'g', 'm', 'k']

    # The number of lines in the plot (not including section title
    # lines).
    nrows = len(vals) - nsect

    # The position of the highest guideline in axes coordinates.
    top = (aheight - props["top"]) / float(aheight)

    # The position of the lowest guideline in axes coordinates.
    bottom = props["bottom"] / float(aheight)

    # x coordinate is data, y coordinate is axes
    trans = transforms.blended_transform_factory(ax.transData,
                                                 ax.transAxes)

    # Space for a section title, in axes coordinates
    title_space_axes = props["section_space"] / aheight

    # Space between lines
    dy = (top - bottom - nsect*title_space_axes) / float(nrows - 1)

    # Determine the spacing for stacked points
    def len1(x):
        if np.isscalar(x):
            return 1
        else:
            return len(x)
    # The maximum number of points on one line.
    nval = max([len1(val) for val in vals])
    if nval > 1:
        stackd = dy / (2.5*(float(nval)-1))
    else:
        stackd = 0.

    # Extend the color list (by recycling) if needed.
    while len(colors) < nval:
        colors.extend(colors)

    points, intervals = [], []

    rslt["guidelines"] = []
    rslt["section titles"] = []
    rslt["left labels"] = []
    rslt["right labels"] = []

    # The vertical position of the current line.
    y = top

    # Loop through the lines (rows of data in the plot)
    for j,val in enumerate(vals):

        # Draw a section title
        if np.isscalar(val) and isinstance(val, str):
            y -= title_space_axes / 2
            txt = ax.text(0.5, y, val, horizontalalignment='center',
                          transform=ax.transAxes)
            txt.set_fontweight("bold")
            rslt["section titles"].append(txt)
            y -= title_space_axes / 2
            continue

        # The left margin label of the current line.
        label = left_labels[j] if left_labels is not None else ""

        # Special case: one scalar plots as one point
        if np.isscalar(val):
            val = [[val,],]

        # Special case: one tuple plots as one interval
        if all([np.isscalar(x) for x in val]):
            val = [val,]

        # Draw the left margin label
        txt = ax.text(-0.02, y, label, horizontalalignment="right",
                       verticalalignment='center',
                       transform=ax.transAxes)
        rslt["left labels"].append(txt)

        # Draw the right margin label
        if right_labels:
            txt = ax.text(1.02, y, right_labels[j],
                          horizontalalignment="left",
                          verticalalignment='center',
                          transform=ax.transAxes)
            rslt["right labels"].append(txt)

        # Draw the guide line
        gl = ax.axhline(y, color='grey')
        rslt["guidelines"].append(gl)

        # Draw the stripe
        if stripe and j % 2 == 0:
            ax.fill_between((0, 1), (y-dy/2, y-dy/2),
                            (y+dy/2, y+dy/2),
                            color='lightgrey',
                            transform=ax.transAxes)

        # Loop over the points in one line
        points_row, intervals_row = [], []
        for jp,valp in enumerate(val):

            # Calculate the vertical offset
            yo = 0
            if stack:
                yo = -dy/5 + jp*stackd

            # Plot the interval
            if len(valp) > 1:

                # Symmetric interval
                if len(valp) == 2:
                    lcb, ucb = valp[0] - valp[1], valp[0] + valp[1]

                # Nonsymmetric interval
                elif len(valp) == 3:
                    lcb, ucb = valp[0] - valp[1], valp[0] + valp[2]

                # Invalid data
                else:
                    msg = "dotplot: Element %d of `val` is %s\n" %\
                        (j, str(val))
                    msg += "It should be a numeric scalar, "\
                        "or a sequence of 1-3 numeric scalars."
                    raise Exception(msg)

                # Draw the interval
                lw = 2 if stack else 8
                interval, = ax.plot([lcb, ucb], [y+yo, y+yo], '-',
                                    color='grey', lw=lw, transform=trans)
                intervals_row.append(interval)

            # Plot the point
            ms = 5 if stack else 10
            a, = ax.plot([valp[0],], [y+yo,], 'o', ms=ms, color=colors[jp],
                         transform=trans)

            points_row.append(a)

        points.append(points_row)
        intervals.append(intervals_row)

        y -= dy

    rslt["points"] = points
    rslt["intervals"] = intervals

    # Set up the axis
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("none")
    ax.set_yticklabels([])
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # The position of the axis in axes coordinates.
    hp = props["axis_pos"] / aheight
    ax.spines['bottom'].set_position(('axes', hp))

    ax.set_ylim(0, 1)

    ax.autoscale_view(scaley=False, tight=True)
    ax.autoscale(enable=False, axis='y')

    return rslt
