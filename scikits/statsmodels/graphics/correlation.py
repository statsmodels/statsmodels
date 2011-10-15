def plot_corr(rrcorr, xnames=None, ynames=None, title=None, normcolor=False):
    nvars = rrcorr.shape[0]
    #rrcorr[range(nvars), range(nvars)] = np.nan

    if (ynames is None) and (not xnames is None):
        ynames = xnames
    if title is None:
        title = 'Correlation Matrix'
    if normcolor:
        vmin, vmax = -1.0, 1.0
    else:
        vmin, vmax = None, None


    fig = plt.figure()
    ax = fig.add_subplot(111)
    axim = ax.imshow(rrcorr, cmap=plt.cm.jet, interpolation='nearest',
                     extent=(0,30,0,30), vmin=vmin, vmax=vmax)
    if ynames:
        ax.set_yticks(np.arange(nvars)+0.5)
        ax.set_yticklabels(ynames[::-1], minor=True, fontsize='small',
                           horizontalalignment='right')
    if xnames:
        ax.set_xticks(np.arange(nvars)+0.5)
        ax.set_xticklabels(xnames, minor=True, fontsize='small',rotation=45, horizontalalignment='right')
        #some keywords don't work in previous line ?
        plt.setp( ax.get_xticklabels(), fontsize='small', rotation=45,
                 horizontalalignment='right')
    fig.colorbar(axim)
    ax.set_title(title)