import numpy as np

from dotplot import dotplot

try:
    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib.backends.backend_pdf import PdfPages
    if matplotlib.__version__ < '1':
        raise
    have_matplotlib = True
except:
    have_matplotlib = False

pdf = PdfPages("test_dotplot.pdf")

plt.clf()

# Basic dotplot with points only
ax = plt.axes()
vals = range(20)
dotplot(vals, ax)
ax.set_title("Basic dotplot (input is list)")
pdf.savefig(facecolor="yellow")

# Basic dotplot with points only
ax = plt.axes()
vals = np.arange(20)
dotplot(vals, ax)
ax.set_title("Basic dotplot (input is 1D ndarray)")
pdf.savefig(facecolor="yellow")

# Tall and skinny
plt.figure(figsize=(4,12))
ax = plt.axes()
vals = np.arange(40)
dotplot(vals, ax)
ax.set_title("Tall and skinny dotplot")
pdf.savefig()

# Basic dotplot with few points
plt.figure()
ax = plt.axes()
vals = np.arange(4)
dotplot(vals, ax)
ax.set_title("Basic dotplot with few lines")
pdf.savefig()

# Manually set the x axis limits
plt.figure()
ax = plt.axes()
vals = np.arange(20)
dotplot(vals, ax)
ax.set_xlim(-10, 30)
ax.set_title("Dotplot with adjusted horizontal range")
pdf.savefig()

# Left row labels
plt.clf()
ax = plt.axes()
vals = np.random.normal(size=20)
names = ["left %d" % k for k in range(1,21)]
dotplot(vals, ax, names)
ax.set_title("Dotplot with labels in the left margin")
pdf.savefig()

# Right row labels
plt.clf()
ax = plt.axes()
vals = np.random.normal(size=20)
right_names = ["right %d" % k for k in range(1,21)]
dotplot(vals, ax, right_labels=right_names)
ax.set_title("Dotplot with labels in the right margin")
pdf.savefig()

# Left and right row labels
plt.clf()
ax = plt.axes([0.1, 0.07, 0.78, 0.85])
vals = np.random.normal(size=20)
dotplot(vals, ax, names, right_labels=right_names)
ax.set_title("Dotplot with labels in both margins")
pdf.savefig()

# Adjust the points after plotting
plt.clf()
ax = plt.axes([0.1, 0.07, 0.78, 0.85])
vals = np.random.normal(size=20)
fig,rslt = dotplot(vals, ax)
for pr in rslt["points"]:
    for p in pr:
        p.set_ms(7)
        p.set_mec("blue")
        p.set_mfc("none")
        p.set_marker("s")
ax.set_title("Dotplot with custom colors and symbols")
pdf.savefig()

# Basic dotplot with symmetric intervals
plt.clf()
ax = plt.axes()
vals = range(20)
vals = [(x, 2) for x in vals]
dotplot(vals, ax)
ax.set_title("Dotplot with symmetric intervals (input is list of lists)")
pdf.savefig()

# Basic dotplot with symmetric intervals
plt.clf()
ax = plt.axes()
vals = np.zeros((20,2))
vals[:,0] = range(20)
vals[:,1] = 2
dotplot(vals, ax)
ax.set_title("Dotplot with symmetric intervals (input is 2D ndarray)")
pdf.savefig()

# Basic dotplot with nonsymmetric intervals
plt.clf()
ax = plt.axes()
vals = np.arange(20)
vals = [(x, np.random.uniform(0, 5), np.random.uniform(0, 5))
        for x in vals]
dotplot(vals, ax)
ax.set_title("Dotplot with nonsymmetric intervals")
pdf.savefig()

# Dotplot with nonsymmetric intervals, adjust line properties
plt.clf()
ax = plt.axes()
vals = np.arange(20)
vals = [(x, 1, 3) for x in vals]
fig,rslt = dotplot(vals, ax)
for interval in rslt["intervals"]:
    interval[0].set_solid_capstyle("round")
    interval[0].set_color("lightgrey")
ax.set_title("Dotplot with custom line properties")
pdf.savefig()

# Dotplot with two points per line and a legend
plt.clf()
ax = plt.axes([0.1, 0.1, 0.7, 0.8])
vals = [((j-np.random.uniform(-5, 5),), (j+np.random.uniform(-3, 3),))
        for j in range(20)]
fig,rslt = dotplot(vals, ax)
for point in rslt["points"]:
    point[0].set_alpha(0.6)
    point[0].set_marker("s")
    point[0].set_color("orange")
    point[1].set_alpha(0.6)
    point[1].set_color("purple")
leg = plt.figlegend(rslt["points"][0], ("Apple", "Orange"),
                    "center right", numpoints=1, handletextpad=0.001)
leg.draw_frame(False)
ax.set_title("Dotplot with two points per line")
pdf.savefig()

# Stacked dotplot with two points per line
plt.clf()
ax = plt.axes()
vals = np.arange(20)
vals = [((x, 1, 3), (x+2, 1)) for x in vals]
fig,rslt = dotplot(vals, ax, stack="above")
ax.set_title("Dotplot with stacked lines")
pdf.savefig()

# Stacked dotplot with three points per line and stripes
plt.clf()
plt.figure(figsize=(7,15))
ax = plt.axes([0.1, 0.1, 0.7, 0.8])
vals = []
for k in range(20):
    val = []
    for j in range(3):
        val.append([np.random.uniform(0, 20), 2, 2])
    vals.append(val)
names = [str(j+1) for j in range(20)]
fig,rslt = dotplot(vals, ax, names, stack=True, stripe=True)
leg = plt.figlegend(rslt["points"][0],
                    ("Apple", "Orange", "Pear"),
                    "center right", numpoints=1,
                    handletextpad=0.001)
leg.draw_frame(False)
ax.set_xlim(-5, 25)
ax.set_title("Stacked dotplot with three points per line and stripes")
pdf.savefig()

# Dotplot with sections
plt.clf()
ax = plt.axes()
vals = np.arange(20)
vals = [(x, 2) for x in vals]
vals[0] = "Axx"
vals[10] = "Byy"
vals[15] = "Czz"
dotplot(vals, ax)
ax.set_title("Dotplot with sections")
pdf.savefig()

pdf.close()
