Adding a dataset.

Main Steps
1) Obtain permission to use the data.

1) Obtain permission!  This is really important!  I usually look up an e-mail
address and politely (and briefly) explain why I would like to use the data.
Most people get back to me almost immediately, and I have never had anyone say
no.  After all, I think most academics are sympathetic to the idea that
information wants to be free...

2) Make a directory in the datasets folder.  For this example I will be using
the Spector and Mazzeo data from Greene's Econometric Analysis, so I make a
folder called statsmodels/datasets/spector

3) Copy the template_data.py file over to the new directory, but rename it data.py.  It contains all the meta information for the datasets.  So we now have datasets/spector/data.py

4) Put the raw data into this folder and convert it.

Sometimes the data used for examples is different than the raw data.  If this
is the case then the datasets/spector directory should contain a folder named
src for the original data.  In this case, the data is clean, so I just put a
file name spector.csv into datasets/spector.  This file is just an ascii file
with spaces as delimiters.  If the file requires a little cleaning, then put the
raw data in src and create a file called spector.csv in the spector folder for
the cleaned data.

After this is done, we use the convert function in scikits.statsmodels.datasets.data_utils to convert the data into the format needed. In the folder with our .csv file, just do.

from scikits.statsmodels.datasets.data_utils import convert
convert('./spector.csv', delimiter=" ")

This creates a spector.py file, which contains all of the variables as lists of strings.

5) Edit data.py to reflect the correct meta information.

Usually, this will require editing the COPYRIGHT, TITLE, SOURCE,
DESCRSHORT (and/or DESCRLONG), and "NOTE"

6) Edit the Load class of data.py to load the newly created dataset.

In this case, we change the following lines to read

from spector import __dict__, names
self.endog = np.array(self._d[self._names[4]], dtype=float)
self.exog = np.column_stack(self._d[i] \
    for i in self._names[1:4]).astype(np.float)

This is probably not the best way to handle the datasets class, and will
probably change in the future as the datasets package becomes more robust.
Suggetions are very welcome.

7) Create an __init__.py in the new folder

The __init__.py file should contain

from data import *

8) Edit the datasets.__init__.py to import the new directory

9) Make sure everything is correct, and you've saved everything,
   and put the directory under version control.

bzr add spector


