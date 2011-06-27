This directory is only of interest to developers.  It contains files needed to build the docs
automatically and to do code maintenance.

How to update the main entry page
---------------------------------

If you want to update the main docs page from the most recent release then from the docs directory 
run the following (with your credentials).

make clean
make html
rsync -avPr -e ssh build/html/* jseabold,statsmodels@web.sourceforge.net:htdocs/

How to update the nightly builds
--------------------------------
Note that this is done automatically with the update_web.py script except for
new releases.  They should be done by hand if there are any backported changes.

Important: Make sure you have the version installed for which you are building 
the documentation.

To update devel branch (from the master branch)

Make sure you have master installed
cd to docs directory
make clean
make html

rsync -avPr -e ssh build/html/* jseabold,statsmodels@web.sourceforge.net:htdocs/devel

How to add a new directory
---------------------------
If you want to create a new directory on the sourceforge site.  
This can be done on linux as follows

sftp jseabold,statsmodels@web.sourceforge.net
<enter password>

mkdir 0.2release
bye

Then make sure you have the release installed, cd to the docs directory and run

make clean
make html

rsync -avPr -e ssh build/html/* jseabold,statsmodels@web.sourceforge.net:htdocs/0.2release
