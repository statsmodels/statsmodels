#!/usr/bin/env python
import sys
import os


BUILDDIR = sys.argv[-1]
read_file_path = os.path.join(BUILDDIR,'latex','statsmodels.tex')
write_file_path = os.path.join(BUILDDIR, 'latex','statsmodels_tmp.tex')

read_file = open(read_file_path,'r')
write_file = open(write_file_path, 'w')

for line in read_file:
    if 'longtable}{LL' in line:
        line = line.replace('longtable}{LL', 'longtable}{|l|l|')
    write_file.write(line)

read_file.close()
write_file.close()

os.remove(read_file_path)
os.rename(write_file_path, read_file_path)
