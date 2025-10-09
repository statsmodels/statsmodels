from __future__ import print_function

#!/usr/bin/env python
#
# texttable - module for creating simple ASCII tables
# Copyright (C) 2003-2009 Gerome Fournier <jefke(at)free.fr>
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

# Modified by Roger Lew 05/10/2011 to make the float and integer printing easier
# for humans to read.
#   Specifically:
#      - a public method to set float precision (set_float_precision)
#      - a private method to format the cells (_str)
#      - a private array to hold the formatting information (self._dtype)
#      - some modifications to add_row()

"""module for creating simple ASCII tables


Example:

    table = Texttable()
    table.set_cols_align(["l", "r", "c"])
    table.set_cols_valign(["t", "m", "b"])
    table.add_rows([ ["Name", "Age", "Nickname"], 
                     ["Mr\\nXavier\\nHuon", 32, "Xav'"],
                     ["Mr\\nBaptiste\\nClement", 1, "Baby"] ])
    print table.draw()

Result:

    +----------+-----+----------+
    |   Name   | Age | Nickname |
    +==========+=====+==========+
    | Mr       |     |          |
    | Xavier   |  32 |          |
    | Huon     |     |   Xav'   |
    +----------+-----+----------+
    | Mr       |     |          |
    | Baptiste |   1 |          |
    | Clement  |     |   Baby   |
    +----------+-----+----------+
"""

__all__ = ["Texttable", "ArraySizeError"]

__author__ = 'Gerome Fournier <jefke(at)free.fr>'
__license__ = 'GPL'
__version__ = '0.7'
__revision__ = '$Id: texttable.py 128 2009-10-04 15:16:22Z jef $'
__credits__ = """\
Jeff Kowalczyk:
    - textwrap improved import
    - comment concerning  output

Anonymous:
    - add_rows method, for adding rows in one go

Sergey Simonenko:
    - redefined len() function to deal with non-ASCII characters

Roger Lew:
    - made the float and integer printing easier for humans to read

"""

import sys
import string
import math

try:
    if sys.version >= '2.3':
        import textwrap
    elif sys.version >= '2.2':
        from optparse import textwrap
    else:
        from optik import textwrap
except ImportError:
    sys.stderr.write("Can't import textwrap module!\n")
    raise


##if sys.version_info[0] != 3:
##    try:
##        True, False
##    except NameError:
##        (True, False) = (1, 0)

def len(iterable):
    """Redefining len here so it will be able to work with non-ASCII characters
    """
    if not isinstance(iterable, str):
        return iterable.__len__()
    
    try:
        return len(unicode(iterable, 'utf'))
    except:
        return iterable.__len__()



def _str(x, dtype='a', n=3):
    """Handles string formatting of cell data

        x- is the item being added
        dtype- is so we can index self._dtype 
    """
    
    if x == None : return '--'
    
    try    : f=float(x)
    except : return str(x)

    if   math.isnan(f) : return '--'
    elif math.isinf(f) : return 'inf'
    elif dtype == 'i'  : return str(int(round(f)))
    elif dtype == 'f'  : return '%.*f'%(n, f)
    elif dtype == 'e'  : return '%.*e'%(n, f)
    elif dtype == 't'  : return str(x)
    else:
        if f-round(f) == 0:
            if abs(f) > 1e8:
                return '%.*e'%(n, f)
            else:
                return str(int(round(f)))
        else:
            if abs(f) > 1e8 or abs(f) <= float('1e-%i'%n):
                return '%.*e'%(n, f)
            else:
                return '%.*f'%(n, f)            
class ArraySizeError(Exception):
    """Exception raised when specified rows don't fit the required size
    """

    def __init__(self, msg):
        self.msg = msg
        Exception.__init__(self, msg, '')

    def __str__(self):
        return self.msg


class Texttable:

    BORDER = 1
    HEADER = 1 << 1
    HLINES = 1 << 2
    VLINES = 1 << 3
    FOOTER = 1 << 4

    def __init__(self, max_width=80):
        """Constructor

        - max_width is an integer, specifying the maximum width of the table
        - if set to 0, size is unlimited, therefore cells won't be wrapped
        """

        if max_width <= 0:
            max_width = False
        self._max_width = max_width
        self._float_precision=3
        
        self._deco = Texttable.VLINES | Texttable.HLINES | Texttable.BORDER | \
            Texttable.HEADER | Texttable.FOOTER
        self.set_chars(['-', '|', '+', '='])
        self.reset()

    def reset(self):
        """Reset the instance

        - reset rows and header
        """

        self._hline_string = None
        self._row_size = None
        self._header = []
        self._footer = []
        self._rows = []

    def header(self, array):
        """Specify the header of the table
        """

        self._check_row_size(array)
        self._header = map(_str, array)

    def footer(self, array):
        """Specify the footer of the table
        """

        self._check_row_size(array)
        self._footer = map(_str, array)
        
    def add_row(self, array):
        """Add a row in the rows stack

        - cells can contain newlines and tabs
        """

        self._check_row_size(array)
        
        if not hasattr(self, "_dtype"):
            self._dtype = ["a"]*self._row_size
            
        cells=[]
        for i,x in enumerate(array):
            cells.append(_str(x,self._dtype[i], self._float_precision))
        self._rows.append(cells)

    def add_rows(self, rows, header=True):
        """Add several rows in the rows stack

        - The 'rows' argument can be either an iterator returning arrays,
          or a by-dimensional array
        - 'header' specifies if the first row should be used as the header
          of the table
        """

        # nb: don't use 'iter' on by-dimensional arrays, to get a 
        #     usable code for python 2.1
        if header:
            if hasattr(rows, '__iter__') and hasattr(rows, 'next'):
                self.header(rows.next())
            else:
                self.header(rows[0])
                rows = rows[1:]
        for row in rows:
            self.add_row(row)

    def set_chars(self, array):
        """Set the characters used to draw lines between rows and columns

        - the array should contain 4 fields:

            [horizontal, vertical, corner, header]

        - default is set to:

            ['-', '|', '+', '=']
        """

        if len(array) != 4:
            raise Exception("array should contain 4 characters")
        array = [ x[:1] for x in [ str(s) for s in array ] ]
        (self._char_horiz, self._char_vert,
            self._char_corner, self._char_header) = array

    def set_deco(self, deco):
        """Set the table decoration

        - 'deco' can be a combinaison of:

            Texttable.BORDER: Border around the table
            Texttable.HEADER: Horizontal line below the header
            Texttable.HLINES: Horizontal lines between rows
            Texttable.VLINES: Vertical lines between columns
            Texttable.FOOTER: Horizontal line above the footer

           All of them are enabled by default

        - example:

            Texttable.BORDER | Texttable.HEADER
        """

        self._deco = deco

    def set_float_precision(self, width):
        if int(width)<0:
            raise ValueError('width must be greater then 0')
        self._float_precision=int(width)

    def set_cols_align(self, array):
        """Set the desired columns alignment

        - the elements of the array should be either "l", "c" or "r":

            * "l": column flushed left
            * "c": column centered
            * "r": column flushed right
        """

        self._check_row_size(array)
        self._align = array

    def set_cols_valign(self, array):
        """Set the desired columns vertical alignment

        - the elements of the array should be either "t", "m" or "b":

            * "t": column aligned on the top of the cell
            * "m": column aligned on the middle of the cell
            * "b": column aligned on the bottom of the cell
        """

        self._check_row_size(array)
        self._valign = array

    def set_cols_dtype(self, array):
        """Set the desired columns datatype for the cols.
           Must be set BEFORE adding rows.

        - the elements of the array should be either "a", "t", "f", or "i":

            * "a": automatic datatyping (default)
            * "t": treat as text
            * "f": treat as float in decimal format
            * "e": treat as float in exponential format
            * "i": treat as int
        """

        self._check_row_size(array)
        self._dtype = array

    def set_cols_width(self, array):
        """Set the desired columns width

        - the elements of the array should be integers, specifying the
          width of each column. For example:

                [10, 20, 5]
        """

        self._check_row_size(array)
        try:
            array = map(int, array)
            if reduce(min, array) <= 0:
                raise ValueError
        except ValueError:
            sys.stderr.write("Wrong argument in column width specification\n")
            raise
        self._width = array

    def draw(self):
        """Draw the table

        - the table is returned as a whole string
        """

        if not self._header and not self._rows:
            return
        self._compute_cols_width()
        self._check_align()
        out = ""
        if self._has_border():
            out += self._hline()
        if self._header:
            out += self._draw_line(self._header, isheader=True)
            if self._has_header():
                out += self._hline_header()
        length = 0
        for row in self._rows:
            length += 1
            out += self._draw_line(row)
            if self._has_hlines() and length < len(self._rows):
                out += self._hline()
        if self._footer:
            if self._has_footer():
                out += self._hline_header()
            out += self._draw_line(self._footer)
        if self._has_border():
            out += self._hline()
        return out[:-1]

    def _check_row_size(self, array):
        """Check that the specified array fits the previous rows size
        """
        actual_len = len(array)
        if not self._row_size:
            self._row_size = actual_len
        elif self._row_size != actual_len:
            raise Exception("array should contain %d elements, contains %d" \
                %(self._row_size, actual_len))

    def _has_vlines(self):
        """Return a boolean, if vlines are required or not
        """

        return self._deco & Texttable.VLINES > 0

    def _has_hlines(self):
        """Return a boolean, if hlines are required or not
        """

        return self._deco & Texttable.HLINES > 0

    def _has_border(self):
        """Return a boolean, if border is required or not
        """

        return self._deco & Texttable.BORDER > 0

    def _has_header(self):
        """Return a boolean, if header line is required or not
        """

        return self._deco & Texttable.HEADER > 0

    def _has_footer(self):
        """Return a boolean, if header line is required or not
        """

        return self._deco & Texttable.FOOTER > 0

    def _hline_header(self):
        """Print header's horizontal line
        """

        return self._build_hline(True)

    def _hline(self):
        """Print an horizontal line
        """

        if not self._hline_string:
            self._hline_string = self._build_hline()
        return self._hline_string

    def _build_hline(self, is_header=False):
        """Return a string used to separated rows or separate header from
        rows
        """
        horiz = self._char_horiz
        if (is_header):
            horiz = self._char_header
        # compute cell separator
        s = "%s%s%s" % (horiz, [horiz, self._char_corner][self._has_vlines()],
            horiz)
        # build the line
        l = string.join([horiz*n for n in self._width], s)
        # add border if needed
        if self._has_border():
            l = "%s%s%s%s%s\n" % (self._char_corner, horiz, l, horiz,
                self._char_corner)
        else:
            l += "\n"
        return l

    def _len_cell(self, cell):
        """Return the width of the cell

        Special characters are taken into account to return the width of the
        cell, such like newlines and tabs
        """

        cell_lines = cell.split('\n')
        maxi = 0
        for line in cell_lines:
            length = 0
            parts = line.split('\t')
            for part, i in zip(parts, range(1, len(parts) + 1)):
                length = length + len(part)
                if i < len(parts):
                    length = (length/8 + 1)*8
            maxi = max(maxi, length)
        return maxi

    def _compute_cols_width(self):
        """Return an array with the width of each column

        If a specific width has been specified, exit. If the total of the
        columns width exceed the table desired width, another width will be
        computed to fit, and cells will be wrapped.
        """

        if hasattr(self, "_width"):
            return
        maxi = []
        if self._header:
            maxi = [ self._len_cell(x) for x in self._header ]
        if self._footer:
            for cell,i in zip(self._footer, range(len(self._footer))):
                maxi[i] = max(maxi[i], self._len_cell(cell))
        for row in self._rows:
            for cell,i in zip(row, range(len(row))):
                try:
                    maxi[i] = max(maxi[i], self._len_cell(cell))
                except (TypeError, IndexError):
                    maxi.append(self._len_cell(cell))
        items = len(maxi)
        length = reduce(lambda x,y: x+y, maxi)
        if self._max_width and length + items*3 + 1 > self._max_width:
            maxi = [(self._max_width - items*3 -1) / items \
                for n in range(items)]
        self._width = maxi
    

    def _check_align(self):
        """Check if alignment has been specified, set default one if not
        """

        if not hasattr(self, "_align"):
            self._align = ["l"]*self._row_size
        if not hasattr(self, "_valign"):
            self._valign = ["t"]*self._row_size

    def _draw_line(self, line, isheader=False):
        """Draw a line

        Loop over a single cell length, over all the cells
        """

        line = self._splitit(line, isheader)
        space = " "
        out  = ""
        for i in range(len(line[0])):
            if self._has_border():
                out += "%s " % self._char_vert
            length = 0
            for cell, width, align in zip(line, self._width, self._align):
                length += 1
                cell_line = cell[i]
                fill = width - len(cell_line)
                if isheader:
                    align = "c"
                if align == "r":
                    out += "%s " % (fill * space + cell_line)
                elif align == "c":
                    out += "%s " % (fill/2 * space + cell_line \
                            + (fill/2 + fill%2) * space)
                else:
                    out += "%s " % (cell_line + fill * space)
                if length < len(line):
                    out += "%s " % [space, self._char_vert][self._has_vlines()]
            out += "%s\n" % ['', self._char_vert][self._has_border()]
        return out

    def _splitit(self, line, isheader):
        """Split each element of line to fit the column width

        Each element is turned into a list, result of the wrapping of the
        string to the desired width
        """

        line_wrapped = []
        for cell, width in zip(line, self._width):
            array = []
            for c in cell.split('\n'):
                array.extend(textwrap.wrap(unicode(c, 'utf'), width))
            line_wrapped.append(array)
        max_cell_lines = reduce(max, map(len, line_wrapped))
        for cell, valign in zip(line_wrapped, self._valign):
            if isheader:
                valign = "t"
            if valign == "m":
                missing = max_cell_lines - len(cell)
                cell[:0] = [""] * (missing / 2)
                cell.extend([""] * (missing / 2 + missing % 2))
            elif valign == "b":
                cell[:0] = [""] * (max_cell_lines - len(cell))
            else:
                cell.extend([""] * (max_cell_lines - len(cell)))
        return line_wrapped

if __name__ == '__main__':
    table = Texttable()
    table.set_cols_align(["l", "r", "c"])
    table.set_cols_valign(["t", "m", "b"])
    table.add_rows([ ["Name", "Age", "Nickname"], 
                     ["Mr\nXavier\nHuon", 32, "Xav'"],
                     ["Mr\nBaptiste\nClement", 1, "Baby"] ])
    print(table.draw())


    # Roger Lew also add this example/test
    table= Texttable()
    table.set_deco(Texttable.HEADER | Texttable.FOOTER)

    # set the datatypes, this must be called before we start adding rows
    table.set_cols_dtype(['t',  # text 
                          'f',  # float (decimal)
                          'e',  # float (exponent)
                          'i',  # integer
                          'a']) # automatic typing
    
    # It default to 'a' (in the add_row function.) Setting the default
    # to 't' would make it 100% backwards compatible.
    
    table.add_rows([['text','float',  'exp',   'int',   'auto'],
                    ['abcd', '67',    654,     None,      128.001],
                    ['efgh', 67.5434, .654,    89.6,    12800000000000000000000.00023],
                    ['ijkl', 5e-78,   5e-78,   89.4,    .000000000000128],
                    ['mnop', float('nan'),    5e+78,   92.,     12800000000000000000000]])
    table.footer(['a','b','c','d','e'])
    print(table.draw())

