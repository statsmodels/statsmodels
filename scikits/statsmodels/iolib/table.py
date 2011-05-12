"""
Simple table class.
Note that this module depends only on the Python standard library.
You can "install" it just by dropping it into your working directory.

A SimpleTable is inherently (but not rigidly) rectangular.
You should create it from a *rectangular* (2d!) iterable of data.
A SimpleTable can be concatenated with another SimpleTable
or extended by another SimpleTable. ::

	table1.extend_right(table2)
	table1.extend(table2)

Note that although a SimpleTable allows only one column (the first) of
stubs at initilization, concatenation of tables allows you to produce
tables with interior stubs.  (You can also assign the datatype 'stub'
to the cells in any column, or use ``insert_stubs``.)

A SimpleTable can be initialized with `datatypes`: a list of ints that
provide indexes into `data_fmts` and `data_aligns`.  Each data cell is
assigned a datatype, which will control formatting.  If you do not
specify the `datatypes` list, it will be set to ``range(ncols)`` where
`ncols` is the number of columns in the data.  (I.e., cells in a
column have their own datatype.) This means that you can just specify
`data_fmts` without bother to provide a `datatypes` list.  If
``len(datatypes)<ncols`` then datatype assignment will cycle across a
row.  E.g., if you provide 10 rows of data with ``datatypes=[0,1]``
then you will have 5 columns of datatype 0 and 5 columns of datatype
1, alternating.  Correspoding to this specification, you should provide
a list of two ``data_fmts`` and a list of two ``data_aligns``.

Potential problems for Python 3
-------------------------------

- Calls ``next`` instead of ``__next__``.
  The 2to3 tool should handle that no problem.
  (We will switch to the `next` function if 2.5 support is ever dropped.)
- from __future__ import division, with_statement
- from itertools import izip as zip
- Let me know if you find other problems.

:contact: alan dot isaac at gmail dot com
:requires: Python 2.5.1+
:note: current version
:note: HTML data format currently specifies tags
:todo: support a bit more of http://www.oasis-open.org/specs/tr9503.html
:todo: add colspan support to Cell
:since: 2008-12-21
:change: 2010-05-02 eliminate newlines that came before and after table
"""
from __future__ import division, with_statement
try: #accommodate Python 3
    from itertools import izip as zip
    pass  #JP: try to avoid empty try with 2to3
except ImportError:
    pass
from itertools import cycle
from collections import defaultdict
import csv

def csv2st(csvfile, headers=False, stubs=False, title=None):
    """Return SimpleTable instance,
    created from the data in `csvfile`,
    which is in comma separated values format.
    The first row may contain headers: set headers=True.
    The first column may contain stubs: set stubs=True.
    Can also supply headers and stubs as tuples of strings.
    """
    rows = list()
    with open(csvfile,'r') as fh:
        reader = csv.reader(fh)
        if headers is True:
            headers = reader.next()
        elif headers is False:
            headers=()
        if stubs is True:
            stubs = list()
            for row in reader:
                if row:
                    stubs.append(row[0])
                    rows.append(row[1:])
        else: #no stubs, or stubs provided
            for row in reader:
                if row:
                    rows.append(row)
        if stubs is False:
            stubs = ()
    nrows = len(rows)
    ncols = len(rows[0])
    if any(len(row)!=ncols for row in rows):
        raise IOError('All rows of CSV file must have same length.')
    return SimpleTable(data=rows, headers=headers, stubs=stubs)


class SimpleTable(list):
    """Produce a simple ASCII, CSV, HTML, or LaTeX table from a
    *rectangular* (2d!) array of data, not necessarily numerical.
    Directly supports at most one header row,
    which should be the length of data[0].
    Directly supports at most one stubs column,
    which must be the length of data.
    (But see `insert_stubs` method.)
    See globals `default_txt_fmt`, `default_csv_fmt`, `default_html_fmt`,
    and `default_latex_fmt` for formatting options.

    Sample uses::

    	mydata = [[11,12],[21,22]]  # data MUST be 2-dimensional
    	myheaders = [ "Column 1", "Column 2" ]
    	mystubs = [ "Row 1", "Row 2" ]
    	tbl = sm.iolib.SimpleTable(mydata, myheaders, mystubs, title="Title")
    	print( tbl )
    	print( tbl.as_html() )
    	# set column specific data formatting
    	tbl = sm.iolib.SimpleTable(mydata, myheaders, mystubs,
    		fmt={'data_fmts':["%3.2f","%d"]})
    	print( tbl.as_csv() )
    	with open('./temp.tex','w') as fh:
    		fh.write( tbl.as_latex_tabular() )
    """
    def __init__(self, data, headers=None, stubs=None, title='',
                 datatypes=None,
                 csv_fmt=None, txt_fmt=None, ltx_fmt=None, html_fmt=None,
                 celltype= None, rowtype=None,
                 **fmt_dict):
        """
        Parameters
        ----------
        data : list of lists or 2d array (not matrix!)
        	R rows by K columns of table elements
        headers : list (or tuple) of str
        	sequence of K strings, one per header
        stubs : list (or tuple) of str
        	sequence of R strings, one per stub
        title : string
        	title of the table
        datatypes : list of int
        	indexes to `data_fmts`
        txt_fmt : dict
        	text formatting options
        ltx_fmt : dict
        	latex formatting options
        csv_fmt : dict
        	csv formatting options
        hmtl_fmt : dict
        	hmtl formatting options
        celltype : class
        	the cell class for the table (default: Cell)
        rowtype : class
        	the row class for the table (default: Row)
        fmt_dict : dict
        	general formatting options
        """
        #self._raw_data = data
        self.title = title
        self._datatypes = datatypes or range(len(data[0]))
        #start with default formatting
        self._text_fmt = default_txt_fmt.copy()
        self._latex_fmt = default_latex_fmt.copy()
        self._csv_fmt = default_csv_fmt.copy()
        self._html_fmt = default_html_fmt.copy()
        #substitute any general user specified formatting
        #:note: these will be overridden by output specific arguments
        self._csv_fmt.update(fmt_dict)
        self._text_fmt.update(fmt_dict)
        self._latex_fmt.update(fmt_dict)
        self._html_fmt.update(fmt_dict)
        #substitute any output-type specific formatting
        self._csv_fmt.update(csv_fmt or dict())
        self._text_fmt.update(txt_fmt or dict())
        self._latex_fmt.update(ltx_fmt or dict())
        self._html_fmt.update(html_fmt or dict())
        self.output_formats = dict(
            text=self._text_fmt,
            txt=self._text_fmt,
            csv=self._csv_fmt,
            htm=self._html_fmt,
            html=self._html_fmt,
            latex=self._latex_fmt,
            ltx=self._latex_fmt)
        self._Cell = celltype or Cell
        self._Row = rowtype or Row
        rows = self._data2rows(data)  # a list of Row instances
        list.__init__(self, rows)
        self._add_headers_stubs(headers, stubs)
    def __str__(self):
        return self.as_text()
    def _add_headers_stubs(self, headers, stubs):
        """Return None.  Adds headers and stubs to table,
        if these were provided at initialization.
        Parameters
        ----------
        headers : list of strings
        	K strings, where K is number of columns
        stubs : list of strings
        	R strings, where R is number of non-header rows

        :note: a header row does not receive a stub!
        """
        _Cell = self._Cell
        _Row = self._Row
        if headers:
            headers = [ _Cell(h,datatype='header') for h in headers ]
            headers = _Row(headers, datatype='header')
            headers.table = self
            for cell in headers:
                cell.row = headers
            self.insert(0, headers)
        if stubs:
            self.insert_stubs(0, stubs)
    def _data2rows(self, raw_data):
        """Return list of Row,
        the raw data as rows of cells.
        """
        _Cell = self._Cell
        _Row = self._Row
        rows = []
        for datarow in raw_data:
            dtypes = cycle(self._datatypes)
            newrow = _Row([_Cell(datum) for datum in datarow])
            newrow.table = self  #row knows its SimpleTable
            for cell in newrow:
                cell.datatype = dtypes.next()
                cell.row = newrow  #a cell knows its row
            rows.append(newrow)
        return rows
    def pad(self, s, width, align):
        """DEPRECATED: just use the pad function"""
        return pad(s, width, align)
    def get_colwidths(self, output_format, **fmt_dict):
        fmt = self.output_formats[output_format].copy()
        fmt.update(fmt_dict)
        ncols = max(len(row) for row in self)
        request = fmt.get('colwidths')
        if request is 0: #assume no extra space desired (e.g, CSV)
            return [0] * ncols
        elif request is None: #assume no extra space desired (e.g, CSV)
            request = [0] * ncols
        elif isinstance(request, int):
            request = [request] * ncols
        elif len(request) < ncols:
            request = [request[i%len(request)] for i in range(ncols)]
        min_widths = []
        for col in zip(*self):
            maxwidth = max(len(c.format(0,output_format,**fmt)) for c in col)
            min_widths.append(maxwidth)
        result = map(max, min_widths, request)
        return result
    def _get_fmt(self, output_format, **fmt_dict):
        """Return dict, the formatting options.
        """
        #first get the default formatting
        try:
            fmt = self.output_formats[output_format].copy()
        except KeyError:
            raise ValueError('Unknown format: %s' % output_format)
        #then, add formatting specific to this call
        fmt.update(fmt_dict)
        return fmt
    def as_csv(self, **fmt_dict):
        """Return string, the table in CSV format.
        Currently only supports comma separator."""
        #fetch the format, which may just be default_csv_format
        fmt = self._get_fmt('csv', **fmt_dict)
        return self.as_text(**fmt)
    def as_text(self, **fmt_dict):
        """Return string, the table as text."""
        #fetch the text format, override with fmt_dict
        fmt = self._get_fmt('txt', **fmt_dict)
        #get rows formatted as strings
        formatted_rows = [ row.as_string('text', **fmt) for row in self ]
        rowlen = len(formatted_rows[-1]) #don't use header row

        #place decoration above the table body, if desired
        table_dec_above = fmt.get('table_dec_above','=')
        if table_dec_above:
            formatted_rows.insert(0, table_dec_above * rowlen)
        #next place a title at the very top, if desired
        #:note: user can include a newlines at end of title if desired
        title = self.title
        if title:
            title = pad(self.title, rowlen, fmt.get('title_align','c'))
            formatted_rows.insert(0, title)
        #add decoration below the table, if desired
        table_dec_below = fmt.get('table_dec_below','-')
        if table_dec_below:
            formatted_rows.append(table_dec_below * rowlen)
        return '\n'.join(formatted_rows)
    def as_html(self, **fmt_dict):
        """Return string.
        This is the default formatter for HTML tables.
        An HTML table formatter must accept as arguments
        a table and a format dictionary.
        """
        #fetch the text format, override with fmt_dict
        fmt = self._get_fmt('html', **fmt_dict)
        formatted_rows = ['<table class="simpletable">']
        if self.title:
            title = '<caption>%s</caption>' % self.title
            formatted_rows.append(title)
        formatted_rows.extend( row.as_string('html', **fmt) for row in self )
        formatted_rows.append('</table>')
        return '\n'.join(formatted_rows)
    def as_latex_tabular(self, **fmt_dict):
        '''Return string, the table as a LaTeX tabular environment.
        Note: will equire the booktabs package.'''
        #fetch the text format, override with fmt_dict
        fmt = self._get_fmt('latex', **fmt_dict)
        aligns = self[-1].get_aligns('latex', **fmt)
        formatted_rows = [ r'\begin{tabular}{%s}' % aligns ]

        table_dec_above = fmt['table_dec_above']
        if table_dec_above:
            formatted_rows.append(table_dec_above)

        formatted_rows.extend(
            row.as_string(output_format='latex', **fmt) for row in self )

        table_dec_below = fmt['table_dec_below']
        if table_dec_below:
            formatted_rows.append(table_dec_below)

        formatted_rows.append(r'\end{tabular}')
        #tabular does not support caption, but make it available for figure environment
        if self.title:
            title = r'%%\caption{%s}' % self.title
            formatted_rows.append(title)
        return '\n'.join(formatted_rows)
        """
		if fmt_dict['strip_backslash']:
			ltx_stubs = [stub.replace('\\',r'$\backslash$') for stub in self.stubs]
			ltx_headers = [header.replace('\\',r'$\backslash$') for header in self.headers]
			ltx_headers = self.format_headers(fmt_dict, ltx_headers)
		else:
			ltx_headers = self.format_headers(fmt_dict)
		ltx_stubs = self.format_stubs(fmt_dict, ltx_stubs)
		"""
    def extend_right(self, table):
        """Return None.
        Extend each row of `self` with corresponding row of `table`.
        Does **not** import formatting from ``table``.
        This generally makes sense only if the two tables have
        the same number of rows, but that is not enforced.
        :note: To extend append a table below, just use `extend`,
        which is the ordinary list method.  This generally makes sense
        only if the two tables have the same number of columns,
        but that is not enforced.
        """
        for row1, row2 in zip(self, table):
            row1.extend(row2)
    def insert_stubs(self, loc, stubs):
        """Return None.  Insert column of stubs at column `loc`.
        If there is a header row, it gets an empty cell.
        So ``len(stubs)`` should equal the number of non-header rows.
        """
        _Cell = self._Cell
        stubs = iter(stubs)
        for row in self:
            if row.datatype == 'header':
                empty_cell = _Cell('', datatype='empty')
                row.insert(loc, empty_cell)
            else:
                row.insert_stub(loc, stubs.next())
    @property
    def data(self):
        return [row.data for row in self]
#END: class SimpleTable

def pad(s, width, align):
    """Return string padded with spaces,
    based on alignment parameter."""
    if align == 'l':
        s = s.ljust(width)
    elif align == 'r':
        s = s.rjust(width)
    else:
        s = s.center(width)
    return s


class Row(list):
    """A Row is a list of cells;
    a row can belong to a SimpleTable.
    """
    def __init__(self, cells, datatype='', table=None, celltype=None, **fmt_dict):
        """
        Parameters
        ----------
        table : SimpleTable
        """
        list.__init__(self, cells)
        self.datatype = datatype # data or header
        self.table = table
        if celltype is None:
            try:
                celltype = table._Cell
            except AttributeError:
                celltype = Cell
        self._Cell = celltype
        self._fmt = fmt_dict
    def insert_stub(self, loc, stub):
        """Return None.  Inserts a stub cell
        in the row at `loc`.
        """
        _Cell = self._Cell
        if not isinstance(stub, _Cell):
            stub = stub
            stub = _Cell(stub, datatype='stub', row=self)
        self.insert(loc, stub)
    def _get_fmt(self, output_format, **fmt_dict):
        """Return dict, the formatting options.
        """
        #first get the default formatting
        try:
            fmt = default_fmts[output_format].copy()
        except KeyError:
            raise ValueError('Unknown format: %s' % output_format)
        #second get table specific formatting (if possible)
        try:
            fmt.update(self.table.output_formats[output_format])
        except AttributeError:
            pass
        #finally, add formatting for this cell and this call
        fmt.update(self._fmt)
        fmt.update(fmt_dict)
        return fmt
    def get_aligns(self, output_format, **fmt_dict):
        """Return string, sequence of column alignments.
        Ensure comformable data_aligns in `fmt_dict`."""
        fmt = self._get_fmt(output_format, **fmt_dict)
        return ''.join( cell.alignment(output_format, **fmt) for cell in self )
    def as_string(self, output_format='txt', **fmt_dict):
        """Return string: the formatted row.
        This is the default formatter for rows.
        Override this to get different formatting.
        A row formatter must accept as arguments
        a row (self) and an output format,
        one of ('html', 'txt', 'csv', 'latex').
        """
        fmt = self._get_fmt(output_format, **fmt_dict)

        #get column widths
        try:
            colwidths = self.table.get_colwidths(output_format, **fmt)
        except AttributeError:
            colwidths = fmt.get('colwidths')
        if colwidths is None:
            colwidths = (0,) * len(self)

        colsep = fmt['colsep']
        row_pre = fmt.get('row_pre','')
        row_post = fmt.get('row_post','')
        formatted_cells = []
        for cell, width in zip(self, colwidths):
            content = cell.format(width, output_format=output_format, **fmt)
            formatted_cells.append(content)
        header_dec_below = fmt.get('header_dec_below')
        formatted_row = row_pre + colsep.join(formatted_cells) + row_post
        if self.datatype == 'header' and header_dec_below:
            formatted_row = self.decorate_header(formatted_row, output_format, header_dec_below)
        return formatted_row
    def decorate_header(self, header_as_string, output_format, header_dec_below):
        """This really only makes sense for the text and latex output formats."""
        if output_format in ('text','txt'):
            row0len = len(header_as_string)
            result = header_as_string + "\n" + (header_dec_below * row0len)
        elif output_format == 'latex':
            result = header_as_string + "\n" + header_dec_below
        else:
            raise ValueError("I can't decorate a %s header."%output_format)
        return result
    @property
    def data(self):
        return [cell.data for cell in self]
#END class Row


class Cell(object):
    def __init__(self, data='', datatype=0, row=None, **fmt_dict):
        self.data = data
        self.datatype = datatype
        self.row = row
        self._fmt = fmt_dict
    def __str__(self):
        return '%s' % self.data
    def _get_fmt(self, output_format, **fmt_dict):
        """Return dict, the formatting options.
        """
        #first get the default formatting
        try:
            fmt = default_fmts[output_format].copy()
        except KeyError:
            raise ValueError('Unknown format: %s' % output_format)
        #then get any table specific formtting
        try:
            fmt.update(self.row.table.output_formats[output_format])
        except AttributeError:
            pass
        #then get any row specific formtting
        try:
            fmt.update(self.row._fmt)
        except AttributeError:
            pass
        #finally add formatting for this instance and call
        fmt.update(self._fmt)
        fmt.update(fmt_dict)
        return fmt
    def alignment(self, output_format, **fmt_dict):
        fmt = self._get_fmt(output_format, **fmt_dict)
        datatype = self.datatype
        data_aligns = fmt.get('data_aligns','c')
        if isinstance(datatype, int):
            align = data_aligns[datatype % len(data_aligns)]
        elif datatype == 'header':
            align = fmt.get('header_align','c')
        elif datatype == 'stub':
            align = fmt.get('stubs_align','c')
        elif datatype == 'empty':
            align = 'c'
        else:
            raise ValueError('Unknown cell datatype: %s'%datatype)
        return align
    def format(self, width, output_format='txt', **fmt_dict):
        """Return string.
        This is the default formatter for cells.
        Override this to get different formating.
        A cell formatter must accept as arguments
        a cell (self) and an output format,
        one of ('html', 'txt', 'csv', 'latex').
        It will generally respond to the datatype,
        one of (int, 'header', 'stub').
        """
        fmt = self._get_fmt(output_format, **fmt_dict)

        data = self.data
        datatype = self.datatype
        data_fmts = fmt.get('data_fmts')
        if data_fmts is None:
            #chk allow for deprecated use of data_fmt
            data_fmt = fmt.get('data_fmt')
            if data_fmt is None:
                data_fmt = '%s'
            data_fmts = [data_fmt]
        data_aligns = fmt.get('data_aligns','c')
        if isinstance(datatype, int):
            datatype = datatype % len(data_fmts) #constrain to indexes
            content = data_fmts[datatype] % data
        elif datatype == 'header':
            content = fmt.get('header_fmt','%s') % data
        elif datatype == 'stub':
            content = fmt.get('stub_fmt','%s') % data
        elif datatype == 'empty':
            content = fmt.get('empty_cell','')
        else:
            raise ValueError('Unknown cell datatype: %s'%datatype)
        align = self.alignment(output_format, **fmt)
        return pad(content, width, align)
#END class Cell





#########  begin: default formats for SimpleTable  ##############
""" Some formatting suggestions:

- if you want rows to have no extra spacing,
  set colwidths=0 and colsep=''.
  (Naturally the columns will not align.)
- if you want rows to have minimal extra spacing,
  set colwidths=1.  The columns will align.
- to get consistent formatting, you should leave
  all field width handling to SimpleTable:
  use 0 as the field width in data_fmts.  E.g., ::

        data_fmts = ["%#0.6g","%#0.6g","%#0.4g","%#0.4g"],
        colwidths = 14,
        data_aligns = "r",
"""
default_csv_fmt = dict(
    data_fmts = ['%s'],
    data_fmt = '%s',  #deprecated; use data_fmts
    empty_cell = '',
    colwidths = None,
    colsep = ',',
    row_pre = '',
    row_post = '',
    table_dec_above = '',
    table_dec_below = '',
    header_dec_below = '',
    header_fmt = '"%s"',
    stub_fmt = '"%s"',
    title_align = '',
    header_align = 'c',
    data_aligns = "l",
    stubs_align = "l",
    fmt = 'csv',
)

default_html_fmt = dict(
    data_fmts = ['<td>%s</td>'],
    data_fmt = "<td>%s</td>",  #deprecated; use data_fmts
    empty_cell = '<td></td>',
    colwidths = None,
    colsep=' ',
    row_pre = '<tr>\n  ',
    row_post = '\n</tr>',
    table_dec_above=None,
    table_dec_below=None,
    header_dec_below=None,
    header_fmt = '<th>%s</th>',
    stub_fmt = '<th>%s</th>',
    title_align='c',
    header_align = 'c',
    data_aligns = "c",
    stubs_align = "l",
    fmt = 'html',
)

default_txt_fmt = dict(
    data_fmts = ["%s"],
    data_fmt = "%s",  #deprecated; use data_fmts
    empty_cell = '',
    colwidths = None,
    colsep=' ',
    row_pre = '',
    row_post = '',
    table_dec_above='=',
    table_dec_below='-',
    header_dec_below='-',
    header_fmt = '%s',
    stub_fmt = '%s',
    title_align='c',
    header_align = 'c',
    data_aligns = "c",
    stubs_align = "l",
    fmt = 'txt',
)

default_latex_fmt = dict(
    data_fmts = ['%s'],
    data_fmt = '%s',  #deprecated; use data_fmts
    empty_cell = '',
    colwidths = None,
    colsep=' & ',
    table_dec_above = r'\toprule',
    table_dec_below = r'\bottomrule',
    header_dec_below = r'\midrule',
    strip_backslash = True,
    header_fmt = r'\textbf{%s}',
    stub_fmt =r'\textbf{%s}',
    header_align = 'c',
    data_aligns = 'c',
    stubs_align = 'l',
    fmt = 'ltx',
    row_post = r'  \\'
)
default_fmts = dict(
html= default_html_fmt,
htm= default_html_fmt,
txt=default_txt_fmt,
text=default_txt_fmt,
latex=default_latex_fmt,
ltx=default_latex_fmt,
csv=default_csv_fmt
)
#########  end: default formats  ##############
