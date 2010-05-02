"""
SimpleTable was created as part of econpy.

http://code.google.com/p/econpy/
"""

class SimpleTable:
        """Produce a simple ASCII, CSV, HTML, or LaTeX table from a
        rectangular array of data, not necessarily numerical.
        Supports at most one header row,
        which must be the length of data[0] (or +1 if stubs).
        Supports at most one stubs column, which must be the length of data.
        See globals `default_txt_fmt`, `default_csv_fmt`, `default_html_fmt`,
        and `default_ltx_fmt` for formatting options.

        Sample uses::

                mydata = [[11,12],[21,22]]
                myheaders = "Column 1", "Column 2"
                mystubs = "Row 1", "Row 2"
                tbl = text.SimpleTable(mydata, myheaders, mystubs, title="Title")
                print( tbl )
                print( tbl.as_html() )
                # set column specific data formatting
                tbl = text.SimpleTable(mydata, myheaders, mystubs,
                        fmt={'data_fmt':["%3.2f","%d"]})
                print( tbl.as_csv() )
                with open('c:/temp/temp.tex','w') as fh:
                        fh.write( tbl.as_latex_tabular() )
        """
        def __init__(self, data, headers=(), stubs=(), title='', fmt=None,
                csv_fmt=None, txt_fmt=None, ltx_fmt=None, html_fmt=None):
                """
                :Parameters:
                        data : list of lists or 2d array
                                R rows by K columns of table elements
                        headers: tuple
                                sequence of K strings, one per header
                        stubs: tuple
                                sequence of R strings, one per stub
                        fmt : dict
                                formatting options
                        txt_fmt : dict
                                text formatting options
                        ltx_fmt : dict
                                latex formatting options
                        csv_fmt : dict
                                csv formatting options
                        hmtl_fmt : dict
                                hmtl formatting options
                """
                self.raw_data = data
                self.headers = headers
                self.stubs = tuple(str(stub) for stub in stubs)
                self.title = title
                #start with default formatting
                self.txt_fmt = default_txt_fmt
                self.ltx_fmt = default_ltx_fmt
                self.csv_fmt = default_csv_fmt
                self.html_fmt = default_html_fmt
                #substitute any user specified formatting
                if fmt:
                        self.csv_fmt.update(fmt)
                        self.txt_fmt.update(fmt)
                        self.ltx_fmt.update(fmt)
                        self.html_fmt.update(fmt)
                self.csv_fmt.update(csv_fmt or dict())
                self.txt_fmt.update(txt_fmt or dict())
                self.ltx_fmt.update(ltx_fmt or dict())
                self.html_fmt.update(html_fmt or dict())
        def __str__(self):
                return self.as_text()
        def _format_rows(self, tablestrings, fmt_dict):
                """Return: list of strings,
                the formatted table data *including* headers and stubs.
                Note that `tablestrings` is a *rectangular* iterable of strings.
                """
                fmt = fmt_dict['fmt']
                colwidths = self.get_colwidths(tablestrings, fmt_dict)
                cols_aligns = self.get_cols_aligns(fmt_dict)
                colsep = fmt_dict['colsep']
                row_pre = fmt_dict.get('row_pre','')
                row_post = fmt_dict.get('row_post','')
                rows = []
                for row in tablestrings:
                        cols = []
                        for content, width, align in izip(row, colwidths,
                            cols_aligns):
                                content = self.pad(content, width, align)
                                cols.append(content)
                        rows.append( row_pre + colsep.join(cols) + row_post )
                return rows
        def pad(self, s, width, align):
                """Return string padded with spaces,
                based on alignment parameter."""
                if align == 'l':
                        s = s.ljust(width)
                elif align == 'r':
                        s = s.rjust(width)
                else:
                        s = s.center(width)
                return s
        def merge_table_parts(self, fmt_dict=dict()):
                """Return list of lists of strings,
                all table parts merged.
                Inserts stubs and headers into `data`."""
                data = self.format_data(fmt_dict)
                headers = self.format_headers(fmt_dict)
                stubs = self.format_stubs(fmt_dict)
                for i in range(len(stubs)):
                        data[i].insert(0,stubs[i])
                if headers:
                        data.insert(0,headers)
                return data
        def format_data(self, fmt_dict):
                """Return list of lists,
                the formatted data (without headers or stubs).
                Note: does *not* change `self.raw_data`."""
                data_fmt = fmt_dict.get('data_fmt','%s')
                if isinstance(data_fmt, str):
                        result = [[(data_fmt%datum) for datum in row] \
                                for row in self.raw_data]
                else:
                        fmt = cycle( data_fmt )
                        result = [[(fmt.next()%datum) for datum in row] \
                                for row in self.raw_data]
                return result
        def format_headers(self, fmt_dict, headers=None):
                """Return list, the formatted headers."""
                dcols = len(self.raw_data[0])
                headers2fmt = list(headers or self.headers)
                header_fmt = fmt_dict.get('header_fmt') or '%s'
                if self.stubs and len(headers2fmt)==dcols:
                        headers2fmt.insert(0,'')
                return [header_fmt%(header) for header in headers2fmt]
        def format_stubs(self, fmt_dict, stubs=None):
                """Return list, the formatted stubs."""
                stub_fmt = fmt_dict.get('stub_fmt','%s')
                stubs2fmt = stubs or self.stubs
                return [stub_fmt%stub for stub in stubs2fmt]
        def get_colwidths(self, tablestrings, fmt_dict):
                """Return list of int, the column widths.
                Ensure comformable colwidths in `fmt_dict`.
                Other, compute as the max width for each column of
                `tablestrings`.

                Note that `tablestrings` is a rectangular iterable of strings.
                """
                ncols = len(tablestrings[0])
                request_widths = fmt_dict.get('colwidths')
                if request_widths is None:
                        result = [0] * ncols
                else:
                        min_widths = [max(len(d) for d in c) for c in \
                                izip(*tablestrings)]
                        if isinstance(request_widths, int):
                                request_widths = cycle([request_widths])
                        elif len(request_widths) != ncols:
                                request_widths = min_widths
                        result = [max(m,r) for m,r in izip(min_widths,
                            request_widths)]
                return result
        def get_cols_aligns(self, fmt_dict):
                """Return string, sequence of column alignments.
                Ensure comformable data_aligns in `fmt_dict`."""
                dcols = len(self.raw_data[0])  # number of data columns
                has_stubs = bool(self.stubs)
                cols_aligns = fmt_dict.get('cols_aligns')
                if cols_aligns is None or len(cols_aligns) != dcols + has_stubs:
                        if has_stubs:
                                stubs_align = fmt_dict.get('stubs_align') or 'l'
                                assert len(stubs_align)==1
                        else:
                                stubs_align = ''
                        data_aligns = fmt_dict.get('data_aligns') or 'c'
                        if len(data_aligns) != dcols:
                                c = cycle(data_aligns)
                                data_aligns = ''.join(c.next() for _ in \
                                        range(dcols))
                        cols_aligns = stubs_align + data_aligns
                return cols_aligns
        def as_csv(self, **fmt):
                """Return string, the table in CSV format.
                Currently only supports comma separator."""
                #fetch the format, which may just be default_csv_format
                fmt_dict = self.csv_fmt.copy()
                #update format using `fmt`
                fmt_dict.update(fmt)
                return self.as_text(**fmt_dict)
        def as_text(self, **fmt):
                """Return string, the table as text."""
                #fetch the format, which may just be default_txt_format
                fmt_dict = self.txt_fmt.copy()
                fmt_dict.update(fmt)
                #format the 3 table parts (data, headers, stubs)
                #and merge in list of lists
                txt_data = self.merge_table_parts(fmt_dict)
                rows = self._format_rows(txt_data, fmt_dict)
                row0len = len(rows[0])
                begin = ''
                if self.title:
                        begin += self.pad(self.title, row0len,
                                fmt_dict.get('title_align','c'))
                #decoration above the table, if desired
                table_dec_above = fmt_dict['table_dec_above']
                if table_dec_above:
                        begin += "\n" + table_dec_above*row0len
                if self.headers:
                        hdec = fmt_dict['header_dec_below']
                        if hdec:
                                rows[0] = rows[0] + "\n" + hdec*row0len
                below = fmt_dict['table_dec_below']
                end = (below*row0len + "\n") if below else ''
                return begin + '\n' + '\n'.join(rows) + '\n' + end
        def as_html(self, **fmt):
                """Return string, the table as an HTML table."""
                fmt_dict = self.html_fmt.copy()
                fmt_dict.update(fmt)
                datastrings = self.merge_table_parts(fmt_dict)
                rows = self._format_rows(datastrings, fmt_dict)
                begin = "<table class='%s'>\n"%"simpletable"
                if self.title:
                        begin += "<caption>%s</caption>\n"%(self.title,)
                end = r'</table>'
                return begin + '\n'.join(rows) + "\n" + end
        def as_latex_tabular(self, **fmt):
                '''Return string, the table as a LaTeX tabular environment.
                Note: will equire the booktabs package.'''
                fmt_dict = self.ltx_fmt.copy()
                fmt_dict.update(fmt)
                """
                if fmt_dict['strip_backslash']:
                        ltx_stubs = [stub.replace('\\',r'$\backslash$') for stub in self.stubs]
                        ltx_headers = [header.replace('\\',r'$\backslash$') for header in self.headers]
                        ltx_headers = self.format_headers(fmt_dict, ltx_headers)
                else:
                        ltx_headers = self.format_headers(fmt_dict)
                ltx_stubs = self.format_stubs(fmt_dict, ltx_stubs)
                """
                datastrings = self.merge_table_parts(fmt_dict)
                #this just formats output; add real colwidths?
                rows = self._format_rows(datastrings, fmt_dict)
                begin = r'\begin{tabular}{%s}'%(self.get_cols_aligns(fmt_dict))
                above = fmt_dict['table_dec_above']
                if above:
                        begin += "\n" + above + "\n"
                if self.headers:
                        hdec = fmt_dict['header_dec_below']
                        if hdec:
                                rows[0] = rows[0] + "\n" + hdec
                end = r'\end{tabular}'
                below = fmt_dict['table_dec_below']
                if below:
                        end = below + "\n" + end
                return begin + '\n'.join(rows) + "\n" + end


#########  begin: default formats for SimpleTable  ##############
default_csv_fmt = dict(
                data_fmt = '%s',
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
                stubs_align = "l",
                data_aligns = "l",
                fmt = 'csv',
                )

default_html_fmt = dict(
                data_fmt = "<td>%s</td>",
                colwidths = None,
                colsep='',
                row_pre = '<tr>\n  ',
                row_post = '\n</tr>',
                table_dec_above=None,
                table_dec_below=None,
                header_dec_below=None,
                header_fmt = '<th>%s</th>',
                stub_fmt = '<th>%s</th>',
                title_align='c',
                data_aligns = "c",
                stubs_align = "l",
                fmt = 'html',
                )

default_txt_fmt = dict(
                data_fmt = "%s",
                colwidths = 0,
                colsep=' ',
                row_pre = '',
                row_post = '',
                table_dec_above='=',
                table_dec_below='-',
                header_dec_below='-',
                header_fmt = '%s',
                stub_fmt = '%s',
                title_align='c',
                data_aligns = "c",
                stubs_align = "l",
                fmt = 'txt',
                )

default_ltx_fmt = dict(
                data_fmt = "%s",
                colwidths = 0,
                colsep=' & ',
                table_dec_above = r'\toprule',
                table_dec_below = r'\bottomrule',
                header_dec_below = r'\midrule',
                strip_backslash = True,
                header_fmt = "\\textbf{%s}",
                stub_fmt = "\\textbf{%s}",
                data_aligns = "c",
                stubs_align = "l",
                fmt = 'ltx',
                row_post = r'  \\'
                )
#########  end: default formats  ##############

