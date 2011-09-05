"""
Summary Table formating
This is here to help keep the formating consistant across the different models
"""

gen_fmt = dict(
        data_fmts = ["%s", "%s", "%s", "%s", "%s"],
        empty_cell = '',
        colwidths = 7, #17,
        colsep='   ',
        row_pre = '  ',
        row_post = '  ',
        table_dec_above='=',
        table_dec_below='',
        header_dec_below=None,
        header_fmt = '%s',
        stub_fmt = '%s',
        title_align='c',
        header_align = 'r',
        data_aligns = "r",
        stubs_align = "l",
        fmt = 'txt'
        )
        # Note table_1l_fmt over rides the below formating unless it is not 
        # appended to table_1l
fmt_1_right = dict(
        data_fmts = ["%s", "%s", "%s", "%s", "%S"],
        empty_cell = '',
        colwidths = 16,
        colsep='   ',
        row_pre = '',
        row_post = '',
        table_dec_above='=',
        table_dec_below='',
        header_dec_below=None,
        header_fmt = '%s',
        stub_fmt = '%s',
        title_align='c',
        header_align = 'r',
        data_aligns = "r",
        stubs_align = "l",
        fmt = 'txt'
        )

fmt_2 = dict(
        data_fmts = ["%s", "%s", "%s", "%s"],
        empty_cell = '',
        colwidths = 8,
        colsep=' ',
        row_pre = '  ',
        row_post = '   ',
        table_dec_above='=',
        table_dec_below='=',
        header_dec_below='-',
        header_fmt = '%s',
        stub_fmt = '%s',
        title_align='c',
        header_align = 'r',
        data_aligns = 'r',
        stubs_align = 'l',
        fmt = 'txt'
        )

