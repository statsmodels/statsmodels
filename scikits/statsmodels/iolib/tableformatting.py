"""
Summary Table formating
This is here to help keep the formating consistent across the different models
"""

gen_fmt = dict(
        data_fmts = ["%s", "%s", "%s", "%s", "%s"],
        empty_cell = '',
        colwidths = 7, #17,
        colsep='   ',
        row_pre = '  ',
        row_post = '  ',
        table_dec_above='=',
        table_dec_below=None,
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
        data_fmts = ["%s", "%s", "%s", "%s", "%s"],
        empty_cell = '',
        colwidths = 16,
        colsep='   ',
        row_pre = '',
        row_post = '',
        table_dec_above='=',
        table_dec_below=None,
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
        colwidths = 10,
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


# new version
fmt_base = dict(
        data_fmts = ["%s", "%s", "%s", "%s", "%s"],
        empty_cell = '',
        colwidths = 10,
        colsep=' ',
        row_pre = '',
        row_post = '',
        table_dec_above='=',
        table_dec_below='=', #TODO need '=' at the last subtable
        header_dec_below='-',
        header_fmt = '%s',
        stub_fmt = '%s',
        title_align='c',
        header_align = 'r',
        data_aligns = 'r',
        stubs_align = 'l',
        fmt = 'txt'
        )

import copy
fmt_2cols = copy.deepcopy(fmt_base)

fmt2 = dict(
            data_fmts = ["%18s", "-%19s", "%18s", "%19s"], #TODO: 
            colsep=' ',
            colwidths = 18,
            stub_fmt = '-%21s',
            )
fmt_2cols.update(fmt2)

fmt_params = copy.deepcopy(fmt_base)

fmt3 = dict(
            data_fmts = ["%s", "%s", "%8s", "%s", "%23s"],
            )
fmt_params.update(fmt3)

