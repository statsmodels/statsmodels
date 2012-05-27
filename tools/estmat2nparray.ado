*
* Save estimation results and matrices to a Python module
*
* Based on mat2nparray by Skipper
* Changes by Josef
*
* changes
* -------
* write also column and row names of matrices to py module
* replace missing values by np.nan in matrices
* make namelist optional
* add estimation results from e(), e(scalars) and e(macros), not the matrices in e
* make estimation result optional

capture program drop estmat2nparray
program define estmat2nparray
    version 11.0
    syntax [namelist(min=1)], SAVing(str) [ Format(str) APPend REPlace NOEst]
    if "`format'"=="" local format "%16.0g"
    local saving: subinstr local saving "." ".", count(local ext)
    if !`ext' local saving "`saving'.py"
    tempname myfile
    file open `myfile' using "`saving'", write text `append' `replace'
    file write `myfile' "import numpy as np" _n _n

	/* get results from e()*/
	if "`noest'" == "" {
		file write `myfile' "est = dict(" _n

		local escalars : e(scalars)
		foreach ii in `escalars'{
			file write `myfile' "           " "`ii'" " = " "`e(`ii')'" "," _n
		}

		local emacros : e(macros)
		foreach ii in `emacros' {
			file write `myfile' "           " "`ii'" " = "  `"""'   "`e(`ii')'" `"""' ","   _n
		}
		file write `myfile' "          )" _n _n
	}
	/* end write e()*/


    foreach mat of local namelist {
        mkarray `mat' `myfile' `format'
    }
    file write `myfile' "class Bunch(dict):" _n
    file write `myfile' "    def __init__(self, **kw):" _n
    file write `myfile' "        dict.__init__(self, kw)" _n
    file write `myfile' "        self.__dict__  = self" _n _n _n
    file write `myfile' "results = Bunch(" _n
    foreach mat of local namelist {
        file write `myfile' "                `mat'=`mat', " _n
		file write `myfile' "                `mat'_colnames=`mat'_colnames, " _n
		file write `myfile' "                `mat'_rownames=`mat'_rownames, " _n
    }

	if "`noest'" == "" {
		file write `myfile' "                **est" _n
	}
    file write `myfile' "                )" _n _n
file close `myfile'
end

capture program drop mkarray
program define mkarray

    args mat myfile fmt
    local nrows = rowsof(`mat')
    local ncols = colsof(`mat')
    local i 1
    local j 1
    file write `myfile' "`mat' = np.array(["
    local justifyn = length("`mat' = np.array([")
    forvalues i=1/`nrows' {
        forvalues j = 1/`ncols' {
            if `i' > 1 | `j' > 1 { // then we need to indent
                forvalues k=1/`justifyn' {
                    file write `myfile' " "
                }
            }
            if `i' < `nrows' | `j' < `ncols' {
			    if mi(`mat'[`i',`j']) {
					file write `myfile' "np.nan" ", " _n
				}
				else {
					file write `myfile' `fmt' (`mat'[`i',`j']) ", " _n
				}
            }
            else {
				if mi(`mat'[`i',`j']) {
					file write `myfile' "np.nan"
				}
				else {
					file write `myfile' `fmt' (`mat'[`i',`j'])
				}
            }
        }
    }

    if `nrows' == 1 | `ncols' == 1 {
        file write `myfile' "])" _n _n
    }
    else {
        file write `myfile' "]).reshape(`nrows',`ncols')" _n _n
    }
	capture drop colnms
	local colnms: coln `mat'
	*gen str `col_names' = "`colnms'"
	*file write `myfile' "# " `col_names' _n _n
	capture file write `myfile' "`mat'_colnames = '" "`colnms'" "'.split()" _n _n

	capture drop colnms
	local rownms: rown `mat'
	capture file write `myfile' "`mat'_rownames = '" "`rownms'" "'.split()" _n _n
end
