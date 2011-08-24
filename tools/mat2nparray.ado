capture program drop mat2array
program define mat2array 
    version 11.2
    syntax namelist(min=1), SAVing(str) [ Format(str) APPend REPlace ]
    if "`format'"=="" local format "%16.0g"
    local saving: subinstr local saving "." ".", count(local ext)
    if !`ext' local saving "`saving'.py"
    tempname myfile
    file open `myfile' using "`saving'", write text `append' `replace'
    file write `myfile' "import numpy as np" _n _n

    foreach mat of local namelist {
        mkarray `mat' `myfile' `format'
    }
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
                file write `myfile' `fmt' (`mat'[`i',`j']) ", " _n
            }
            else {
                file write `myfile' `fmt' (`mat'[`i',`j'])
            }
        }
    } 

    if `nrows' == 1 | `ncols' == 1 {
        file write `myfile' "])" _n _n
    }
    else {
        file write `myfile' "]).reshape(`nrows',`ncols')" _n _n
    }
end