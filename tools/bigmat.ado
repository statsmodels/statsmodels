/* 
Put a variable or variable list into a Stata matrix even if it's bigger than 
matsize.

Usage:

If you have some variables xb1 and xb2

bigmat xb1 xb2, mat(new_mat)

You can check for new_mat by using

mat dir
*/
capture program drop bigmat
program bigmat
    version 11.0
    syntax namelist(min=1), MAT(str)
    mata: _bigmat("`namelist'", "`mat'")
end

mata:

void function _bigmat(string varlist, string Bigmat)
{
    /* creates a matrix with name givem by bigmat */
    real matrix Big
    real matrix Data

    V = st_varindex(tokens(varlist))
    Data = J(1,1,0)
    st_view(Data,.,V)
    Big = J(rows(Data), cols(Data), 0)
    for(i=1; i<=rows(Data); i++) {
        for(j=1; j<=cols(Data); j++) {
            Big[i,j] = Data[i,j]
        }
    }
    st_matrix(Bigmat, Big)
}

end
