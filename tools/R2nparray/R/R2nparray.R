# This function respects the digits option

mkarray <- function(X, name) {
    cat(name); cat(" = np.array(["); cat(X, sep=","); cat("])")
    if (is.matrix(X)) {
    i <- as.character(nrow(X))
    j <- as.character(ncol(X))
    cat(".reshape("); cat(i); cat(","); cat(j); cat(", order='F')")
    }
    cat("\n\n")
}

R2nparray <- function(..., fname, append=FALSE) {
    if (!is.list(...)) {
        to_write <- list(...)
    }
    else {
        to_write <- (...)
    }
    sink(file=fname, append=append)
    # assumes appended file already imports numpy
    if (file.info(fname)$size == 0) {
        cat("import numpy as np\n\n")
        }
    for (i in c(1:length(to_write))) {
        name <- names(to_write)[i]
        X <- to_write[[i]]
        name <- gsub("\\.", "_", name) # make name pythonic
        mkarray(X=X, name=name)
    }
    sink()   
}



#fname <- "RResults.py"
#R2array(A=A,B=B,params=params,fname=fname)
#also takes a lsit
#R2array(list(A=A,B=B,params=params),fname=fname)
