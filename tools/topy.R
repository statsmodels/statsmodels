
cat_items <- function(object, prefix="", blacklist=NULL, trans=list())    {
    #print content (names) of object into python expressions for defining variables
    #
    #Parameters
    #----------
    #object : object with names attribute
    #   names(object) will be written as python assignments
    #prefix : string
    #   string that is prepended to the variable names
    #blacklist : list of strings
    #   names that are in the blacklist are ignored
    #trans : named list (dict_like)
    #   names that are in trans will be replaced by the corresponding value
    #
    #example cat_items(fitresult, blacklist=c("eq"))
    #
    #currently limited type inference, mainly numerical

    items = names(object)
    for (name in items) {
        if (is.element(name, blacklist)) next
        #cat(name); cat("\n")
        item = object[[name]]

        #fix name
        #Skipper's sanitize name
        name_ <- gsub("\\.", "_", name) # make name pythonic
        #translate name
        newname = trans[[name_]]   #translation table on sanitized names
        if (!is.null(newname)) {
            name_ = newname
        }
        name_ = paste(prefix, name_, sep="")

        if (is.numeric(item)) {
            if (!is.null(names(item))) {    #named list, class numeric ?
               mkarray2(as.matrix(item), name_);
                if (!is.null(dimnames(item))) write_dimnames(item, prefix=name_)
            }
            else if (class(item) == 'matrix') {
               mkarray2(item, name_);
                if (!is.null(dimnames(item))) write_dimnames(item, prefix=name_)
            }
            else if (class(item) == 'numeric') {   #scalar
                  cat(name_); cat(" = "); cat(item); cat("\n")
            }
        }
        else if (is.character(item)) {
            #assume string doesn't contain single quote
            cat(name_); cat(" = '"); cat(item); cat("'\n")
        }
        else {
            cat(name_); cat(" = '''"); cat(deparse(item)); cat("'''"); cat("\n")
        }

    }
} #end function

write_dimnames <- function(mat, prefix="") {
    if (prefix != "") {
        prefix = paste(prefix, "_", sep="")
    }
    dimn = list("rownames", "colnames", "thirdnames")  #up to 3 dimension ?
    for (ii in c(1:length(dimnames(mat)))) {
        cat(paste(prefix, dimn[[ii]], sep=""))
        cat (" = [");
        for (dname in dimnames(mat)[[ii]]) {
            cat("'"); cat(dname); cat("', ")
        }
        cat("]\n")
    }
}

write_header <-function() {
    cat("import numpy as np\n\n")
    cat("class Bunch(dict):\n")
    cat("    def __init__(self, **kw):\n")
    cat("        dict.__init__(self, kw)\n")
    cat("        self.__dict__  = self\n\n")
}


mkhtest <- function(ht, name, distr="f") {
    #function to write results of a statistical test of class htest to a python dict
    #
    #Parameters
    #----------
    #ht : instance of ht
    #   return of many statistical tests
    #name : string
    #   name of variable that holds results dict
    #distr : string
    #   distribution of the test statistic
    #
    cat(name); cat(" = dict(");
    cat("statistic="); cat(ht$statistic); cat(", ");
    cat("pvalue="); cat(ht$p.value); cat(", ");
    cat("parameters=("); cat(ht$parameter, sep=","); cat(",), ");
    cat("distr='"); cat(distr); cat("'");
    cat(")");
    cat("\n\n")
}

mkarray2 <- function(X, name) {
    indent = "    "
    cat(name); cat(" = np.array([\n"); cat(X, sep=", ", fill=76, labels=indent); cat(indent); cat("])")
    if (is.matrix(X)) {
        i <- as.character(nrow(X))
        j <- as.character(ncol(X))
        cat(".reshape("); cat(i); cat(","); cat(j); cat(", order='F')")
    }
    cat("\n")
}
