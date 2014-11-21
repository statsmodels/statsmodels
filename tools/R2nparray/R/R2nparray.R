#' Print R object contents to python expression
#'
#' @param object with names attribute names(object) will be written as python assignments
#' @param prefix string string that is prepended to the variable names
#' @param blacklist list of strings names that are in the blacklist are ignored
#' @param trans named list (dict_like) names that are in trans will be replaced by the corresponding value
#' @param strict skip content that cannot be pretty-printed. Otherwise, will print as str(deparse(x))
#' @note currently limited type inference, mainly numerical
#' @export
#' @examples
#' mod = lm(rnorm(100) ~ rnorm(100))
#' cat_items(mod)
cat_items <- function(object, prefix="", strict=TRUE, trans=list(),
                      blacklist=NULL){
    out = convert_items(object, prefix, strict, trans, blacklist)
    cat(out)
}

#' Same as cat_items() except it returns a string instead of printing it
#'
#' @export
#' @examples
#' mod = lm(rnorm(100) ~ rnorm(100))
#' convert_items(mod)
convert_items <- function(object, prefix="", strict=TRUE, trans=list(),
                          blacklist=NULL){
    out = list()
    for (n in names(object)){
        if (n %in% blacklist) next
        newname = sanitize_name(n)
        if (newname %in% names(trans)){
            newname = trans[[newname]]
        }
        newname = paste(prefix, newname, sep='')
        tmp = try(convert(object[[n]], name=newname, strict=strict))
        if (class(tmp) != 'try-error'){
            out[[newname]] = tmp
        }
    }
    out = paste(out, collapse='\n')
    return(out)
}

convert <- function (x, name='default', strict=TRUE) {
       UseMethod("convert", x)
}

convert.default <- function(X, name='default', strict=TRUE){
    head = paste(name, '= ')
    mid = paste(deparse(X), collapse='\n')
    out = paste(head, "'''", mid, "'''")
    if(!strict){
        return(out)
    }
}

convert.data.frame <- function(X, name='default', strict=TRUE){
    X = as.matrix(X)
    out = convert(X, name=name)
    return(out)
}

convert.numeric <- function(X, name='default', strict=TRUE) {
    if (length(X) > 1){
        head = paste(name, '= np.array([\n')
        mid = strwrap(paste(X, collapse=', '), width=76, prefix='    ')
        mid = paste(mid, collapse='\n')
        tail = "\n    ])"
        if (is.matrix(X)) {
            i <- nrow(X)
            j <- ncol(X)
            tail = paste(tail, ".reshape(", i, ", ", j, ", order='F')\n", sep='')
        }
        out = paste(head, mid, tail, collapse='\n') 
    }else{
        out = paste(name, '=', X)
    }
    return(out)
}

convert.character <- function(X, name='default', strict=TRUE) {
    if (length(X) > 1){
        head = paste(name, '= np.array([\n')
        mid = paste("'", X, "'", sep='')
        mid = paste(mid, collapse=', ')
        tail = "\n    ])"
        out = paste(head, mid, tail)
    }else{
        out = paste(name, " = '", X, "'", sep='')
    }
    return(out)
}

sanitize_name <- function(name) {
    #"[%s]" % "]|[".join(map(re.escape, list(string.punctuation.replace("_","")
    punctuation <-  '[\\!]|[\\"]|[\\#]|[\\$]|[\\%]|[\\&]|[\\\']|[\\(]|[\\)]|[\\*]|[\\+]|[\\,]|[\\-]|[\\.]|[\\/]|[\\:]|[\\;]|[\\<]|[\\=]|[\\>]|[\\?]|[\\@]|[\\[]|[\\\\]|[\\]]|[\\^]|[\\`]|[\\{]|[\\|]|[\\}]|[\\~]'
    # handle spaces,tabs,etc. and periods specially
    name <- gsub("[[:blank:]\\.]", "_", name)
    name <- gsub(punctuation, "", name)
    return(name)
}

get_dimnames <- function(mat, prefix="", asstring=FALSE) {
    dimension_names = c('rownames', 'colnames', 3:100)
    dimn = dimnames(mat)
    if(is.null(dimn)){
        dimn = names(mat)
        if(is.null(dimn)){
            dimn = NULL
        }else{
            dimn = list(dimn)
        }
    }
    for (i in 1:length(dimn)){
        pref = paste(prefix, dimension_names[i], sep='')
        dimn[[i]] = sapply(dimn[[i]], sanitize_name)
        dimn[[i]] = paste(dimn[[i]], collapse="', '")
        dimn[[i]] = paste(pref, " = ['", dimn[[i]], "']", sep='')
    }
    dimn = paste(dimn, collapse='\n')[1]
    return(dimn)
}

# Dictionary of lambda functions
d = list()
d$df_resid = function(x) x$df.residual
d$deviance = function(x) x$deviance
d$null_deviance = function(x) x$null.deviance
d$rank = function(x) x$rank
d$aic = function(x) x$aic
d$nobs = function(x) dim(model.matrix(x))[1]
d$conf_int = function(x) as.matrix(confint(x))
d$resid = function(x) resid(x)
d$predict = function(x) predict(x)
d$fittedvalues = function(x) predict(x)
d$params = function(x) summary(x)$coefficients[,1]
d$bse = function(x) summary(x)$coefficients[,2]
d$tvalues = function(x) summary(x)$coefficients[,3]
d$pvalues = function(x) summary(x)$coefficients[,4]
d$rsquared = function(x) summary(x)$r.squared[1]
d$rsquared_adj = function(x) summary(x)$r.squared[2]
d$fvalue = function(x) summary(x)$fstatistic$statistic
d$f_pvalue = function(x) summary(x)$fstatistic$p.value
d$fstatistic = function(x) summary(x)$fstatistic

#' Applies functions to an object (skips on error)
#'
#' @param object an object on which we want to apply the lambda functions
#' @param lambdas a named list of lambda functions
#' @export
#' @examples
#' mod = lm(rnorm(100) ~ rnorm(100))
#' cat_items(apply_functions(mod))
apply_functions = function(object, lambdas=d){
    out = list()
    for (n in names(lambdas)){
        tmp = try(lambdas[[n]](object))
        if (class(tmp) != 'try-error'){
            out[[n]] = tmp
        }
    }
    return(out)
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

header = "
import numpy as np

class Bunch(dict):
    def __init__(self, **kw):
        dict.__init__(self, kw)
        self.__dict__  = self

"
write_header = function(){
    cat(header)
}

# Example
#library(plm)
#library(systemfit)
#data(Grunfeld)
#panel <- plm.data(Grunfeld, c('firm','year'))
#SUR <- systemfit(inv ~ value + capital, method='SUR',data=panel)

##translation table for names  (could be dict in python)
#translate = list(coefficients="params",
                 #coefCov="cov_params",
                 #residCovEst="resid_cov_est",
                 #residCov="resid_cov",
                 #df_residual="df_resid",
                 #df_residual_sys="df_resid_sys",
                 ##nCoef="k_vars",    #not sure about this
                 #fitted_values="fittedvalues"
                 #)

#fname = "tmp_sur_0.py"
#append = FALSE #TRUE
#sink(file=fname, append=append) #redirect output to file
#cat(header)
#cat("\nsur = Bunch()\n")

#cat_items(SUR, prefix="sur.", blacklist=c("eq", "control"), trans=translate)
#equations = SUR[["eq"]]
#for (ii in c(1:length(equations))) {
    #equ_name = paste("sur.equ", ii, sep="")
    #cat("\n\n", equ_name, sep=""); cat(" = Bunch()\n")
    #cat_items(equations[[ii]], prefix=paste(equ_name, ".", sep=""), trans=translate)
#}
#sink()
