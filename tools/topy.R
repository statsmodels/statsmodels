
cat_items <- function(object, prefix="", blacklist=NULL, trans=list())    {
#print items into python expression for defining variables
#example cat_items(names(object), blacklist=c("eq"))
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
	newname = trans[[name]]
      if (!is.null(newname)) {
          name_ = newname}
	name_ = paste(prefix, name_, sep="")

	if (is.numeric(item)) {
		if (!is.null(names(item))) {    #named list, class numeric ?
               mkarray(as.matrix(item), name_); cat("\n")
                  if (!is.null(dimnames(item))) write_dimnames(item, prefix=name_)
		}
		else if (class(item) == 'matrix') {
               mkarray(item, name_); cat("\n")
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
