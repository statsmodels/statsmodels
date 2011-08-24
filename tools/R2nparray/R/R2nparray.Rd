\name{R2nparray}
\alias{R2nparray}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
%%  ~~function to do ... ~~
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
R2nparray(..., fname, append = FALSE)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{\dots}{
%%     ~~Describe \code{\dots} here~~
}
  \item{fname}{
%%     ~~Describe \code{fname} here~~
}
  \item{append}{
%%     ~~Describe \code{append} here~~
}
}
\details{
%%  ~~ If necessary, more details than the description above ~~
}
\value{
%%  ~Describe the value returned
%%  If it is a LIST, use
%%  \item{comp1 }{Description of 'comp1'}
%%  \item{comp2 }{Description of 'comp2'}
%% ...
}
\references{
%% ~put references to the literature/web site here ~
}
\author{
%%  ~~who you are~~
}
\note{
%%  ~~further notes~~
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
}
\examples{
##---- Should be DIRECTLY executable !! ----
##-- ==>  Define data, use random,
##--	or do  help(data=index)  for the standard data sets.

## The function is currently defined as
function(..., fname, append=FALSE) {
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
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
