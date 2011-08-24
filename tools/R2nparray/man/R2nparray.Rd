\name{R2nparray}
\alias{R2nparray}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Write R data to file as Numpy Arrays
%%  ~~function to do ... ~~
}
\description{Takes a matrix, scalar, or vector in R and dumps it to a file
as a NumPy array.
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
R2nparray(..., fname, append = FALSE)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{\dots}{Scalars, vectors, or matrices. Can be part of a list or not.
%%     ~~Describe \code{\dots} here~~
}
  \item{fname}{Filename to write the arrays to.
%%     ~~Describe \code{fname} here~~
}
  \item{append}{Whether or not to append to an existing file or overwrite it.
%%     ~~Describe \code{append} here~~
}
}
\details{The names of the arugments are the names of the arrays in the file.
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
\author{Skipper Seabold
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
mdat <- matrix(c(1,2,3, 11,12,13), nrow = 2, ncol=3)
scalar <- 127.5
vec <- c(1,2,3)

R2nparray(mdat=mdat, myscalar=scalar, vec=vec, fname="./numpyarrays")

# or use a list

lst = list(mdat=mdat, myscalar=scalar, vec=vec)
R2nparray(lst, fname="./numpyarrays")
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ IO }
