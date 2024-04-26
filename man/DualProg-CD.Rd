\name{dualCD}
\alias{dualCD}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
 Coordinate descent algorithm to solve the dual debiasing program
}
\description{
  Solves the dual debiasing program for a giving tuning parameter via coordinate descent.
}
\usage{
dualCD(X, x, Psi=NULL, v_init=NULL, gamma, eps=1e-6, max_iter=5000)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{X}{Covariates.
}
  \item{x}{Querry point at which to evaluate the conditional quantile function of Y given X = x.
}
  \item{Psi}{Estimate of the density matrix.
}
  \item{v_init}{
  Initialized value of the solution to the dual debiasing progra
}
  \item{gamma}{Tuning parameter for primal debiasing program.
}
  \item{eps}{
 Tolerance level to determine convergence of the coordinate descent algorithm. Default value is 1e-06.
}
  \item{max_iter}{
   Maximum number of iterations tge ADMM algorithm. Default value is 5000.
}
}

\value{
  Solution to the dula debiasing program.
}
\references{
  Giessing, A. and Wang, J. (2023). Debiased inference on heterogeneous quantile treatment effects with regression rank-scores. \emph{Journal of the Royal Statistical Society: Series B (Statistical Methodology)}, 85(5): 1561–1588. \url{https://doi.org/10.1093/jrsssb/qkad075}
}

\author{
  Alexander Giessing, \email{giessing@umich.edu}
}

\examples{
##---- Should be DIRECTLY executable !! ----
##-- ==>  Define data, use random,
##--	or do  help(data=index)  for the standard data sets.

## The function is currently defined as
function (x)
{
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory (show via RShowDoc("KEYWORDS")):
% \keyword{ ~kwd1 }
% \keyword{ ~kwd2 }
% Use only one keyword per line.
% For non-standard keywords, use \concept instead of \keyword:
 \concept{Dual debiasing program}
 \concept{Coordinate descent algorithm}
% Use only one concept per line.
