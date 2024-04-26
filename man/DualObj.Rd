\name{dualObj}
\alias{dualObj}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Objective function of the dual debiasing program.
}
\description{
Returns the the objective function of the dual debiasing program.
}
\usage{
dualObj(X, x, Psi, v, gamma)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{X}{
  Covariates.
}
  \item{x}{
  Querry point at which to evaluate the conditional quantile function of Y given X = x.
}
  \item{Psi}{
  Estimate of the density matrix.
}
  \item{v}{
  Solution to the dual debiasing program.
}
  \item{gamma}{Tuning parameter for primal debiasing program.
}
}

\value{
 Value of the dual objective function.
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
\keyword{Utilities}
% \keyword{ ~kwd2 }
% Use only one keyword per line.
% For non-standard keywords, use \concept instead of \keyword:
 \concept{Dual objective function}
% \concept{ ~cpt2 }
% Use only one concept per line.
