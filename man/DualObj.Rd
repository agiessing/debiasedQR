\name{dualObj}
\alias{dualObj}

\title{
Objective function of the dual debiasing program.
}

\description{
Returns the the objective function of the dual debiasing program.
}

\usage{
dualObj(X, x, Psi, v, gamma)
}

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

\keyword{Utilities}
\concept{Dual objective function}
