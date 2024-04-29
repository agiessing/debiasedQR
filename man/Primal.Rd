\name{primal}
\alias{primal}

\title{
  Primal rank-score debiasing program
}

\description{
  Solves primal debiasing program using CVXR.
}

\usage{
primal(X, x, Psi, gamma, quiet = TRUE)
}

\arguments{
  \item{X}{Covariates.
}
  \item{x}{Querry point at which to evaluate the conditional quantile function of Y given X = x.
}
  \item{Psi}{Estimate of the density matrix.
}
  \item{gamma}{Tuning parameter for primal debiasing program.
}
  \item{quiet}{
  If quiet = TRUE, then no messages will be displayed.
}
}

\value{
Value of the primal objective function. Returns NA if the program is infeasible.
}

\references{
  Giessing, A. and Wang, J. (2023). Debiased inference on heterogeneous quantile treatment effects with regression rank-scores. \emph{Journal of the Royal Statistical Society: Series B (Statistical Methodology)}, 85(5): 1561–1588.  \doi{10.1093/jrsssb/qkad075}
}

\author{
  Alexander Giessing, \email{giessing@umich.edu}
}

\concept{Primal debiasing program}
\concept{CVXR}
