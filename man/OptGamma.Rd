\name{optGamma}
\alias{optGamma}

\title{
  Optimal tuning parameter
}

\description{
  Finds the optimal tuning parameter for primal and dual program from a list of
cross-validated losses.
}

\usage{
optGamma(dual_loss, gamma_lst, cv_fold, cv_rule, robust = TRUE)
}

\arguments{
  \item{dual_loss}{
  List of values of the cross-validated dual objective function.
}
  \item{gamma_lst}{
  List of tuning parameters "\eqn{\gamma > 0}" for the primal debiasing program from which to choose the optimal value. The default value NULL results in a list with 41 equally spaced values from 0 to the maximum norm of the querry point x. If manually set, end point should always be the maximum norm of the querry point x.
}
  \item{cv_fold}{
  Number of folds of the cross-validation procedure.
}
  \item{cv_rule}{Cross-validation rule to be applied to .drqcv object to select optimal tuning parameter "\eqn{gamma >0}". Choices are `1se', `mincv', and `minfeas'. For details see documentatio of optGamma().
}
  \item{robust}{
  If robust = TRUE, then median and mean absolute deviation (instead of mean and standard deviation) are used when finding the optimal tuning parameter.
}
}

\value{
Optimal tuning parameter "\eqn{\gamma > 0}" of the primal problem chosen via cross-validation.
}

\references{
  Giessing, A. and Wang, J. (2023). Debiased inference on heterogeneous quantile treatment effects with regression rank-scores. \emph{Journal of the Royal Statistical Society: Series B (Statistical Methodology)}, 85(5): 1561–1588. \url{https://doi.org/10.1093/jrsssb/qkad075}
}

\author{
  Alexander Giessing, \email{giessing@umich.edu}
}

\keyword{Utilities}
\concept{Optimal cross-validated tuning parameter}
