\name{drqcv}
\alias{drqcv}

\title{
Rank-score debiasing program with cross-validation
}
\description{
Solves primal and dual debiasing program using cross-validation and returns cross-validated losses (which can be used to determine the optimal tuning parameter).
}

\usage{
drqcv(
  Y,
  X,
  x,
  tau,
  density = "nid",
  sparsity = NULL,
  cv_fold = 5,
  gamma_lst = NULL,
  screening = TRUE,
  max_iter = 500,
  tol = c(1e-6, 1e-6),
  algo ="CD",
  parallel=FALSE
)
}

\arguments{
  \item{Y}{Responses.
}
  \item{X}{Covariates.
}
  \item{x}{Querry point at which to evaluate the conditional quantile function of Y given X = x.
}
  \item{tau}{Quantile level "\eqn{\tau \in (0,1)}".
}
  \item{density}{Options for estimating the density matrix: "nid" (non-iid data), "iid" (iid data), and "iidGaussian" (iid Gaussian data).
}
  \item{sparsity}{Only relevant if screening = TRUE. Upper bound on sparsity. Default value NULL.
}
  \item{cv_fold}{Cross-validation rule to be applied to .drqcv object to select optimal tuning parameter "\eqn{gamma >0}". Choices are `1se', `mincv', and `minfeas'. For details see documentatio of optGamma().
}
  \item{gamma_lst}{List of tuning parameters "\eqn{\gamma > 0}" for the primal debiasing program from which to choose the optimal value. The default value NULL results in a list with 41 equally spaced values from 0 to the maximum norm of the querry point x. If manually set, end point should always be the maximum norm of the querry point x.
}
  \item{screening}{If screening = TRUE, then a version of the (iterative) sure independence screening for linear quantile regression is used to select covariates and then estimate the density matrix based on the selected covariates. If screening = FALSE, then the provided value for beta is used to estimate the density matrix.
}
  \item{max_iter}{
Maximum number of iterations of the coordinate descent or alternating direction of multiplier method algorithm. Default value is 500.
}
  \item{tol}{
  Only relevant if algo = "ADMM". Default value is c(1e-6, 1e-6).
}
  \item{algo}{
  Algorithm for solving the dual program. Options are "CD" (coordinate descent) and "ADMM" (alternating direction of multiplier method).
}
  \item{parallel}{
  If parallel == TRUE, cross-validation is solved in parallel using foreach().
}
}

\value{
  \item{Y}{Returns responses.
}
  \item{X}{Returns covariates.
}
  \item{x}{Returns querry point at which to evaluate the conditional quantile function of Y given X = x.
}
  \item{tau}{Returns quantile level.
}
\item{loss}{List of values of the cross-validated dual objective function.
}
\item{gamma}{Returns the porovided list of tuning parameters "\eqn{\gamma > 0}".
}
\item{fold}{Number of folds of the cross-validation procedure.
}
\item{density}{Returns chosen option for estimating the density matrix.
}
\item{sparsity}{Returns sparsity level if screening = TRUE.
}
\item{iter}{Returns maximum number of iterations of the coordinate descent or alternating direction of multiplier method algorithm.
}
\item{algo}{Returns chosen akgorithm to solve the dual debiasing program.
}
}

\references{
  Giessing, A. and Wang, J. (2023). Debiased inference on heterogeneous quantile treatment effects with regression rank-scores. \emph{Journal of the Royal Statistical Society Series B: Statistical Methodology}, 85(5): 1561–1588.
  \doi{10.1093/jrsssb/qkad075}
}

\author{
  Alexander Giessing, \email{giessing@umich.edu}
}

\keyword{Utilities}
\concept{Cross-validation}
