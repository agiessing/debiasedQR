\name{densityMatrix}
\alias{densityMatrix}

\title{Estimate of the density matrix
}

\description{
Various estimates of the density matrix (`sparsity' in QR lingo). The density matrix is a diagonal matrix with conditional densities of Y given X evaluated at the conditional quantile of Y given X = X_i i.e. "\eqn{f_{Y \mid X}(X_i'\beta_0 \mid X_i)}" for   "\eqn{i = 1, \ldots, n}".
}

\usage{
densityMatrix(Y, X, beta, tau, density="nid", sparsity, screening = TRUE)
}

\arguments{
  \item{Y}{
  Responses.
}
 \item{X}{
  Covariates.
}
 \item{tau}{
  Quantile level "\eqn{\tau \in (0,1)}".
}
 \item{beta}{
  Estiamte of the conditional quantile regression vector based on Y and X.
}
 \item{density}{
  Options for estimating the density matrix: "nid" (non-iid data), "iid" (iid data), and "iidGaussian" (iid Gaussian data).
}
 \item{sparsity}{
  Only relevant if screening = TRUE. Upper bound on sparsity. Default value NULL.
}
 \item{screening}{
  If screening = TRUE, then a version of the (iterative) sure indendence screening for linear quantile regression is used to select covariates and then estimate the density matrix based on the selected covariates. If screening = FALSE, then the provided value for beta is used to estimate the density matrix.
}
}

\value{
  Estimate of the density matrix.
}

\references{
  Giessing, A. and Wang, J. (2023). Debiased inference on heterogeneous quantile treatment effects with regression rank-scores. \emph{Journal of the Royal Statistical Society: Series B (Statistical Methodology)}, 85(5): 1561–1588.  \doi{10.1093/jrsssb/qkad075}
}

\author{
  Alexander Giessing, \email{giessing@umich.edu}
}

\keyword{Utilities}
\concept{Density matrix}
