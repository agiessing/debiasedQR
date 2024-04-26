\name{densityMatrix}
\alias{densityMatrix}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Estimate of the density matrix
}
\description{
Various estimates of the density matrix (`sparsity' in QR lingo). The density matrix is a diagonal matrix with conditional densities of Y given X evaluated at the conditional quantile of Y given X = X_i i.e. "\eqn{f_{Y \mid X}(X_i'\beta_0 \mid X_i)}" for   "\eqn{i = 1, \ldots, n}".
}
\usage{
densityMatrix(Y, X, beta, tau = tau, density="nid", sparsity, screening = TRUE)
}
%- maybe also 'usage' for other objects documented here.
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
\concept{Density matrix}
% \concept{ ~cpt2 }
% Use only one concept per line.
