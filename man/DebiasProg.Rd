\name{drq}
\alias{drq}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Debiasing program}
\description{
Solves primal and dual debiasing program for a given tuning parameter (provided
that the primal problem is feasible and that strong duality holds).
}
\usage{
drq(Y, X, x, tau, density = "nid", sparsity = NULL, lambda = NULL, gamma = NULL,
max_iter = 5000, algo="CD")
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
  \item{lambda}{Regularization parameter for L1-penalized quantile regression problem.
}
  \item{gamma}{Tuning parameter for primal debiasing program.
}
  \item{max_iter}{
Maximum number of iterations of the coordinate descent algorithm or alternating direction of multiplier method. Default value is 500.
}
  \item{algo}{
  Algorithm for solving the dual program. Options are "CD" (coordinate descent) and "ADMM" (alternating direction of multiplier method).
}
}

\value{
%%  ~Describe the value returned
%%  If it is a LIST, use
\item{w}{Solution to the primal debiasing program.
}
\item{v}{Solution to the dual debiasing program.
}
\item{psi}{Density matrix, i.e. diagonal matrix with conditional densities of Y given X evaluated at the conditional quantile of Y given X = X_i i.e. "\eqn{f_{Y \mid X}(X_i'\beta_0 \mid X_i)}" for   "\eqn{i = 1, \ldots, n}".
}
\item{gamma}{Tuning parameter for primal debiasing program.
}
\item{lambda}{Regularization parameter for L1-penalized quantile regression problem.
}
\item{pilot}{Pilot estimate based on L1-penalized quantile regression vector and querry point x.
}
\item{residuals}{Residuals fo L1-penalized quantile regression program.
}
\item{gradL}{Gradient of the check loss with respect to the quantile function, a.k.a. rank scores.
}
\item{density}{Returns chosen option for estimating the density matrix.
}
\item{algo}{Returns chosen option of the algorithm for solving the dual program.
}
\item{tau}{Returns quantile level.
}
\item{X}{Returns covariates.
}
\item{Y}{Returns responses.
}
\item{x}{Returns querry point x.
}
\item{s}{Returns sparsity level (if screening = TRUE).
}
%% ...
}
\references{
  Giessing, A. and Wang, J. (2023). Debiased inference on heterogeneous quantile treatment effects with regression rank-scores. \emph{Journal of the Royal Statistical Society Series B: Statistical Methodology}, 85(5): 1561–1588. \url{https://doi.org/10.1093/jrsssb/qkad075}
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
 \concept{Debiasing program}
% \concept{ ~cpt2 }
% Use only one concept per line.
