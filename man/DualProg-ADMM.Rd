\name{dualADMM}
\alias{dualADMM}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
  ADMM algorithm to solve the dual debiasing program
}
\description{
  Solves the dual debiasing program for a giving tuning parameter via the alternating direction method of multipliers (ADMM).
}
\usage{
dualADMM(X, x, Psi=NULL, v_init=NULL, gamma, rho=5, max_iter = 500,
tol = c(1e-6, 1e-6), quiet = TRUE)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{X}{Covariates.
}
  \item{x}{Querry point at which to evaluate the conditional quantile function of Y given X = x.
}
  \item{Psi}{
  Estimate of the density matrix.
}
  \item{v_init}{
  Initialized value of the solution to the dual debiasing program.
}
 \item{gamma}{Tuning parameter for primal debiasing program.
}
  \item{rho}{
  Step size.
}
  \item{max_iter}{
  Maximum number of iterations tge ADMM algorithm. Default value is 500.
}
  \item{tol}{
  Tolerance levels to determine convergence of the ADMM algorithm. Default value is c(1e-6, 1e-6).
}
  \item{quiet}{
  If quiet = TRUE, then now messages are displayed.
}
}

\value{
  Solution to the dula debiasing program.
}
\references{
  Giessing, A. and Wang, J. (2023). Debiased inference on heterogeneous quantile treatment effects with regression rank-scores. \emph{Journal of the Royal Statistical Society: Series B (Statistical Methodology)}, 85(5): 1561–1588. \url{https://doi.org/10.1093/jrsssb/qkad075}

  Boyd, S., Parikh, N., Chu, E., Peleato, B., and Eckstein, J. (2011). Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers. \emph{Foundations and Trends in Machine Learning}, 3(1): 1–122. \url{https://doi.org/10.1561/2200000016}
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
 \concept{ADMM algorithm}
% Use only one concept per line.
