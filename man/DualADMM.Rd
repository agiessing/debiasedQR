\name{dualADMM}
\alias{dualADMM}

\title{
  ADMM algorithm to solve the dual rank-score debiasing program
}
\description{
  Solves the dual debiasing program for a giving tuning parameter via the alternating direction method of multipliers (ADMM).
}
\usage{
dualADMM(X, x, Psi=NULL, v_init=NULL, gamma, rho=5, max_iter = 500,
tol = c(1e-6, 1e-6), quiet = TRUE)
}

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
  If quiet = TRUE, then no messages are displayed.
}
}

\value{
  Solution to the dual debiasing program.
}
\references{
  Giessing, A. and Wang, J. (2023). Debiased inference on heterogeneous quantile treatment effects with regression rank-scores. \emph{Journal of the Royal Statistical Society: Series B (Statistical Methodology)}, 85(5): 1561–1588.  \doi{10.1093/jrsssb/qkad075}

  Boyd, S., Parikh, N., Chu, E., Peleato, B., and Eckstein, J. (2011). Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers. \emph{Foundations and Trends in Machine Learning}, 3(1): 1–122. \doi{10.1561/2200000016}
}

\author{
  Alexander Giessing, \email{giessing@umich.edu}
}

\concept{Dual debiasing program}
\concept{ADMM algorithm}
