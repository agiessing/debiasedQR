\name{debiasedPredict}
\alias{debiasedPredict}

\title{
Summary function for .drq and .drqcv objects
}

\description{
Returns the debiased estimate of the conditional quantile of Y given X = x, where x is a pre-specified querry point, and an estimate of the asymptotic variance of the debaised estimator.
}

\usage{debiasedPredict(object, cv_rule = "1se", robust = FALSE)
}

\arguments{
  \item{object}{An .drq or .drqcv object obtained from drq() or drqcv().}
  \item{cv_rule}{Cross-validation rule to be applied to .drqcv object to select optimal tuning parameter "\eqn{gamma >0}". Options are `1se', `mincv', and `minfeas'. For details see documentatio of optGamma().}
  \item{robust}{Only relevant for .drqcv objects. If robust = TRUE, then median and mean absolute deviation are used in cross-validation procedure. If robust = FALSE,  then mean and standard deviation are used.}
}

\value{
  \item{pilot}{Pilot estimate of the conditional quantile function based on L1-penalized quantile regression vector.
  }
  \item{debias}{Debiased estimate of the conditional quantile function based on primal variable "\eqn{w}".
  }
  \item{dual}{Debiased estimate of the conditional quantile function based on dual variable "\eqn{v}".
  }
  \item{avar}{Estimate of the asymptotic variance of either debiased estimate of the conditional quantile function.
  }
  \item{gamma_opt}{Optimal tuning parameter "\eqn{\gamma > 0}" of the primal problem chosen via cross-valdiation.
  }
  \item{density}{Estimate of the conditional densities of Y given X evaluated at the conditional quantile of Y given X = X_i i.e. "\eqn{f_{Y \mid X}(X_i'\beta_0 \mid X_i)}" for   "\eqn{i = 1, \ldots, n}".
  }
  \item{rule}{Cross-validatio rule.
  }
}
\references{
  Giessing, A. and Wang, J. (2023). Debiased inference on heterogeneous quantile treatment effects with regression rank-scores.
  \emph{Journal of the Royal Statistical Society Series B: Statistical Methodology}, 85(5): 1561–1588.
  \doi{10.1093/jrsssb/qkad075}
}
\author{
  Alexander Giessing, \email{giessing@umich.edu}
}

\examples{
\dontrun{
library(MASS)
library(quantreg)
library(CVXR)
library(caret)
library(doParallel)

set.seed(2024)

d <- 700
n <- 400
tau <- 0.6

Sigma <- toeplitz((1/2)^{0:(d-1)})
sig <- 1

## Current query point
x <- rep(0, d+1)
#x[c(1, 2, 3, 7, 8)] <- c(1, 1/2, 1/4, 1/2, 1/8)
x[c(1, 2, 3, 4, 5, 7, 8)] <- rep(1, 7)

## True regression coefficient
s_beta <- 5
beta_0 <- rep(0, d)
beta_0[1:s_beta] <- 1/sqrt(s_beta)

## Design matrix
X <-  mvrnorm(n, mu = rep(0, d), Sigma)
eps <- sig * rnorm(n, 0, 1)
Y <- drop(X \%*\% beta_0) + eps

## True conditional quantile function
q_0 <- x[-1] \%*\% beta_0 + x[1]*sig * qnorm(tau, 0,1)

## Debiased quantile function, tuning parameter selected by cross-validation

fit1 <- drqcv(Y, X, x, tau, density = "nid", sparsity = 6, cv_fold = 5,
             max_iter = 1000, parallel = FALSE)
dQR1 <- debiasedPredict(fit1, cv_rule = "1se", robust=FALSE)

# Debiased estimate (based on primal variables)
dQR1$debias

# Debiased estimate (based on dual variables)
dQR1$dual

# Biased ilot estimate (based on L1-penalized QR)
dQR1$pilot

# Estimate of asymptotic variance of debiased estimate
dQR1$avar

## Debiased quantile function, tuning parameter selected by cross-validation
## (parallelized, can be substantially faster)

ncpu <- 2 # for compatibility reasons with CRAN
#ncpu <- max(1L, detectCores() - 2L, na.rm = TRUE) # for general use
cl <- makeCluster(ncpu)
registerDoParallel(cl)
fit2 <- drqcv(Y, X, x, tau, density = "nid", sparsity = 6, cv_fold = 5,
              max_iter = 1000, parallel = TRUE)
dQR2 <- debiasedPredict(fit2, cv_rule = "1se", robust=FALSE)
stopCluster(cl)
}
}

\keyword{Utilities}
\concept{Debiased prediction}
\concept{Debiased estimator}
