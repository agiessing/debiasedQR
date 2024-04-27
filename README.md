# ``debiasedQR``: Debiased inference on L1-penalized high-dimensional linear quantile regression models via regression rank-scores

## Installation

The development version can be installed from github:

```R
devtools::install_github("agiessing/debiasedQR")
```

## Illustrative Toy Example
```R
library(MASS)
library(quantreg)
library(CVXR)
library(caret)
library(doParallel)
library(mvtnorm)
library(debiasedQR)

set.seed(2024)

d <- 500
n <- 200
tau <- 0.6

Sigma <- toeplitz((1/2)^{0:(d-1)})
sig <- 1

## Querry point at which to evaluate the conditional quantile function
x <- rep(0, d+1)
x[c(1, 2, 3, 4, 5, 7, 8)] <- rep(1, 7)

## True regression coefficient
s_beta <- 5
beta_0 <- rep(0, d)
beta_0[1:s_beta] <- 1/sqrt(s_beta)

## Design matrix
X <-  mvrnorm(n, mu = rep(0, d), Sigma)
eps <- sig * rnorm(n, 0, 1)
Y <- drop(X %*% beta_0) + eps

## True conditional quantile function at querry point x
q_0 <- x[-1] %*% beta_0 + x[1]*sig * qnorm(tau, 0,1)
q_0

## Debiased quantile function for fixed tuning parameter gamma = 0.33
fit1 <- drq(Y, X, x, tau, density="nid", sparsity = 10,
            lambda = lambdaBC(X=X, tau=tau), gamma = 0.33)
dqr1 <- debiasedPredict(fit1, robust=FALSE)

dqr1$debias # debiased estimate (based on primal variables)
dqr1$dual   # debiased estimate (based on dual variables)
dqr1$pilot  # biased pilot estimate (based on L1-penalized QR)
dqr1$avar   # estimate of asymptotic variance of debiased estimate

## Debiased quantile function，tuning parameter gamma selected by cross-validation
## (single CPU)
fit2 <- drqcv(Y, X, x, tau, density = "nid", sparsity = 6, cv_fold = 5,
              max_iter = 1000, parallel = FALSE)
dqr2 <- debiasedPredict(fit2, cv_rule = "1se", robust=FALSE)

dqr2$debias # debiased estimate (based on primal variables)
dqr2$dual   # debiased estimate (based on dual variables)
dqr2$pilot  # biased pilot estimate (based on L1-penalized QR)
dqr2$avar   # estimate of asymptotic variance of debiased estimate

## Debiased quantile function, tuning parameter selected by cross-validation
## (multiple CPUs, substantially faster than single CPU)
ncpu <- max(1L, detectCores() - 2L, na.rm = TRUE)
cl <- makeCluster(ncpu)
registerDoParallel(cl)
fit3 <- drqcv(Y, X, x, tau, density = "nid", sparsity = 6, cv_fold = 5,
              max_iter = 1000, parallel = TRUE)
stopCluster(cl)
dqr3 <- debiasedPredict(fit3, cv_rule = "1se", robust=FALSE)

dqr3$debias # debiased estimate (based on primal variables)
dqr3$dual   # debiased estimate (based on dual variables)
dqr3$pilot  # biased pilot estimate (based on L1-penalized QR)
dqr3$avar   # estimate of asymptotic variance of debiased estimate

## Asymptotic 95% confidence intervals for q_0 at querry point x
cat("The 95% confidence interval for q_0 is [",
    dqr3$debias - sqrt(dqr3$avar) / sqrt(n) * qnorm(1-0.05/2), ", ",
    dqr3$debias + sqrt(dqr3$avar) / sqrt(n) * qnorm(1-0.05/2), "].\n", sep = "")
```

