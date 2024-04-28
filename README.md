# ``debiasedQR``

Debiasing procedure for $\ell_1$-penalized quantile regression in high-dimensional sparse models via regression rank-scores, as proposed by Giessing and Wang (2023).

To install a development version of this package in R, run the following commands:

```R
library(devtools)
install_github("agiessing/debiasedQR")
```

The package currently offers two algorithmic options: Alternating Direction of Method of Multipliers (ADMM) and Proximal Coordinate Descent (CD). The simulation study and data analysis in Giessing and Wang (2023) were conducted using the ADMM algorithm. We have found that the newly implemented Proximal CD algorithm converges substantially faster, is more accurate, and also more robust when applied to extreme quantiles. We therefore recommend to use the default setting with the Proximal CD algorithm.

## Usage
The R package has three main functions: 

- ``drq()`` Solves the primal and dual rank-score debiasing programs in Giessing and Wang (2023) for a given tuning parameter $\gamma > 0$, provided that the primal problem is feasible and that strong duality holds. [more](https://github.com/agiessing/debiasedQR/blob/main/R/DebiasProg.R)
- ``drqcv()`` Solves primal and dual rank-score debiasing programs for a range of tuning parameters and returns the values of the optimal cross-validated dual losses, which can be used to determine the optimal tuning parameter $\gamma^* > 0$. We provide the option to parallelize these computations. [more](https://github.com/agiessing/debiasedQR/blob/main/R/DebiasProgCV.R) 
- ``debiasedPredict()`` A wrapper function which takes either `.drq` or `.drqcv` objects and returns the biased $\ell_1$-penalized pilot estimate of the conditional quantile function, two debiased estimates of the conditional quantile function (based on the solutions to the primal and dual programs), and an estimate of the asymptotic variance of the debiased estimate. When applied to a `.drqcv` object, the wrapper function first finds the optimal tuning parameter $\gamma^* > 0$ which minimizes the cross-validated dual loss function. [more](https://github.com/agiessing/debiasedQR/blob/main/R/DebiasedPredict.R)

Occasionally, the two debiased estimates produced by `debiasedPredict()` differ. This only happens, if either ADMM or Proximal CD algorithms did not converge. In this case, we suggest to increase the maximum number of iterations of these algorithms.

When using `drqcv()` the penalty parameter of the $\ell_1$-penalized quantile regression problem is chosen data adaptively, exploiting the pivotal properties of the gradient of the check-loss function (e.g. Belloni and Chernozhukov, 2011).

## Examples
Below code illustrates the main functionalities of the R package.

```R
library(MASS)
library(CVXR)
library(caret)
library(mvtnorm)
library(quantreg)
library(debiasedQR)

set.seed(2024)

d <- 500
n <- 200
tau <- 0.6

Sigma <- toeplitz((1/2)^{0:(d-1)})
sig <- 1

# Querry point at which to evaluate the conditional quantile function
x <- rep(0, d+1)
x[c(1, 2, 3, 4, 5, 7, 8)] <- rep(1, 7)

# True regression coefficient
s_beta <- 5
beta_0 <- rep(0, d)
beta_0[1:s_beta] <- 1/sqrt(s_beta)

# Data generating process
X <-  mvrnorm(n, mu = rep(0, d), Sigma)
eps <- sig * rnorm(n, 0, 1)
Y <- drop(X %*% beta_0) + eps

# True conditional quantile function at querry point x
q_0 <- x[-1] %*% beta_0 + x[1] * sig * qnorm(tau, 0,1)
q_0

# EXAMPLE 1
# Debiased quantile function for fixed tuning parameter gamma = 0.33
fit1 <- drq(Y, X, x, tau, density="nid", sparsity = 10,
            lambda = lambdaBC(X=X, tau=tau), gamma = 0.33)
dqr1 <- debiasedPredict(fit1, robust=FALSE)

dqr1$debias # debiased estimate (based on primal variable w)
dqr1$dual   # debiased estimate (based on dual variable v)
dqr1$pilot  # biased pilot estimate (based on L1-penalized QR estimate)
dqr1$avar   # estimate of the asymptotic variance of the debiased estimate

# Asymptotic 95% confidence intervals for q_0 at querry point x
cat("The 95% confidence interval for q_0 is [",
    dqr1$debias - sqrt(dqr1$avar) / sqrt(n) * qnorm(1-0.05/2), ", ",
    dqr1$debias + sqrt(dqr1$avar) / sqrt(n) * qnorm(1-0.05/2), "].\n", sep = "")

# EXAMPLE 2
# Debiased quantile function，tuning parameter selected via cross-validation
# (single CPU)
fit2 <- drqcv(Y, X, x, tau, density = "nid", sparsity = 6, cv_fold = 5,
              max_iter = 1000, parallel = FALSE)
dqr2 <- debiasedPredict(fit2, cv_rule = "1se", robust=FALSE)

dqr2$debias # debiased estimate (based on primal variable w)
dqr2$dual   # debiased estimate (based on dual variable v)
dqr2$pilot  # biased pilot estimate (based on L1-penalized QR estimate)
dqr2$avar   # estimate of the asymptotic variance of the debiased estimate

# EXAMPLE 3
# Debiased quantile function, tuning parameter selected via cross-validation
# (multiple CPUs, parallelized cross-validation)

library(doParallel)

ncpu <- max(1L, detectCores() - 2L, na.rm = TRUE)
cl <- makeCluster(ncpu)
registerDoParallel(cl)
fit3 <- drqcv(Y, X, x, tau, density = "nid", sparsity = 6, cv_fold = 5,
              max_iter = 1000, parallel = TRUE)
stopCluster(cl)
dqr3 <- debiasedPredict(fit3, cv_rule = "1se", robust=FALSE)

dqr3$debias # debiased estimate (based on primal variable w)
dqr3$dual   # debiased estimate (based on dual variable v)
dqr3$pilot  # biased pilot estimate (based on L1-penalized QR estimate)
dqr3$avar   # estimate of the asymptotic variance of the debiased estimate
```

References
--------
<a >[1]</a> A. Giessing and J. Wang (2023). Debiased inference on heterogeneous quantile treatment effects with regression rank-scores. *Journal of the Royal Statistical Society Series B: Statistical Methodology*. 85(5), 1561–1588 [https://doi.org/10.1093/jrsssb/qkad075]

<a >[2]</a> A. Belloni and V. Chernozhukov (2011). $\ell_1$-penalized quantile regression in high-dimensional sparse models. *The Annals of Statistics*. 39(1), 82-130. [https://doi.org/10.1214/10-AOS827]

