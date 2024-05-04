# ``debiasedQR``

Debiasing procedure for $\ell_1$-penalized quantile regression in high-dimensional sparse models via regression rank-scores, as proposed by Giessing and Wang (2023).

To install a development version of this package in R, run the following commands:

```R
library(devtools)
install_github("agiessing/debiasedQR")
```
The current package differs in two ways from the original code that we used in the simulation study and data analysis in Giessing and Wang (2023):

First, we now offer two algorithms for solving the dual debiasing program: Alternating Direction Method of Multipliers (ADMM) and Proximal Coordinate Descent (CD). We have found that the newly implemented Proximal CD algorithm converges substantially faster, is more accurate, and also more robust when applied to extreme quantiles. We have therefore made the Proximal CD algorithm the default.

Second, we now estimate the conditional densities (needed for the primal and dual rank-score debiasing programs) in a two-step procedure: first, we apply a version of the iterative sure independence screening to select a model (e.g. Fan and Lv 2008), then we compute the conditional densities using only the selected model. This procedure proves to be more robust, especially for extreme quantiles. The downside of this approach is that it requires an upper bound on the model size.

The theoretical results in Giessing and Wang (2023) are unaffected by these modifications.

## Usage
The R package has three main functions: 

- ``drq()`` Solves the primal and dual rank-score debiasing programs in Giessing and Wang (2023) for a given tuning parameter $\gamma > 0$, provided that the primal program is feasible and that strong duality holds. [more](https://github.com/agiessing/debiasedQR/blob/main/R/DebiasProg.R)
- ``drqcv()`` Solves the primal and dual rank-score debiasing programs for a range of tuning parameters and returns the values of the cross-validated dual losses, which can be used to determine the optimal tuning parameter $\gamma^* > 0$. [more](https://github.com/agiessing/debiasedQR/blob/main/R/DebiasProgCV.R) 
- ``debiasedPredict()`` A wrapper function which takes either `.drq` or `.drqcv` objects and returns the biased $\ell_1$-penalized pilot estimate of the conditional quantile function, two debiased estimates of the conditional quantile function (based on the solutions of the primal and dual program, respectively), and an estimate of the asymptotic variance of the debiased estimate. When applied to a `.drqcv` object, the wrapper function first finds the optimal tuning parameter $\gamma^* > 0$ which minimizes the cross-validated dual loss function. [more](https://github.com/agiessing/debiasedQR/blob/main/R/DebiasedPredict.R)

If the two debiased estimates returned by `debiasedPredict()` differ, then either the ADMM or the Proximal CD algorithm did not converge. In this case, we recommend to increase the maximum number of iterations of these algorithms.

When using `drqcv()` the penalty parameter of the $\ell_1$-penalized quantile regression program is chosen data adaptively, exploiting the pivotal properties of the gradient of the check-loss function (e.g. Belloni and Chernozhukov, 2011).

## Examples

Below code illustrates the main functionalities of the R package. 

Draw samples from a homoscedastic high-dimensional sparse regression model and evaluate the true conditional quantile function $q_0$ at querry point $x$ and quantile level $\tau = 0.6$.

```R
# Sampling from a homoscedastic high-dimensional sparse regression model

library(debiasedQR)
library(MASS)

set.seed(2024)

d <- 800
n <- 300
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
```

Compute the debiased estimate of the conditional quantile function at querry point $x$ for fixed tuning parameter $\gamma = 0.33$ and construct an asymptotic 95\% confidence interval for the true conditional quantile function $q_0$.

```R
# EXAMPLE 1
# Debiased quantile function for fixed tuning parameter gamma = 0.33

fit1 <- drq(Y, X, x, tau, density = "nid", sparsity = 6,
            lambda = lambdaBC(X = X, tau = tau), gamma = 0.33)
dqr1 <- debiasedPredict(fit1)

dqr1$debias # debiased estimate (based on primal variable w)
dqr1$dual   # debiased estimate (based on dual variable v)
dqr1$pilot  # biased pilot estimate (based on L1-penalized QR estimate)
dqr1$avar   # estimate of the asymptotic variance of the debiased estimate

# Asymptotic 95% confidence intervals for q_0 at querry point x
cat("An asymptotic 95% confidence interval for q_0 is [",
    dqr1$debias - sqrt(dqr1$avar) / sqrt(n) * qnorm(1-0.05/2), ", ",
    dqr1$debias + sqrt(dqr1$avar) / sqrt(n) * qnorm(1-0.05/2), "].\n", sep = "")
```

Use cross-validation to determine a suitable value of the tuning parameter $\gamma^* > 0$ and compute the debiased estimate of the conditional quantile function.

```R
# EXAMPLE 2
# Debiased quantile function，tuning parameter selected via cross-validation

fit2 <- drqcv(Y, X, x, tau, density = "nid", screening = FALSE,
              cv_fold = 5, max_iter = 5000, parallel = FALSE)
dqr2 <- debiasedPredict(fit2, cv_rule = "1se", robust = FALSE)

dqr2$debias # debiased estimate (based on primal variable w)
dqr2$dual   # debiased estimate (based on dual variable v)
dqr2$pilot  # biased pilot estimate (based on L1-penalized QR estimate)
dqr2$avar   # estimate of the asymptotic variance of the debiased estimate
```

Parallelized version of the previous example.

```R
# EXAMPLE 3
# Debiased quantile function, tuning parameter selected via cross-validation
# multiple cores, parallelized cross-validation

library(doParallel)

ncpu <- max(1L, detectCores() - 2L, na.rm = TRUE)
cl <- makeCluster(ncpu)
registerDoParallel(cl)
fit3 <- drqcv(Y, X, x, tau, density = "nid", screening = FALSE, 
              cv_fold = 5, max_iter = 5000, parallel = TRUE)
stopCluster(cl)
dqr3 <- debiasedPredict(fit3, cv_rule = "1se", robust = FALSE)

dqr3$debias # debiased estimate (based on primal variable w)
dqr3$dual   # debiased estimate (based on dual variable v)
dqr3$pilot  # biased pilot estimate (based on L1-penalized QR estimate)
dqr3$avar   # estimate of the asymptotic variance of the debiased estimate
```

References
--------
</a> A. Belloni and V. Chernozhukov (2011). $\ell_1$-penalized quantile regression in high-dimensional sparse models. *The Annals of Statistics*. 39(1), 82-130. https://doi.org/10.1214/10-AOS827

</a> J. Fan and J. Lv (2008). Sure independence screening for ultrahigh dimensional feature space (with discussion). *Journal of the Royal Statistical Society B: Statistical Methodology*, 70(5), 849-911. https://doi.org/10.1111/j.1467-9868.2008.00674.x

</a> A. Giessing and J. Wang (2023). Debiased inference on heterogeneous quantile treatment effects with regression rank-scores. *Journal of the Royal Statistical Society Series B: Statistical Methodology*. 85(5), 1561–1588. https://doi.org/10.1093/jrsssb/qkad075



