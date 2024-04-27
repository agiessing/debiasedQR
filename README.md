# ``debiasedQR``

Debiasing procedure for $\ell_1$-penalized high-dimensional linear quantile regression models via regression rank-scores, as proposed by Giessing and Wang (2023).

To install a development version of this package in R, run the following commands:

```R
library(devtools)
install_github("agiessing/debiasedQR")
```

This package currently offers two algorithmic options: Alternating Direction of Multiplier Method (ADMM) and Proximal Coordinate Descent (CD). The simulation study and data analysis in Giessing and Wang (2023) were conducted using the ADMM algorithm. We found that the newly implemented Proximal CD algorithm converges substantially faster, is more accurate, and also more robust when applied to extreme quantiles. We therefore recommend to only use the default setting with the Proximal CD algorithm.

## Usage
The R package has three major functions: 

- ``drq()``
- ``drqcv()``
- ``debiasedPredict()``


Below code illustrates how to use these functions.

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

## Querry point at which to evaluate the conditional quantile function
x <- rep(0, d+1)
x[c(1, 2, 3, 4, 5, 7, 8)] <- rep(1, 7)

## True regression coefficient
s_beta <- 5
beta_0 <- rep(0, d)
beta_0[1:s_beta] <- 1/sqrt(s_beta)

## Data generating process
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

dqr1$debias # debiased estimate (based on primal variable w)
dqr1$dual   # debiased estimate (based on dual variable v)
dqr1$pilot  # biased pilot estimate (based on L1-penalized QR estimate)
dqr1$avar   # estimate of the asymptotic variance of the debiased estimate

## Debiased quantile function，tuning parameter gamma selected via cross-validation
## (single CPU)
fit2 <- drqcv(Y, X, x, tau, density = "nid", sparsity = 6, cv_fold = 5,
              max_iter = 1000, parallel = FALSE)
dqr2 <- debiasedPredict(fit2, cv_rule = "1se", robust=FALSE)

dqr2$debias # debiased estimate (based on primal variable w)
dqr2$dual   # debiased estimate (based on dual variable v)
dqr2$pilot  # biased pilot estimate (based on L1-penalized QR estimate)
dqr2$avar   # estimate of the asymptotic variance of the debiased estimate

## Debiased quantile function, tuning parameter selected via cross-validation
## (multiple CPUs, parallelized cross-validation)

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

## Asymptotic 95% confidence intervals for q_0 at querry point x
cat("The 95% confidence interval for q_0 is [",
    dqr3$debias - sqrt(dqr3$avar) / sqrt(n) * qnorm(1-0.05/2), ", ",
    dqr3$debias + sqrt(dqr3$avar) / sqrt(n) * qnorm(1-0.05/2), "].\n", sep = "")
```



References
--------
<a name="debias">[1]</a> Y. Zhang, A. Giessing, Y.-C. Chen (2023+) Efficient Inference on High-Dimensional Linear Models with Missing Outcomes [arXiv:2309.06429](https://arxiv.org/abs/2309.06429).

<a name="scaledlasso">[2]</a> T. Sun and C.-H. Zhang (2012). Scaled Sparse Linear Regression. *Biometrika*, **99**, no.4: 879-898.


optimizers: mosek, pogs, and quadprog. Mosek is a commercial interior point solver, pogs is a first-order optimizer, based on ADMM, while quadprog is a standard R optimization library. In general, we achieved best performance with mosek, and recommend trying optimizers in the order listed above. We found pogs to be somewhat slower than mosek on the problems we tried. (Note that we offer two solution strategies based on pogs: pogs and pogs.dual. We usually recommend the former, except when p is much larger than n.) Finally, quadprog performors well on small problems, but can be much slower for larger problems.


[![PyPI pyversions](https://img.shields.io/pypi/pyversions/Debias-Infer.svg)](https://pypi.python.org/pypi/Debias-Infer/)
[![PyPI version](https://badge.fury.io/py/Debias-Infer.svg)](https://badge.fury.io/py/Debias-Infer)
[![Downloads](https://static.pepy.tech/badge/Debias-Infer)](https://pepy.tech/project/Debias-Infer)
[![Documentation Status](https://readthedocs.org/projects/sconce-scms/badge/?version=latest)](http://debias-infer.readthedocs.io/?badge=latest)

# Efficient Inference on High-Dimensional Linear Models With Missing Outcomes

This package implements the proposed debiasing method for conducting valid inference on the high-dimensional linear regression function with missing outcomes. We also document all the code for the simulations and real-world applications in our paper [here](https://github.com/zhangyk8/Debias-Infer/tree/main/Paper_Code).

* Free software: MIT license
* Python Package Documentation: [https://debias-infer.readthedocs.io](https://debias-infer.readthedocs.io).
* You may also consider using our R package [DebiasInfer](https://cran.r-project.org/web/packages/DebiasInfer/index.html), though the Python package will be computationally faster.

Installation guide
--------

```Debias-Infer``` requires Python 3.8+ (earlier version might be applicable), [NumPy](http://www.numpy.org/), [SciPy](https://www.scipy.org/), [scikit-learn](https://scikit-learn.org/stable/), [CVXPY](https://www.cvxpy.org/), [statsmodels](https://www.statsmodels.org/). To install the latest version of ```Debias-Infer``` from this repository, run:

```
python setup.py install
```

To pip install a stable release, run:
```
pip install Debias-Infer
```

References
--------

<a name="debias">[1]</a> Y. Zhang, A. Giessing, Y.-C. Chen (2023+) Efficient Inference on High-Dimensional Linear Models with Missing Outcomes [arXiv:2309.06429](https://arxiv.org/abs/2309.06429).

<a name="scaledlasso">[2]</a> T. Sun and C.-H. Zhang (2012). Scaled Sparse Linear Regression. *Biometrika*, **99**, no.4: 879-898.





## Installation

The development version can be installed from github:

```R
devtools::install_github("agiessing/debiasedQR")
```

## Toy Example


