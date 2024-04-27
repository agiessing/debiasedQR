# ``debiasedQR``: An R Package for efficient debiased inference on high-dimensional linear quantile regression models

<!-- badges: start -->
[![CRAN status](https://www.r-pkg.org/badges/version/DebiasInfer)](https://CRAN.R-project.org/package=DebiasInfer)
<!-- badges: end -->

## Installation

The latest release of the R package can be installed through CRAN:

```R
install.packages("DebiasInfer")
```

The development version can be installed from github:

```R
devtools::install_github("agiessing/debiasedQR", subdir = "R_Package")
```

## Toy Example

```R
require(MASS)
require(glmnet)
require(scalreg)
require(DebiasInfer)

d = 1000
n = 900

Sigma = array(0, dim = c(d,d)) + diag(d)
rho = 0.1
for(i in 1:(d-1)){
  for(j in (i+1):d){
    if ((j < i+6) | (j > i+d-6)){
      Sigma[i,j] = rho
      Sigma[j,i] = rho
    }
  }
}
sig = 1

## Current query point
x_cur = rep(0, d)
x_cur[c(1, 2, 3, 7, 8)] = c(1, 1/2, 1/4, 1/2, 1/8)
x_cur = array(x_cur, dim = c(1,d))

## True regression coefficient
s_beta = 5
beta_0 = rep(0, d)
beta_0[1:s_beta] = sqrt(5)

## Generate the design matrix and outcomes
set.seed(123)
X_sim = mvrnorm(n, mu = rep(0, d), Sigma)
eps_err_sim = sig * rnorm(n)
Y_sim = drop(X_sim %*% beta_0) + eps_err_sim

obs_prob = 1 / (1 + exp(-1 + X_sim[, 7] - X_sim[, 8]))
R_sim = rep(1, n)
R_sim[runif(n) >= obs_prob] = 0

## Lasso Pilot Estimate
lasso_pilot = scalreg(X_sim[R_sim == 1,], Y_sim[R_sim == 1], lam0 = "univ", LSE = FALSE)
beta_pilot = lasso_pilot$coefficients
sigma_pilot = lasso_pilot$hsigma

## Estimate the propensity scores via the Lasso-type generalized linear model with cross-validations
zeta = 10^seq(-1, log10(300), length.out = 40) * sqrt(log(d) / n)
lr1 = cv.glmnet(X_sim, R_sim, family = 'binomial', alpha = 1, type.measure = 'deviance', 
                lambda = zeta, nfolds = 5, parallel = TRUE)
lr1 = glmnet(X_sim, R_sim, family = "binomial", alpha = 1, lambda = lr1$lambda.min, 
             standardize = TRUE, thresh=1e-6)
prop_score = drop(predict(lr1, newx = X_sim, type = 'response'))

## Estimate the debiasing weights with the tuning parameter selected by cross-validations
deb_res = DebiasProgCV(X_sim, x_cur, prop_score, gamma_lst = c(0.2, 0.4, 0.6, 1),
                       cv_fold = 5, cv_rule = '1se')

## Construct the 95% confidence intervals for the true regression function
m_deb = sum(x_cur %*% beta_pilot) + sum(deb_res$w_obs * R_sim * (Y_sim - X_sim %*% beta_pilot)) / sqrt(n)
asym_sd = sqrt(sum(prop_score * deb_res$w_obs^2) / n)

cat("The 95% confidence interval yielded by our debiasing method is [",
    m_deb - asym_sd*sigma_pilot*qnorm(1-0.05/2), ", ",
    m_deb + asym_sd*sigma_pilot*qnorm(1-0.05/2), "].\n", sep = "")
```

