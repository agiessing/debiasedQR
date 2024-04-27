library(MASS)
library(quantreg)
library(CVXR)
library(caret)
library(doParallel)

source("PrimalProg.R")
source("DualProg-ADMM.R")
source("DualProg-CD.R")
source("DensityMatrix.R")
source("SoftThres.R")
source("LambdaBC.R")
source("DualObj.R")
source("OptGamma.R")
source("DebiasProg.R")
source("DebiasProgCV.R")
source("SummaryDRQ.R")
source("SISRQ.R")


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
Y <- drop(X %*% beta_0) + eps

## True conditional quantile function
q_0 <- x[-1] %*% beta_0 + x[1]*sig * qnorm(tau, 0,1)

## Debiased quantile function with tuning parameter selected by cross-validation
## of the dual

fit2 <- drqcv(Y, X, x, tau, density = "nid", sparsity = 6, cv_fold = 5,
              max_iter = 1000, parallel = FALSE)
debiasedPredict(fit2, cv_rule = "1se", robust=FALSE)

############################# PARALLEL VERSION #################################
##
## ncpu <- detectCores()
## cl <- makeCluster(ncpu-1)
## registerDoParallel(cl)
## fit2 <- drqcv(Y, X, x, tau, density = "nid", sparsity = 6, cv_fold = 5,
##              max_iter = 1000, parallel = TRUE)
## debiasedPredict(fit2, cv_rule = "1se", robust=FALSE)
## stopCluster(cl)
##
################################################################################
