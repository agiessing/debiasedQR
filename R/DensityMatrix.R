#' This function returns an estimate of the density matrix, i.e. a diagonal matrix
#' with conditional densities of Y given X evaluated at the conditional quantile
#' of Y given X = X_i i.e. "\eqn{f_{Y \mid X}(X_i'\beta_0 \mid X_i)}" for
#' "\eqn{i = 1, \ldots, n}".
#'
#' @param Y         Responses.
#' @param X         Covariates.
#' @param beta      Estimate of the quantile regression vector.
#' @param tau       Quantile level "\eqn{\tau \in (0,1)}".
#' @param density   Options for estimating the density matrix: "nid" (non-iid data),
#'                  "iid" (iid data), and "iidGaussian" (iid Gaussian data).
#' @param sparsity  Only relevant if screening = TRUE. Upper bound on sparsity of
#'                  the quantile regression function. Default value NULL.
#' @param screening If screening = TRUE, then a version of the (iterative) sure
#'                  indendence screening for linear quantile regression is used
#'                  to select covariates and then estimate the density matrix based
#'                  on the selected covariates. If screening = FALSE, then the
#'                  provided value for beta is used to estimate the density matrix.
#'
#' @import quantreg
#' @import stats
#' @export

densityMatrix <- function(Y, X, beta, tau, density="nid", sparsity, screening = TRUE) {

  n <- dim(X)[1]
  d <- dim(X)[2]
  eps <- .Machine$double.eps^(1/2)
  h <- bandwidth.rq(tau, n, hs = TRUE, alpha = 0.2)

  if (screening == TRUE) {
    s <- sisrq(Y, X, tau, kM = sparsity, max_iter = 100)
  } else {
    s <- which( (abs(beta) > 1e-03) )
  }

  if (density == "nid") {

    if (tau + h > 1)
      stop("tau + h > 1:  error in function `density.matrix'")
    if (tau - h < 0)
      stop("tau - h < 0:  error in function `density.matrix'")

    if(length(s) == 0){
      bhi <- quantile(Y, tau + h)
      blo <- quantile(Y, tau - h)
      dyhat <- bhi - blo
    } else {
      bhi <- rq(Y ~ X[, s], tau = tau + h)$coef
      blo <- rq(Y ~ X[, s], tau = tau - h)$coef
      dyhat <- as.matrix(X[, s]) %*% (bhi[-1] - blo[-1]) + bhi[1] - blo[1]
    }

    f <- diag( pmax(0, (2 * h) / (dyhat - eps)) )

  } else if (density == "iid") {

    if(length(s) == 0){
      yq <- quantile(Y, tau)
      res <- Y - yq
    } else {
      res <- rq(Y ~ X[, s], tau)$res
    }

    nz <- sum(abs(res) < eps)
    h <- max(length(s) + 1, ceiling(n * h))
    ir <- (nz + 1):(h + nz + 1)
    ord.resid <- sort(res[order(abs(res))][ir])
    xt <- ir/(n - nz)
    sparsity <- rq(ord.resid ~ xt)$coef[2]

    f <- diag(1/sparsity, n)

  } else if (density == "iidGaussian") {

    f <- diag(dnorm(qnorm(tau, 0,1), 0,1),n)

  } else if (density == "oracle") {

    if (tau + h > 1)
      stop("tau + h > 1:  error in function `density.matrix'")
    if (tau - h < 0)
      stop("tau - h < 0:  error in function `density.matrix'")

    bhi <- rq(Y ~ X[, 1:5], tau = tau + h)$coef
    blo <- rq(Y ~ X[, 1:5], tau = tau - h)$coef
    dyhat <- as.matrix(X[, 1:5]) %*% (bhi[-1] - blo[-1]) + bhi[1] - blo[1]

    f <- diag( pmax(0, (2 * h) / (dyhat - eps)) )
  }

  return(f)
}
