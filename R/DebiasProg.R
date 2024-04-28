#' This function returns a .drq object, which can be fed into the function
#' debiasedPredict() to compute the rank-score debiased estimate of the
#' conditional quantile function at querry point x.
#'
#' @param Y          Responses.
#' @param X          Covariates.
#' @param x          Querry point at which to evaluate the conditional quantile
#'                   function of Y given X = x.
#' @param tau        Quantile level "\eqn{\tau \in (0,1)}".
#' @param density    Options for estimating the density matrix: "nid" (non-iid data),
#'                   "iid" (iid data), and "iidGaussian" (iid Gaussian data).
#' @param sparsity   Only relevant if screening = TRUE in densityMatrix(). Upper
#'                   bound on sparsity of the quantile regression function. Default
#'                   value is NULL.
#' @param lambda     Regularization parameter for L1-penalized quantile regression
#'                   problem.
#' @param gamma      Tuning parameter for primal debiasing program.
#' @param max_iter   Maximum number of iterations of the coordinate descent
#'                   algorithm or alternating direction of multiplier method.
#'                   Default value is 500.
#' @param algo       Algorithm for solving the dual program. Options are "CD"
#'                   (coordinate descent) and "ADMM" (alternating direction of
#'                   multiplier method).
#'
#' @return w         Solution to the primal debiasing program.
#' @return v         Solution to the dual debiasing program.
#' @return psi       Density matrix, i.e. diagonal matrix with conditional densities
#'                   of Y given X evaluated at the conditional quantile of Y given
#'                   X = X_i i.e. "\eqn{f_{Y \mid X}(X_i'\beta_0 \mid X_i)}" for
#'                   "\eqn{i = 1, \ldots, n}".
#' @return pilot     Pilot estimate based on L1-penalized quantile regression vector
#'                   and querry point x.
#' @return residuals Residuals of L1-penalized quantile regression program.
#' @return gradL     Gradient of the check loss with respect to the quantile
#'                   function, a.k.a. rank scores.
#'
#' @import quantreg
#' @export

drq <- function(Y, X, x, tau, density = "nid", sparsity = NULL, lambda = NULL, gamma = NULL, max_iter = 5000, algo="CD") {
  n = dim(X)[1]
  if (is.null(gamma)) {
    stop(paste0("Missing value for gamma!\n"))
  }

  if (is.null(lambda)) {
    stop(paste0("Missing regularization parameter lambda!\n"))
  }

  fit <- rq(Y ~ X, tau=tau, method="lasso", lambda = lambda)
  beta_hat <- fit$coef
  Psi_hat <- densityMatrix(Y, X, beta = beta_hat[-1], tau = tau, density = density, sparsity = sparsity)

  w_hat <- primal(X = X, x = x[-1], Psi = diag(diag(1/Psi_hat)^2), gamma = gamma)
  if (is.na(sum(w_hat))) {
    stop(paste0("Primal debiasing program is infeasible for gamma = ", gamma," ! Program aborted..."))
  }

  if(algo=="ADMM") {
    v_hat <- dualADMM(X, x[-1], Psi=diag(diag(Psi_hat)^2), gamma = gamma, max_iter = max_iter)$v
  } else if (algo == "CD") {
    v_hat <- dualCD(X, x[-1], Psi=diag(diag(Psi_hat)^2), gamma = gamma, max_iter = max_iter)$v
  }

  duality_gap <- sum( abs(w_hat + drop(diag(diag(Psi_hat)^2) %*% X %*% v_hat) / (2 * sqrt(n)) ) > 1/sqrt(n) ) #1e-03
  if (duality_gap > 0) {
    stop(paste0("Strong duality does not hold for gamma = ", gamma,". Primal and dual relationship violated ", duality_gap, " times out of ", n," !" ))
  }

  dual_loss <- dualObj(X, x = x[-1], Psi = diag(diag(Psi_hat)^2), v = drop(v_hat), gamma = gamma)

  dfit <- list()
  class(dfit) <- "drq"
  dfit$w <- drop(w_hat)
  dfit$v <- drop(v_hat)
  dfit$psi <- diag(Psi_hat)
  dfit$gamma <- gamma
  dfit$lambda <- fit$lambda
  dfit$pilot <- drop(t(x) %*% beta_hat)
  dfit$residuals <- drop(fit$residuals)
  dfit$gradL <- tau - (drop(fit$residuals) < 0)
  dfit$density <- density
  dfit$algo <- algo
  dfit$tau <- tau
  dfit$X <- X
  dfit$Y <- Y
  dfit$x <- x
  dfit$s <- sparsity


  return(dfit)
}
