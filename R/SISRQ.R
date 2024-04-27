#' Iterative sure indendence screening for linear quantile regression
#'
#' Iterative sure indendence screening for linear quantile regression to improve
#' and robustify the estimation of the density matrix.
#'
#' @param Y        Responses.
#' @param X        Covariates.
#' @param tau      Quantile level "\eqn{\tau \in (0,1)}".
#' @param kM       Only relevant if screening = TRUE. Upper bound on sparsity of
#'                 the quantile regression function. Default value NULL.
#' @param max_iter Maximum number of iterations of sure independence screening.
#'                 Default value is 100.
#' @param quiet    If quiet = TRUE, then no messages are displayed.
#'
#' @return M       Selected model/ covariates.
#'
#' @import quantreg
#' @export

sisrq <- function(Y, X, tau, kM, max_iter = 100, quiet = TRUE) {
  n <- dim(X)[1]
  d <- dim(X)[2]
  kA=ceiling(n/log(n))
  kB=ceiling(n/log(n))

  L <- array(NA, dim = c(1,d))
  cnt <- 0

  # Step 1: Marginal utilities
  for (k in 1:d) {
    cnt <- cnt + 1
    fit <- rq(Y ~ X[, k], tau)
    L[cnt] <- mean(fit$res*(tau - (fit$res < 0)))
  }
  A <- order(L, decreasing=FALSE)[1:kA]
  fit <- rq(Y~ X[, A], tau=tau, method="lasso", lambda = lambdaBC(X=X[, A], tau=tau, c=2))
  M_old <- A[(abs(fit$coef[-1]) > 1e-03)]

  cntt <- 0
  M_new <- array(NA, dim = c(1,d))

  while ( (length(M_new) > kM) && (cntt <= max_iter) ) {
    cntt <- cntt + 1

    LL <- array(NA, dim = c(1, n - length(M_old)))

    # Step 2: Conditional marginal utilities
    cnt <- 0
    for (k in setdiff(1:n, M_old)) {
      cnt <- cnt + 1
      fit <- rq(Y ~ X[, c(M_old, k)], tau)
      LL[cnt] <- mean(fit$res*(tau - (fit$res < 0)))
    }
    B <- order(LL, decreasing=FALSE)[1:min(kB, length(LL))]

    # Step 3: Lasso on union(M, B)
    U <- union(M_old, B)
    fit <- rq(Y~X[, U], tau = tau, method="lasso", lambda = lambdaBC(X=X[, U], tau=tau, c=2))
    M_new <- U[(abs(fit$coef[-1]) > 1e-03)]

    M_old <- M_new

    if(!quiet){
      message(paste0(c("Selected Model: ", M_old), collapse = " "))
    }
  }

  return(M_new)
}
