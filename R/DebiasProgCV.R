#' This function returns a .drqcv object, which can be fed into the function
#' debiasedPredict() to compute the rank-score debiased estimate of the
#' conditional quantile function at querry point x.
#'
#' @param Y          Responses.
#' @param X          Covariates.
#' @param x          Querry point at which to evaluate the conditional quantile
#'                   function of Y given X = x.
#' @param tau        Quantile level "\eqn{\tau \in (0,1)}".
#' @param density    Options for estimating the density matrix: "nid" (non-iid
#'                   data), "iid" (iid data), and "iidGaussian" (iid Gaussian data).
#' @param sparsity   Only relevant if screening = TRUE in densityMatrix(). Upper
#'                   bound on sparsity of the quantile regression function. Default
#'                   value is NULL.
#' @param cv_fold    Cross-validation rule to be applied to .drqcv object to select
#'                   optimal tuning parameter "\eqn{gamma >0}". Default value is 5.
#'                   For details see docu of optGamma().
#' @param gamma_lst  List of tuning parameters "\eqn{\gamma > 0}" for the primal
#'                   debiasing program from which to choose the optimal value.
#'                   The default value NULL results in a list with 41 equally
#'                   spaced values from 0 to the maximum norm of the querry point
#'                   x. If set manually, the end point should always be the
#'                   maximum norm of the querry point x. Default value is NULL.
#' @param screening  If screening = TRUE, then a version of the (iterative) sure
#'                   independence screening for linear quantile regression is used
#'                   to select covariates and then estimate the density matrix
#'                   based on the selected covariates. If screening = FALSE, then
#'                   the provided value for beta is used to estimate the density
#'                   matrix.
#' @param max_iter   Maximum number of iterations of the coordinate descent
#'                   algorithm or alternating direction of multiplier method.
#'                   Default value is 500.
#' @param tol        Only relevant if algo = "ADMM". Default value is c(1e-6, 1e-6).
#' @param algo       Algorithm for solving the dual program. Options are "CD"
#'                   (coordinate descent) and "ADMM" (alternating direction method
#'                   of multipliers).
#' @param parallel   If parallel = TRUE, cross-validation is solved in parallel
#'                   using foreach().
#'
#'
#' @return dual_loss List of values of the cross-validated dual objective function.
#'
#' @import MASS
#' @import quantreg
#' @importFrom caret createFolds
#' @import doParallel
#' @import foreach
#' @export

drqcv <- function(Y, X, x, tau, density = "nid", sparsity = NULL, cv_fold = 5,
                  gamma_lst = NULL, screening = TRUE, max_iter = 500,
                  tol = c(1e-6, 1e-6), algo ="CD", parallel=FALSE)  {
  n <- dim(X)[1]
  d <- dim(X)[2]
  if (is.null(gamma_lst)) {
    gamma_lst <- seq(0.001, max(abs(x)), length.out = 41)
  }

  fit <- rq(Y ~ X, tau=tau, method="lasso", lambda = lambdaBC(X=X, tau=tau))
  beta_hat <- fit$coef
  Z <- cbind(rep(1, n), X)
  Psi <- densityMatrix(Y, X, beta = beta_hat, tau = tau, density = density,
                       sparsity = sparsity, screening = screening)

  kf <- createFolds(1:n, cv_fold, list = FALSE, returnTrain = TRUE)
  dual_loss <- matrix(0, nrow = cv_fold, ncol = length(gamma_lst))
  f_ind <- 1

  for (fold in 1:cv_fold) {

    message(paste0("\nFold ", fold))

    train_ind <- (kf != fold)
    test_ind <- (kf == fold)
    Z_train <- Z[train_ind, ]
    Z_test <- Z[test_ind, ]
    Y_train <- Y[train_ind]
    Y_test <- Y[test_ind]

    `%infix%` <- ifelse(parallel, `%dopar%`, `%do%`)

    j <- NULL

    out <- foreach(j=1:length(gamma_lst),
                   .packages = c("quantreg", "CVXR"),
                   .combine = cbind,
                   .export = c("primal", "dualADMM", "dualCD", "dualObj", "SoftThres")
                   ) %infix% {
      Psi_train <- diag(diag(Psi)[train_ind])
      w_train <- primal(X = Z_train, x = x, Psi = diag(diag(1/Psi_train)^2),
                        gamma = gamma_lst[j])

      if (any(is.na(w_train))) {
        message(paste0("Primal debiasing program infeasible for gamma = ",
                       round(gamma_lst[j], 4), "."))
        NA
      } else {

        if(algo=="ADMM") {
          v_train  <- dualADMM(Z_train, x, Psi=diag(diag(Psi_train)^2),
                              gamma = gamma_lst[j], max_iter = max_iter, tol = tol)$v
        } else if (algo == "CD") {
          v_train <- dualCD(Z_train, x, Psi=diag(diag(Psi_train)^2),
                            gamma = gamma_lst[j], max_iter = max_iter)$v
        }

        duality_gap <- sum( abs(w_train + drop(diag(diag(Psi_train)^2) %*% Z_train %*% v_train)
                                / (2 * sqrt(dim(Z_train)[1])) ) > 1/sqrt(4*n) ) #1e-03

        if (duality_gap > 0) {
          message(paste0("Primal and dual relationship violated ", duality_gap,
                         " times out of ", length(w_train)," !"))
          NA
        } else {
          Psi_test <- diag(diag(Psi)[test_ind])
          dualObj(Z_test, x = x, Psi = diag(diag(Psi_test)^2), v = drop(v_train),
                  gamma = gamma_lst[j])
        }
      }
    }
    dual_loss[f_ind, ] <- out
    f_ind = f_ind + 1
  }

  dfit <- list()
  class(dfit) <- "drqcv"
  dfit$X <- X
  dfit$Y <- Y
  dfit$x <- x
  dfit$tau <- tau
  dfit$loss <- dual_loss
  dfit$gamma <- gamma_lst
  dfit$psi <- Psi
  dfit$fold <- cv_fold
  dfit$lambda <- fit$lambda
  dfit$pilot <- drop(t(x) %*% beta_hat)
  dfit$residuals <- drop(fit$residuals)
  dfit$gradL <- tau - (drop(fit$residuals) < 0)
  dfit$density <- density
  dfit$s <- sparsity
  dfit$iter <- max_iter
  dfit$algo <- algo
  return(dfit)
}
