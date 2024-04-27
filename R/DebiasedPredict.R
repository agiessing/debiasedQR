#' This function takes .drq and .drqcv objects and returns the rank-score debiased
#' estimate of the conditional quantile function and its asymptotic variance.
#'
#' @param object An .drq or .drqcv object obtained from drq() or drqcv().
#' @param cv_rule Cross-validation rule to be applied to .drqcv object to select
#' optimal tuning parameter "\eqn{gamma >0}". Options are `1se', `mincv', and
#' `minfeas'. For details see documentatio of optGamma().
#' @param robust  Only relevant for .drqcv objects. If robust = TRUE, then median
#' and mean absolute deviation are used in cross-validation procedure.If robust
#' = FALSE, then mean and standard deviation are used.
#'
#' @export

debiasedPredict <- function(object, cv_rule= "1se", robust = FALSE) {

  if ( (class(object) != "drq") && (class(object) != "drqcv") ) {
    stop(paste0("Wrong object!"))
  }

  X <- object$X
  Y <- object$Y
  x <- object$x
  gradL <- object$gradL

  n <- dim(X)[1]
  d <- dim(X)[2]

  sfit<- list()

  if (class(object) == "drq") {

    w <- object$w
    v <- object$v
    Psi_hat <- object$psi

    tau <- object$tau

    mask <- rep(TRUE, n)
    mask[which(1/Psi_hat == Inf)] <- FALSE
    Psi_inv <- rep(0, n)
    Psi_inv[mask] <- 1/Psi_hat[mask]
    Psi_inv2 <- rep(0, n)
    Psi_inv2[mask] <- 1/Psi_hat[mask]^2

    sfit$pilot <- object$pilot
    sfit$debias <- object$pilot + t(w) %*% diag(Psi_inv)  %*% gradL / sqrt(n)
    sfit$dual <- object$pilot - t(v) %*% t(X) %*% diag(Psi_hat)  %*% gradL / (2 * n)
    sfit$avar <- t(w) %*% diag(Psi_inv2) %*% w * tau * (1-tau)
    sfit$density <- object$psi

    class(sfit) <- "summary.drq"

  } else {
    gamma <- optGamma(dual_loss = object$loss, gamma_lst=object$gamma, cv_fold = object$fold, cv_rule=cv_rule, robust=robust)
    lambda <- lambdaBC(X=X, tau=tau)

    fit <- rq(Y~ X, tau=tau, method="lasso", lambda = lambda)
    beta_hat <- fit$coef
    q_pilot <- drop(t(x) %*% beta_hat)
    Psi_hat <- densityMatrix(Y, X, beta = beta_hat[-1], tau = tau, sparsity = object$s, density = object$density)

    w_hat <- primal(X = X, x = x[-1], Psi = diag(1/diag(Psi_hat)^2), gamma = gamma)

    if(object$algo=="ADMM") {
      v_hat <- dualADMM(X = X, x = x[-1], Psi = diag(diag(Psi_hat)^2), gamma = gamma, max_iter = object$iter)$v
    } else if (object$algo == "CD") {
      v_hat <- dualCD(X = X, x = x[-1], Psi = diag(diag(Psi_hat)^2), gamma = gamma, max_iter = object$iter)$v
    }

    duality_gap <- sum(abs(w_hat + drop(diag(diag(Psi_hat)^2) %*% X %*% v_hat) / (2 * sqrt(n) )) > 1/sqrt(4*n) ) #1e-03
    dual_loss_opt <- dualObj(X, x = x[-1], Psi = diag(diag(Psi_hat)^2), v = v_hat, gamma = gamma)

    mask <- rep(TRUE, n)
    mask[which(1/diag(Psi_hat) == Inf)] <- FALSE
    Psi_inv <- rep(0, n)
    Psi_inv[mask] <- 1/diag(Psi_hat)[mask]
    Psi_inv2 <- rep(0, n)
    Psi_inv2[mask] <- 1/diag(Psi_hat)[mask]^2

    sfit$pilot <- q_pilot
    sfit$debias <- q_pilot + t(w_hat) %*% diag(Psi_inv)  %*% (tau - (fit$res < 0)) / sqrt(n)
    sfit$dual <- q_pilot - t(v_hat) %*% t(X) %*% Psi_hat  %*% (tau - (fit$res < 0)) / (2 * n)
    sfit$avar <- t(w_hat) %*% diag(Psi_inv2) %*% w_hat * tau * (1-tau)
    sfit$gamma_opt <- gamma
    sfit$density <- diag(Psi_hat)
    sfit$rule <- cv_rule

    class(sfit) <- "debiasedPredict"
  }

  return(sfit)
}
