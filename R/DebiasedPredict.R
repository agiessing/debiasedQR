#' This function takes .drq and .drqcv objects and returns the rank-score debiased
#' estimate of the conditional quantile function and its asymptotic variance.
#'
#' @param object   An .drq or .drqcv object obtained from drq() or drqcv().
#' @param cv_rule  Cross-validation rule to be applied to .drqcv object to select
#'                 optimal tuning parameter "\eqn{gamma >0}". Options are `1se',
#'                 `mincv', and `minfeas'. For details see docu of optGamma().
#' @param robust   Only relevant for .drqcv objects. If robust = TRUE, then median
#'                 and mean absolute deviation are used in cross-validation
#'                 procedure. If robust = FALSE, then mean and standard deviation
#'                 are used.
#'
#' @return pilot   Pilot estimate of the conditional quantile function based on
#'                 L1-penalized quantile regression vector.
#' @return debias  Debiased estimate of the conditional quantile function based
#'                 on primal variable "\eqn{w}".
#' @return dual    Debiased estimate of the conditional quantile function based
#'                 on dual variable "\eqn{v}".
#' @return avar    Estimate of the asymptotic variance of either debiased estimate
#'                 of the conditional quantile function.
#'
#' @export

debiasedPredict <- function(object, cv_rule= "1se", robust = FALSE) {

  if ( !inherits(object, c("drq", "drqcv"), which = FALSE)) {
    stop(paste0("Wrong object!"))
  }

  X <- object$X
  Y <- object$Y
  x <- object$x
  gradL <- object$gradL
  tau <- object$tau
  Z <- cbind(rep(1, n), X)

  n <- dim(X)[1]
  d <- dim(X)[2]

  sfit <- list()

  if (inherits(object, "drq") ) {

    w <- object$w
    v <- object$v
    Psi_hat <- object$psi

    mask <- rep(TRUE, n)
    mask[which(1/Psi_hat == Inf)] <- FALSE
    Psi_inv <- rep(0, n)
    Psi_inv[mask] <- 1/Psi_hat[mask]
    Psi_inv2 <- rep(0, n)
    Psi_inv2[mask] <- 1/Psi_hat[mask]^2

    sfit$pilot <- object$pilot
    sfit$debias <- object$pilot + t(w) %*% diag(Psi_inv)  %*% gradL / sqrt(n)
    sfit$dual <- object$pilot - t(v) %*% t(Z) %*% diag(Psi_hat)  %*% gradL / (2 * n)
    sfit$avar <- t(w) %*% diag(Psi_inv2) %*% w * tau * (1-tau)
    sfit$density <- object$psi

    class(sfit) <- "summary.drq"

  } else {
    gamma <- optGamma(dual_loss = object$loss, gamma_lst=object$gamma,
                      cv_fold = object$fold, cv_rule=cv_rule, robust=robust)

    Psi_hat <- object$psi
    w_hat <- primal(X = Z, x = x, Psi = diag(1/diag(Psi_hat)^2), gamma = gamma)

    if(object$algo=="ADMM") {
      dual <- dualADMM(X = Z, x = x, Psi = diag(diag(Psi_hat)^2),
                        gamma = gamma, max_iter = object$iter)
      v_hat <- dual$v
    } else if (object$algo == "CD") {
      dual <- dualCD(X = Z, x = x, Psi = diag(diag(Psi_hat)^2),
                      gamma = gamma, max_iter = object$iter)
      v_hat <- dual$v
    }

    duality_gap <- sum(abs(w_hat + drop(diag(diag(Psi_hat)^2) %*% Z %*% v_hat)
                           / (2 * sqrt(n) )) > 1/sqrt(4*n) ) #1e-03
    dual_loss_opt <- dualObj(Z, x = x, Psi = diag(diag(Psi_hat)^2), v = v_hat,
                             gamma = gamma)

    mask <- rep(TRUE, n)
    mask[which(1/diag(Psi_hat) == Inf)] <- FALSE
    Psi_inv <- rep(0, n)
    Psi_inv[mask] <- 1/diag(Psi_hat)[mask]
    Psi_inv2 <- rep(0, n)
    Psi_inv2[mask] <- 1/diag(Psi_hat)[mask]^2

    sfit$pilot <- object$pilot
    sfit$debias <- object$pilot + t(w_hat) %*% diag(Psi_inv) %*% gradL / sqrt(n)
    sfit$dual <- object$pilot - t(v_hat) %*% t(Z) %*% Psi_hat %*% gradL / (2 * n)
    sfit$avar <- t(w_hat) %*% diag(Psi_inv2) %*% w_hat * tau * (1-tau)
    sfit$avard <- t(v_hat) %*% t(Z) %*% diag(diag(Psi_hat)^2) %*% Z %*% v_hat * tau * (1-tau) / (4*n) # check!
    sfit$density <- object$psi
    sfit$gamma_opt <- gamma
    sfit$rule <- cv_rule

    class(sfit) <- "debiasedPredict"
  }

  return(sfit)
}
