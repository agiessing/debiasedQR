#' @export

dualCD <- function(X, x, Psi=NULL, v_init=NULL, gamma=0.05, eps=1e-6, max_iter=5000) {
  n <- nrow(X)
  d <- ncol(X)

  if (is.null(Psi)) {
    Psi <- diag(n)
  }

  if (length(gamma) == 0) {
    gamma <- 1
  }

  A <- t(X) %*% Psi %*% X

  if (is.null(v_init)) {
    v_new <- rep(1, d)
  } else {
    v_new <- v_init
  }

  v_old <- 100 * rep(1, d)
  cnt <- 0
  flag <- 0

  while ( (norm(v_old - v_new, type = "2") > eps) && ((cnt <= max_iter) || (flag == 0))) {
    v_old <- v_new
    cnt <- cnt + 1

    # Proximal coordinate descent
    for (j in 1:d) {
      v_cur <- v_new
      mask <- rep(TRUE, d)
      mask[j] <- FALSE
      A_kj <- A[mask, j]
      v_cur <- v_cur[mask]
      v_new[j] <- SoftThres(-(A_kj %*% v_cur) / (2 * n) - x[j], lambda = gamma) / (A[j, j] / (2 * n))
    }

    if ((cnt > max_iter) && (flag == 0)) {
      message(paste0("CD algorithm has reached max. no. of iterations for gamma = ", round(gamma, 4),
                     ". Re-running CD with small perturbation to design matrix..."))
      A <- A + 1e-9*diag(d)
      cnt <- 0
      flag <- 1
    }
  }

  return(list(v = v_new))
}
